#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include "Tle.h"
#include "SGP4.h"
#include "SGP4Batch.h"
#include "Interpolation.h"
#include "SpatialPartition.h"
#include "TemporalPruner.h"
#include "Globals.h"
#include "Eci.h"
#include "DateTime.h"

using namespace libsgp4;
using namespace std;

const double SIGMA_KM = 1.5; 
const double COARSE_THRESHOLD_KM = 50.0; 
const double MAX_REL_VEL_KMS = 16.0;

struct SatState {
    int id;
    DateTime epoch;
    string l1, l2;
};

Tle create_probe_tle(double alt, double inc, double raan) {
    double MU = 398600.4418;
    double RE = 6378.137;
    double a = RE + alt;
    double n_rad_sec = sqrt(MU / pow(a, 3));
    double n_rev_day = n_rad_sec * 86400.0 / (2.0 * kPI);
    char l1[75], l2[75];
    sprintf(l1, "1 99999U 20001A   20001.00000000  .00000000  00000-0  00000-0 0  9990");
    sprintf(l2, "2 99999 %8.4f %8.4f 0000001 %8.4f %8.4f %11.8f    00", inc, raan, 0.0, 0.0, n_rev_day);
    return Tle(string(l1).substr(0,69), string(l2).substr(0,69));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <tle_file> [limit]" << endl;
        return 1;
    }

    double target_alt = 751.0;
    double target_inc = 37.0;
    double target_raan = 0.0;
    DateTime sim_time(2024, 1, 1);
    DateTime end_time(2024, 1, 5);
    
    map<int, SatState> catalog;
    string fname = argv[1];
    ifstream infile(fname);
    string l1, l2, line;
    int limit = (argc > 2) ? stoi(argv[2]) : 5000;

    // Track "sleeping" pairs to skip checks
    unordered_map<int, int> probe_sleep_map;

    cout << "Starting Temporal-Optimized Simulation..." << endl;

    while (sim_time < end_time) {
        while (getline(infile, line)) {
            if (line.empty()) continue;
            if (line[0] == '1') l1 = line;
            else if (line[0] == '2') {
                l2 = line;
                try {
                    Tle tle(l1.substr(0,69), l2.substr(0,69));
                    if (tle.Epoch() > sim_time.AddDays(1)) break;
                    int id = tle.NoradNumber();
                    catalog[id] = {id, tle.Epoch(), l1.substr(0,69), l2.substr(0,69)};
                } catch(...) {}
            }
        }

        vector<Tle> active_tles;
        vector<int> sat_ids;
        for(auto const& [id, sat] : catalog) {
            if (abs((sim_time - sat.epoch).TotalDays()) < 30.0) {
                active_tles.push_back(Tle(sat.l1, sat.l2));
                sat_ids.push_back(id);
                if (active_tles.size() >= (size_t)limit) break;
            }
        }

        if (active_tles.empty()) { sim_time = sim_time.AddDays(1); continue; }

        SGP4Batch batch(active_tles);
        Tle probe_tle = create_probe_tle(target_alt, target_inc, target_raan);
        SGP4 probe_prop(probe_tle);
        int n = active_tles.size();
        double daily_hazard = 0;

        for (int s = 0; s < 86400; s += 30) {
            DateTime t = sim_time.AddSeconds(s);
            Vector p_pos0 = probe_prop.FindPosition((t - probe_tle.Epoch()).TotalMinutes()).Position();
            
            vector<Eci> res0, res1;
            batch.Propagate((t - sim_time).TotalMinutes(), res0);
            batch.Propagate((t.AddSeconds(30) - sim_time).TotalMinutes(), res1);

            #pragma omp parallel for reduction(+:daily_hazard)
            for (int i = 0; i < n; i++) {
                int sid = sat_ids[i];
                
                // --- PHYSICS OPTIMIZATION: Temporal Sleep ---
                // Only check if sleep counter is zero
                bool skip = false;
                #pragma omp critical(sleep_access)
                {
                    if (probe_sleep_map.count(sid) && probe_sleep_map[sid] > 0) {
                        probe_sleep_map[sid]--;
                        skip = true;
                    }
                }
                if (skip) continue;

                Vector s_pos0 = res0[i].Position();
                Vector delta = p_pos0 - s_pos0;
                double d2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                double d = sqrt(d2);

                if (d < COARSE_THRESHOLD_KM) {
                    // Refine using Interpolation
                    const Eci& e0 = res0[i]; const Eci& e1 = res1[i];
                    const Vector& v0 = e0.Velocity(); const Vector& v1 = e1.Velocity();
                    double min_d2 = d2;
                    for (int rs = 1; rs < 30; rs++) {
                        double u = (double)rs / 30.0;
                        double u2 = u * u; double u3 = u2 * u;
                        double h00 = 2*u3 - 3*u2 + 1; double h10 = u3 - 2*u2 + u;
                        double h01 = -2*u3 + 3*u2; double h11 = u3 - u2;
                        Vector rsat(
                            h00 * e0.Position().x + h10 * 30.0 * v0.x + h01 * e1.Position().x + h11 * 30.0 * v1.x,
                            h00 * e0.Position().y + h10 * 30.0 * v0.y + h01 * e1.Position().y + h11 * 30.0 * v1.y,
                            h00 * e0.Position().z + h10 * 30.0 * v0.z + h01 * e1.Position().z + h11 * 30.0 * v1.z
                        );
                        double p_dt = (t.AddSeconds(rs) - probe_tle.Epoch()).TotalMinutes();
                        Vector rp = probe_prop.FindPosition(p_dt).Position();
                        Vector rd = rp - rsat;
                        double rd2 = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
                        if (rd2 < min_d2) min_d2 = rd2;
                    }
                    daily_hazard += exp(-min_d2 / (2.0 * SIGMA_KM * SIGMA_KM));
                } else {
                    // Calculate sleep steps for this satellite
                    int steps = TemporalPruner::EstimateSleepSteps(d, COARSE_THRESHOLD_KM, 30.0, MAX_REL_VEL_KMS);
                    if (steps > 0) {
                        #pragma omp critical(sleep_access)
                        probe_sleep_map[sid] = steps;
                    }
                }
            }
        }
        cout << "Simulated: " << sim_time.ToString() << " (H: " << daily_hazard << ", N: " << n << ")" << endl;
        sim_time = sim_time.AddDays(1);
    }
    return 0;
}
