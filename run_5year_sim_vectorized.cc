#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include "Tle.h"
#include "SGP4.h"
#include "SGP4Batch.h"
#include "Interpolation.h"
#include "Globals.h"
#include "Eci.h"
#include "DateTime.h"

using namespace libsgp4;
using namespace std;

const double SIGMA_KM = 1.5; 
const double REFINEMENT_THRESHOLD_KM = 200.0; 
const double SQ_REFINEMENT_THRESHOLD = 40000.0;

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

    cout << "Starting Vectorized Interpolated Simulation..." << endl;

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
        for(auto const& [id, sat] : catalog) {
            if (abs((sim_time - sat.epoch).TotalDays()) < 30.0) {
                active_tles.push_back(Tle(sat.l1, sat.l2));
                if (active_tles.size() >= limit) break;
            }
        }

        if (active_tles.empty()) {
            sim_time = sim_time.AddDays(1); continue;
        }

        SGP4Batch batch(active_tles);
        Tle probe_tle = create_probe_tle(target_alt, target_inc, target_raan);
        SGP4 probe_prop(probe_tle);
        int n = active_tles.size();

        double daily_hazard = 0;
        vector<Eci> res0, res1;
        vector<Vector> interp_pos;

        // Optimized Step: We only propagate at the boundaries of the 30s step.
        // We use Hermite Interpolation for the 1s sub-steps during refinement.
        for (int s = 0; s < 86400; s += 30) {
            double t_min0 = (double)s / 60.0;
            double t_min1 = (double)(s + 30) / 60.0;
            
            batch.Propagate(t_min0, res0, SGP4Batch::MathMode::Standard);
            batch.Propagate(t_min1, res1, SGP4Batch::MathMode::Standard);
            
            Vector p_pos0 = probe_prop.FindPosition(t_min0).Position();
            Vector p_pos1 = probe_prop.FindPosition(t_min1).Position();

            #pragma omp parallel for reduction(+:daily_hazard)
            for (int i = 0; i < n; i++) {
                // Coarse distance check at start of interval
                Vector s_pos0 = res0[i].Position();
                double dx = p_pos0.x - s_pos0.x;
                double dy = p_pos0.y - s_pos0.y;
                double dz = p_pos0.z - s_pos0.z;
                double d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < SQ_REFINEMENT_THRESHOLD) {
                    // Refine using INTERPOLATION instead of SGP4
                    // Spline coefficients for the 30s window
                    double min_d2 = d2;
                    const Eci& e0 = res0[i];
                    const Eci& e1 = res1[i];
                    
                    // Velocity at boundaries (km/s)
                    const Vector& v0 = e0.Velocity();
                    const Vector& v1 = e1.Velocity();

                    // Probe also needs interpolation or 1s steps
                    // For simplicity, we'll use 1s steps for the probe (only 1 object)
                    for (int rs = 1; rs < 30; rs++) {
                        double u = (double)rs / 30.0;
                        double u2 = u * u; double u3 = u2 * u;
                        double h00 = 2*u3 - 3*u2 + 1;
                        double h10 = u3 - 2*u2 + u;
                        double h01 = -2*u3 + 3*u2;
                        double h11 = u3 - u2;

                        // Interpolated Satellite Position
                        Vector rsat(
                            h00 * e0.Position().x + h10 * 30.0 * v0.x + h01 * e1.Position().x + h11 * 30.0 * v1.x,
                            h00 * e0.Position().y + h10 * 30.0 * v0.y + h01 * e1.Position().y + h11 * 30.0 * v1.y,
                            h00 * e0.Position().z + h10 * 30.0 * v0.z + h01 * e1.Position().z + h11 * 30.0 * v1.z
                        );

                        // Interpolated Probe Position (or exact if cheap)
                        double p_dt = t_min0 + (double)rs/60.0;
                        Vector rp = probe_prop.FindPosition(p_dt).Position();

                        double rdx = rp.x - rsat.x, rdy = rp.y - rsat.y, rdz = rp.z - rsat.z;
                        double rd2 = rdx*rdx + rdy*rdy + rdz*rdz;
                        if (rd2 < min_d2) min_d2 = rd2;
                    }
                    daily_hazard += exp(-min_d2 / (2.0 * SIGMA_KM * SIGMA_KM));
                }
            }
        }
        cout << "Simulated: " << sim_time.ToString() << " (H: " << daily_hazard << ", N: " << n << ")" << endl;
        sim_time = sim_time.AddDays(1);
    }
    return 0;
}
