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
#include "Globals.h"
#include "Eci.h"
#include "DateTime.h"

using namespace libsgp4;
using namespace std;

const double SIGMA_KM = 1.5; 
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

    cout << "Starting 5-Day Optimized Simulation..." << endl;

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
        vector<SGP4*> props;
        for(auto const& [id, sat] : catalog) {
            if (abs((sim_time - sat.epoch).TotalDays()) < 30.0) {
                active_tles.push_back(Tle(sat.l1, sat.l2));
                props.push_back(new SGP4(active_tles.back()));
            }
        }

        if (active_tles.empty()) {
            sim_time = sim_time.AddDays(1);
            continue;
        }

        Tle probe_tle = create_probe_tle(target_alt, target_inc, target_raan);
        SGP4 probe_prop(probe_tle);
        int n = active_tles.size();

        double daily_hazard = 0;
        for (int s = 0; s < 86400; s += 30) {
            DateTime t = sim_time.AddSeconds(s);
            Vector p_pos;
            try { p_pos = probe_prop.FindPosition((t - probe_tle.Epoch()).TotalMinutes()).Position(); } catch(...) { continue; }
            
            #pragma omp parallel for reduction(+:daily_hazard)
            for (int i = 0; i < n; i++) {
                try {
                    double dt_min = (t - active_tles[i].Epoch()).TotalMinutes();
                    Vector s_pos = props[i]->FindPosition(dt_min).Position();
                    
                    double dx = p_pos.x - s_pos.x;
                    double dy = p_pos.y - s_pos.y;
                    double dz = p_pos.z - s_pos.z;
                    double d2 = dx*dx + dy*dy + dz*dz;

                    if (d2 < SQ_REFINEMENT_THRESHOLD) {
                        double min_d2 = d2;
                        for (int rs = 1; rs < 30; rs++) {
                            double rdt = dt_min + (double)rs/60.0;
                            double pdt = (t.AddSeconds(rs) - probe_tle.Epoch()).TotalMinutes();
                            Vector rp = probe_prop.FindPosition(pdt).Position();
                            Vector rsat = props[i]->FindPosition(rdt).Position();
                            double rdx = rp.x - rsat.x, rdy = rp.y - rsat.y, rdz = rp.z - rsat.z;
                            double rd2 = rdx*rdx + rdy*rdy + rdz*rdz;
                            if (rd2 < min_d2) min_d2 = rd2;
                        }
                        daily_hazard += exp(-min_d2 / (2.0 * SIGMA_KM * SIGMA_KM));
                    }
                } catch(...) {}
            }
        }
        cout << "Simulated: " << sim_time.ToString() << " (H: " << daily_hazard << ", N: " << n << ")" << endl;
        sim_time = sim_time.AddDays(1);
        for(auto p : props) delete p;
    }
    return 0;
}
