#include <SGP4.h>
#include <SGP4Batch.h>
#include <Tle.h>
#include <DateTime.h>
#include <Vector.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>

using namespace libsgp4;

struct HistoryPoint {
    Tle tle;
    DateTime epoch;
    Vector position;
    Vector velocity;
};

void load_history(const std::string& filename, std::vector<HistoryPoint>& history) {
    std::ifstream file(filename);
    std::string line;
    std::string line1, line2;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Clean up trailing backslashes if present
        if (line.back() == '\\') line.pop_back();
        
        if (line[0] == '1') {
            line1 = line.substr(0, 69);
        } else if (line[0] == '2') {
            line2 = line.substr(0, 69);
            try {
                Tle tle("18960", line1, line2);
                SGP4 model(tle);
                Eci eci = model.FindPosition(0.0);
                history.push_back({tle, tle.Epoch(), eci.Position(), eci.Velocity()});
            } catch (...) {
                // Skip invalid lines
            }
        }
    }
}

double distance(const Vector& v1, const Vector& v2) {
    double dx = v1.x - v2.x;
    double dy = v1.y - v2.y;
    double dz = v1.z - v2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

int main() {
    std::vector<HistoryPoint> history;
    load_history("/home/troyrock/.openclaw/workspace/sgp4_project/data/18960_history.txt", history);

    if (history.size() < 2) {
        std::cerr << "Insufficient history data!" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << history.size() << " TLEs for Object 18960." << std::endl;
    std::cout << "Starting validation from initial epoch: " << history[0].epoch << std::endl;

    // Use the first TLE as the base for long-term propagation
    Tle base_tle = history[0].tle;
    SGP4 scalar_model(base_tle);
    
    // Setup Batch (just for this one object to compare precision)
    std::vector<Tle> batch_tles = {base_tle};
    SGP4Batch batch_model(batch_tles);
    std::vector<Eci> batch_results;

    std::ofstream csv("validation_results.csv");
    csv << "MinutesSinceEpoch,ScalarError_km,BatchError_km,ScalarVsBatch_km" << std::endl;

    for (size_t i = 1; i < history.size(); ++i) {
        double tsince = (history[i].epoch - history[0].epoch).TotalMinutes();
        
        // Scalar Propagation
        Eci scalar_eci = scalar_model.FindPosition(tsince);
        double scalar_err = distance(scalar_eci.Position(), history[i].position);

        // Batch Propagation
        batch_model.Propagate(tsince, batch_results);
        double batch_err = distance(batch_results[0].Position(), history[i].position);
        
        // Internal Consistency (Scalar vs Batch)
        double consistency = distance(scalar_eci.Position(), batch_results[0].Position());

        csv << std::fixed << std::setprecision(4) 
            << tsince << "," 
            << scalar_err << "," 
            << batch_err << "," 
            << consistency << std::endl;

        if (i % 500 == 0) {
            std::cout << "Processed " << i << " points... Current Error: " << scalar_err << " km" << std::endl;
        }
    }

    std::cout << "Validation complete. Results saved to validation_results.csv" << std::endl;

    return 0;
}
