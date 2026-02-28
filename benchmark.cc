#include <SGP4.h>
#include <SGP4Batch.h>
#include <Tle.h>
#include <DateTime.h>
#include <Vector.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace libsgp4;
using namespace std::chrono;

void load_tles(const std::string& filename, std::vector<Tle>& tles) {
    std::ifstream file(filename);
    std::string line;
    std::string line1, line2;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == '1') {
            line1 = line.substr(0, 69);
        } else if (line[0] == '2') {
            line2 = line.substr(0, 69);
            try {
                tles.emplace_back("Benchmark", line1, line2);
            } catch (...) {}
        }
    }
}

int main(int argc, char* argv[]) {
    std::vector<Tle> tles;
    load_tles("SGP4-VER.TLE", tles);

    if (tles.empty()) {
        std::cerr << "No TLEs loaded!" << std::endl;
        return 1;
    }

    // Scale up for significant results
    std::vector<Tle> scaled_tles;
    for(int i=0; i<300; i++) scaled_tles.insert(scaled_tles.end(), tles.begin(), tles.end());
    tles = scaled_tles;

    int iterations = 1000;
    if (argc > 1) iterations = std::stoi(argv[1]);

    std::cout << "Benchmarking " << tles.size() << " satellites and " << iterations << " iterations each..." << std::endl;

    // Batch Benchmark (Standard results.resize inside)
    {
        SGP4Batch batch(tles);
        std::vector<Eci> results;
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            batch.Propagate(static_cast<double>(i), results);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        std::cout << "[Standard Batch] Time: " << duration/1e6 << "s" << std::endl;
    }

    // Pool Benchmark (Pre-allocated outside)
    {
        SGP4Batch batch(tles);
        std::vector<Eci> pool(tles.size(), Eci(DateTime(), Vector()));
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            batch.PropagatePool(static_cast<double>(i), pool);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        std::cout << "[Pool Batch]     Time: " << duration/1e6 << "s" << std::endl;
    }

    return 0;
}
