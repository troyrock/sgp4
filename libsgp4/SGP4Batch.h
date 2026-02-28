#pragma once

#include "Tle.h"
#include "OrbitalElements.h"
#include "Eci.h"
#include <vector>

namespace libsgp4
{

class SGP4Batch
{
public:
    static constexpr int BATCH_SIZE = 8;

    struct alignas(64) BatchConstants {
        // Common
        double cosio[BATCH_SIZE];
        double sinio[BATCH_SIZE];
        double eta[BATCH_SIZE];
        double t2cof[BATCH_SIZE];
        double x1mth2[BATCH_SIZE];
        double x3thm1[BATCH_SIZE];
        double x7thm1[BATCH_SIZE];
        double aycof[BATCH_SIZE];
        double xlcof[BATCH_SIZE];
        double xnodcf[BATCH_SIZE];
        double c1[BATCH_SIZE];
        double c4[BATCH_SIZE];
        double omgdot[BATCH_SIZE];
        double xnodot[BATCH_SIZE];
        double xmdot[BATCH_SIZE];

        // Near Space
        double c5[BATCH_SIZE];
        double omgcof[BATCH_SIZE];
        double xmcof[BATCH_SIZE];
        double delmo[BATCH_SIZE];
        double sinmo[BATCH_SIZE];
        double d2[BATCH_SIZE];
        double d3[BATCH_SIZE];
        double d4[BATCH_SIZE];
        double t3cof[BATCH_SIZE];
        double t4cof[BATCH_SIZE];
        double t5cof[BATCH_SIZE];

        // Elements
        double xmo[BATCH_SIZE];
        double nodeo[BATCH_SIZE];
        double omegao[BATCH_SIZE];
        double ecco[BATCH_SIZE];
        double inclo[BATCH_SIZE];
        double bstar[BATCH_SIZE];
        double aodp[BATCH_SIZE];
        double no_kozai[BATCH_SIZE];
        
        // Results (re-using buffer to avoid alloc)
        double res_x[BATCH_SIZE];
        double res_y[BATCH_SIZE];
        double res_z[BATCH_SIZE];
        double res_vx[BATCH_SIZE];
        double res_vy[BATCH_SIZE];
        double res_vz[BATCH_SIZE];
        
        bool use_simple_model[BATCH_SIZE];
        bool use_deep_space[BATCH_SIZE];
        bool active[BATCH_SIZE];
        
        DateTime epoch[BATCH_SIZE];
    };

    SGP4Batch(const std::vector<Tle>& tles);
    
    void Propagate(double tsince, std::vector<Eci>& results) const;

    // Optimized propagation using pre-allocated memory pool
    void PropagatePool(double tsince, std::vector<Eci>& pool) const;

private:
    std::vector<BatchConstants> batches_;
    int total_satellites_;
};

} // namespace libsgp4
