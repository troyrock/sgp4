#pragma once

#include "Tle.h"
#include <vector>
#include "Eci.h"
#include "DateTime.h"
#include <memory>

namespace libsgp4
{

/**
 * @brief SGP4Batch - High-performance vectorized SGP4 propagator.
 */
class SGP4Batch
{
public:
    enum class MathMode {
        Standard,   // libmvec (High precision)
        Minimax     // Fast Polynomial Approx (~1e-12 precision)
    };

    SGP4Batch(const std::vector<Tle>& tles);
    void Propagate(double tsince, std::vector<Eci>& results, MathMode mode = MathMode::Standard) const;

    int total_satellites() const { return total_satellites_; }

private:
    int total_satellites_;
    int padded_satellites_;

    template<typename T>
    struct AlignedDeleter {
        void operator()(T* p) const { free(p); }
    };

    template<typename T>
    using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

    template<typename T>
    static AlignedPtr<T> make_aligned(size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, size * sizeof(T)) != 0) throw std::bad_alloc();
        return AlignedPtr<T>(static_cast<T*>(ptr));
    }

    // SoA storage
    AlignedPtr<double> cosio, sinio, eta, t2cof, x1mth2, x3thm1, x7thm1, aycof, xlcof, xnodcf, c1, c4, omgdot, xnodot, xmdot;
    AlignedPtr<double> c5, omgcof, xmcof, delmo, sinmo, d2, d3, d4, t3cof, t4cof, t5cof;
    AlignedPtr<double> xmo, nodeo, omegao, ecco, inclo, bstar, aodp, no_kozai;
    AlignedPtr<double> use_simple_model, use_deep_space, active;

    std::vector<DateTime> epochs;

    mutable AlignedPtr<double> last_tsince, memo_e, memo_a, memo_omega, memo_xl, memo_xnode;
};

} // namespace libsgp4
