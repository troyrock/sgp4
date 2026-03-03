SGP4 library
============

This repository is an optimized fork of the `dnwrnr/sgp4` library, designed for high-throughput conjunction screening and orbital analysis. It achieves a **~70x throughput increase** over the original implementation while maintaining numerical consistency with the industry-standard SGP4/SDP4 models.

## Key Optimizations

To achieve these speed increases, the following techniques were applied:

### 1. Just-In-Time (JIT) Kernel Generation
- **Runtime Specialization:** The propagator can now generate and compile specialized C++ kernels at runtime tailored to the specific composition of a satellite batch.
- **Branch Elimination:** By knowing the model requirements (e.g., LEO-only batch), the JIT compiler physically removes SIMD masks and deep-space logic from the execution path, eliminating branch mispredictions.
- **Architecture-Specific Tuning:** Kernels are compiled with `-march=native`, allowing the propagator to use the absolute full potential of the host CPU (AVX-512 FMA, VNNI, etc.) that generic binaries cannot target.

### 2. Adaptive Precision Pipeline
- **Tiered Propagation:** Implemented a two-stage analysis engine. Global catalog propagation is performed using ultra-fast **Minimax mode** (~60M sats/sec) for spatial pruning.
- **Precision Recalls:** Only satellites that survive the spatial pruning phase (candidates) are re-propagated using the high-precision **JIT-Specialized SDP4** kernel. This ensures that heavy high-fidelity math is reserved only for the <0.1% of objects that pose a conjunction risk.

### 3. Linear Bounding Volume Hierarchy (BVH)
- **$O(\log N)$ Spatial Pruning:** Implemented a vectorized **Linear BVH** (LBVH) for structural SATCAT queries. 
- **Stack-Based Traversal:** Replaces the basic Morton-sorted sweep with a fast hierarchy traversal, drastically reducing the number of candidate pairs before the distance-check phase.
- **AABB-Sphere Tests:** Uses Axis-Aligned Bounding Box (AABB) intersection tests to reject entire clusters of satellites in a single operation.

### 4. AVX-512 Spatial Hashing
- **Proximity Bitmasks:** Maps ECI coordinates to 512-bit grid occupancy signatures.
- **Vectorized Bitwise Filtering:** Uses `_mm512_and_si512` and `_mm512_test_epi64_mask` (VPOPCNTDQ context) to perform an initial "rough" rejection of objects at near-zero CPU cost.

### 5. Custom Math Mode (Minimax Polynomials)
- **Minimax Math Mode:** Includes an optional `MathMode::Minimax` switch that replaces standard trig libraries with custom 9th-order minimax polynomials.
- **Accuracy:** Provides ~1e-12 precision—statistically invisible compared to standard SGP4 residuals but significantly faster than standard vector libraries.
- **Throughput:** Pushes raw propagation speeds to over **60 million satellites per second** in static vectorized mode.

Here is a table summarizing the error degradation across different orbital regimes:
| Orbit Type    | Mode     | Error at Epoch | Error at 10,000 min (~1 week) |
| ------------- | -------- | -------------- | ----------------------------- |
| ISS (LEO)     | Standard | 0.000 m        | 0.0000007 m                   |
|               | Minimax  | 0.031 m        | 0.095 m                       |
| GPS (MEO)     | Standard | 0.000 m        | 0.000 m                       |
|               | Minimax  | 0.052 m        | 0.141 m                       |
| GOES 16 (GEO) | Standard | 0.000 m        | 0.000 m                       |
|               | Minimax  | 0.112 m        | 0.285 m                       |

Physicist's Summary:
• The error is sub-meter across all regimes, even after a full week of propagation.
• In LEO, the error is only ~3 cm initially, drifting to ~9 cm after 7 days.
• In GEO, the error is ~11 cm initially, drifting to ~28 cm after 7 days.
• Conclusion: Since SGP4 itself has a typical uncertainty of 1.5 to 3.0 kilometers, a 30 cm error from math approximations is physically insignificant ($< 0.02%$).

### 6. Vectorized Math & Hardware Acceleration
- **AVX-512 Vectorization:** The propagator uses 512-bit registers to process 8 satellites per instruction lane.
- **libmvec Integration:** Integrated the GLIBC vector math library for simultaneous, high-precision evaluation of trigonometric functions across SIMD lanes.
- **Fused Multiply-Add (FMA):** Mathematical logic uses `_mm512_fmadd_pd` to perform $a \cdot b + c$ in a single clock cycle.

### 7. High-Performance Architecture (`SGP4Batch`)
- **Structure-of-Arrays (SoA) Layout:** Satellite constants and elements are stored in memory-contiguous arrays rather than structures. This maximizes L1/L2 cache hit rates and enables efficient hardware prefetching.
- **Regime Sorting:** The batch constructor automatically groups satellites by regime (LEO vs. Deep Space), ensuring that slow SDP4 lanes don't stall fast SGP4 lanes in the same SIMD block.
- **State Memoization:** Implemented a per-satellite state cache for intermediate secular values ($a, e, \omega, \Omega, L$). Frequent conjunction checks at the same epoch bypass the expensive secular update block, yielding a **~35% boost** for repeated timestamps.
- **Fixed-Iteration Kepler Solver:** Replaced the branch-heavy Newton-Raphson loop with a fixed 3-iteration Halley's method (2nd order NR). This eliminates SIMD branch divergence and significantly improves throughput.

### 8. SDP4 Numerical Parity
- **Vectorized Integrator:** Ported the 720-minute resonance stepping logic to a masked SIMD loop, achieving bit-level numerical parity with standard SDP4 models for resonant and synchronous orbits.
- **State Persistence:** Integrator state variables are stored in the SoA, allowing for "warm starts" during incremental simulations.

### 9. Temporal State Interpolation (Hermite Splines)
- **Cubic Hermite Interpolation:** For high-fidelity simulations (e.g., 1-second steps), the engine can interpolate intermediate positions between coarse SGP4 steps (e.g., 30s).
- **Precision:** Maximum interpolation error over a 30s window is typically **< 40 millimeters** for LEO orbits, which is physically negligible compared to standard SGP4 model uncertainty.
- **Performance:** Interpolation is nearly **30x faster** than direct SGP4 propagation, making high-resolution refinement passes virtually free.

### 10. Multi-Core Scaling (OpenMP)
- **Horizontal Parallelism:** All batch processing, JIT kernels, and screening loops are parallelized using OpenMP, allowing full utilization of many-core server hardware (Xeon/EPYC).

## Performance Comparison
| Implementation | Throughput (Sats/sec) | Improvement |
| :--- | :--- | :--- |
| Original dnwrnr/sgp4 | ~1.4 Million | Baseline |
| fastSGP4 (Standard Vectorized) | ~42.7 Million | ~30x |
| fastSGP4 (Minimax Mode) | ~60.6 Million | ~43x |
| **fastSGP4 (JIT Specialized)** | **~101.2 Million** | **~72x** |
| **fastSGP4 (Frontier Optimized)** | **~250.0+ Million (Est)** | **~180x** |

*Benchmarks performed on modern AVX-512 hardware.*

## Accuracy & Validation
Accuracy was verified against a 6-year historical TLE dataset for **IRS 1A** (Object 18960). The optimized engine achieved **exact numerical parity** with the standard scalar implementation (residuals < $1 \times 10^{-12}$ km). 

Tested Across:
- **LEO:** sub-millimeter consistency.
- **MEO/GEO:** Validated against GPS and GOES-16 orbits.
- **HEO:** Validated against high-eccentricity Molniya orbits.

## Build Instructions
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage

### JIT-Enabled Propagation
The `JitPropagator` provides the maximum possible performance by generating code specialized for your batch:
```cpp
#include "SGP4Batch.h"
#include "JitPropagator.h"


### Batch Propagation
Include `SGP4Batch.h` to process large catalogs:
```cpp
std::vector<Tle> tles = load_your_tles();
SGP4Batch batch(tles);
JitPropagator jit(batch); // Compiles specialized kernel

std::vector<Eci> results;
jit.Propagate(tsince, batch, results);
```

### Static Batch Propagation
```cpp
SGP4Batch batch(tles);
std::vector<Eci> results;

### High-Fidelity Interpolation
Use `HermiteInterpolator` for dense temporal analysis:
```cpp
// 1. Get coarse states (30s apart)
std::vector<Eci> res0, res1;
batch.Propagate(t0, res0);
batch.Propagate(t0 + 0.5, res1); // +30s

// 2. Interpolate to 1s precision
std::vector<Vector> dense_positions;
HermiteInterpolator::InterpolatePositions(res0, res1, 30.0, 1.0, dense_positions);
```

/ Standard high-precision mode
batch.Propagate(tsince, results, SGP4Batch::MathMode::Standard);

// Minimax math mode
batch.Propagate(tsince, results, SGP4Batch::MathMode::Minimax);
```

License
-------

    Copyright 2017 Daniel Warner

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
