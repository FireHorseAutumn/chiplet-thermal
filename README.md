# chiplet-thermal

FP64 finite-difference solver for steady-state heat conduction in a 2D
cross-section of a 2.5D chiplet package. The model maps compute-die and HBM
power to temperature through a layered substrate, interposer, die, and TIM
stack. Serial-host and HIP-API GPU Jacobi implementations are included.

## Status

- CTest 1/1 with sixteen checks, about 14 seconds: analytic limit,
  energy balance, host/GPU field agreement, nondefault-TIM cases, and a
  fresh-GPU/preserved-host field regression.
- Recorded with CUDA 13.2 on an RTX 5060 Laptop GPU (`sm_120`), 600 x 36
  grid at h=0.05 mm, host and GPU on the same FP64 Jacobi iteration.

## Headline: TIM conductivity

Two full GPU sweeps, a polymer-like baseline (0.005 W/(mm·K), 5 W/m·K)
against the 86 W/m·K bulk-indium value published by Indium Corporation.
All eight matched power points converged in both runs.

| Compute power | Polymer-like 5 W/m·K | Indium bulk 86 W/m·K | Reduction |
|--:|--:|--:|--:|
| 25 W | 376.39 K | 329.00 K | 47.39 K |
| 100 W (host reference) | 535.55 K | 350.16 K | 185.39 K |
| 150 W | 644.01 K | 365.84 K | 278.17 K |

At 150 W the HBM maximum falls from 489.26 K to 360.05 K.

## Key numbers

- 1D analytic limit (die widened to 30 mm, 100 W, HBM off):
  T_max=391.4200 K versus 391.4234 K analytic, an error of 0.0047%.
- Default energy balance (100 W compute + 20 W HBM): P_in=120.000000 W,
  Q_top=119.99963 W, residual 0.0003%.
- Default 2D case: compute-die hot spot 535.5 K at (x=15.1, z=1.12) mm,
  at the die/interposer interface; HBM peak 449.7 K.
- Host/GPU field agreement: max|dT|=3.1e-5 K (1D) and 5.0e-10 K (2D);
  the indium 100 W field agrees to 2.99e-5 K.
- Sweep slopes: 2.15 K/W compute die, 0.72 K/W HBM (least squares).

## Honest limits

- The 2D slice carries each die's full power through 1 mm of unit depth,
  raising areal power density and absolute temperatures relative to a
  real 3D package. Use the Kelvin values for comparisons within the
  model; the useful outputs are hot-spot placement and lateral spreading
  under the stated boundary conditions.
- Residual-window timing: window 128 used one quarter of window 32's
  readbacks and cut median wall time 18.6%, but window-64 variability
  blocks any monotonic-scaling or optimal-window claim.
- Not modeled: convection, transient response, 3D geometry, per-layer
  TIM stacks, contact resistance, package warpage, liquid-cooling
  correlations. The 86 W/m·K value is bulk conductivity, not total
  thermal resistance; neither TIM case is a qualified commercial stack.
- The GPU source uses the HIP API through `include/hip_shim.h`; thermal
  AMD runtime validation has not been performed.

## Build and test

Requirements: CMake 3.20+, a CUDA toolchain, and Python 3 with NumPy and
Matplotlib for plots.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Reproduce the matched TIM sweeps and comparison artifacts:

```bash
./build/thermal_batched --k-tim 0.005 --out data/sweep_polymer.csv
./build/thermal_batched --k-tim 0.086 --out data/sweep_indium.csv
python3 tools/compare_tim.py
```

## Repository layout

- `docs/THERMAL.md`: geometry, materials, boundary conditions, and numerics
- `include/hip_shim.h`: HIP-API to CUDA-runtime development shim
- `src/thermal_reference.cpp`: host reference solver
- `src/thermal_batched.cu`: HIP-API GPU solver
- `tools/plot_thermal.py`: heatmap and hot-spot sweep plot
- `tools/compare_tim.py`: matched TIM sweep merger and comparison plot
- `data/`: numerical outputs and plots

## Deeper reading

- `docs/RUNGS.md`: the full record: every verification, the sweep
  narrative, the window-timing table.
- `data/tim_indium_100W/MANIFEST.md`: preserved host/GPU indium fields
  with scenario parameters, hashes, and full event timings.
