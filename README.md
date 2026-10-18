# chiplet-thermal

FP64 finite-difference solver for steady-state heat conduction in a 2D
cross-section of a 2.5D chiplet package. The model maps compute-die and HBM
power to temperature through a layered substrate, interposer, die, and TIM
stack. Serial-host and HIP-API GPU Jacobi implementations are included.

The solver is checked against an analytic 1D thermal-resistance limit and an
energy-conservation condition. The 2D calculation resolves lateral spreading and
hot-spot movement that a 1D model cannot capture.

## Verification

- In the laterally uniform 1D limit, the temperature profile matches the
  series-resistance/parabolic solution within discretization error.
- Integrated outward heat flux equals total input power within 1%.
- Convergence is reported as max |Delta T| per iteration with a 1e-6 K
  threshold; non-convergence returns a nonzero status.
- GPU and host temperature fields are compared directly.

## Default results

Recorded 2026-10-11 with CUDA 13.2 on an RTX 5060 Laptop GPU (`sm_120`). The
grid has 600 x 36 cells with h=0.05 mm. Host and GPU use the same FP64 Jacobi
iteration. CTest passes 1/1 with ten checks and takes about 14 seconds.

- 1D analytic limit, with the die widened to 30 mm, 100 W compute power, and HBM
  off: T_max=391.4200 K versus 391.4234 K analytic, an error of 0.0047%.
- Default energy balance at 100 W compute plus 20 W HBM:
  P_in=120.000000 W and Q_top=119.99963 W, a residual of 0.0003%.
- Default 2D case: the compute-die hot spot is 535.5 K at
  (x=15.1, z=1.12) mm, at the die/interposer interface. The HBM peak is 449.7 K.
- Host/GPU field differences: max|dT|=3.1e-5 K in the 1D case and 5.0e-10 K in
  the default 2D case.

The power sweep covers 25 to 150 W in eight steps with HBM fixed at 20 W. Every
point converged. The first point took about 155k cold sweeps; subsequent points
took 35-42k warm-started sweeps. Warm starts scale the previous solution but are
only initial guesses because HBM power stays fixed. Cold re-solves at 25 W and
150 W reproduce the converged values.

Least-squares sweep slopes are 2.15 K/W for the compute die and 0.72 K/W for
HBM; endpoint secants are 2.14 and 0.68 K/W. The curves are piecewise linear
because the maximum is taken over cells and the argmax cell changes. Below about
43 W, the HBM hot cell is its own center and its curve is nearly flat. At higher
compute power, the hottest HBM cell faces the compute die and the slope rises to
0.79 K/W.

Below about 40 W compute power, HBM is the package hot spot: at 25 W, HBM reaches
404 K while the compute die reaches 376 K. This follows from the fixed 20 W HBM
source and its low effective conductivity, 0.020 W/(mm·K).

The default artifacts are `data/field_100W.csv`,
`data/field_reference_100W.csv`, `data/sweep.csv`, `data/heatmap_100W.png`, and
`data/sweep.png`.

## TIM conductivity comparison

`Config::k_tim` controls the TIM conductivity; the default is 0.005 W/(mm·K),
or 5 W/m·K. Two full GPU sweeps compared that polymer-like baseline with
0.086 W/(mm·K), the 86 W/m·K bulk-indium value published by Indium Corporation.
All eight matched power points converged in both runs. The merged artifacts are
`data/tim_comparison.csv` and `data/tim_comparison.png`, produced by
`tools/compare_tim.py`.

Selected compute-die results:

| Compute power | Polymer-like 5 W/m·K | Indium bulk 86 W/m·K | Reduction |
|--:|--:|--:|--:|
| 25 W | 376.39 K | 329.00 K | 47.39 K |
| 100 W (host reference) | 535.55 K | 350.16 K | 185.39 K |
| 150 W | 644.01 K | 365.84 K | 278.17 K |

At 150 W, the HBM maximum falls from 489.26 K to 360.05 K. The indium host/GPU
100 W field difference is max|dT|=2.99e-5 K. The test suite also includes a
nondefault-TIM analytic case, an energy check, and a fresh-GPU/preserved-host
field regression. Its 1e-4 K field tolerance is an empirical regression bound
for this deterministic configuration, approximately 3.3 times the measured
difference.

The host and GPU fields are preserved in `data/tim_indium_100W/` with scenario
parameters, hashes, convergence results, and reproduction commands. A fresh
reproduction matched both fields exactly.

The 86 W/m·K value is [bulk conductivity](https://www.indium.com/products/thermal-interface-materials/solder-tims/),
not total thermal resistance. Neither TIM case represents a qualified commercial
stack.

## Residual-window timing

`DevSolve::run` reports host wall time, HIP-event time, kernel launches, and
residual readbacks. The `--window N` option changes the residual-check frequency;
the default is 32. Power points are not batched.

A 100 W indium solve was repeated three times at each window. All runs converged
to the same hot-spot values.

| Window | Iterations | Residual readbacks | Wall ms, median [min,max] |
|--:|--:|--:|--:|
| 32 | 135,520 | 4,235 | 667.388 [627.935, 736.388] |
| 64 | 135,552 | 2,118 | 1,006.155 [794.739, 1,352.049] |
| 128 | 135,552 | 1,059 | 543.377 [529.735, 561.360] |

Window 128 used one quarter as many residual readbacks as window 32 and reduced
median wall time by 18.6%. Residual checks therefore have measurable overhead,
but they do not dominate this solve. The variability at window 64 prevents a
monotonic-scaling or optimal-window conclusion, and the measurement does not
isolate launch overhead. Full event timings and commands are in
`data/tim_indium_100W/MANIFEST.md`.

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

The GPU source uses the HIP API through `include/hip_shim.h`. Recorded numerical
results use CUDA 13.2 on an RTX 5060 Laptop GPU; thermal AMD runtime validation
has not been performed.

## Repository layout

- `docs/THERMAL.md`: geometry, materials, boundary conditions, and numerics
- `include/hip_shim.h`: HIP-API to CUDA-runtime development shim
- `src/thermal_reference.cpp`: host reference solver
- `src/thermal_batched.cu`: HIP-API GPU solver
- `tools/plot_thermal.py`: heatmap and hot-spot sweep plot
- `tools/compare_tim.py`: matched TIM sweep merger and comparison plot
- `data/`: numerical outputs and plots

## Model limits

The model covers steady-state 2D conduction with a uniform TIM conductivity.
Convection, transient response, 3D geometry, per-layer TIM stacks, contact
resistance, package warpage, and liquid-cooling correlations are not modeled.

The 2D slice carries each die's full power through 1 mm of unit depth, which
raises the areal power density and absolute temperatures relative to a real 3D
package. Use the Kelvin values for comparisons within the model; the useful
outputs are hot-spot placement and lateral spreading under the stated boundary
conditions.
