# Thermal model specification

The solver computes steady-state heat conduction in a 2D vertical cross-section
of a 2.5D package. Units are mm, W, K, and W/(mm·K). The default ambient and
top-boundary temperature is T_amb=318 K (45 °C).

## Equation and discretization

With no convection inside the solid,

    div(k(x,z) grad T) = -q'''(x,z).

The uniform finite-difference grid has dx=dz=h. Harmonic-mean conductivities on
cell bonds enforce flux continuity across material interfaces:

    sum_bonds k_half (T_nb - T_c)/h² = -q'''_c.

The implementation uses a double-buffered FP64 Jacobi update and stops when
max|T_new-T| < 1e-6 K.

## Geometry and materials

The default stack is 30 mm wide and 1.8 mm tall. The bottom is the substrate/PCB
side and the top is the cold plate.

| Layer | Thickness | k [W/(mm·K)] | Notes |
|---|---:|---:|---|
| Substrate (organic) | 1.0 mm | 0.020 | Passive |
| Interposer (Si) | 0.1 mm | 0.148 | Passive |
| HBM stack (left) | 0.6 mm | 0.020 effective | 8 mm wide, 20 W default |
| Compute chiplet | 0.6 mm | 0.148 | 10 mm wide, 100 W default |
| Gap/mold between dies | 0.6 mm | 0.001 | Fills non-die width |
| TIM | 0.1 mm | 0.005 default | Configurable as `Config::k_tim` |
| Cold-plate boundary | — | — | Dirichlet T=T_amb |

The compute chiplet and HBM sit side by side on the interposer in the same z
band. Volumetric heating is uniform within each die and equals power divided by
die volume. The top boundary is Dirichlet at T_amb; the sides and bottom are
adiabatic, so all heat leaves through the top.

## Verification

1. **1D analytic limit.** Widen the dies to span the full width and disable HBM,
   giving laterally uniform heating. With an adiabatic bottom, all heat conducts
   through the TIM to the top:

       T_max = T_amb + (P/A) (t_TIM/k_TIM + t_die/(2 k_die)).

   The `t_die/(2 k_die)` term is the rise from the die's top surface to its
   hottest point at the zero-flux bottom. The layer average would instead carry
   `t_die/(3 k_die)`. The numerical and analytic values must agree to about 1%
   or within a stated discretization estimate.
2. **Energy conservation.** Integrated outward flux across the top boundary must
   equal total input power within 1%.
3. **Hot-spot behavior.** Peaks must colocate with source footprints. At default
   and high compute power, the compute die is hottest. Below roughly 40 W, fixed
   20 W HBM self-heating can make HBM the package hot spot.
4. **Nondefault TIM.** With `k_tim=0.086 W/(mm·K)`, the laterally uniform 1D case
   must satisfy the same analytic and energy checks. Measured errors are 0.0021%
   and 0.0016%, respectively.
5. **Indium host/GPU field regression.** A fresh 100 W GPU solve must agree with
   `data/tim_indium_100W/host_field.csv` within 1e-4 K. The measured maximum
   difference is 2.991e-5 K. This threshold is an empirical regression bound for
   the identical deterministic configuration, not a general fixed-point or
   discretization error estimate.

## Outputs

- Steady-state temperature CSV and heatmap, with stack layers and hot-spot
  location.
- An eight-point compute-power sweep from 25 to 150 W, using sequential warm
  starts in the measured protocol. T_max(compute) and T_max(HBM) are piecewise
  linear because the argmax cell changes.
- A matched TIM comparison plot and CSV.
- Preserved host/GPU indium fields in `data/tim_indium_100W/`, with hashes,
  convergence results, and reproduction commands.

## GPU implementation and timing

The GPU uses a Jacobi sweep kernel with double-buffered global-memory fields.
Kernel relaunches provide grid-wide synchronization. Red-black Gauss-Seidel is a
possible alternative but is not implemented or benchmarked. The power sweep runs
sequentially by power point. Every launch is checked, and the solver reports the
iteration count and final residual.

`--window N` controls residual-readback frequency and defaults to 32. HIP events
and host wall time support the recorded 32/64/128 comparison. Window 128 used one
quarter as many readbacks as window 32 and reduced median wall time by 18.6%;
window 64 was highly variable. These data do not establish an optimal window or
separate launch overhead from other costs. Protocol and raw output are preserved
in `data/tim_indium_100W/MANIFEST.md` and `window_ab.log`.

## Model limits

Convection, transient response, 3D geometry, per-layer TIM stacks, contact
resistance, package warpage, and liquid-cooling correlations are not modeled.

The 2D slice carries each die's full power through 1 mm of unit depth. Areal
power density, and therefore absolute temperature, is roughly an order of
magnitude above a comparable 3D package. Verification checks are internal
consistency tests and are unaffected by this scaling. Interpret absolute
temperatures as model values; the principal results are hot-spot placement and
lateral spreading.
