# 100 W indium TIM host/GPU field evidence

Updated 2026-10-17.

This directory preserves the host-reference and GPU temperature fields for the
100 W compute / 86 W/(m·K) bulk-indium conductivity scenario. They are outputs of
the 2D unit-depth conductivity model, not commercial TIM or package qualification
data.

## Files

- `host_field.csv`: preserved copy of the original `build/tim_host_100.csv`
- `gpu_field.csv`: preserved copy of the original `build/tim_gpu_100.csv`
- `window_ab.log`: complete residual-window timing output

Each field has three header lines followed by 21,600 `(x_mm,z_mm,T_K)` rows.

SHA-256:

```text
host_field.csv  b9c5c31cc49c1025058ca648e7013146a969be7a24357b08b29771ee99ae041c
gpu_field.csv   81009694374d1580a742f5a8305214e10b83ca3c0dcc046dc6dabac888c91d61
```

The original build-directory copies were removed after byte-for-byte comparison
with these files.

## Scenario

- FP64 Jacobi, 600 x 36 cells, h=0.05 mm, unit depth 1 mm
- T_amb=318 K, top Dirichlet boundary, remaining boundaries adiabatic
- Compute: 100 W over the default 10-20 mm footprint
- HBM: 20 W over the default 1-9 mm footprint
- TIM: `k_tim=0.086 W/(mm·K)`, equivalent to 86 W/(m·K) bulk indium
- Convergence: max|dT| < 1e-6 K, with a 50,000,000-iteration safety cap
- Other geometry and material settings are the defaults in
  `include/thermal_core.h`

## Field verification

Preserved-field comparison:

- Host/GPU coordinates match exactly; max|dT|=2.9908e-5 K.
- Host compute-die maximum: 350.160278852 K.
- Host HBM/global maximum: 358.245772186 K.

Fresh reproduction on Oct 17:

- Host: 135,490 iterations, final max|dT|=9.999e-7 K, converged.
- GPU: 135,520 iterations, final max|dT|=9.985e-7 K, converged.
- Both: P_in=120.000000 W, Q_top=119.999662 W,
  T_max_compute=350.160 K, and T_max_hbm/global=358.246 K.
- Fresh host versus preserved host: max|dT|=0 K.
- Fresh GPU versus preserved GPU: max|dT|=0 K.

The full CTest suite passes 1/1 with ten checks:

- Nondefault-TIM 1D analytic case: T_max=328.6325 K versus 328.6327 K analytic,
  an error of 0.0021% within the 1% tolerance.
- Nondefault-TIM energy check: Q_top=99.99840 W versus P_in=100.00000 W, a
  residual of 0.0016% within the 1% tolerance.
- Fresh GPU versus preserved indium host: max|dT|=2.991e-5 K within the 1e-4 K
  regression tolerance. The threshold is approximately 3.3 times the preserved
  measured difference and applies to this identical deterministic configuration.

CTest writes fresh GPU output only to `build/test_field_gpu_indium.csv`; it does
not overwrite the preserved files.

## Reproduce the fields

From the repository root, build and write fresh fields to a temporary directory:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
tmp_dir=$(mktemp -d /tmp/thermal-indium-100W.XXXXXX)
./build/thermal_reference \
  --power-compute 100 --power-hbm 20 --k-tim 0.086 --h 0.05 --tol 1e-6 \
  --out "$tmp_dir/host_field.csv"
./build/thermal_batched \
  --solve 100 --power-hbm 20 --k-tim 0.086 --h 0.05 --tol 1e-6 \
  --field-out "$tmp_dir/gpu_field.csv"
```

The CSVs have two comment lines and one text column-header line before the
numeric data. The recorded comparisons used NumPy with `skiprows=3` and
`delimiter=','`.

## Residual-window timing

Protocol: one 100 W indium solve per run, three independent repetitions per
residual window, using the same binary and default geometry. All nine runs
converged with T_max_compute=350.160 K and T_max_hbm/global=358.246 K. HIP-event
and host wall times are reported as median [min,max].

| Window | Iterations | Residual readbacks | Wall ms, median [min,max] | HIP-event ms, median [min,max] |
|--:|--:|--:|--:|--:|
| 32 | 135,520 | 4,235 | 667.388 [627.935, 736.388] | 667.339 [627.920, 736.367] |
| 64 | 135,552 | 2,118 | 1,006.155 [794.739, 1,352.049] | 1,005.996 [794.684, 1,351.931] |
| 128 | 135,552 | 1,059 | 543.377 [529.735, 561.360] | 543.357 [529.674, 561.327] |

Window 128 used one quarter as many residual readbacks as window 32 and reduced
median wall time by 18.6%. Window 64 was highly variable, so these runs do not
support an optimal-window or monotonic-scaling conclusion. Event time includes
initial and final copies and is not a pure kernel-time or launch-overhead
measurement. Power-point batching remains unimplemented.

Reproduce from the repository root:

```bash
for window in 32 64 128; do
  for rep in 1 2 3; do
    ./build/thermal_batched \
      --solve 100 --power-hbm 20 --k-tim 0.086 --h 0.05 --tol 1e-6 \
      --window "$window"
  done
done
```
