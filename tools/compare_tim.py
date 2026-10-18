#!/usr/bin/env python3
"""Merge two thermal power sweeps into a conductivity-only TIM exhibit."""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_sweep(path):
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows:
        raise ValueError(f"no sweep rows in {path}")
    result = []
    for row in rows:
        if int(row["converged"]) != 1:
            raise ValueError(f"unconverged point in {path}: {row}")
        result.append({
            "power": float(row["P_compute_W"]),
            "hbm_power": float(row["P_hbm_W"]),
            "compute": float(row["T_max_compute_K"]),
            "hbm": float(row["T_max_hbm_K"]),
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polymer", default="data/sweep_polymer.csv")
    ap.add_argument("--indium", default="data/sweep_indium.csv")
    ap.add_argument("--out-csv", default="data/tim_comparison.csv")
    ap.add_argument("--out-png", default="data/tim_comparison.png")
    args = ap.parse_args()
    polymer = read_sweep(args.polymer)
    indium = read_sweep(args.indium)
    if len(polymer) != len(indium):
        raise ValueError("sweep lengths differ")
    rows = []
    for p, i in zip(polymer, indium):
        if abs(p["power"] - i["power"]) > 1e-6 or abs(p["hbm_power"] - i["hbm_power"]) > 1e-6:
            raise ValueError("power grids differ")
        rows.append({
            "P_compute_W": p["power"],
            "P_hbm_W": p["hbm_power"],
            "T_compute_polymer_K": p["compute"],
            "T_compute_indium_K": i["compute"],
            "delta_compute_K": p["compute"] - i["compute"],
            "T_hbm_polymer_K": p["hbm"],
            "T_hbm_indium_K": i["hbm"],
            "delta_hbm_K": p["hbm"] - i["hbm"],
            "T_max_polymer_K": max(p["compute"], p["hbm"]),
            "T_max_indium_K": max(i["compute"], i["hbm"]),
            "delta_max_K": max(p["compute"], p["hbm"]) - max(i["compute"], i["hbm"]),
        })
    fields = list(rows[0])
    with open(args.out_csv, "w", newline="") as f:
        f.write("# Conductivity-only 2D unit-depth model; polymer-like k=0.005 and indium bulk k=0.086 W/(mm*K).\n")
        f.write("# Absolute temperatures are model values, not qualified package predictions; see docs/THERMAL.md.\n")
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.6f}" for k, v in row.items()} for row in rows)

    power = [row["P_compute_W"] for row in rows]
    fig, (ax_compute, ax_hbm) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, prefix, title in (
        (ax_compute, "compute", "Compute-die maximum"),
        (ax_hbm, "hbm", "HBM maximum"),
    ):
        ax.plot(power, [row[f"T_{prefix}_polymer_K"] for row in rows], "o-", label="polymer-like, 5 W/m·K")
        ax.plot(power, [row[f"T_{prefix}_indium_K"] for row in rows], "s-", label="indium bulk, 86 W/m·K")
        ax.set_xlabel("compute power [W] (HBM fixed at 20 W)")
        ax.set_ylabel("T_max [K]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("TIM conductivity-only comparison — 2D unit-depth model")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    print(f"wrote {args.out_csv} and {args.out_png}")
    for row in rows:
        print(f"P={row['P_compute_W']:.3f} W: compute ΔT={row['delta_compute_K']:.3f} K, HBM ΔT={row['delta_hbm_K']:.3f} K")


if __name__ == "__main__":
    main()
