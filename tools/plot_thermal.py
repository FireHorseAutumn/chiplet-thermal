#!/usr/bin/env python3
# plot_thermal.py — plot chiplet-thermal outputs:
#   1. steady-state temperature field with stack layers and hot-spot location
#   2. compute-die and HBM maxima versus compute power
# Footprint and layer constants mirror the defaults in docs/THERMAL.md.

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# docs/THERMAL.md layer table (z boundaries, mm) and die footprints.
Z_SUB, Z_INT, Z_DIE, Z_TOP = 1.0, 1.1, 1.7, 1.8
HBM_X0, HBM_X1 = 1.0, 9.0
COMPUTE_X0, COMPUTE_X1 = 10.0, 20.0


def read_field(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("x"):
                continue  # comments and the column header
            rows.append([float(v) for v in line.split(",")])
    data = np.array(rows)
    x = np.unique(data[:, 0])
    z = np.unique(data[:, 1])
    T = data[:, 2].reshape(len(z), len(x))
    return x, z, T


def plot_heatmap(field_csv, out_png):
    x, z, T = read_field(field_csv)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(
        T,
        extent=[x[0], x[-1], z[0], z[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    for zb in (Z_SUB, Z_INT, Z_DIE):
        ax.axhline(zb, color="white", lw=0.7, alpha=0.8)
    for x0, x1, label in ((HBM_X0, HBM_X1, "HBM"), (COMPUTE_X0, COMPUTE_X1, "compute")):
        ax.add_patch(
            Rectangle((x0, Z_INT), x1 - x0, Z_DIE - Z_INT,
                      fill=False, edgecolor="cyan", lw=1.2)
        )
        ax.text((x0 + x1) / 2, Z_DIE + 0.03, label, color="cyan",
                ha="center", va="bottom", fontsize=9)
    imax = np.unravel_index(np.argmax(T), T.shape)
    tmax, hx, hz = T[imax], x[imax[1]], z[imax[0]]
    ax.annotate(
        f"hot spot {tmax:.1f} K\n({hx:.2f}, {hz:.2f}) mm",
        xy=(hx, hz), xytext=(hx + 4.5, hz - 0.55),
        color="white", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="white"),
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("2.5D package cross-section, steady state (compute 100 W, HBM 20 W)")
    fig.colorbar(im, ax=ax, label="T [K]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png}")


def plot_sweep(sweep_csv, out_png):
    rows = []
    with open(sweep_csv) as f:
        for line in f:
            if line.startswith("#") or line.startswith("P"):
                continue  # comments and the column header
            rows.append([float(v) for v in line.split(",")])
    d = np.array(rows)
    P, Tc, Th = d[:, 0], d[:, 2], d[:, 3]
    sc = np.polyfit(P, Tc, 1)[0]
    sh = np.polyfit(P, Th, 1)[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(P, Tc, "o-", label=f"compute die (slope {sc:.2f} K/W)")
    ax.plot(P, Th, "s-", label=f"HBM stack (slope {sh:.2f} K/W)")
    ax.set_xlabel("compute power [W]  (HBM fixed at 20 W)")
    ax.set_ylabel("T_max [K]")
    ax.set_title("Hot spot vs power — HBM slope is lateral spreading")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png}; slopes: compute {sc:.3f} K/W, HBM {sh:.3f} K/W")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="data/field_100W.csv")
    ap.add_argument("--sweep", default="data/sweep.csv")
    ap.add_argument("--out-heatmap", default="data/heatmap_100W.png")
    ap.add_argument("--out-sweep", default="data/sweep.png")
    args = ap.parse_args()
    plot_heatmap(args.field, args.out_heatmap)
    plot_sweep(args.sweep, args.out_sweep)


if __name__ == "__main__":
    main()
