#pragma once

// thermal_core.h — shared geometry/materials/host-solver core for
// chiplet-thermal. Physics per docs/THERMAL.md:
// steady-state conduction div(k grad T) = -q''' on a uniform cell-centered
// FD grid (dx = dz = h), harmonic-mean bond conductivities, top Dirichlet
// T = T_amb (cold plate, half-bond from the last cell row), sides and
// bottom adiabatic. Units: mm, W, K, W/(mm*K). Unit depth (1 mm) in y.

#include <cmath>
#include <cstdio>
#include <vector>

namespace thermal {

// Layer table (docs/THERMAL.md), z bottom to top: thickness [mm], k [W/(mm*K)].
inline constexpr double T_SUB = 1.0, K_SUB = 0.020;  // organic substrate
inline constexpr double T_INT = 0.1, K_INT = 0.148;  // Si interposer
inline constexpr double T_DIE = 0.6;                 // die band (compute/HBM/mold)
inline constexpr double K_COMPUTE = 0.148, K_HBM = 0.020, K_MOLD = 0.001;
inline constexpr double T_TIM = 0.1, K_TIM = 0.005;
inline constexpr double H_STACK = T_SUB + T_INT + T_DIE + T_TIM; // 1.8 mm

struct Config {
    double W = 30.0;                        // domain width [mm]
    double h = 0.05;                        // grid pitch [mm]
    double T_amb = 318.0;                   // cold plate / ambient [K]
    double P_compute = 100.0;               // compute-die power [W]
    double P_hbm = 20.0;                    // HBM-stack power [W]
    double k_tim = K_TIM;                   // TIM conductivity [W/(mm*K)]
    double hbm_x0 = 1.0, hbm_x1 = 9.0;      // HBM footprint in x [mm]
    double compute_x0 = 10.0, compute_x1 = 20.0; // compute footprint [mm]
    double tol = 1e-6;                      // convergence: max|dT| per sweep [K]
    long max_iter = 50000000;               // sweep cap (safety)
};

// Region tags for die-band cells (region extrema in reports/plots).
inline constexpr int REGION_NONE = 0, REGION_COMPUTE = 1, REGION_HBM = 2;

// Cell-centered fields, nx x nz cells, index c = j*nx + i, cell center at
// x = (i+0.5)*h, z = (j+0.5)*h. Jacobi update in precomputed form:
//   T_new[c] = aE*T[c+1] + aW*T[c-1] + aN*T[c+nx] + aS*T[c-nx] + src[c]
// Adiabatic faces carry a = 0 (no bond). The top Dirichlet face is a
// half-bond (2*k): it enters the denominator (ksum) and folds 2*k*T_amb
// into src, but its aN multiplier is 0 — there is no neighbor cell above.
struct Grid {
    int nx = 0, nz = 0;
    double h = 0.0, T_amb = 0.0;
    std::vector<double> aE, aW, aN, aS, src;
    std::vector<double> k;       // cell conductivity [W/(mm*K)]
    std::vector<double> q;       // cell volumetric source [W/mm^3]
    std::vector<int> region;     // REGION_* per cell
};

// Harmonic mean of two cell conductivities: exact series resistance of the
// two half-cells straddling a material interface.
inline double k_hm(double ka, double kb) { return 2.0 * ka * kb / (ka + kb); }

// Build the grid: materials, sources, and Jacobi coefficients from Config.
// Die-band cell classification is by cell-center x against the footprints.
inline Grid make_grid(const Config& c) {
    Grid g;
    g.nx = (int)llround(c.W / c.h);
    g.nz = (int)llround(H_STACK / c.h);
    g.h = c.h;
    g.T_amb = c.T_amb;
    const int n = g.nx * g.nz;
    g.aE.assign(n, 0.0); g.aW.assign(n, 0.0);
    g.aN.assign(n, 0.0); g.aS.assign(n, 0.0);
    g.src.assign(n, 0.0);
    g.k.assign(n, 0.0); g.q.assign(n, 0.0); g.region.assign(n, REGION_NONE);

    const double z_int = T_SUB, z_die = T_SUB + T_INT, z_tim = z_die + T_DIE;
    const double q_compute = c.P_compute / ((c.compute_x1 - c.compute_x0) * T_DIE);
    const double q_hbm = c.P_hbm / ((c.hbm_x1 - c.hbm_x0) * T_DIE);

    for (int j = 0; j < g.nz; ++j) {
        const double z = (j + 0.5) * c.h;
        for (int i = 0; i < g.nx; ++i) {
            const int cell = j * g.nx + i;
            const double x = (i + 0.5) * c.h;
            double k = K_SUB;
            if (z >= z_tim) k = c.k_tim;
            else if (z >= z_die) {
                if (x >= c.compute_x0 && x < c.compute_x1) {
                    k = K_COMPUTE; g.q[cell] = q_compute; g.region[cell] = REGION_COMPUTE;
                } else if (x >= c.hbm_x0 && x < c.hbm_x1) {
                    k = K_HBM; g.q[cell] = q_hbm; g.region[cell] = REGION_HBM;
                } else {
                    k = K_MOLD;
                }
            } else if (z >= z_int) k = K_INT;
            g.k[cell] = k;
        }
    }

    for (int j = 0; j < g.nz; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const int cell = j * g.nx + i;
            const double kc = g.k[cell];
            double kE = 0, kW = 0, kN = 0, kS = 0;
            if (i + 1 < g.nx) kE = k_hm(kc, g.k[cell + 1]);
            if (i > 0) kW = k_hm(kc, g.k[cell - 1]);
            if (j + 1 < g.nz) kN = k_hm(kc, g.k[cell + g.nx]);
            if (j > 0) kS = k_hm(kc, g.k[cell - g.nx]);
            // Top row: Dirichlet half-bond adds 2*kc to the denominator and
            // 2*kc*T_amb to src, but has no neighbor multiplier.
            const double kN_diag = (j + 1 < g.nz) ? kN : 2.0 * kc;
            const double ksum = kE + kW + kN_diag + kS;
            g.aE[cell] = kE / ksum; g.aW[cell] = kW / ksum;
            g.aN[cell] = kN / ksum; g.aS[cell] = kS / ksum;
            g.src[cell] = g.q[cell] * c.h * c.h / ksum;
            if (j + 1 == g.nz) g.src[cell] += 2.0 * kc * c.T_amb / ksum;
        }
    }
    return g;
}

struct SolveReport {
    long iterations = 0;
    int converged = 0;
    double final_max_dT = 0.0;   // max|T_new - T| on the final sweep [K]
    double P_in = 0.0;           // integrated source power [W, per mm depth]
    double Q_top = 0.0;          // outward flux through the cold plate [W]
    double T_max = 0.0;          // global hot spot [K]
    int hot_i = -1, hot_j = -1;  // hot-spot cell indices
    double T_max_compute = 0.0, T_max_hbm = 0.0; // region maxima [K]
    double wall_ms = 0.0;        // host wall time [ms]
    double device_ms = 0.0;      // HIP-event elapsed time [ms]
    long launches = 0;           // Jacobi kernel launches
    long residual_checks = 0;    // host residual readbacks
};

// Fill a SolveReport's diagnostics (flux, extrema) for a converged field.
inline void fill_diagnostics(const Grid& g, const std::vector<double>& T,
                             SolveReport& r) {
    r.P_in = 0.0; r.Q_top = 0.0;
    r.T_max = -1e300; r.T_max_compute = -1e300; r.T_max_hbm = -1e300;
    for (int j = 0; j < g.nz; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const int cell = j * g.nx + i;
            r.P_in += g.q[cell] * g.h * g.h;
            if (T[cell] > r.T_max) {
                r.T_max = T[cell]; r.hot_i = i; r.hot_j = j;
            }
            if (g.region[cell] == REGION_COMPUTE && T[cell] > r.T_max_compute)
                r.T_max_compute = T[cell];
            if (g.region[cell] == REGION_HBM && T[cell] > r.T_max_hbm)
                r.T_max_hbm = T[cell];
        }
    }
    for (int i = 0; i < g.nx; ++i) {
        const int cell = (g.nz - 1) * g.nx + i;
        r.Q_top += 2.0 * g.k[cell] * (T[cell] - g.T_amb); // half-bond flux
    }
}

// Host Jacobi solver, FP64, double-buffered. T carries the initial guess
// in and the converged field out. Converged when max|T_new - T| < tol.
inline SolveReport solve_jacobi(const Grid& g, double tol, long max_iter,
                                std::vector<double>& T) {
    const int n = g.nx * g.nz;
    std::vector<double> Tn(n);
    SolveReport r;
    for (long it = 1; it <= max_iter; ++it) {
        double max_dT = 0.0;
        for (int j = 0; j < g.nz; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                const int cell = j * g.nx + i;
                double v = g.src[cell];
                if (g.aE[cell] != 0.0) v += g.aE[cell] * T[cell + 1];
                if (g.aW[cell] != 0.0) v += g.aW[cell] * T[cell - 1];
                if (g.aN[cell] != 0.0) v += g.aN[cell] * T[cell + g.nx];
                if (g.aS[cell] != 0.0) v += g.aS[cell] * T[cell - g.nx];
                Tn[cell] = v;
                const double d = fabs(v - T[cell]);
                if (d > max_dT) max_dT = d;
            }
        }
        T.swap(Tn);
        r.iterations = it;
        r.final_max_dT = max_dT;
        if (max_dT < tol) { r.converged = 1; break; }
    }
    fill_diagnostics(g, T, r);
    return r;
}

// Analytic 1D-limit hot spot (docs/THERMAL.md verification 1): laterally
// uniform source, all heat out the top. P_over_A in W/mm^2 (unit depth).
inline double analytic_1d_Tmax(double P_over_A, double T_amb,
                               double k_tim = K_TIM) {
    return T_amb + P_over_A * (T_TIM / k_tim + T_DIE / (2.0 * K_COMPUTE));
}

// Write a converged field as CSV: comment header, then x_mm,z_mm,T_K rows.
// Returns 0 on success, 1 if the file cannot be opened.
inline int write_field_csv(const Grid& g, const std::vector<double>& T,
                           const char* path, const char* tag) {
    FILE* f = fopen(path, "w");
    if (!f) return 1;
    fprintf(f, "# chiplet-thermal steady-state field (%s)\n", tag);
    fprintf(f, "# grid %d x %d, h = %.4f mm, cell-centered, unit depth (1 mm)\n",
            g.nx, g.nz, g.h);
    fprintf(f, "x_mm,z_mm,T_K\n");
    for (int j = 0; j < g.nz; ++j)
        for (int i = 0; i < g.nx; ++i)
            fprintf(f, "%.5f,%.5f,%.9f\n", (i + 0.5) * g.h, (j + 0.5) * g.h,
                    T[j * g.nx + i]);
    fclose(f);
    return 0;
}

} // namespace thermal
