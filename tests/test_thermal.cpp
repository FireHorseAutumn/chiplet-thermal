// Acceptance tests for docs/THERMAL.md:
//   (a) 1D analytic limit within discretization error.
//   (b) Top-boundary energy conservation within 1%.
//   (c) Default-case hot-spot ordering.
//   (d) Convergence at every GPU power-sweep point.
//   (e-g) Nondefault-TIM analytic, energy, and preserved-host field checks.
// argv[1] is the thermal_batched path; argv[2] is the preserved indium host CSV.

#include "thermal_core.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(int ok, const char* name, const char* detail) {
    printf("%s: %s (%s)\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) ++g_failures;
}

// Read a field CSV written by thermal::write_field_csv. Returns false on
// any parse or size mismatch.
static bool read_field_csv(const char* path, int nx, int nz,
                           std::vector<double>& T) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    T.assign(nx * nz, 0.0);
    char line[256];
    int got = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == 'x') continue; // comments, header
        double x, z, t;
        if (sscanf(line, "%lf,%lf,%lf", &x, &z, &t) != 3) continue;
        if (got >= nx * nz) { fclose(f); return false; }
        T[got++] = t;
    }
    fclose(f);
    return got == nx * nz;
}

static double max_abs_diff(const std::vector<double>& a,
                           const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = fmax(m, fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test_thermal <path-to-thermal_batched> "
                        "<preserved-indium-host-field>\n");
        return 2;
    }
    const std::string gpu = argv[1];
    const std::string indium_host_path = argv[2];
    char buf[1024];

    // (a) 1D analytic limit: compute die spans full width, HBM off.
    thermal::Config c1;
    c1.compute_x0 = 0.0; c1.compute_x1 = c1.W; c1.P_hbm = 0.0;
    thermal::Grid g1 = thermal::make_grid(c1);
    std::vector<double> T1(g1.nx * g1.nz, c1.T_amb);
    thermal::SolveReport r1 = thermal::solve_jacobi(g1, c1.tol, c1.max_iter, T1);
    const double analytic = thermal::analytic_1d_Tmax(c1.P_compute / c1.W, c1.T_amb);
    snprintf(buf, sizeof buf, "T_max = %.4f K, analytic = %.4f K, err = %.4f %%",
             r1.T_max, analytic,
             100.0 * fabs(r1.T_max - analytic) / (analytic - c1.T_amb));
    check(r1.converged &&
              fabs(r1.T_max - analytic) / (analytic - c1.T_amb) < 0.01,
          "(a) 1D analytic limit within 1%", buf);

    // (b)+(c) default 2D case on the host reference.
    thermal::Config c2;
    thermal::Grid g2 = thermal::make_grid(c2);
    std::vector<double> T2(g2.nx * g2.nz, c2.T_amb);
    thermal::SolveReport r2 = thermal::solve_jacobi(g2, c2.tol, c2.max_iter, T2);
    snprintf(buf, sizeof buf, "Q_top = %.5f W, P_in = %.5f W, residual = %.4f %%",
             r2.Q_top, r2.P_in, 100.0 * fabs(r2.Q_top - r2.P_in) / r2.P_in);
    check(r2.converged && fabs(r2.Q_top - r2.P_in) / r2.P_in < 0.01,
          "(b) energy conservation within 1%", buf);
    const int hot_region = g2.region[r2.hot_j * g2.nx + r2.hot_i];
    snprintf(buf, sizeof buf,
             "hot spot %.2f K at (x = %.3f, z = %.3f) region = %d; "
             "T_max_compute = %.2f K, T_max_hbm = %.2f K",
             r2.T_max, (r2.hot_i + 0.5) * g2.h, (r2.hot_j + 0.5) * g2.h,
             hot_region, r2.T_max_compute, r2.T_max_hbm);
    check(hot_region == thermal::REGION_COMPUTE &&
              r2.T_max_hbm < r2.T_max_compute,
          "(c) hot spot over compute die, HBM peak below compute peak", buf);

    // GPU cross-check, 1D case. Both solvers converge to per-sweep
    // max|dT| < 1e-6 K; the laterally-uniform fixed point's slowest mode
    // is the vertical one (damping ~5e-4/sweep), so both fields sit
    // within ~2e-3 K of the fixed point, so 0.01 K bounds their difference.
    std::vector<double> Tg;
    std::string f1 = "test_field_gpu_1d.csv";
    snprintf(buf, sizeof buf, "\"%s\" --solve 100 --uniform-1d --field-out %s",
             gpu.c_str(), f1.c_str());
    int rc = system(buf);
    bool ok1 = (rc == 0) && read_field_csv(f1.c_str(), g1.nx, g1.nz, Tg);
    snprintf(buf, sizeof buf, "max|dT| vs host 1D = %.3e K",
             ok1 ? max_abs_diff(Tg, T1) : -1.0);
    check(ok1 && max_abs_diff(Tg, T1) < 0.01,
          "GPU reproduces reference field (1D, tol 0.01 K)", buf);

    // GPU cross-check, default 2D case. The slowest lateral Jacobi mode
    // damps at ~7e-6/sweep, so each converged field can sit ~0.15 K from
    // the exact fixed point; 0.5 K bounds the difference rigorously.
    std::string f2 = "test_field_gpu_2d.csv";
    snprintf(buf, sizeof buf, "\"%s\" --solve 100 --field-out %s", gpu.c_str(),
             f2.c_str());
    rc = system(buf);
    bool ok2 = (rc == 0) && read_field_csv(f2.c_str(), g2.nx, g2.nz, Tg);
    snprintf(buf, sizeof buf, "max|dT| vs host 2D = %.3e K",
             ok2 ? max_abs_diff(Tg, T2) : -1.0);
    check(ok2 && max_abs_diff(Tg, T2) < 0.5,
          "GPU reproduces reference field (2D, tol 0.5 K)", buf);

    // (d) GPU power sweep: every point converged, ordering sane.
    std::string fs = "test_sweep_gpu.csv";
    snprintf(buf, sizeof buf, "\"%s\" --out %s", gpu.c_str(), fs.c_str());
    rc = system(buf);
    FILE* f = rc == 0 ? fopen(fs.c_str(), "r") : nullptr;
    std::vector<double> P, Tc, Th;
    std::vector<long> conv;
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == 'P') continue;
            double p, ph, tc, th, hx, hz, md;
            long it; int cv;
            if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%ld,%d,%lf", &p, &ph,
                       &tc, &th, &hx, &hz, &it, &cv, &md) != 9)
                continue;
            P.push_back(p); Tc.push_back(tc); Th.push_back(th);
            conv.push_back(cv);
        }
        fclose(f);
    }
    int all_conv = !P.empty();
    int ordering = !P.empty();
    for (size_t i = 0; i < P.size(); ++i) {
        if (!conv[i]) all_conv = 0;
        if (i && (Tc[i] <= Tc[i - 1] || Th[i] <= Th[i - 1])) ordering = 0;
    }
    snprintf(buf, sizeof buf, "%zu sweep points, all converged = %d",
             P.size(), all_conv);
    check(rc == 0 && all_conv,
          "(d) convergence achieved for all sweep points", buf);
    // At low compute power, fixed HBM self-heating can make HBM the package
    // hot spot. The compute-die ordering check applies only to the 100 W case.
    snprintf(buf, sizeof buf,
             "T_max(compute) and T_max(HBM) both monotone rising with "
             "compute power = %d", ordering);
    check(ordering, "(d) sweep trends monotone", buf);

    // (e) Nondefault TIM: 86 W/(m*K) bulk-indium conductivity in the 1D limit.
    thermal::Config c_tim;
    c_tim.compute_x0 = 0.0; c_tim.compute_x1 = c_tim.W; c_tim.P_hbm = 0.0;
    c_tim.k_tim = 0.086;
    thermal::Grid g_tim = thermal::make_grid(c_tim);
    std::vector<double> T_tim(g_tim.nx * g_tim.nz, c_tim.T_amb);
    thermal::SolveReport r_tim =
        thermal::solve_jacobi(g_tim, c_tim.tol, c_tim.max_iter, T_tim);
    const double analytic_tim =
        thermal::analytic_1d_Tmax(c_tim.P_compute / c_tim.W, c_tim.T_amb,
                                  c_tim.k_tim);
    snprintf(buf, sizeof buf,
             "T_max = %.4f K, analytic = %.4f K, err = %.4f %%, k_tim = %.3f",
             r_tim.T_max, analytic_tim,
             100.0 * fabs(r_tim.T_max - analytic_tim) /
                 (analytic_tim - c_tim.T_amb),
             c_tim.k_tim);
    check(r_tim.converged &&
              fabs(r_tim.T_max - analytic_tim) /
                      (analytic_tim - c_tim.T_amb) < 0.01,
          "(e) nondefault-TIM 1D analytic limit within 1%", buf);
    snprintf(buf, sizeof buf,
             "Q_top = %.5f W, P_in = %.5f W, residual = %.4f %%",
             r_tim.Q_top, r_tim.P_in,
             100.0 * fabs(r_tim.Q_top - r_tim.P_in) / r_tim.P_in);
    check(r_tim.converged &&
              fabs(r_tim.Q_top - r_tim.P_in) / r_tim.P_in < 0.01,
          "(f) nondefault-TIM energy conservation within 1%", buf);

    // (g) Fresh GPU field against preserved host evidence. The preserved
    // host/GPU difference is 2.9908e-5 K. The 1e-4 K threshold is an empirical
    // regression bound (3.3x measured) for this deterministic configuration,
    // not a general fixed-point or discretization error bound.
    std::vector<double> T_indium_host;
    std::string f_tim = "test_field_gpu_indium.csv";
    snprintf(buf, sizeof buf,
             "\"%s\" --solve 100 --power-hbm 20 --k-tim 0.086 --h 0.05 "
             "--tol 1e-6 --field-out %s",
             gpu.c_str(), f_tim.c_str());
    rc = system(buf);
    bool ok_tim = (rc == 0) &&
                  read_field_csv(indium_host_path.c_str(), g2.nx, g2.nz,
                                 T_indium_host);
    if (ok_tim)
        ok_tim = read_field_csv(f_tim.c_str(), g2.nx, g2.nz, Tg);
    const double d_tim = ok_tim ? max_abs_diff(T_indium_host, Tg) : -1.0;
    snprintf(buf, sizeof buf,
             "max|dT| vs preserved indium host = %.3e K, tol = 1e-4 K", d_tim);
    check(ok_tim && d_tim < 1e-4,
          "(g) GPU reproduces preserved indium host field", buf);

    if (g_failures) {
        printf("TESTS FAILED: %d check(s) — see FAIL lines above\n", g_failures);
        return 1;
    }
    printf("all acceptance checks passed\n");
    return 0;
}
