// thermal_reference.cpp — host reference steady-state heat solver (plain
// C++, std-only). Jacobi iteration, FP64, double-buffered, harmonic-mean
// bond conductivities. Physics per docs/THERMAL.md.
//
// CLI: --power-compute W --power-hbm W --k-tim W/(mm*K) --h MM --tol K
//      --max-iter N --width MM --compute-x0 MM --compute-x1 MM
//      --hbm-x0 MM --hbm-x1 MM
//      --uniform-1d (compute die spans full width, HBM off; analytic test)
//      --out field.csv

#include "thermal_core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    thermal::Config c;
    std::string out;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--power-compute")) c.P_compute = atof(need("--power-compute"));
        else if (!strcmp(argv[i], "--power-hbm")) c.P_hbm = atof(need("--power-hbm"));
        else if (!strcmp(argv[i], "--k-tim")) c.k_tim = atof(need("--k-tim"));
        else if (!strcmp(argv[i], "--h")) c.h = atof(need("--h"));
        else if (!strcmp(argv[i], "--tol")) c.tol = atof(need("--tol"));
        else if (!strcmp(argv[i], "--max-iter")) c.max_iter = atol(need("--max-iter"));
        else if (!strcmp(argv[i], "--width")) c.W = atof(need("--width"));
        else if (!strcmp(argv[i], "--compute-x0")) c.compute_x0 = atof(need("--compute-x0"));
        else if (!strcmp(argv[i], "--compute-x1")) c.compute_x1 = atof(need("--compute-x1"));
        else if (!strcmp(argv[i], "--hbm-x0")) c.hbm_x0 = atof(need("--hbm-x0"));
        else if (!strcmp(argv[i], "--hbm-x1")) c.hbm_x1 = atof(need("--hbm-x1"));
        else if (!strcmp(argv[i], "--uniform-1d")) {
            c.compute_x0 = 0.0; c.compute_x1 = c.W; c.P_hbm = 0.0;
        }
        else if (!strcmp(argv[i], "--out")) out = need("--out");
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    thermal::Grid g = thermal::make_grid(c);
    printf("chiplet-thermal reference (host, serial, fp64, Jacobi)\n");
    printf("grid %d x %d, h = %.4f mm; compute %.1f W, HBM %.1f W, T_amb = %.1f K, k_TIM = %.6f W/(mm*K)\n",
           g.nx, g.nz, g.h, c.P_compute, c.P_hbm, c.T_amb, c.k_tim);

    std::vector<double> T(g.nx * g.nz, c.T_amb);
    thermal::SolveReport r = thermal::solve_jacobi(g, c.tol, c.max_iter, T);

    printf("iterations %ld, final max|dT| = %.3e K, converged = %d\n",
           r.iterations, r.final_max_dT, r.converged);
    printf("P_in = %.6f W, Q_top = %.6f W, imbalance = %.4f %%\n",
           r.P_in, r.Q_top, 100.0 * (r.Q_top - r.P_in) / r.P_in);
    printf("T_max = %.3f K at (x = %.3f mm, z = %.3f mm)\n", r.T_max,
           (r.hot_i + 0.5) * g.h, (r.hot_j + 0.5) * g.h);
    if (r.T_max_compute > 0.0)
        printf("T_max_compute = %.3f K\n", r.T_max_compute);
    if (r.T_max_hbm > 0.0)
        printf("T_max_hbm = %.3f K\n", r.T_max_hbm);
    if (!r.converged)
        printf("WARNING: NOT CONVERGED at max_iter = %ld; field is invalid\n",
               c.max_iter);

    if (!out.empty()) {
        if (thermal::write_field_csv(g, T, out.c_str(), "host reference")) {
            fprintf(stderr, "cannot open %s\n", out.c_str());
            return 1;
        }
        printf("wrote %s\n", out.c_str());
    }
    return r.converged ? 0 : 1;
}
