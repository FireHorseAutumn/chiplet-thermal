// GPU steady-state heat solver using the HIP API through include/hip_shim.h.
// FP64 Jacobi sweeps use double-buffered global-memory fields and kernel
// relaunches for grid-wide synchronization. Power points run sequentially with
// a scaled warm start; fixed HBM power means this is only an initial guess.
// HIP events and host wall time support residual-window timing measurements.
//
// CLI: (default) power sweep -> CSV; --solve P for a single solve.
//   --p-min W --p-max W --n-points N --out sweep.csv
//   --solve W --field-out field.csv [--uniform-1d]
//   --power-hbm W --k-tim W/(mm*K) --h MM --tol K --max-iter N
//   --window N (sweeps between residual readbacks, default 32)

#include "hip_shim.h"
#include "thermal_core.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                    __FILE__, __LINE__);                                      \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

constexpr int NT = 256;      // threads per block

// One Jacobi sweep over all cells. With RESIDUAL, also fold |T_new - T|
// into res via atomicMax (non-negative doubles: bit pattern is monotone).
template <bool RESIDUAL>
__global__ void jacobi_sweep(const double* __restrict__ T,
                             double* __restrict__ Tn,
                             const double* __restrict__ aE,
                             const double* __restrict__ aW,
                             const double* __restrict__ aN,
                             const double* __restrict__ aS,
                             const double* __restrict__ src, int nx, int nz,
                             unsigned long long* __restrict__ res) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nx * nz) return;
    double v = src[c];
    const double ae = aE[c], aw = aW[c], an = aN[c], as = aS[c];
    if (ae != 0.0) v += ae * T[c + 1];
    if (aw != 0.0) v += aw * T[c - 1];
    if (an != 0.0) v += an * T[c + nx];
    if (as != 0.0) v += as * T[c - nx];
    Tn[c] = v;
    if (RESIDUAL)
        atomicMax(res, (unsigned long long)__double_as_longlong(fabs(v - T[c])));
}

struct DevSolve {
    int n = 0, nx = 0, nz = 0, nblocks = 0;
    double *aE = nullptr, *aW = nullptr, *aN = nullptr, *aS = nullptr,
           *src = nullptr;
    double *T0 = nullptr, *T1 = nullptr;
    unsigned long long* res = nullptr;

    void alloc(const thermal::Grid& g) {
        n = g.nx * g.nz; nx = g.nx; nz = g.nz;
        nblocks = (n + NT - 1) / NT;
        HIP_CHECK(hipMalloc(&aE, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&aW, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&aN, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&aS, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&src, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&T0, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&T1, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&res, sizeof(unsigned long long)));
        HIP_CHECK(hipMemcpy(aE, g.aE.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(aW, g.aW.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(aN, g.aN.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(aS, g.aS.data(), n * sizeof(double), hipMemcpyHostToDevice));
    }

    void free() {
        hipFree(aE); hipFree(aW); hipFree(aN); hipFree(aS);
        hipFree(src); hipFree(T0); hipFree(T1); hipFree(res);
    }

    // Solve from the initial guess in T (host), field returned in T.
    thermal::SolveReport run(const thermal::Grid& g, double tol, long max_iter,
                             std::vector<double>& T, int window) {
        const auto wall_start = std::chrono::steady_clock::now();
        hipEvent_t ev_start, ev_stop;
        HIP_CHECK(hipEventCreate(&ev_start));
        HIP_CHECK(hipEventCreate(&ev_stop));
        HIP_CHECK(hipEventRecord(ev_start));
        HIP_CHECK(hipMemcpy(src, g.src.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(T0, T.data(), n * sizeof(double), hipMemcpyHostToDevice));
        double *cur = T0, *nxt = T1;
        thermal::SolveReport r;
        long checks = 0;
        for (long it = 0; it < max_iter;) {
            HIP_CHECK(hipMemset(res, 0, sizeof(unsigned long long)));
            for (int w = 0; w < window - 1; ++w) {
                jacobi_sweep<false><<<nblocks, NT>>>(cur, nxt, aE, aW, aN, aS,
                                                     src, nx, nz, res);
                HIP_CHECK(hipGetLastError());
                double* t = cur; cur = nxt; nxt = t;
            }
            jacobi_sweep<true><<<nblocks, NT>>>(cur, nxt, aE, aW, aN, aS, src,
                                                nx, nz, res);
            HIP_CHECK(hipGetLastError());
            double* t = cur; cur = nxt; nxt = t;
            it += window;
            ++checks;
            unsigned long long h_res = 0;
            HIP_CHECK(hipMemcpy(&h_res, res, sizeof(unsigned long long),
                                hipMemcpyDeviceToHost));
            double res_d;
            static_assert(sizeof res_d == sizeof h_res);
            memcpy(&res_d, &h_res, sizeof res_d); // non-negative double bits
            r.final_max_dT = res_d;
            r.iterations = it;
            if (r.final_max_dT < tol) { r.converged = 1; break; }
        }
        r.launches = r.iterations;
        r.residual_checks = checks;
        HIP_CHECK(hipMemcpy(T.data(), cur, n * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float device_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&device_ms, ev_start, ev_stop));
        r.device_ms = device_ms;
        r.wall_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - wall_start)
                        .count();
        HIP_CHECK(hipEventDestroy(ev_start));
        HIP_CHECK(hipEventDestroy(ev_stop));
        thermal::fill_diagnostics(g, T, r);
        return r;
    }
};

static void print_report(const char* tag, const thermal::Grid& g,
                         const thermal::SolveReport& r) {
    const double us_per_launch =
        r.launches > 0 ? 1000.0 * r.device_ms / r.launches : 0.0;
    printf("%s: iterations %ld, final max|dT| = %.3e K, converged = %d\n", tag,
           r.iterations, r.final_max_dT, r.converged);
    printf("%s: wall %.3f ms, HIP event %.3f ms, launches %ld (%.3f us/launch "
           "including copies), residual readbacks %ld\n",
           tag, r.wall_ms, r.device_ms, r.launches, us_per_launch,
           r.residual_checks);
    printf("%s: P_in = %.6f W, Q_top = %.6f W, imbalance = %.4f %%\n", tag,
           r.P_in, r.Q_top, 100.0 * (r.Q_top - r.P_in) / r.P_in);
    printf("%s: T_max = %.3f K at (x = %.3f mm, z = %.3f mm);", tag, r.T_max,
           (r.hot_i + 0.5) * g.h, (r.hot_j + 0.5) * g.h);
    if (r.T_max_compute > 0.0) printf(" T_max_compute = %.3f K;", r.T_max_compute);
    if (r.T_max_hbm > 0.0) printf(" T_max_hbm = %.3f K", r.T_max_hbm);
    printf("\n");
    if (!r.converged)
        printf("%s: WARNING: NOT CONVERGED; field is invalid\n", tag);
}

int main(int argc, char** argv) {
    thermal::Config c;
    double p_min = 25.0, p_max = 150.0, solve_power = -1.0;
    int n_points = 8, window = 32;
    std::string out = "data/sweep.csv", field_out;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--solve")) solve_power = atof(need("--solve"));
        else if (!strcmp(argv[i], "--field-out")) field_out = need("--field-out");
        else if (!strcmp(argv[i], "--uniform-1d")) {
            c.compute_x0 = 0.0; c.compute_x1 = c.W; c.P_hbm = 0.0;
        }
        else if (!strcmp(argv[i], "--p-min")) p_min = atof(need("--p-min"));
        else if (!strcmp(argv[i], "--p-max")) p_max = atof(need("--p-max"));
        else if (!strcmp(argv[i], "--n-points")) n_points = atoi(need("--n-points"));
        else if (!strcmp(argv[i], "--out")) out = need("--out");
        else if (!strcmp(argv[i], "--power-hbm")) c.P_hbm = atof(need("--power-hbm"));
        else if (!strcmp(argv[i], "--k-tim")) c.k_tim = atof(need("--k-tim"));
        else if (!strcmp(argv[i], "--h")) c.h = atof(need("--h"));
        else if (!strcmp(argv[i], "--tol")) c.tol = atof(need("--tol"));
        else if (!strcmp(argv[i], "--max-iter")) c.max_iter = atol(need("--max-iter"));
        else if (!strcmp(argv[i], "--window")) window = atoi(need("--window"));
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (window <= 0) {
        fprintf(stderr, "--window must be positive\n");
        return 2;
    }

    thermal::Grid g = thermal::make_grid(c);
    printf("chiplet-thermal GPU solver (HIP API, fp64, Jacobi relaunch batches)\n");
    printf("grid %d x %d, h = %.4f mm, T_amb = %.1f K, k_TIM = %.6f W/(mm*K)\n",
           g.nx, g.nz, g.h, c.T_amb, c.k_tim);

    DevSolve dev;
    dev.alloc(g);

    if (solve_power >= 0.0) { // single solve
        c.P_compute = solve_power;
        g = thermal::make_grid(c);
        std::vector<double> T(g.nx * g.nz, c.T_amb);
        thermal::SolveReport r = dev.run(g, c.tol, c.max_iter, T, window);
        print_report("gpu", g, r);
        if (!field_out.empty()) {
            if (thermal::write_field_csv(g, T, field_out.c_str(), "GPU")) {
                fprintf(stderr, "cannot open %s\n", field_out.c_str());
                return 1;
            }
            printf("wrote %s\n", field_out.c_str());
        }
        dev.free();
        return r.converged ? 0 : 1;
    }

    // Power sweep, sequential per point. With HBM power fixed, scaling the
    // previous solution is only an initial guess, not the exact fixed point.
    FILE* f = fopen(out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    fprintf(f, "# chiplet-thermal power sweep (GPU, fp64, Jacobi)\n");
    fprintf(f, "# HBM power fixed at %.1f W; sequential per point, warm-started\n",
            c.P_hbm);
    fprintf(f, "# TIM conductivity = %.6f W/(mm*K)\n", c.k_tim);
    fprintf(f, "P_compute_W,P_hbm_W,T_max_compute_K,T_max_hbm_K,"
               "hot_x_mm,hot_z_mm,iterations,converged,final_max_dT\n");

    std::vector<double> T(g.nx * g.nz, c.T_amb);
    double prev_power = 0.0;
    int all_converged = 1;
    for (int p = 0; p < n_points; ++p) {
        c.P_compute = p_min + (p_max - p_min) * p / (n_points - 1);
        g = thermal::make_grid(c);
        if (prev_power > 0.0) {
            const double s = c.P_compute / prev_power;
            for (double& v : T) v = c.T_amb + (v - c.T_amb) * s;
        }
        thermal::SolveReport r = dev.run(g, c.tol, c.max_iter, T, window);
        print_report("sweep", g, r);
        if (!r.converged) all_converged = 0;
        fprintf(f, "%.3f,%.3f,%.9f,%.9f,%.4f,%.4f,%ld,%d,%.3e\n", c.P_compute,
                c.P_hbm, r.T_max_compute, r.T_max_hbm, (r.hot_i + 0.5) * g.h,
                (r.hot_j + 0.5) * g.h, r.iterations, r.converged,
                r.final_max_dT);
        prev_power = c.P_compute;
    }
    fclose(f);
    printf("wrote %s; all points converged = %d\n", out.c_str(), all_converged);
    dev.free();
    return all_converged ? 0 : 1;
}
