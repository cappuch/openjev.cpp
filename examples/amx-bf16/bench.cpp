// F16 GEMM throughput on the CPU AMX buffer.
//
// Build:  cmake -B build -DGGML_NATIVE=ON && cmake --build build --target llama-amx-bf16-bench -j
// Run:    ./build/bin/llama-amx-bf16-bench
//         GGML_AMX_BF16=0 ./build/bin/llama-amx-bf16-bench
//
// GGML_AMX_BF16=0 uses the AVX-512 F16 kernels. Unset (or 1) uses AMX-BF16
// when the CPU and the build both have it. Weights stay F16 in memory.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        const char * key = "model name";
        if (line.compare(0, std::strlen(key), key) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }
    return "unknown";
}

struct Shape {
    const char * name;
    int M, N, K;
    int reps;
};

static ggml_backend_buffer_type_t find_amx_buft() {
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t * bufts = get_extra(dev);
    if (!bufts) {
        return nullptr;
    }
    for (int i = 0; bufts[i] != nullptr; ++i) {
        const char * name = ggml_backend_buft_name(bufts[i]);
        if (name && std::strcmp(name, "AMX") == 0) {
            return bufts[i];
        }
    }
    return nullptr;
}

struct Gemm {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * w = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * y = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_h = nullptr;
    int M = 0, N = 0;

    ~Gemm() {
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (buf_h) ggml_backend_buffer_free(buf_h);
        if (ctx) ggml_free(ctx);
    }
};

static bool gemm_init(Gemm & g, ggml_backend_buffer_type_t buft_w, ggml_backend_buffer_type_t buft_h, int M, int N, int K) {
    const size_t mem = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 1024;
    ggml_init_params ip = { mem, nullptr, true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) {
        return false;
    }
    g.w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, K, N);
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    g.y = ggml_mul_mat(g.ctx, g.w, g.x);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.y);
    g.M = M;
    g.N = N;

    const size_t sw = ggml_backend_buft_get_alloc_size(buft_w, g.w);
    g.buf_w = ggml_backend_buft_alloc_buffer(buft_w, sw);
    if (!g.buf_w) {
        return false;
    }
    if (ggml_backend_tensor_alloc(g.buf_w, g.w, ggml_backend_buffer_get_base(g.buf_w)) != GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t sx = ggml_backend_buft_get_alloc_size(buft_h, g.x);
    const size_t sy = ggml_backend_buft_get_alloc_size(buft_h, g.y);
    const size_t off_y = (sx + 63u) & ~size_t(63);
    g.buf_h = ggml_backend_buft_alloc_buffer(buft_h, off_y + sy);
    if (!g.buf_h) {
        return false;
    }
    char * base = (char *) ggml_backend_buffer_get_base(g.buf_h);
    if (ggml_backend_tensor_alloc(g.buf_h, g.x, base) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (ggml_backend_tensor_alloc(g.buf_h, g.y, base + off_y) != GGML_STATUS_SUCCESS) {
        return false;
    }
    return true;
}

static bool gemm_compute(ggml_backend_t backend, Gemm & g, int reps, double & sec) {
    const double t0 = now_sec();
    for (int i = 0; i < reps; ++i) {
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    sec = (now_sec() - t0) / reps;
    return true;
}

static void nmse_of(const std::vector<float> & got, const std::vector<float> & ref, double & nmse, double & max_abs) {
    double se = 0, e = 0;
    max_abs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        se += d * d;
        e += (double) ref[i] * (double) ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    nmse = e > 0 ? se / e : (se == 0 ? 0 : 1);
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "CPU backend init failed\n");
        return 1;
    }
    ggml_backend_buffer_type_t amx = find_amx_buft();
    if (!amx) {
        std::printf("AMX buffer type is not available in this build. Nothing to measure.\n");
        ggml_backend_free(backend);
        return 0;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t host = ggml_backend_dev_buffer_type(dev);
    const int nthreads = std::max(1, (int) std::thread::hardware_concurrency());
    ggml_backend_cpu_set_n_threads(backend, nthreads);

    const char * env = std::getenv("GGML_AMX_BF16");
    std::printf("cpu: %s\n", cpu_model().c_str());
    std::printf("threads: %d\n", nthreads);
    std::printf("GGML_AMX_BF16=%s\n", env ? env : "(unset, AMX-BF16 on when the build has it)");
    std::printf("method: F16 weight x F32 activation -> F32, AMX buffer vs host buffer\n");
    std::printf("host path is the non-AMX CPU GEMM. nmse is AMX-buffer output vs that host output.\n");

    const Shape shapes[] = {
        { "qkv512", 512, 3072, 1024, 8 },
        { "out512", 512, 1024, 1024, 10 },
        { "up512",  512, 5248, 1024, 6 },
        { "dn512",  512, 1024, 2624, 8 },
        { "qkv32",   32, 3072, 1024, 12 },
        { "qkv64",   64, 3072, 1024, 12 },
        { "up64",    64, 5248, 1024, 10 },
        { "tail20",  20,   64,  128, 4 },
        { "gemv",     1,  256,  128, 4 },
    };

    int failed = 0;
    for (const Shape & sh : shapes) {
        std::vector<float> x((size_t) sh.M * sh.K);
        std::vector<ggml_fp16_t> w((size_t) sh.N * sh.K);
        uint32_t state = 1;
        auto rnd = [&]() {
            state = state * 1664525u + 1013904223u;
            return (state >> 8) * (1.f / 16777216.f) - 0.5f;
        };
        for (auto & v : x) v = rnd() * 0.25f;
        for (auto & v : w) v = ggml_fp32_to_fp16(rnd() * 0.25f);

        Gemm ref;
        Gemm tst;
        if (!gemm_init(ref, host, host, sh.M, sh.N, sh.K) || !gemm_init(tst, amx, host, sh.M, sh.N, sh.K)) {
            std::printf("%-8s setup failed\n", sh.name);
            failed = 1;
            break;
        }
        ggml_backend_tensor_set(ref.w, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ref.x, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(tst.w, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(tst.x, x.data(), 0, x.size() * sizeof(float));

        double sec_ref = 0, sec = 0;
        if (!gemm_compute(backend, ref, 1, sec_ref) || !gemm_compute(backend, tst, 1, sec)) {
            std::printf("%-8s compute failed\n", sh.name);
            failed = 1;
            break;
        }
        // warmup is the call above; time the next reps
        if (!gemm_compute(backend, ref, sh.reps, sec_ref) || !gemm_compute(backend, tst, sh.reps, sec)) {
            std::printf("%-8s compute failed\n", sh.name);
            failed = 1;
            break;
        }

        std::vector<float> y_ref((size_t) sh.M * sh.N), y((size_t) sh.M * sh.N);
        ggml_backend_tensor_get(ref.y, y_ref.data(), 0, y_ref.size() * sizeof(float));
        ggml_backend_tensor_get(tst.y, y.data(), 0, y.size() * sizeof(float));
        double nmse = 0, max_abs = 0;
        nmse_of(y, y_ref, nmse, max_abs);
        const double flops = 2.0 * sh.M * sh.N * sh.K;
        const bool bad = !std::isfinite(nmse) || nmse > 1e-4 || max_abs > 0.5;
        if (bad) {
            failed = 1;
        }
        std::printf("%-8s host %7.2f ms %6.1f G  amx %7.2f ms %6.1f G  x%5.2f  nmse %.3e max %.3g%s\n",
                    sh.name,
                    sec_ref * 1e3, flops / sec_ref / 1e9,
                    sec * 1e3, flops / sec / 1e9,
                    sec_ref / sec,
                    nmse, max_abs,
                    bad ? "  FAIL" : "");
    }

    // Same inputs, many evals, to catch a tile that sometimes stores zeros.
    {
        const int M = 128, N = 256, K = 128;
        std::vector<float> x((size_t) M * K);
        std::vector<ggml_fp16_t> w((size_t) N * K);
        for (int i = 0; i < M * K; ++i) x[i] = ((i * 17) % 11) * 0.01f - 0.05f;
        for (int i = 0; i < N * K; ++i) w[i] = ggml_fp32_to_fp16(((i * 13) % 9) * 0.01f - 0.04f);
        Gemm ref, tst;
        if (!gemm_init(ref, host, host, M, N, K) || !gemm_init(tst, amx, host, M, N, K)) {
            std::printf("flake setup failed\n");
            ggml_backend_free(backend);
            return 1;
        }
        ggml_backend_tensor_set(ref.w, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ref.x, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(tst.w, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(tst.x, x.data(), 0, x.size() * sizeof(float));
        double sec = 0;
        gemm_compute(backend, ref, 1, sec);
        std::vector<float> y_ref((size_t) M * N), y((size_t) M * N);
        ggml_backend_tensor_get(ref.y, y_ref.data(), 0, y_ref.size() * sizeof(float));
        int flakes = 0;
        const int reps = 30;
        for (int i = 0; i < reps; ++i) {
            if (!gemm_compute(backend, tst, 1, sec)) {
                flakes++;
                continue;
            }
            ggml_backend_tensor_get(tst.y, y.data(), 0, y.size() * sizeof(float));
            double nmse = 0, max_abs = 0;
            nmse_of(y, y_ref, nmse, max_abs);
            if (!std::isfinite(nmse) || nmse > 1e-4 || max_abs > 0.5) {
                ++flakes;
            }
        }
        std::printf("flake128  %d/%d\n", flakes, reps);
        if (flakes) {
            failed = 1;
        }
    }

    ggml_backend_free(backend);
    return failed;
}
