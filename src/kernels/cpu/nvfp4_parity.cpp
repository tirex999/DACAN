#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels::cpu;

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 64;
    ggml_cpu_init();
    if (!nvfp4_avx512_ok()) { std::printf("no AVX-512 VNNI on this CPU\n"); return 2; }
    const ggml_type_traits_cpu* tn = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const ggml_type_traits_cpu* tq = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (tn->vec_dot == nullptr || tn->vec_dot_type != GGML_TYPE_Q8_0 || tq->from_float == nullptr) {
        std::printf("ggml-cpu: NVFP4 does not dot against Q8_0 here\n");
        return 2;
    }
    std::mt19937 rng(28092026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int bad = 0, checked = 0;
    for (const int n : {2560, 640}) {
        const size_t rb = ggml_row_size(GGML_TYPE_NVFP4, n), ab = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<uint8_t> w((size_t) rows * rb);
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < n / 64; ++b) {
                uint8_t* blk = w.data() + (size_t) r * rb + (size_t) b * 36;
                for (int k = 0; k < 4; ++k) {
                    const uint32_t v = rng();
                    blk[k] = (v % 50 == 0) ? 0 : (v % 50 == 1) ? 0x7F : (v % 50 == 2) ? (uint8_t) (v >> 8 & 7)
                                                                                      : (uint8_t) (0x28 + (v >> 8) % 40);
                }
                for (int j = 0; j < 32; ++j) blk[4 + j] = (uint8_t) rng();
            }
        std::vector<std::vector<uint8_t>> act(8, std::vector<uint8_t>(ab));
        std::vector<float> x((size_t) n);
        for (int t = 0; t < 8; ++t) {
            const float scale = 0.05f + 3.0f * (float) t;
            for (int i = 0; i < n; ++i) x[(size_t) i] = nd(rng) * scale;
            tq->from_float(x.data(), act[(size_t) t].data(), n);
        }
        const void* ap[8];
        for (int t = 0; t < 8; ++t) ap[t] = act[(size_t) t].data();
        for (int nt = 1; nt <= 8; ++nt)
            for (int r = 0; r + 1 < rows; ++r) {
                const uint8_t* r0 = w.data() + (size_t) r * rb;
                const uint8_t* r1 = r0 + rb;
                float m1[8], m0[8], mu[8];
                nvfp4_row_dots(r0, n, ap, nt, m1);
                nvfp4_row2_dots(r0, r1, n, ap, nt, m0, mu);
                for (int t = 0; t < nt; ++t) {
                    float g0 = 0.f, g1 = 0.f;
                    tn->vec_dot(n, &g0, 0, r0, 0, ap[t], 0, 1);
                    tn->vec_dot(n, &g1, 0, r1, 0, ap[t], 0, 1);
                    checked += 3;
                    if (std::memcmp(&g0, &m1[t], 4) != 0 || std::memcmp(&g0, &m0[t], 4) != 0 || std::memcmp(&g1, &mu[t], 4) != 0) {
                        if (bad < 10)
                            std::printf("MISMATCH n=%d nt=%d row %d token %d: ggml %.9g %.9g, avx512 %.9g %.9g %.9g\n", n, nt, r,
                                        t, g0, g1, m1[t], m0[t], mu[t]);
                        ++bad;
                    }
                }
            }
    }
    std::printf("parity: %d of %d dots differ from ggml-cpu\n", bad, checked);

    {
        const int n = 2560, ff = 640;
        const size_t rb = ggml_row_size(GGML_TYPE_NVFP4, n), ab = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<uint8_t> w((size_t) 2 * ff * rb);
        for (auto& b : w) b = (uint8_t) rng();
        for (size_t r = 0; r < (size_t) 2 * ff; ++r)
            for (int b = 0; b < n / 64; ++b)
                for (int k = 0; k < 4; ++k) w[r * rb + (size_t) b * 36 + k] = (uint8_t) (0x30 + k);
        std::vector<std::vector<uint8_t>> act(8, std::vector<uint8_t>(ab));
        std::vector<float> x((size_t) n);
        for (int t = 0; t < 8; ++t) {
            for (int i = 0; i < n; ++i) x[(size_t) i] = nd(rng);
            tq->from_float(x.data(), act[(size_t) t].data(), n);
        }
        const void* ap[8];
        for (int t = 0; t < 8; ++t) ap[t] = act[(size_t) t].data();
        static Nvfp4Act prep[8];
        const Nvfp4Act* pa[8];
        for (int t = 0; t < 8; ++t) { nvfp4_prepare(ap[t], n, &prep[t]); pa[t] = &prep[t]; }
        const double mb = (double) w.size() / 1e6;
        volatile float sink = 0.f;
        for (const int nt : {1, 2, 3, 4, 8}) {
            const int reps = 20;
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < reps; ++k)
                for (int r = 0; r < ff; ++r)
                    for (int t = 0; t < nt; ++t) {
                        float g = 0.f, u = 0.f;
                        tn->vec_dot(n, &g, 0, w.data() + (size_t) r * rb, 0, ap[t], 0, 1);
                        tn->vec_dot(n, &u, 0, w.data() + (size_t) (ff + r) * rb, 0, ap[t], 0, 1);
                        sink = sink + g + u;
                    }
            const double us_g = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < reps; ++k)
                for (int r = 0; r < ff; ++r) {
                    float g[8], u[8];
                    nvfp4_dots2(w.data() + (size_t) r * rb, w.data() + (size_t) (ff + r) * rb, n, pa, nt, g, u);
                    sink = sink + g[0] + u[0];
                }
            const double us_a = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("gate+up of one expert (%.2f MB), %d token(s), one core: ggml-cpu %7.1f us (%5.2f GB/s)   avx512 %7.1f us "
                        "(%5.2f GB/s)   x%.2f\n", mb, nt, us_g, mb / us_g * 1e3, us_a, mb / us_a * 1e3, us_g / us_a);
        }
        (void) sink;
    }
    return bad == 0 ? 0 : 1;
}
