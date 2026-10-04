#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using strata::kernels::SamplerParams;
using strata::kernels::sample_tokens;

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int V = 248320, T = 4;
    std::mt19937 rng(20260927);
    std::normal_distribution<float> nd(0.0f, 2.5f);
    std::vector<float> h((size_t) T * V);
    for (auto& x : h) x = nd(rng);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 30; ++i) h[(size_t) t * V + (size_t) ((i * 7919 + t * 104729) % V)] = 14.0f - 0.35f * (float) i;
    for (int i = 0; i < 6; ++i) h[(size_t) 1 * V + (size_t) (1000 + i * 3)] = 16.0f;
    for (int i = 0; i < 25; ++i) h[(size_t) 3 * V + (size_t) (5000 + i * 11)] = 15.5f;

    float* d_logits = nullptr;
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    unsigned long long *h_rng = nullptr, *d_rng = nullptr;
    cudaMalloc(&d_logits, h.size() * sizeof(float));
    cudaMalloc(&d_a, T * sizeof(int));
    cudaMalloc(&d_b, T * sizeof(int));
    cudaMalloc(&d_c, T * sizeof(int));
    cudaHostAlloc((void**) &h_rng, 16, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**) &d_rng, h_rng, 0);
    cudaMemcpy(d_logits, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);

    const struct { float temp, top_p; int top_k; bool tf; int seeds; } cfg[] = {
        {0.6f, 0.95f, 20, false, 4}, {1.0f, 1.0f, 20, false, 4}, {0.3f, 0.8f, 5, false, 4}, {1.5f, 0.99f, 64, false, 4},
        {0.6f, 0.95f, 1, false, 4},
        {0.6f, 0.95f, 20, true, 12}, {0.3f, 0.8f, 5, true, 12}, {1.5f, 0.99f, 64, true, 12}};
    int checked = 0, bad = 0, order_diff = 0, calls = 0;
    double ms_ref = 0, ms_fast = 0;
    for (const auto& c : cfg) {
        int cfg_bad = 0, cfg_diff = 0;
        for (int s = 0; s < c.seeds; ++s) {
            SamplerParams p;
            p.temperature = c.temp;
            p.top_p = c.top_p;
            p.top_k = c.top_k;
            p.temp_first = c.tf;
            p.rng_dev = d_rng;
            h_rng[0] = 1000ull * (unsigned long long) s + 17;
            h_rng[1] = 0x9e3779b97f4a7c15ull ^ (unsigned long long) s;
            setenv("STRATA_SAMPLER_REFERENCE", "1", 1);
            auto t0 = std::chrono::steady_clock::now();
            sample_tokens(d_logits, T, V, nullptr, 0, p, d_a, nullptr);
            auto t1 = std::chrono::steady_clock::now();
            unsetenv("STRATA_SAMPLER_REFERENCE");
            sample_tokens(d_logits, T, V, nullptr, 0, p, d_b, nullptr);
            auto t2 = std::chrono::steady_clock::now();
            SamplerParams q = p;
            q.temp_first = !p.temp_first;
            sample_tokens(d_logits, T, V, nullptr, 0, q, d_c, nullptr);
            ms_ref += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_fast += std::chrono::duration<double, std::milli>(t2 - t1).count();
            ++calls;
            int a[T], b[T], o[T];
            cudaMemcpy(a, d_a, sizeof a, cudaMemcpyDeviceToHost);
            cudaMemcpy(b, d_b, sizeof b, cudaMemcpyDeviceToHost);
            cudaMemcpy(o, d_c, sizeof o, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T; ++t) {
                ++checked;
                if (b[t] != o[t]) ++cfg_diff;
                if (a[t] != b[t]) {
                    ++bad;
                    ++cfg_bad;
                    if (bad <= 10)
                        std::printf("MISMATCH temp %.2f top_p %.2f top_k %d temp_first %d seed #%d row %d: reference %d, fast %d\n",
                                    c.temp, c.top_p, c.top_k, (int) c.tf, s, t, a[t], b[t]);
                }
            }
        }
        order_diff += (c.tf && c.temp < 1.0f) ? cfg_diff : 0;
        std::printf("temp %.2f top_p %.2f top_k %2d temp_first %d: %d rows, %d differ from the reference, %d differ "
                    "from the other order\n", c.temp, c.top_p, c.top_k, (int) c.tf, c.seeds * T, cfg_bad, cfg_diff);
    }
    {
        cudaStream_t st;
        cudaStreamCreate(&st);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int R = 5;
        float* d_l5 = nullptr;
        int* d_o5 = nullptr;
        cudaMalloc(&d_l5, (size_t) R * V * sizeof(float));
        cudaMalloc(&d_o5, R * sizeof(int));
        for (int t = 0; t < R; ++t) cudaMemcpy(d_l5 + (size_t) t * V, h.data() + (size_t) (t % T) * V, V * sizeof(float), cudaMemcpyHostToDevice);
        SamplerParams sp;
        sp.temperature = 0.6f; sp.top_p = 0.95f; sp.top_k = 20; sp.temp_first = true; sp.rng_dev = d_rng;
        SamplerParams gp;
        gp.greedy = true;
        for (int w = 0; w < 3; ++w) {
            const SamplerParams& q = w == 2 ? gp : sp;
            if (w == 1) setenv("STRATA_SAMPLER_BLOCK", "1", 1);
            if (w == 2) unsetenv("STRATA_SAMPLER_BLOCK");
            for (int i = 0; i < 10; ++i) sample_tokens(d_l5, R, V, nullptr, 0, q, d_o5, st);
            cudaEventRecord(e0, st);
            for (int i = 0; i < 200; ++i) sample_tokens(d_l5, R, V, nullptr, 0, q, d_o5, st);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("%s kernel, 5 rows: %.1f us per launch (GPU events)\n",
                        w == 0 ? "sampled, top_k 20 (default: warp lists)"
                               : w == 1 ? "sampled, top_k 20 (block kernel)" : "greedy",
                        1000.0f * ms / 200);
        }
        cudaFree(d_l5);
        cudaFree(d_o5);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    const bool ok = bad == 0 && e == cudaSuccess && order_diff > 0;
    std::printf("%s: %d of %d picks differ; the orders differ in %d rows at T < 1; reference %.3f ms, fast %.3f ms "
                "per call (4 rows, %d vocab)\n", ok ? "PASS" : "FAIL", bad, checked, order_diff, ms_ref / calls,
                ms_fast / calls, V);
    return ok ? 0 : 1;
}
