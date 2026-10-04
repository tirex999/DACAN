#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    check(cudaMalloc(&d, h.size() * sizeof(T)), "malloc");
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
uint64_t fnv(const std::vector<float>& v, uint64_t h) {
    const auto* b = (const uint8_t*) v.data();
    for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
}

int main(int argc, char** argv) {
    using namespace strata::kernels;
    const int T = argc > 1 ? std::atoi(argv[1]) : 4, iters = argc > 2 ? std::atoi(argv[2]) : 2000;
    constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
    if (T < 1 || T > kFusedGrMaxT) { std::fprintf(stderr, "T must be 1..%d\n", kFusedGrMaxT); return 2; }
    std::mt19937 rng(28092026);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto rnd = [&](size_t n, float scale) { std::vector<float> v(n); for (auto& x : v) x = u(rng) * scale; return v; };
    auto rnd16 = [&](size_t n, float scale) { std::vector<uint16_t> v(n); for (auto& x : v) x = to_bf16(u(rng) * scale); return v; };
    const std::vector<float> hR = rnd((size_t) T * D, 1.0f), hbo = rnd((size_t) T * N, 0.5f), hinj = rnd((size_t) T * HC, 1.0f);
    float* R = upload(hR);
    float* bo = upload(hbo);
    float* inj = upload(hinj);
    float* w_norm = upload(rnd(D, 1.0f));
    uint16_t* w_down = upload(rnd16((size_t) LR * D, 0.02f));
    uint16_t* w_up = upload(rnd16((size_t) D * LR, 0.05f));
    uint16_t* w_inject = upload(rnd16((size_t) HC * D, 0.02f));
    float *lo, *rs, *injo, *mixed, *xn;
    check(cudaMalloc(&lo, (size_t) T * LR * 4), "lo");
    check(cudaMalloc(&rs, (size_t) T * HC * 4), "rs");
    check(cudaMalloc(&injo, (size_t) T * HC * 4), "inj");
    check(cudaMalloc(&mixed, (size_t) T * N * 4), "mixed");
    check(cudaMalloc(&xn, (size_t) T * D * 4), "xn");
    FusedGrArgs a[kFusedGrMaxT];
    for (int t = 0; t < T; ++t) {
        a[t].R = R + (size_t) t * D; a[t].R_out = R + (size_t) t * D; a[t].apply = true;
        a[t].bo_prev = bo + (size_t) t * N; a[t].inj_prev = inj + (size_t) t * HC;
        a[t].w_norm = w_norm; a[t].w_down = w_down; a[t].w_up = w_up; a[t].w_inject = w_inject; a[t].eps = 1e-6f;
        a[t].lo = lo + (size_t) t * LR; a[t].rs = rs + (size_t) t * HC; a[t].inject_out = injo + (size_t) t * HC;
        a[t].mixed = mixed + (size_t) t * N;
    }
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    fused_gr_read_multi(a, T, xn, s);
    check(cudaStreamSynchronize(s), "first call");
    std::vector<float> o1((size_t) T * N), o2((size_t) T * LR), o3((size_t) T * HC), o4((size_t) T * D);
    cudaMemcpy(o1.data(), mixed, o1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o2.data(), lo, o2.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o3.data(), injo, o3.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o4.data(), R, o4.size() * 4, cudaMemcpyDeviceToHost);
    const uint64_t h = fnv(o4, fnv(o3, fnv(o2, fnv(o1, 1469598103934665603ull))));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 200; ++i) fused_gr_read_multi(a, T, xn, s);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) fused_gr_read_multi(a, T, xn, s);
    cudaEventRecord(e1, s);
    check(cudaEventSynchronize(e1), "run");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = 1e3 * ms / iters;
    const double mb = (2.0 * LR * D + HC * D) * 2 / 1e6;
    std::printf("gr_read T=%d%s: %.1f us per call (%.1f MB of weights -> %.0f GB/s), hash %016llx\n", T,
                std::getenv("STRATA_GR_V1") ? " (v1)" : "", us, mb, mb / us * 1e3, (unsigned long long) h);
    return 0;
}
