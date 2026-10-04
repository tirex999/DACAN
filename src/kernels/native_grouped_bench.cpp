#include "strata/kernels/iq_kernels.hpp"

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
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: native_grouped_bench <type 40|8> <groups> <entries per group> [tokens] [iterations]\n");
        return 2;
    }
    const int type = std::atoi(argv[1]), G = std::atoi(argv[2]), per = std::atoi(argv[3]);
    const int T = argc > 4 ? std::atoi(argv[4]) : 5, iters = argc > 5 ? std::atoi(argv[5]) : 500;
    const int64_t H = 2560, FF = 640;
    auto L = strata::kernels::native_expert_layout(type, type, H, FF);
    const bool scaled = type == 40;
    if (scaled) { L.scaled = 1; L.scale_off = L.bytes; L.bytes += 16; }
    const size_t blob = (L.bytes + 255) / 256 * 256;
    std::mt19937 rng(28092026);
    std::vector<uint8_t> hb(blob * (size_t) G);
    for (auto& b : hb) b = (uint8_t) rng();
    if (type == 40) {
        for (int g = 0; g < G; ++g) {
            uint8_t* p = hb.data() + (size_t) g * blob;
            for (size_t off = 0; off + 36 <= L.scale_off; off += 36)
                for (int s = 0; s < 4; ++s) p[off + s] = (uint8_t) (0x30 + (rng() % 16));
            const float s4[4] = {1.0f, 1.0f, 1.0f, 0.0f};
            std::memcpy(p + L.scale_off, s4, sizeof s4);
        }
    } else if (type == 8) {
        for (int g = 0; g < G; ++g) {
            uint8_t* p = hb.data() + (size_t) g * blob;
            for (size_t off = 0; off + 34 <= (size_t) L.bytes; off += 34) { p[off] = 0x00; p[off + 1] = 0x20; }
        }
    }
    uint8_t* db = nullptr;
    check(cudaMalloc(&db, hb.size()), "blobs");
    check(cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice), "blobs up");
    auto per_of = [&](int g) { return per > 0 ? per : 1 + (g % 3 == 2 ? 1 : 0); };
    int E = 0;
    for (int g = 0; g < G; ++g) E += per_of(g);
    std::vector<unsigned long long> ptr(G);
    std::vector<int32_t> start(G + 1), dst(E), tok(E);
    for (int g = 0, e = 0; g < G; ++g) {
        ptr[g] = (unsigned long long) (db + (size_t) g * blob);
        start[g] = e;
        for (int j = 0; j < per_of(g); ++j, ++e) {
            tok[e] = (g + j) % T;
            dst[e] = e;
        }
    }
    start[G] = E;
    unsigned long long* d_ptr;
    int32_t *d_start, *d_n, *d_dst, *d_tok;
    check(cudaMalloc(&d_ptr, G * 8), "ptr");
    check(cudaMalloc(&d_start, (G + 1) * 4), "start");
    check(cudaMalloc(&d_n, 4), "n");
    check(cudaMalloc(&d_dst, E * 4), "dst");
    check(cudaMalloc(&d_tok, E * 4), "tok");
    check(cudaMemcpy(d_ptr, ptr.data(), G * 8, cudaMemcpyHostToDevice), "up");
    check(cudaMemcpy(d_start, start.data(), (G + 1) * 4, cudaMemcpyHostToDevice), "up");
    check(cudaMemcpy(d_n, &G, 4, cudaMemcpyHostToDevice), "up");
    check(cudaMemcpy(d_dst, dst.data(), E * 4, cudaMemcpyHostToDevice), "up");
    check(cudaMemcpy(d_tok, tok.data(), E * 4, cudaMemcpyHostToDevice), "up");
    std::vector<float> hx((size_t) T * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    float *dx, *dout;
    void *dxq, *dscr;
    check(cudaMalloc(&dx, hx.size() * 4), "x");
    check(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
    check(cudaMalloc(&dxq, (size_t) T * H / 32 * 36), "xq");
    check(cudaMalloc(&dscr, strata::kernels::native_expert_scratch_bytes(E, FF)), "scratch");
    check(cudaMalloc(&dout, (size_t) E * H * 4), "out");
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    strata::kernels::quantize_q8_1_rows(dx, T, H, dxq, s);
    auto run = [&] {
        strata::kernels::native_expert_grouped(L, d_ptr, d_start, d_n, d_dst, d_tok, G, E, dxq, dscr, dout, s);
    };
    for (int i = 0; i < 2000; ++i) run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) run();
    cudaEventRecord(e1, s);
    check(cudaEventSynchronize(e1), "run");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = 1e3 * ms / iters;
    const double bytes = (double) G * (double) L.bytes;
    std::printf("type %d: %d groups, %d entries (%s), %d tokens: %.1f us per grouped call, %.1f MB of weights, %.0f GB/s\n",
                type, G, E, per > 0 ? "uniform" : "mix 1/1/2", T, us, bytes / 1e6, bytes / us / 1e3);
    std::vector<float> ho((size_t) E * H);
    check(cudaMemcpy(ho.data(), dout, ho.size() * 4, cudaMemcpyDeviceToHost), "out");
    double cs = 0;
    uint64_t fnv = 1469598103934665603ull;
    for (size_t i = 0; i < ho.size(); ++i) {
        cs += (double) ho[i] * (double) ((i % 97) + 1);
        uint32_t u;
        std::memcpy(&u, &ho[i], 4);
        fnv = (fnv ^ u) * 1099511628211ull;
    }
    std::printf("checksum %.9e  bits %016llx\n", cs, (unsigned long long) fnv);
    return 0;
}
