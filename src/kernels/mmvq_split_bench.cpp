// src/kernels/mmvq_split_bench.cpp - 28.09.2026: what a tensor-parallel split of the dense projections would save.
// Times native_mmvq (Q8_0 weights, q8_1 activations, T columns) on the model's projection shapes, whole and split
// in two - by output rows (column-parallel: in_proj, the shared expert's gate/up) or by input columns (row-parallel:
// out_proj, the shared expert's down; each half then needs its partial sums added across the cards).
//     build/mmvq_split_bench [T] [iterations]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 4, iters = argc > 2 ? std::atoi(argv[2]) : 2000;
    struct Shape { const char* name; int n_in, n_out; bool by_rows; };
    const Shape shapes[] = {
        {"GDN in_proj qkvz 2560->12288", 2560, 12288, true},
        {"GDN in_proj ba   2560->96", 2560, 96, true},
        {"GDN out_proj     6144->2560", 6144, 2560, false},
        {"QSA q+gate       2560->12288", 2560, 12288, true},
        {"QSA o_proj       6144->2560", 6144, 2560, false},
        {"shared gate/up   2560->640", 2560, 640, true},
        {"shared down      640->2560", 640, 2560, false},
    };
    const int type = 8;   // Q8_0: 34 bytes per 32 weights
    // STRATA_MMVQ_UPSTREAM=1: llama.cpp's multi-column layout (2 rows a block; STRATA_MMVQ_ROWS=4 - 4 rows) instead of
    // the exact one-row layout the engine uses by default
    if (const char* up = std::getenv("STRATA_MMVQ_UPSTREAM"); up != nullptr && up[0] == '1') {
        strata::kernels::native_mmvq_set_multi_exact(false);
        std::printf("layout: upstream%s\n", std::getenv("STRATA_MMVQ_ROWS") ? " (STRATA_MMVQ_ROWS)" : "");
    }
    std::mt19937 rng(28092026);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::printf("T = %d columns, %d iterations each, card warmed first\n", T, iters);
    for (const Shape& sh : shapes) {
        const size_t wbytes = (size_t) sh.n_out * (sh.n_in / 32) * 34;
        std::vector<uint8_t> hw(wbytes);
        for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) rng();
        for (size_t b = 0; b + 34 <= wbytes; b += 34) { hw[b] = 0; hw[b + 1] = 0x20; }   // d = 2^-7
        void *dw, *dx;
        float* dy;
        check(cudaMalloc(&dw, wbytes), "w");
        check(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "w up");
        const size_t xbytes = (size_t) T * (sh.n_in / 32) * 36 + 256;
        check(cudaMalloc(&dx, xbytes), "x");
        check(cudaMemset(dx, 0, xbytes), "x0");
        check(cudaMalloc(&dy, (size_t) T * sh.n_out * 4), "y");
        auto time = [&](int n_in, int n_out) {
            for (int i = 0; i < 300; ++i) strata::kernels::native_mmvq(type, dw, dx, dy, n_in, n_out, T, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) strata::kernels::native_mmvq(type, dw, dx, dy, n_in, n_out, T, s);
            cudaEventRecord(e1, s);
            check(cudaEventSynchronize(e1), "run");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1e3 * ms / iters;
        };
        const double full = time(sh.n_in, sh.n_out);
        const double half = sh.by_rows ? time(sh.n_in, sh.n_out / 2) : time(sh.n_in / 2, sh.n_out);
        std::printf("%-30s %7.1f MB  whole %6.1f us (%4.0f GB/s)  half %6.1f us  -> %3.0f %% of whole\n", sh.name,
                    wbytes / 1e6, full, wbytes / full / 1e3, half, 100.0 * half / full);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dy);
    }
    return 0;
}
