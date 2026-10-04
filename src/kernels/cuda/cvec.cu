// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include "strata/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;
int* g_on = nullptr;
bool g_on_host = false;

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// one block per (stream, token): the pending write, then h . v over the stream, then the update
__global__ void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                            const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                            const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj,
                            int64_t inj_ld, int write) {
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fmaf(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fmaf(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
        __shared__ float part[THREADS / 32];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fmaf(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    float* d_dir = nullptr;
    float* d_s = nullptr;
    int* d_on = nullptr;
    const int one = 1;
    if (cudaMalloc(&d_dir, dir.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_s, s.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&d_on, sizeof(int)) != cudaSuccess ||
        cudaMemcpy(d_dir, dir.data(), dir.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_s, s.data(), s.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_on, &one, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: device allocation failed";
        return false;
    }
    g_cvec.dir = d_dir;
    g_cvec.s = d_s;
    g_cvec.on = d_on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    g_on = d_on;
    g_on_host = true;
    return true;
}

void cvec_set_enabled(bool on) {
    if (g_on == nullptr || on == g_on_host) return;
    cudaDeviceSynchronize();   // nothing in flight may still read the flag
    const int v = on ? 1 : 0;
    cudaMemcpy(g_on, &v, sizeof(int), cudaMemcpyHostToDevice);
    g_on_host = on;
}

bool cvec_enabled() { return g_on != nullptr && g_on_host; }

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const dim3 grid((unsigned) g_cvec.hc, (unsigned) T);
    cvec_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(R, g_cvec.dir, g_cvec.s, g_cvec.on, g_cvec.mode, layer,
                                                            (int) g_cvec.n_embd, (int) g_cvec.hc, r_ld, bo, bo_ld,
                                                            inj, inj_ld, write ? 1 : 0);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply: launch failed");
}

}  // namespace strata::kernels
