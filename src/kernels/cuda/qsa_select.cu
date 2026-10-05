// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

constexpr int TK = 32, TQ = 64, TU = 4;
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_tiled_kernel(const float* __restrict__ pooled,
                                                                              const float* __restrict__ dead,
                                                                              const float* __restrict__ q_idx,
                                                                              const int32_t* __restrict__ steps,
                                                                              int64_t nq, int64_t max_blocks,
                                                                              float* __restrict__ out) {
    __shared__ __align__(16) float sk[TK * IDX_DIM];
    __shared__ __align__(16) float sdead[IDX_DIM];
    __shared__ long long s_maxbid;
    const int64_t q0 = (int64_t) blockIdx.y * TQ;
    const int64_t q1 = q0 + TQ < nq ? q0 + TQ : nq;
    const int64_t b0 = (int64_t) blockIdx.x * TK;
    if (threadIdx.x == 0) {
        long long m = -1;
        for (int64_t qi = q0; qi < q1; ++qi) {
            const long long v = steps[qi * kStepCount + kStepNBid];
            m = v > m ? v : m;
        }
        s_maxbid = m;
    }
    __syncthreads();
    const int64_t maxbid = s_maxbid;
    if (b0 > maxbid || b0 >= max_blocks) return;
    for (int i = threadIdx.x; i < TK * IDX_DIM / 4; i += blockDim.x) {
        const int r = i / (IDX_DIM / 4);
        const int64_t b = b0 + r;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (b < maxbid && b < max_blocks) v = reinterpret_cast<const float4*>(pooled + b * IDX_DIM)[i % (IDX_DIM / 4)];
        reinterpret_cast<float4*>(sk)[i] = v;
    }
    if (threadIdx.x < IDX_DIM / 4)
        reinterpret_cast<float4*>(sdead)[threadIdx.x] = reinterpret_cast<const float4*>(dead)[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int64_t qi = q0 + warp; qi < q1; qi += SCORE_WARPS) {
        const int32_t* st = steps + qi * kStepCount;
        const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
        float4 q4[IDX_HEADS];
        const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) q4[h] = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        int64_t bend = b0 + TK;
        if (bend > max_blocks) bend = max_blocks;
        if (bend > n_bid + 1) bend = n_bid + 1;
        float keep = 0.0f;
        for (int64_t bb = b0; bb < bend; bb += TU) {
            float sc[TU];
#pragma unroll
            for (int u = 0; u < TU; ++u) {
                const int64_t b = bb + u;
                const float* key = (b == n_bid) ? sdead : sk + (b - b0 < TK ? b - b0 : 0) * IDX_DIM;
                const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    float d = k4.x * q4[h].x + k4.y * q4[h].y + k4.z * q4[h].z + k4.w * q4[h].w;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                    score += d > 0.0f ? d : 0.0f;
                }
                if (b == n_bid && n_kv % R != 0) score += 1e9f;
                sc[u] = score;
            }
#pragma unroll
            for (int u = 0; u < TU; ++u) {
                const int64_t b = bb + u;
                if (b < bend) {
                    const int slot = (int) ((b - b0) & 31);
                    if (lane == slot) keep = sc[u];
                    if ((slot == 31 || b + 1 == bend) && lane <= slot) out[qi * max_blocks + (b - slot) + lane] = keep;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

__device__ __forceinline__ int topk_block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < TOPK_T / 32 ? s_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < TOPK_T / 32) s_warp[lane] = w;
    }
    __syncthreads();
    const int warp_before = warp ? s_warp[warp - 1] : 0;
    total = s_warp[TOPK_T / 32 - 1];
    __syncthreads();
    return warp_before + x - v;
}

__global__ void __launch_bounds__(TOPK_T) block_topk_fast_kernel(const float* __restrict__ scores,
                                                                 const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                 int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_warp[TOPK_T / 32];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TOPK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int64_t run_eq = 0, run_out = 0;
    for (int64_t base = 0; base < nb; base += TOPK_T) {
        const int64_t b = base + t;
        int w = 0;
        uint32_t k = 0;
        if (b < nb) {
            w = weight(b);
            if (w) k = order_key(sc[b]);
        }
        const int gt = (w && k > thr) ? w : 0;
        const int eq = (w && k == thr) ? w : 0;
        int eq_tot = 0;
        const int eq_before = topk_block_excl_scan(eq, s_warp, eq_tot);
        int64_t my_eq = eq_budget - (run_eq + eq_before);
        if (my_eq < 0) my_eq = 0;
        if (my_eq > eq) my_eq = eq;
        const int sel = gt + (int) my_eq;
        int sel_tot = 0;
        int64_t wpos = run_out + topk_block_excl_scan(sel, s_warp, sel_tot);
        for (int c = 0; c < sel; ++c) out[wpos++] = (int32_t) (b * R + c);
        run_eq += eq_tot;
        run_out += sel_tot;
    }
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    static const bool old_path = std::getenv("STRATA_SEL_OLD") != nullptr;
    if (nq > 1 && !old_path) {
        const dim3 grid((unsigned) ((max_blocks + TK - 1) / TK), (unsigned) ((nq + TQ - 1) / TQ));
        block_scores_tiled_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps,
                                                                                        nq, max_blocks, scores);
    } else {
        const dim3 grid((unsigned) ((max_blocks + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
        block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps,
                                                                                  max_blocks, scores);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    static const bool old_topk = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (old_topk)
        block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk_fast_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
