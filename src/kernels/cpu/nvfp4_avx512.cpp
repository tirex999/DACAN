#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

struct BlockNvfp4 {
    uint8_t d[4];
    uint8_t qs[32];
};
struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockNvfp4) == 36, "block_nvfp4 is 36 bytes");
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 is 34 bytes");

float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) raw = ldexpf((float) man, -9);
    else raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    return raw * 0.5f;
}
struct Ue4m3Table {
    float v[256];
    Ue4m3Table() { for (int i = 0; i < 256; ++i) v[i] = ue4m3((uint8_t) i); }
};
const float* ue_table() {
    static const Ue4m3Table t;
    return t.v;
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

struct Decoded {
    __m512i mag;
    __mmask64 neg;
    __m256 ue01, ue23;
};
inline Decoded decode(const BlockNvfp4* b, const float* ue) {
    Decoded r;
    const __m256i q = _mm256_loadu_si256((const __m256i*) b->qs);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(q, m4);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
    const __m256i a = _mm256_unpacklo_epi64(lo, hi);
    const __m256i c = _mm256_unpackhi_epi64(lo, hi);
    const __m256i s01 = _mm256_permute2x128_si256(a, c, 0x20);
    const __m256i s23 = _mm256_permute2x128_si256(a, c, 0x31);
    const __m512i codes = _mm512_inserti64x4(_mm512_castsi256_si512(s01), s23, 1);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    r.mag = _mm512_shuffle_epi8(lut, codes);
    r.neg = _mm512_test_epi8_mask(codes, _mm512_set1_epi8(8));
    r.ue01 = _mm256_set_m128(_mm_set1_ps(ue[b->d[1]]), _mm_set1_ps(ue[b->d[0]]));
    r.ue23 = _mm256_set_m128(_mm_set1_ps(ue[b->d[3]]), _mm_set1_ps(ue[b->d[2]]));
    return r;
}

inline void block_fma(const Decoded& w, const Nvfp4Act* a, int ib, __m256& acc) {
    __m512i av = _mm512_load_si512((const void*) (a->q + 64 * ib));
    av = _mm512_mask_sub_epi8(av, w.neg, _mm512_setzero_si512(), av);
    const __m512 f = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w.mag, av));
    const __m256 scales01 = _mm256_mul_ps(w.ue01, _mm256_broadcast_ss(a->d + 2 * ib));
    const __m256 scales23 = _mm256_mul_ps(w.ue23, _mm256_broadcast_ss(a->d + 2 * ib + 1));
    acc = _mm256_fmadd_ps(scales01, _mm512_castps512_ps256(f), acc);
    acc = _mm256_fmadd_ps(scales23, _mm512_extractf32x8_ps(f, 1), acc);
}

template <int NT>
void dots1(const BlockNvfp4* x, int nb, const Nvfp4Act* const* a, float* out) {
    const float* ue = ue_table();
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const Decoded w = decode(x + ib, ue);
        for (int t = 0; t < NT; ++t) block_fma(w, a[t], ib, acc[t]);
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum_float_8(acc[t]);
}

template <int NT>
void dots2(const BlockNvfp4* x0, const BlockNvfp4* x1, int nb, const Nvfp4Act* const* a, float* o0, float* o1) {
    const float* ue = ue_table();
    __m256 a0[NT], a1[NT];
    for (int t = 0; t < NT; ++t) { a0[t] = _mm256_setzero_ps(); a1[t] = _mm256_setzero_ps(); }
    for (int ib = 0; ib < nb; ++ib) {
        const Decoded w0 = decode(x0 + ib, ue);
        const Decoded w1 = decode(x1 + ib, ue);
        for (int t = 0; t < NT; ++t) {
            block_fma(w0, a[t], ib, a0[t]);
            block_fma(w1, a[t], ib, a1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { o0[t] = hsum_float_8(a0[t]); o1[t] = hsum_float_8(a1[t]); }
}

}

bool nvfp4_avx512_ok() noexcept {
    static const bool ok = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
               __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
    }();
    return ok;
}

void nvfp4_prepare(const void* act_q8_0, int n, Nvfp4Act* out) {
    const auto* y = (const BlockQ8_0*) act_q8_0;
    for (int b = 0; b < n / 32; ++b) {
        std::memcpy(out->q + 32 * b, y[b].qs, 32);
        out->d[b] = _cvtsh_ss(y[b].d);
    }
}

void nvfp4_dots(const uint8_t* row, int n, const Nvfp4Act* const* a, int nt, float* out) {
    const auto* x = (const BlockNvfp4*) row;
    const int nb = n / 64;
    switch (nt) {
        case 1: dots1<1>(x, nb, a, out); break;
        case 2: dots1<2>(x, nb, a, out); break;
        case 3: dots1<3>(x, nb, a, out); break;
        case 4: dots1<4>(x, nb, a, out); break;
        case 5: dots1<5>(x, nb, a, out); break;
        case 6: dots1<6>(x, nb, a, out); break;
        case 7: dots1<7>(x, nb, a, out); break;
        default: dots1<8>(x, nb, a, out); break;
    }
}

void nvfp4_dots2(const uint8_t* row0, const uint8_t* row1, int n, const Nvfp4Act* const* a, int nt, float* o0, float* o1) {
    const auto* x0 = (const BlockNvfp4*) row0;
    const auto* x1 = (const BlockNvfp4*) row1;
    const int nb = n / 64;
    switch (nt) {
        case 1: dots2<1>(x0, x1, nb, a, o0, o1); break;
        case 2: dots2<2>(x0, x1, nb, a, o0, o1); break;
        case 3: dots2<3>(x0, x1, nb, a, o0, o1); break;
        case 4: dots2<4>(x0, x1, nb, a, o0, o1); break;
        case 5: dots2<5>(x0, x1, nb, a, o0, o1); break;
        case 6: dots2<6>(x0, x1, nb, a, o0, o1); break;
        case 7: dots2<7>(x0, x1, nb, a, o0, o1); break;
        default: dots2<8>(x0, x1, nb, a, o0, o1); break;
    }
}

void nvfp4_row_dots(const uint8_t* row, int n, const void* const* act, int nt, float* out) {
    thread_local Nvfp4Act prep[8];
    const Nvfp4Act* p[8];
    for (int t = 0; t < nt; ++t) { nvfp4_prepare(act[t], n, &prep[t]); p[t] = &prep[t]; }
    nvfp4_dots(row, n, p, nt, out);
}

void nvfp4_row2_dots(const uint8_t* row0, const uint8_t* row1, int n, const void* const* act, int nt, float* o0,
                     float* o1) {
    thread_local Nvfp4Act prep[8];
    const Nvfp4Act* p[8];
    for (int t = 0; t < nt; ++t) { nvfp4_prepare(act[t], n, &prep[t]); p[t] = &prep[t]; }
    nvfp4_dots2(row0, row1, n, p, nt, o0, o1);
}

}
