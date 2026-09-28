// include/strata/kernels/cpu/nvfp4_avx512.hpp - 28.09.2026: NVFP4 expert rows against Q8_0 activations on AVX-512
// VNNI, for several tokens at once.
//
// ggml-cpu computes an NVFP4 x Q8_0 dot on x86 with its AVX2 kernel only, one token per call - so a verify window's
// CPU experts decoded every row's 4-bit codes again for each of the window's tokens, at ~1.6 GB/s per core.  Here a
// row's codes are decoded once (64 per 512-bit register) and each token's share of a block is one VPDPBUSD against
// its activations, prepared once per call (the 64 bytes of a block in one aligned load, the Q8_0 scales as floats).
// The float side is ggml's AVX2 kernel exactly: the same int32 sum of four products per lane, the same scale
// products, the same eight-lane accumulator with the same two FMAs per block in the same order, the same horizontal
// sum - so every value is bitwise ggml_vec_dot_nvfp4_q8_0's (nvfp4_parity checks it against ggml itself).
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

/// AVX-512 F/BW/VL/DQ/VNNI and F16C present (the caller keeps ggml-cpu's path otherwise).
bool nvfp4_avx512_ok() noexcept;

/// One token's Q8_0 activation of up to kNvfp4MaxN elements, laid out for the kernel.
constexpr int kNvfp4MaxN = 4096;
struct alignas(64) Nvfp4Act {
    int8_t q[kNvfp4MaxN];        // the Q8_0 quants, contiguous
    float d[kNvfp4MaxN / 32];    // the Q8_0 block scales as floats
};
/// From ggml's block_q8_0 row (`n` a multiple of 64, <= kNvfp4MaxN).
void nvfp4_prepare(const void* act_q8_0, int n, Nvfp4Act* out);

/// out[t] = row . a[t] for `nt` (1..8) prepared activations: ggml's block_nvfp4 row of `n` elements.
void nvfp4_dots(const uint8_t* row, int n, const Nvfp4Act* const* a, int nt, float* out);
/// The same for two rows at once (an expert's gate and up rows share the activations): o0[t], o1[t].
void nvfp4_dots2(const uint8_t* row0, const uint8_t* row1, int n, const Nvfp4Act* const* a, int nt, float* o0, float* o1);

/// Convenience forms on raw Q8_0 rows (prepare, then the above) - for tests.
void nvfp4_row_dots(const uint8_t* row, int n, const void* const* act, int nt, float* out);
void nvfp4_row2_dots(const uint8_t* row0, const uint8_t* row1, int n, const void* const* act, int nt, float* o0,
                     float* o1);

}  // namespace strata::kernels::cpu
