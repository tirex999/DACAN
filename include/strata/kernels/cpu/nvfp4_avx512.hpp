#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

bool nvfp4_avx512_ok() noexcept;

constexpr int kNvfp4MaxN = 4096;
struct alignas(64) Nvfp4Act {
    int8_t q[kNvfp4MaxN];
    float d[kNvfp4MaxN / 32];
};
void nvfp4_prepare(const void* act_q8_0, int n, Nvfp4Act* out);

void nvfp4_dots(const uint8_t* row, int n, const Nvfp4Act* const* a, int nt, float* out);
void nvfp4_dots2(const uint8_t* row0, const uint8_t* row1, int n, const Nvfp4Act* const* a, int nt, float* o0, float* o1);

void nvfp4_row_dots(const uint8_t* row, int n, const void* const* act, int nt, float* out);
void nvfp4_row2_dots(const uint8_t* row0, const uint8_t* row1, int n, const void* const* act, int nt, float* o0,
                     float* o1);

}
