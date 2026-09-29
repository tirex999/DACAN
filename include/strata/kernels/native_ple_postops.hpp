#pragma once
#include "strata/kernels/ple.hpp"

namespace strata::kernels {

// Fixed qwen4exp single-token geometry: embd=2560, streams=4, history=9,
// convolution kernel=4/dilation=3. All buffers are caller-owned and capture-safe.
struct NativePlePostopsBuffers {
    float* key;        // 10240, normalized projected key
    float* query;      // 10240, temporary normalized hidden; may equal normalized
    float* gate;       // 4
    float* gated;      // 10240
    float* normalized; // 10240, append this to history AFTER this function
    float* conv;       // 10240
    float* result;     // 10240, may equal hidden exactly
};

// Matches pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d CUDA
// build_ple AFTER the two projections. Only norm_key/query/conv and conv1d_f16
// fields of weights are used. Gamma is full [2560,4]; history is row-fastest
// [9,10240], and F16 taps are [4,10240]. Inputs must be finite.
// Require an explicit stream, nonnull aligned pointers and disjoint writable
// spans, with only the two exact aliases documented above allowed. No allocation
// or synchronization; retain every buffer through any captured graph execution.
void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& weights, const NativePlePostopsBuffers& buffers,
                        void* stream);

// DACAN 29.09.2026: the same for n_tok consecutive tokens (every buffer holds n_tok rows back to back), each output
// bitwise the single-token call's; `hidden_result` is read and overwritten in place, and the history is advanced by
// the batch (what n_tok ple_history_advance calls with the normalized rows would leave).  `query_normalized` holds
// the normalized rows afterwards.
void native_ple_postops_batch(const float* projected_key, float* hidden_result, const float* value, float* history,
                              const PleWeights& weights, float* key, float* query_normalized, float* gate, float* gated,
                              int n_tok, void* stream);

} // namespace strata::kernels
