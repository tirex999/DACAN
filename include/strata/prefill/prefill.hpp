// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    double ms_ple = 0;
    int64_t experts_card2 = 0;      ///< DACAN: expert-layer groups the second card computed (enable_card2)
    int64_t streamed_card2 = 0;     ///< ...of which it streamed from the host over its own PCIe link
    int64_t rows_card2 = 0;         ///< (token, expert) rows it computed
    int64_t fetched_nvlink = 0;     ///< expert blobs one card took from the other card's cache over NVLink
    double ms_moe1 = 0;             ///< the main card's MoE time (after the gather) and the second card's
    double ms_moe2 = 0;
    double balance = 1.0;           ///< rho: the second card's ms per unit of the balance over the main card's
};

class Prefill {
public:
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// DACAN 29.09.2026: the second card takes part in every MoE layer of the prompt.  Each expert goes to the card
    /// that would finish it first: in that card's cache it costs least, in the other card's cache its weights come
    /// over NVLink, in neither it is streamed from the host over that card's own PCIe link; the second card's costs
    /// are scaled by its measured MoE time per unit over the main card's, learned chunk by chunk, so both cards end a
    /// layer's experts together.  The rows travel over NVLink, in groups the second card's buffers hold.  Every
    /// expert is computed with the same kernels either way, so the result is the one-card result.
    /// Allocates ~350 MB on that card; call after `init` (and after the decode helper has taken its share).
    bool enable_card2(const core::SecondCard& card2, std::string& err);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
