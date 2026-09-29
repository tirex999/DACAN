# DACAN (Дацан): Qwen3.8-Flash-Next on 2× RTX 2080 Ti and two Xeon sockets

**DACAN** is our engine, a fork of [Strata](https://github.com/Niko1221/Strata) (Qwen3.8-Flash-Next with the experts
in system RAM) rebuilt for older and server hardware: **Turing cards (sm_75), two GPUs joined by NVLink, two CPU sockets
with AVX-512**. Upstream Strata builds only for sm_80+ and uses one GPU and whatever cores the OS hands it; DACAN runs
the second card as a coprocessor of every layer, computes NVFP4 experts on AVX-512 VNNI and splits the expert arena by
NUMA node. The binary and build targets are still called `strata`; "Strata" below means the upstream engine.

`main` is upstream v0.1.9 (`9e599c0`) plus our changes; the conversation cache between requests, KV streaming, IQ3_S,
prompt lookup and per-request sampling come from upstream. Checked against our v0.1.2 branch token for token: the same
256 greedy tokens at the same speed (60.45 vs 60.48 tok/s; one card, a fixed 4000-expert cache, prompt lookup off).
`turing-v0.1.2` is upstream v0.1.2 (`6da1f66`) plus the same changes, the version most numbers below were measured on.

Our rig: 2× RTX 2080 Ti 22 GB (modded, NVLink bridge, PCIe 3.0 x16 each), 2× Xeon Ice Lake (AVX-512 VNNI/VBMI),
251 GB DDR4-2666, Proxmox LXC, CUDA 13.0, driver 610, gcc 15.

## The model

DACAN runs **one model family: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, architecture
`qwen4exp` (Hugging Face `Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp`):

| | |
|---|---|
| parameters | 125B, 6B activated per token; plus a 51B n-gram embedding (bigrams/trigrams at layer 2) and a 4B MTP layer |
| layers | 48 = 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE)) |
| hidden size | 2560, gated residual over widened residual streams |
| Gated DeltaNet | 48 V heads, 16 QK heads |
| Qwen Sparse Attention | 24 Q heads, 2 KV heads, rotary dimension 64 |
| MoE | 512 experts, 10 routed + 1 shared per token, expert width 640 |
| context | 262,144 tokens natively |

The loader refuses any other geometry (`Qwen4ExpGuard` in `include/strata/artifact/gguf_reader.hpp`: 48 layers, 2560,
512 experts, 10 active, 24 / 2 heads). Fine-tunes with the same shape work; other models and pruned variants (REAP with
288 / 320 experts and the like) do not.

Weights packed for DACAN: **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)**
(26 files, 254.6 GB): Q8_0 experts we requantized from Qwen's FP8 checkpoint, NVIDIA's
[NVFP4 experts](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) repacked, the dense part, the n-gram table and
the MTP layer. Also usable: the [ISTA-DASLab GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
quants. Licenses: the model's [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE);
the NVFP4 experts also NVIDIA's Open Model License.

## Speed

Decode tokens/s through the OpenAI server on long coding answers (8–13K tokens), MTP drafts (`--spec 4`), 131K context:

| quant (experts / dense) | decoding | task, reasoning level | tok/s | checks passed |
|---|---|---|---|---|
| NVIDIA NVFP4 / Q8_0, KV int8 | sampled, T 0.6 | aquarium, low | 71.2 | 24/25 |
| NVIDIA NVFP4 / Q8_0, KV int8 | sampled, T 0.6 | aquarium, medium | 66.0 | 25/25 |
| ISTA GSQ-RCO IQ3_XXS | greedy (before sampling) | hamsters low / medium / xhigh | 68.5 / 60.8 / 50.1 | 8/8 each |
| ISTA GSQ-RCO IQ3_XXS | greedy (before sampling) | aquarium low / medium / xhigh | 68.3 / 64.6 / 57.8 | 23 / 24 / 22 of 25 |

Direct measurements on `main` (NVFP4 experts, greedy, one short prompt, 512 tokens, `--numa`, both cards; every step
below leaves the output identical token for token): 65.97 tok/s after the port to v0.1.9 → 67.24 with the second
card's rows over peer access → 69.11 with one weight read per expert group → 69.88 with the window's per-token
small launches batched. With the second card as the main card's coprocessor (below): 73.07 → 79.49 tok/s against a
control with the same experts on each executor, identical token for token (+8.8 %; 69.82 → 73.56 at ~24K tokens of
context). Q8_0 experts made from the FP8 checkpoint: 64.4 tok/s (the arena is 128 GB and fewer experts
fit on the cards), and the MTP drafts are accepted more often (0.733 vs 0.663). Long prompts, NVFP4: 65.8 tok/s at ~24K tokens of context, 52.8 at ~97K; prefill 310–330 tok/s.

The full 256K context (the OpenAI server, `--max-context 262144`, the coprocessor card on, sampling as served,
no `max_tokens`): a code word hidden at 40 % depth of a code listing was found at every length —

| prompt tokens | prefill | prefill tok/s | answer tok/s | code word |
|---|---|---|---|---|
| 7 844 | 25.4 s | 309 | 56.4 | found |
| 24 246 | 80.6 s | 301 | 54.3 | found |
| 105 017 | 6 min 9 s | 285 | 52.9 | found |
| 195 164 | 12 min 12 s | 267 | 53.5 | found |

Prefill runs on the main card alone and streams the experts it lacks over its one PCIe 3.0 link; the second card, NVLink
and the CPUs sit idle in it (next on the list). A multi-turn chat re-reads only the new text (conversation cache).

The sm_75 port alone, on one card and one socket: 25–39 tok/s on the same tasks. A direct single-prompt measurement of our 4-bit quant
(IQ4_XS/IQ4_NL experts, Q8_0 dense and n-gram table) went 39.7 → 71.7 tok/s with `--numa`, output identical bit for
bit. Full tables, scene screenshots and llama.cpp comparisons (in Russian):
[tirex999.github.io/2x2080ti-nvlink-44gb/flash-next.html](https://tirex999.github.io/2x2080ti-nvlink-44gb/flash-next.html).

## What the fork changes

| change | files | switch |
|---|---|---|
| **sm_75 port.** Scalar path instead of tf32 `mma` in the QSA attention scorer; hyper-connection kernels split their window when shared memory is short (64 KB on sm_75) | `CMakeLists.txt`, `setup.py`, `native_qsa_score.cu`, `fused_gr.cu` | — |
| **Both sockets (NUMA).** Half the rows of every expert (gate/up and down) live in each node's memory, placed by first touch before CUDA registration; each node's worker group computes its own rows, then steals. Host loop on the main card's node | `pool.*`, `pinned.*`, `expert_source.*`, `session.cpp`, `generate.cpp` | `--numa`, `STRATA_NUMA_WORKERS=a,b` |
| **N-gram table stays in RAM.** The GGUF is dropped from the page cache while the arena loads (otherwise 61 GB of arena pushes the 51 GB table out), then the table is warmed and `mlock`ed | `expert_source.cpp`, `ngram.*`, `generate.cpp` | `STRATA_PLE_LOCK=0`, `STRATA_PLE_RESIDENT=0`, `STRATA_GGUF_KEEP_CACHE=1` |
| **Second GPU** holds the experts the main card lacks and computes them alongside it, writing rows where the CPU would, so the main card's CUDA graph is unchanged | `expert_source.*`, `generate.cpp`, `verify.*`, `iq_kernels.*` | `--second-card 1`, `--second-card-usage` |
| Routing statistics accumulated across runs, and a main-card profile built from them | `expert_source.cpp`, `generate.cpp`, `tools/mk_profile_from_usage.py` | `STRATA_USAGE_DUMP=file` |
| **Sampling order.** Upstream's per-request sampling (v0.1.7: temperature, top-k, top-p, min-p, penalties, seed) plus vLLM's order: top-p and min-p over the temperature-scaled distribution (`temp_first` in the request keys; the server sets it unless told otherwise) | `sampler.cu`, `generate.cpp`, `serve/server.py` | `--temp-first`, `STRATA_SAMPLE_ORDER=llama` |
| Fast sampler kernel for top_k ≤ 32: per-warp lists in registers, ballot insertion, merge by warp 0, min-p by ballot; 104.9 µs for a 5-token window vs 1113 µs for a block-per-row kernel (greedy argmax: 48 µs). 0 mismatches against the reference kernel in 224 rows | `sampler.cu`, `sampler_fast_parity.cu` | `STRATA_SAMPLER_BLOCK=1`, `STRATA_SAMPLER_REFERENCE=1` |
| **NVIDIA NVFP4 experts.** CPU (ggml `nvfp4·q8_0` + per-expert global scale), GPU (`vec_dot_nvfp4_q8_1`, software UE4M3 since sm_75 has no FP8), fp16 unpack for prefill; the experts come straight from NVIDIA's ModelOpt checkpoint | `iq_kernels.cu`, `native_expert.*`, `expert_layout.*`, `expert_source.cpp`, `prefill.cpp`, `tools/nvfp4_experts.py` | `native_experts.txt` v4 |
| Pool: one worker per physical core on Linux (dangling `else` in `physical_cores()`), a phase ends only when every worker has woken for it; idle sleep is upstream's (v0.1.3) | `pool.*` | — |
| Hyper-connection kernels over the whole card (was 41 blocks on 68 SMs), equal to the originals up to fp32 rounding; PCIe miss path dropped from the graph at `--pcie-frac 0` | `fused_gr.cu`, `verify.*` | `STRATA_GR_V1=1` |
| IQ4_XS experts on the GPU, Q8_0 token and n-gram tables | `iq_kernels.cu`, `ngram.*` | Q8_0 table: `--ple-io mmap` only |
| **Q8_0 experts from the FP8 checkpoint.** `vec_dot_q8_0_q8_1` on the GPU, ggml on the CPU; `tools/q8_experts.py` requantizes Qwen's FP8 (128×128 block scales) the way `quantize_row_q8_0_ref` does: 0.55–0.59 % from the FP8 weights, NVFP4 is 9.8 % | `iq_kernels.cu`, `tools/q8_experts.py` | `native_experts.txt` v3 |
| **Second card's rows over peer access.** The layer plan says where each row of the window's expert parts comes from; the main card pulls only the CPU's rows from mapped memory (was every row, ~0.5 MB a layer, mostly zeros) and the second card writes its rows straight into the main card's memory (NVLink here) | `verify.*`, `expert_source.*`, `elementwise.cu`, `generate.cpp` | `STRATA_CARD2_NVLINK=0` |
| **Batched per-token launches.** The verify window ran its tiny projections once per token - router, shared-expert gate and its sigmoid, the f32→bf16 copy (and the QSA indexer's k and q in those 12 layers), plus the MTP router; now one launch for the window each (`bf16_gemv_fp32_mmvf_multi`, a block per row and token), every output bitwise as before | `native_bf16.cu`, `verify.cpp`, `shared_expert.cu`, `mtp.cpp` | — |
| **The second card as a coprocessor of every layer.** Each card replays its own CUDA graph of the verify window; they meet on flags in device memory (`(epoch << 10) \| step`, a 5 s timeout), the second card pulls the layer's input over NVLink and writes its outputs into the main card's buffers: GDN alpha/beta and z (`z`), QSA q+gate (`q`), the second half of the head's rows (`h`) and its expert share from its own graph - the host only writes the plan to mapped memory and raises a flag (`e`). Same kernels on the same inputs: bitwise a control with the same experts on the second card (`STRATA_HELPER_RESERVE_ONLY=1`). 1.3 GB of the second card's VRAM for weight copies. `docs/TWO_DOMAINS.md` has the plan | `verify.*`, `verify_kernels.*`, `expert_source.*`, `native_mmvq.*`, `generate.cpp` | `STRATA_HELPER=1` (or letters `zqhe`, `o` = half the output projections, slower), `STRATA_COMMIT_OVERLAP=1` |
| **NVFP4 experts on AVX-512 VNNI.** ggml-cpu has NVFP4 × Q8_0 in AVX2 only and unpacks a weight row again for every token; here a row is unpacked once for all the window's tokens and each token is one `VPDPBUSD` a block, the float part in ggml's exact operation order: bitwise ggml (0 of 13 608 differ), 1.3× / 1.9× / 2.5× / 2.7× a core on 1 / 2 / 3 / 4 tokens | `nvfp4_avx512.*`, `native_expert.cpp`, `nvfp4_parity.cpp` | `STRATA_NO_NVFP4_512=1` |
| Pool: a static row schedule per worker group (was work stealing by atomics across both sockets; `pool_bench` 51 → 35 µs a layer, no gain in the engine yet); router and expert combine for all the window's tokens in one launch each | `pool.*`, `pool_bench.cpp`, `native_router.cu`, `native_moe.cu` | `STRATA_POOL_V1=1` |
| **One weight read per expert group.** An expert is routed from ~1.3 of a window's tokens; for NVFP4 and Q8_0 a group of 2+ entries unpacks each weight once for all of them (bitwise the per-entry result); UE4M3 scales without `ldexpf` | `iq_kernels.cu`, `native_grouped_bench.cpp` | — |

## Build

```bash
git clone https://github.com/tirex999/DACAN && cd DACAN   # main; -b turing-v0.1.2 for the old base
# dependencies, model pack, MTP: as in upstream's README (setup.sh, tools/iq_pack.py, tools/mtp_fetch.py, tools/mtp_rt.py)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_GGML_DIR=/path/to/llama.cpp
cmake --build build -j
```

ggml from llama.cpp `3cf03257`.

## Run on two GPUs and two sockets

The main card becomes device 0, the second one device 1 (ours: the main card is card 1 on socket 1). With `--numa` the
engine finds the main card's node from its PCI address, puts the host loop there and splits the arena by node; do not
use `numactl --membind`, it would put everything on one node.

```bash
CUDA_VISIBLE_DEVICES=1,0 ./build/strata --numa \
  --pack <pack> --native <shard 1> --ple-gguf <shard 2> --ple-io mmap \
  --expert-profile data/profile_other_2609.bin --expert-cache auto --adapt-every 0 \
  --second-card 1 --second-card-usage data/usage_other_2609.bin --pcie-frac 0 \
  --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp <mtp/rt> --max-context 262144 --kv int8 --tokens ...
```

With `STRATA_HELPER=zqhe STRATA_COMMIT_OVERLAP=1` in the environment the second card also computes part of every layer
(+8.8 % on the verify round, output unchanged); it takes 1.3 GB of the second card's VRAM for weight copies. 256K of
context costs the main card ~1.9 GB of expert cache (4043 slots instead of 4786 at 131K).

The log should show `NUMA: the main card (...) is on node 1; the host loop on CPU 32, workers per group: 32 31`,
`expert arena: ... placed by NUMA node` and `n-gram table: 50.7 of 50.7 GiB in memory ...; locked`.

`tools/2x2080ti/run-fast.sh` writes a config for `serve/server.py` with the same switches and starts the OpenAI server
(`NUMA=0`: the old single-socket mode, `KV=fp16|int8`, `EXTRA="..."` for more switches).

### NVFP4 experts

```bash
python3 tools/nvfp4_experts.py <nvidia/Qwen3.8-Flash-Next-NVFP4 dir> <experts.bin> <base pack> <new pack>
ln -s <experts.bin> <directory of --native>/     # the pack names it as a shard beside --native
```

The dense part of NVIDIA's checkpoint is the same Q8_0 / BF16 / F32 as a Q8_0-dense GGUF (checked bit for bit), so
`--native` stays on that GGUF and only the experts come from the raw file (68 GB, written one expert at a time;
llama.cpp's converter holds every expert in RAM, ~100 GB for this model). Checks: our blocks unpack to NVIDIA's values
exactly (layers 0/21/47); CPU and GPU kernels agree to 8e-8 and are 1.15e-2 from float (IQ4_XS: 1.35e-2); prefill's
fp16 unpack 2e-4. The ggml NVFP4 CPU dot product is AVX2 only: 296 µs per expert and thread vs 208 µs for IQ4_XS.

## Routing tables (`data/`)

Taken on ISTA IQ3_XXS, 1024 answer tokens on six prompts (two coding tasks, two conversations, a 4.6K-token probe, a
short bench; not the hamsters or aquarium prompts). `profile_other_2609.bin`: the 9000 most used (layer, expert) pairs
for the main card; `usage_other_2609.bin`: the full counts, from which the second card's experts are chosen. Your own:
`STRATA_USAGE_DUMP=usage.bin ./build/strata ...` (accumulates), then
`python3 tools/mk_profile_from_usage.py usage.bin profile.bin 9000`.

## Things to know

- **RAM.** With `--numa` the n-gram table is locked in RAM: arena (61 GiB for 4-bit experts, 63 GiB NVFP4, 40 GiB ISTA
  IQ3_XXS) plus table (51 GiB Q8_0, 27 GiB IQ4_NL). `STRATA_PLE_LOCK=0` leaves the table in the page cache.
- **Spinning pool.** During generation the workers occupy every physical core. Anything else on those cores (a build,
  git, a VM) stalls the whole layer barrier - and so does heavy work on their SMT siblings (a compile there cost us
  several tok/s): keep measurements and builds apart.
- **SMT.** One worker per logical CPU (`STRATA_SMT=1`, 126 workers) is slower: 76.5 vs 78.3 tok/s, `pool_bench` 59.6
  vs 35.1 µs a layer - the experts are bound by the cores' vector units, not by the number of threads.
- **Main card heat.** With both sockets feeding it, the main card runs at 100 % and reaches 84 °C, where the driver
  lowers clocks (1545 of 2100 MHz). Long answers suffer most; it needs airflow.
- `--spec-split` (the window in two groups so the card and the CPUs overlap): 58.02 vs 73.07 tok/s, output identical -
  experts shared by the two groups are computed twice on the CPUs (3.72 -> 4.41 distinct a layer). Leave it off.
- Measured and left as they are: `STRATA_ASYNC_LAUNCH=1` (graph launched from its own thread) — no gain;
  `--spec-min-p` 0.35 / 0.5 / 0.65 → 67.6 / 69.9 / 69.6 tok/s, 0.5 stays.
- `--pcie-frac`: 0.15 with one card (the default 0.55 assumes PCIe 4.0), 0 with two.
- A Q8_0 output head needs `--vram-reserve-mib 1024`.
- Tried and left off: huge pages for the arena, draft window 6, graph launch from a separate thread,
  `--second-card-dup` (+2.5 % with more misses).
- Without `STRATA_HELPER` the second card computes experts only and sits at 10–15 %; with it, ~35 % of the round.
  Tensor parallel by heads over NVLink is next (`docs/TWO_DOMAINS.md`).
- `"min_max_tokens": 100000` in the server config: a client that asks for less (agents that send `max_tokens: 32000`)
  gets 100,000 - never more than the context has left; `max_tokens` above it and "unlimited" (0 / -1) are unchanged.
- One request at a time. The conversation cache (upstream v0.1.3+, `--prompt-cache`, 6 checkpoints by default) keeps
  the next turn of a conversation from re-reading the whole context; `turing-v0.1.2` has no cache. It holds **one
  conversation**: in a Claude Code session that grew from 32K to 131K tokens, 70 turns in a row took 99 % of the prompt
  from the cache (2–6 s to read a turn). Any other request in between (another chat, a separate short request of the
  same client) replaces it, and the next turn re-reads everything at ~300 tok/s: 88K and 90K tokens took 296 and 303 s.
  The engine log shows it per request: `prompt N tokens = R reused + X read`.
- A GPU-computed expert gives slightly different tokens than a CPU one (different activation quantization, both from
  llama.cpp); `--numa` does not change the output.

Upstream Strata has no license file yet; the model weights carry their own licenses (see upstream's README).
