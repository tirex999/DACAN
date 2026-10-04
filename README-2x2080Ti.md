# DACAN (Дацан): Qwen3.8-Flash-Next on 2× RTX 2080 Ti and two Xeon sockets

Build, run and measurements behind [README.md](README.md) (overview in English) and [README.ru.md](README.ru.md)
(overview in Russian).

**DACAN** is our engine, a fork of [Strata](https://github.com/Niko1221/Strata) rebuilt for older and server hardware:
**Turing cards (sm_75), two GPUs joined by NVLink, two CPU sockets with AVX-512**. The binary and build targets are
still called `strata`; "Strata" below means the upstream engine.

DACAN is based on upstream v0.1.9 (`9e599c0`); the conversation cache between requests, KV streaming, IQ3_S, prompt
lookup and per-request sampling come from upstream.

Our rig: 2× RTX 2080 Ti 22 GB (modded, NVLink bridge, PCIe 3.0 x16 each), 2× Xeon Ice Lake (AVX-512 VNNI/VBMI,
64 cores / 128 threads), 16× 16 GB DDR4 RDIMM at 2666 on all 16 channels (256 GB, 251 GiB visible), Proxmox LXC,
CUDA 13.0, driver 610, gcc 15.

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
(27 files, 260.6 GB): Q8_0 experts we requantized from Qwen's FP8 checkpoint, NVIDIA's
[NVFP4 experts](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) repacked, the dense GGUF for `--native`
(`Qwen3.8-Flash-Next-dense-Q8_0.gguf`, 5.99 GB, cut by `tools/2x2080ti/dense_gguf.py`; with it the engine gave three
greedy answers identical character for character to the service's), the n-gram table and the MTP layer. Also usable: the [ISTA-DASLab GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
quants. Licenses: the model's [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE);
the NVFP4 experts also NVIDIA's Open Model License.

## Speed

**The service as it runs now, measured 03.10.2026 on a free node** (NVFP4 experts, `--prefill 8192 --park 16`,
the neighbouring VM's workload stopped for the run). Greedy, no reasoning unless noted; every request is new text,
not in the engine's cache; speeds from the engine's own line (`strata serve: prompt … read in …, … generated in …`):

| request | prompt tokens | prompt read, tok/s | answer tokens | answer, tok/s | drafts accepted |
|---|---:|---:|---:|---:|---:|
| Python module (in-memory SQL: parser, executor, tests) | 72 | — | 10 250 | **76.9** | 85 % |
| list from a fresh 19K-token document (25 facts of 25 right) | 19 338 | **785.6** | 597 | **80.4** | 89 % |
| list from a fresh 98K-token document (11 of 11) | 97 875 | **714.6** | 259 | **69.9** | 87 % |
| LRU cache, reasoning low (the 27.09 calibration prompt) | 88 | — | 1 984 | **71.9** | 67 % |
| essay in English | 58 | — | 1 931 | **60.4** | 57 % |
| essay in Russian | 77 | — | 2 340 | **47.0** | 32 % |

The spread is the MTP drafts: code and lists are predictable, prose less so, Russian prose least. One "tok/s" without
the kind of text says little.

The tables below are earlier measurements (27–28.09).

Decode tokens/s through the OpenAI server on long coding answers (8–13K tokens), MTP drafts (`--spec 4`), 131K context:

| quant (experts / dense) | decoding | task, reasoning level | tok/s | checks passed |
|---|---|---|---|---|
| NVIDIA NVFP4 / Q8_0, KV int8 | sampled, T 0.6 | aquarium, low | 71.2 | 24/25 |
| NVIDIA NVFP4 / Q8_0, KV int8 | sampled, T 0.6 | aquarium, medium | 66.0 | 25/25 |
| ISTA GSQ-RCO IQ3_XXS | greedy (before sampling) | hamsters low / medium / xhigh | 68.5 / 60.8 / 50.1 | 8/8 each |
| ISTA GSQ-RCO IQ3_XXS | greedy (before sampling) | aquarium low / medium / xhigh | 68.3 / 64.6 / 57.8 | 23 / 24 / 22 of 25 |

On one short prompt (NVFP4 experts, greedy, 512 tokens, both cards and both sockets) DACAN went from 65.97 tok/s right
after the port to v0.1.9 to 79.49 tok/s now, every step checked token for token against the previous output. Q8_0
experts made from the FP8 checkpoint: 64.4 tok/s, and the MTP drafts are accepted more often (0.733 vs 0.663). Long
prompts, NVFP4: 65.8 tok/s at ~24K tokens of context, 52.8 at ~97K.

The full 256K context (the OpenAI server, `--max-context 262144`, sampling as served, no `max_tokens`; measured
28.09, before the faster prompt reading): a code word hidden at 40 % depth of a code listing was found at every
length —

| prompt tokens | prefill | prefill tok/s | answer tok/s | code word |
|---|---|---|---|---|
| 7 844 | 25.4 s | 309 | 56.4 | found |
| 24 246 | 80.6 s | 301 | 54.3 | found |
| 105 017 | 6 min 9 s | 285 | 52.9 | found |
| 195 164 | 12 min 12 s | 267 | 53.5 | found |

Since 29.09 a fresh prompt reads at 786 tok/s at 19K and 715 at 98K (table above). A multi-turn chat re-reads only the
new text (conversation cache).

The sm_75 port alone, on one card and one socket: 25–39 tok/s on the same tasks. Full tables, scene screenshots and
llama.cpp comparisons (in Russian):
[tirex999.github.io/2x2080ti-nvlink-44gb/flash-next.html](https://tirex999.github.io/2x2080ti-nvlink-44gb/flash-next.html).

## What the fork adds

| | switch |
|---|---|
| **Turing (sm_75)**: builds and runs on RTX 20 cards | `-DCMAKE_CUDA_ARCHITECTURES=75` |
| **Two CPU sockets** | `--numa`, `STRATA_NUMA_WORKERS=a,b` |
| **The second GPU** | `--second-card 1`, `--second-card-usage <table>`; `STRATA_HELPER=zqhe STRATA_COMMIT_OVERLAP=1` for more speed |
| **The n-gram table stays in RAM** | `STRATA_PLE_LOCK=0` leaves it in the page cache |
| **Faster prompt reading** | `--prefill 8192`; `STRATA_PREFILL_CARD2=0` keeps the prompt on one card |
| **Parked conversations**: other chats wait in RAM instead of being re-read | `--park N`, `--park-mib`, `--park-min` |
| **NVIDIA NVFP4 experts** straight from NVIDIA's ModelOpt checkpoint, on the GPU and on AVX-512 VNNI | `native_experts.txt` v4, `tools/nvfp4_experts.py` |
| **Q8_0 experts from Qwen's FP8 checkpoint** (0.55–0.59 % from the FP8 weights; NVFP4 is 9.8 %) | `native_experts.txt` v3, `tools/q8_experts.py` |
| IQ4_XS experts on the GPU, Q8_0 token and n-gram tables | Q8_0 table: `--ple-io mmap` only |
| **Sampling order**: upstream's per-request sampling plus vLLM's order (top-p and min-p over the temperature-scaled distribution) | `--temp-first`, `STRATA_SAMPLE_ORDER=llama` |
| Faster sampler kernel for top_k ≤ 32 | `STRATA_SAMPLER_BLOCK=1` turns it off |
| Routing statistics of your own runs | `STRATA_USAGE_DUMP=file`, `tools/mk_profile_from_usage.py` |

## Build

```bash
git clone https://github.com/tirex999/DACAN && cd DACAN
# dependencies, model pack, MTP: as in upstream's README, docs/STRATA_README.md (setup.sh, tools/iq_pack.py, tools/mtp_fetch.py, tools/mtp_rt.py)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_GGML_DIR=/path/to/llama.cpp
cmake --build build -j
```

ggml from llama.cpp `3cf03257`.

## Run on two GPUs and two sockets

The main card becomes device 0, the second one device 1 (ours: the main card is card 1 on socket 1). Do not use
`numactl --membind` with `--numa`.

```bash
CUDA_VISIBLE_DEVICES=1,0 ./build/strata --numa \
  --pack <pack> --native <shard 1> --ple-gguf <shard 2> --ple-io mmap \
  --expert-profile data/profile_other_2609.bin --expert-cache auto --adapt-every 0 \
  --second-card 1 --second-card-usage data/usage_other_2609.bin --pcie-frac 0 \
  --prefill 8192 --spec 4 --spec-min-p 0.5 --mtp <mtp/rt> --max-context 262144 --kv int8 --tokens ...
```

Add `STRATA_HELPER=zqhe STRATA_COMMIT_OVERLAP=1` to the environment for about +8.8 % (output unchanged; it takes
1.3 GB of the second card's VRAM). 256K of context leaves less VRAM for the rest and costs a little speed.

The log should show `NUMA: ...` with both worker groups and `n-gram table: 50.7 of 50.7 GiB in memory ...; locked`.

`tools/2x2080ti/run-fast.sh` writes a config for `serve/server.py` with the same switches and starts the OpenAI server
(`NUMA=0`: the old single-socket mode, `KV=fp16|int8`, `EXTRA="..."` for more switches).

### NVFP4 experts

```bash
python3 tools/nvfp4_experts.py <nvidia/Qwen3.8-Flash-Next-NVFP4 dir> <experts.bin> <base pack> <new pack>
ln -s <experts.bin> <directory of --native>/     # the pack names it as a shard beside --native
```

The dense part of NVIDIA's checkpoint is the same Q8_0 / BF16 / F32 as a Q8_0-dense GGUF (checked bit for bit), so
`--native` stays on that GGUF and only the experts come from the raw file (68 GB, written one expert at a time;
llama.cpp's converter holds every expert in RAM, ~100 GB for this model). Our blocks unpack to NVIDIA's values exactly
(layers 0/21/47).

## Routing tables (`data/`)

`profile_other_2609.bin` and `usage_other_2609.bin` are the tables the command above uses, taken on ISTA IQ3_XXS
(1024 answer tokens on six prompts: two coding tasks, two conversations, a 4.6K-token probe, a short bench). Your own:
`STRATA_USAGE_DUMP=usage.bin ./build/strata ...` (accumulates), then
`python3 tools/mk_profile_from_usage.py usage.bin profile.bin 9000`.

## Things to know

- **RAM.** With `--numa` the n-gram table is locked in RAM: experts (61 GiB for 4-bit, 63 GiB NVFP4, 40 GiB ISTA
  IQ3_XXS) plus table (51 GiB Q8_0, 27 GiB IQ4_NL). `STRATA_PLE_LOCK=0` leaves the table in the page cache.
- **Keep the cores free.** During generation the engine uses every physical core. Anything else on those cores or on
  their SMT siblings (a build, git, a VM) costs several tok/s: keep measurements and builds apart.
- **SMT.** One worker per logical CPU (`STRATA_SMT=1`) is slower (76.5 vs 78.3 tok/s); leave it off.
- **Main card heat.** The main card runs at 100 % and reaches 84 °C, where the driver lowers clocks (1545 of
  2100 MHz). Long answers suffer most; it needs airflow.
- `--spec-split`: slower (58.02 vs 73.07 tok/s). Leave it off.
- `--spec-min-p` 0.35 / 0.5 / 0.65 → 67.6 / 69.9 / 69.6 tok/s, 0.5 stays.
- `--pcie-frac`: 0.15 with one card (the default 0.55 assumes PCIe 4.0), 0 with two.
- A Q8_0 output head needs `--vram-reserve-mib 1024`.
- `"min_max_tokens": 100000` in the server config: a client that asks for less (agents that send `max_tokens: 32000`)
  gets 100,000 - never more than the context has left; `max_tokens` above it and "unlimited" (0 / -1) are unchanged.
- One request at a time. The conversation cache (upstream v0.1.3+, `--prompt-cache`) keeps the next turn of a
  conversation from re-reading the whole context: in a Claude Code session that grew from 32K to 131K tokens, 70 turns
  in a row took 99 % of the prompt from the cache. Other conversations are parked in RAM (`--park N`, default 4;
  `--park-mib`, default 8192; `--park-min`, default 8192 tokens): a 16.8K-token chat parked in 759 ms, brought back in
  55 ms, its next answer identical to the uninterrupted run. The service runs `--park 16 --park-mib 16384`. The engine
  log shows it per request: `prompt N tokens = R reused + X read`, `parked ...`, `brought back ...`.
- A GPU-computed expert gives slightly different tokens than a CPU one (different activation quantization, both from
  llama.cpp); `--numa` does not change the output.

Upstream Strata is MIT-licensed since 0.1.16; the model weights carry their own licenses (see upstream's README).
