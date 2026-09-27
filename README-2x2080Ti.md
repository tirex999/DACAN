# Strata on 2× RTX 2080 Ti and two Xeon sockets

A fork of [Strata](https://github.com/Niko1221/Strata) (Qwen3.8-Flash-Next with the experts in system RAM) for older
and server hardware: **Turing cards (sm_75), two GPUs, two CPU sockets with AVX-512**. Upstream Strata builds only for
sm_80+ and uses one GPU and whatever cores the OS hands it.

`main` is upstream v0.1.9 (`9e599c0`) plus our changes; the conversation cache between requests, KV streaming, IQ3_S,
prompt lookup and per-request sampling come from upstream. Checked against our v0.1.2 branch token for token: the same
256 greedy tokens at the same speed (60.45 vs 60.48 tok/s; one card, a fixed 4000-expert cache, prompt lookup off).
`turing-v0.1.2` is upstream v0.1.2 (`6da1f66`) plus the same changes, the version most numbers below were measured on.

Our rig: 2× RTX 2080 Ti 22 GB (modded, NVLink bridge, PCIe 3.0 x16 each), 2× Xeon Ice Lake (AVX-512 VNNI/VBMI),
251 GB DDR4-2666, Proxmox LXC, CUDA 13.0, driver 610, gcc 15.

## Speed

Decode tokens/s through the OpenAI server on long coding answers (8–13K tokens), MTP drafts (`--spec 4`), 131K context:

| quant (experts / dense) | decoding | task, reasoning level | tok/s | checks passed |
|---|---|---|---|---|
| NVIDIA NVFP4 / Q8_0, KV int8 | sampled, T 0.6 | aquarium, low | 71.2 | 24/25 |
| NVIDIA NVFP4 / Q8_0, KV int8 | sampled, T 0.6 | aquarium, medium | 66.0 | 25/25 |
| ISTA GSQ-RCO IQ3_XXS | greedy (before sampling) | hamsters low / medium / xhigh | 68.5 / 60.8 / 50.1 | 8/8 each |
| ISTA GSQ-RCO IQ3_XXS | greedy (before sampling) | aquarium low / medium / xhigh | 68.3 / 64.6 / 57.8 | 23 / 24 / 22 of 25 |

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

## Build

```bash
git clone https://github.com/tirex999/strata-2x2080ti && cd strata-2x2080ti   # main; -b turing-v0.1.2 for the old base
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
  --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp <mtp/rt> --max-context 131072 --kv int8 --tokens ...
```

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
  git, a VM) stalls the whole layer barrier: keep other work on the SMT siblings or wait.
- **Main card heat.** With both sockets feeding it, the main card runs at 100 % and reaches 84 °C, where the driver
  lowers clocks (1545 of 2100 MHz). Long answers suffer most; it needs airflow.
- `--pcie-frac`: 0.15 with one card (the default 0.55 assumes PCIe 4.0), 0 with two.
- A Q8_0 output head needs `--vram-reserve-mib 1024`.
- Tried and left off: huge pages for the arena, draft window 6, graph launch from a separate thread,
  `--second-card-dup` (+2.5 % with more misses).
- The second card computes experts only; the dense part of every layer runs on the main card, so the second card sits
  at 10–15 %. Tensor-parallel dense layers over NVLink are on the list.
- One request at a time. The conversation cache (upstream v0.1.3+, `--prompt-cache`, 6 checkpoints by default) keeps
  the next turn of a conversation from re-reading the whole context; `turing-v0.1.2` has no cache.
- A GPU-computed expert gives slightly different tokens than a CPU one (different activation quantization, both from
  llama.cpp); `--numa` does not change the output.

Upstream Strata has no license file yet; the model weights carry their own licenses (see upstream's README).
