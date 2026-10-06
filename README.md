# DACAN (Дацан)

🇬🇧 **English** | 🇷🇺 [Русский](README.ru.md) · [❤️ Support the project](#support-the-project)

**DACAN** is an inference engine for **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** — a 125B
mixture-of-experts model — on ordinary hardware: **any NVIDIA card from the RTX 20 series up and any x86-64 CPU with
AVX2; one or two cards, one or two CPU sockets, with or without AVX-512**. It is our fork of
[Strata](https://github.com/Niko1221/Strata) that adds Turing cards (sm_75), a second card and a second CPU socket.
We build and measure it on two RTX 2080 Ti (22 GB, NVLink) and two Xeon sockets with AVX-512; the same engine without
AVX-512 and on one card and one socket is measured below. The binary and build targets are still called `strata`.

## What it runs on

| | needed | notes |
|---|---|---|
| **GPU** | NVIDIA, compute capability 7.5 or newer: RTX 20 (Turing), RTX 30 (Ampere), RTX 40 (Ada), RTX 50 (Blackwell) and their workstation cards; 12 GB of VRAM or more | one or two cards; NVLink is not required; the two cards may be of different generations. Fewer than 12 GB runs, but slowly. AMD and Intel GPUs are not supported (the engine is CUDA) |
| **CPU** | x86-64, Intel or AMD, with **AVX2** (Intel Core and Xeon from Haswell, 2013; AMD Ryzen, Threadripper and EPYC) | **AVX-512 is optional**: when the CPU has AVX-512 with VNNI and VBMI (Intel Xeon from Ice Lake on, Intel Core 11th gen, AMD Zen 4 and Zen 5) the engine uses it by itself; otherwise it runs on AVX2 — that includes Intel Core 12th gen and newer and Skylake / Cascade Lake Xeons. **One or two sockets**: add `--numa` for two |
| **RAM** | 48–64 GB for the GSQ-RCO quants (2–3.5 bit); 128 GB and more for our NVFP4 + Q8 (63 GiB of experts + the 51 GiB n-gram table), 192 GB for our Q8_0 (120 + 51 GiB) | all experts live in RAM; the GPU keeps copies of the most used ones |
| **Driver, OS** | NVIDIA driver 580+ (CUDA 13.0); Linux or Windows (we test on Linux) | the installer checks all of the above (`./setup.sh --check`) |

The build targets the card: `-DCMAKE_CUDA_ARCHITECTURES=75` (RTX 20), `86` (RTX 30), `89` (RTX 40), `120` (RTX 50),
several separated by `;` for mixed cards. Other NVIDIA cards from compute capability 7.5 (data-centre T4, A100, L40,
H100 and the like) take their own value (`75`, `80`, `89`, `90`); we have not tried them. Build on the machine that will
run it, or with `-DSTRATA_PORTABLE=ON` for a binary that runs on any AVX2 CPU.

## How fast

Measured 03.10.2026 on our rig — two cards, two sockets, AVX-512 (NVFP4 experts, greedy, every request new text,
speeds from the engine's own log):

| request | prompt | prompt read | answer | answer speed |
|---|---:|---:|---:|---:|
| a Python module (SQL parser, executor, tests) | 72 tokens | — | 10 250 tokens | **76.9 tok/s** |
| a list from a fresh 19K-token document | 19 338 | **786 tok/s** | 597 | **80.4 tok/s** |
| a list from a fresh 98K-token document | 97 875 | **715 tok/s** | 259 | **69.9 tok/s** |
| an essay in English / in Russian | 58 / 77 | — | 1 931 / 2 340 | 60.4 / 47.0 tok/s |

The full 256K context works (a code word hidden at 40 % of a 195K-token listing was found). The spread between rows is
the MTP draft acceptance: 85–89 % on code and lists, 57 % on English prose, 32 % on Russian. Earlier tables and the
256K test: **[README-2x2080Ti.md](README-2x2080Ti.md)**. Gallery of generated scenes, comparisons with llama.cpp and
FreeToken (in Russian): **[the site](https://tirex999.github.io/2x2080ti-nvlink-44gb/flash-next.html)**.

### Long context: up to 524,288 tokens (05.10.2026)

With `--max-context 524288 --kv int8 --kv-resident 32768` the KV cache lives in RAM (6.2 GiB for the whole window),
twice the model's native 262,144. Checked on Swift 1.5 NVFP4 + Q8 (DACAN), both cards, both sockets:

| prompt | prompt read | check |
|---:|---:|---|
| 234,727 tokens | 350 s (**670 tok/s**) | 7 of 7 needles |
| 461,866 tokens | 833 s (**555 tok/s**) | 7 of 7, 4 of 4 past the 262K mark |
| 452,336 tokens | — | a function at ~340K explained step by step and computed correctly, as at 32K |

Long prompts read faster in this version — engine changes plus reading in 16,384-token chunks (`--prefill 16384`):
235K 409 → 350 s, 462K 1,034 → 833 s. The needles found after the speed-up are the same, 7 of 7.

### Without AVX-512, and on one card and one socket (06–07.10.2026)

The same machine, engine and weights (Swift 1.5 NVFP4 + Q8), greedy, the answer capped at 2,500 tokens, answer speed
from the engine's metrics, tok/s. The code request (a Python module with an SQL parser and tests) was sent twice, then
an essay.

| setup | code, 1st run | code, 2nd run | essay |
|---|---:|---:|---:|
| 2 cards, 2 sockets, AVX-512 | **82.0** | **74.0** | **51.0** |
| 2 cards, 2 sockets, AVX2 only | 70.9 | 65.8 | 47.0 |
| 1 card, 1 socket, AVX-512 | 56.2 | 51.5 | 38.2 |
| 1 card, 1 socket, AVX2 only | 55.5 | 51.9 | 38.1 |

- **AVX2 only**: built with `-DSTRATA_PORTABLE=ON`, run with `STRATA_FORCE_AVX2=1 STRATA_NO_NVFP4_512=1
  STRATA_NO_IQ512=1` and AVX-512 hidden from the C library
  (`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX512F,-AVX512VL,-AVX512BW,-AVX512DQ,-AVX512CD`); the engine then prints
  `this CPU has no AVX-512: the expert kernels run on AVX2`. The answers are the same as with AVX-512, token for token.
- **1 card, 1 socket**: the second card off and the process bound to one NUMA node (`CUDA_VISIBLE_DEVICES=1 numactl
  --cpunodebind=1 --preferred=1`, no `--numa`, no `--second-card`), `--pcie-frac 0.15`, 131,072 of context, `--vram-reserve-mib 1024`.
- Another VM on the host was busy during all four runs.

A CPU without AVX-512 costs 8–14 % here with two cards and two sockets, and at most 1.2 % with one card and one socket.

## DACAN vs upstream Strata 0.1.39 (measured 04.10.2026)

In one week Strata's author released 30 versions (0.1.10 to 0.1.39): a model split by layers across two or three
GPUs, several requests at once, official RTX 20 support, the MIT license. We built 0.1.39 on our rig (sm_75,
CUDA 13.0) and put it next to DACAN on the same model and the same request.

| engine | prompt read (1 902 tokens) | answer (512 tokens) |
|---|---:|---:|
| **DACAN** (two cards, two sockets) | 586 tok/s | **67.4 tok/s** |
| upstream Strata 0.1.39 (its layer split across the two cards) | 605 tok/s | 17.7 tok/s |

Same for both: Swift 1.5 IQ3_XXS by UkisAI, the same pack, MTP draft head and expert profile, the prompt "hamsters —
honest physics" (1 902 tokens), greedy, 512 answer tokens, each engine through its own server. Prompt reading is close
(3 % apart); **DACAN answers 3.8 times faster** on this hardware. Our production model is Swift 1.5 NVFP4 + Q8 (DACAN):
58 tok/s on DACAN (median of 9 runs; Swift 1.5 Q8_0: 49 tok/s). Upstream 0.1.39 cannot load our quants: it takes
the n-gram table only as IQ4_NL, Q5_0 or FP8, and ours is Q8_0.

What we take from upstream next, one change at a time and measured: several requests at once (`parallel`), the
system-prompt checkpoint, the faster sampler (#197), the watchdog that restarts a silent engine.

DACAN is a joint project of Vadim and Claude — one team. On its own hardware it is faster than the original, and we
built it together.

## The model

One model family: Qwen3.8-Flash-Next, architecture `qwen4exp` (Hugging Face `Qwen4ExpForConditionalGeneration`) —
125B parameters with 6B active per token, plus a 51B n-gram embedding table and a 4B MTP layer; 48 layers of
3 × Gated DeltaNet + 1 × Qwen Sparse Attention, each followed by a MoE of 512 experts (10 routed + 1 shared); native
context 262,144 tokens, up to 524,288 on DACAN with KV streaming (`--kv-resident`). The loader checks this geometry and refuses anything else, so fine-tunes of the same shape work —
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next) by UkisAI runs unchanged — while other models and
pruned variants (REAP etc.) do not.

Weights:

- **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)** — the original
  model: **NVFP4 + Q8 (DACAN)** with NVIDIA's NVFP4 experts repacked (68 GB), or our Q8_0
  experts from Qwen's FP8 checkpoint (128 GB); the dense GGUF, the n-gram table, the MTP layer.
- **[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN)** —
  Swift 1.5, our quants from its BF16: 8-bit (Q8_0) and **NVFP4 + Q8 (DACAN)** with our NVFP4 experts — what our service runs (uploaded 04.10.2026: 30 files, 321 GB).
- The GSQ-RCO GGUF quants (2–3.5 bit) — [by ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  for the original model and [by UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) for
  Swift 1.5; the installer sets these up.

**NVFP4 + Q8 (DACAN)** is our name for this mix. The routed experts — about 97 % of the 125B weights — are 4-bit NVFP4
(FP4 E2M1 values, an FP8 E4M3 scale per 16 values, an FP32 scale per expert matrix). The large matrices every token passes
through (attention and DeltaNet projections, the shared expert, embeddings, output head) and the n-gram table are 8-bit
Q8_0; the small sensitive ones (router, hyper-connections, the sparse-attention indexer, DeltaNet α/β, 1.4 GiB in all)
stay BF16.

## What it does

- **One card or two, one CPU socket or two.** On one card and one socket it runs as is; `--second-card 1` adds the
  second card, `--numa` the second socket.
- **CPU experts on AVX2 or AVX-512 (VNNI)** — picked at start by what the CPU has, results bitwise equal to ggml's.
- **MTP speculative decoding**: the model's own draft layer proposes 4 tokens, one pass checks them.
- **Fast prompt reading**: a fresh 19K-token prompt at ~800 tok/s, a 98K one at ~715, 235K at 670, 462K at 555.
- **The n-gram table** (51 GiB as Q8_0) stays in RAM; **conversations are cached** — the next turn reads only what is
  new, and up to 16 other chats are parked in RAM (since 06.10.2026 with KV streaming too), so a returning chat does
  not re-read its whole context.
- **Server**: OpenAI-compatible (`/v1/chat/completions`) and Anthropic-compatible (`/v1/messages`), a web app with
  Chat and a live Monitor of the cards, sockets and RAM, the answer speed and the prompt-read speed (fresh
  tokens per second, live while a prompt is read and per request); images through the model's mmproj encoder.

## Hardware

Requirements for any machine — in [What it runs on](#what-it-runs-on). Ours: 2× RTX 2080 Ti 22 GB (NVLink, PCIe 3.0
x16 each), 2× Xeon Ice Lake (64 cores, AVX-512 VNNI), 256 GB DDR4-2666 on 16 channels. RAM needed: the experts (40–50 GiB for GSQ-RCO, 63 GiB NVFP4, 120 GiB Q8_0) plus the n-gram table: 51 GiB as Q8_0
in our quants; the GSQ-RCO IQ4_NL table (27 GiB) is read straight from disk. The arithmetic is in
[docs/MANUAL.ru.md](docs/MANUAL.ru.md).

Other machines users have run it on:

- **One RTX 4090 48 GB**, Ryzen 7 9700X: 95 tok/s on a long answer at 75–85K of context after filling the card with experts
  — see [docs/ONE_BIG_CARD.md](docs/ONE_BIG_CARD.md) if your big card shows "8 000 experts cached".
- **RTX 4070 Ti SUPER + RTX 5060 Ti (16 GB each, two generations, no NVLink), 2× Xeon E5-2678 v3 (Haswell, AVX2
  only)**: about 45 tok/s while reasoning.

## Install and run

- **One card, Windows or Linux** — the installer (from upstream Strata): `START-HERE.bat` (Windows) or `./setup.sh`
  (Linux) asks for the GSQ-RCO model, size, context, KV cache (8, 4 or 16 bit; 16 for 24 GB cards and up) and images,
  downloads everything and starts the server; from 20 GB of VRAM it uses the full expert profile. Without `--build`
  it installs upstream Strata's ready-made engine, with `--build` it compiles DACAN for your card.
  Details: [docs/DETAILS.md](docs/DETAILS.md).
- **Two cards and/or two sockets**: build with your cards' architecture (ours: `-DCMAKE_CUDA_ARCHITECTURES=75`) and
  add `--second-card 1` for the second card and `--numa` for the second socket — the full command and the switches:
  [README-2x2080Ti.md](README-2x2080Ti.md). `tools/2x2080ti/run-fast.sh` writes the server config and starts it.
- **Without AVX-512** nothing changes: the same build and switches; the engine picks AVX2 by itself.
- **By hand, any model variant, any hardware** - our NVFP4 + Q8 and Q8_0 quants from Hugging Face, the GSQ-RCO
  quants, one or two cards, up to 524K of context: **[docs/MANUAL.ru.md](docs/MANUAL.ru.md)** (in Russian) - which
  files, how to build, the config and the switches.

Once running: the web app at the server's address, `/v1` for OpenAI-compatible clients and coding agents,
`/v1/messages` for Anthropic's API.

## Documents

| | |
|---|---|
| [README-2x2080Ti.md](README-2x2080Ti.md) | build and run on two cards, switches, measurements |
| [docs/MANUAL.ru.md](docs/MANUAL.ru.md) | by hand (in Russian): every model variant, build, config, switches, memory |
| [docs/ONE_BIG_CARD.md](docs/ONE_BIG_CARD.md) | one 24–48 GB card: fill the VRAM with experts, 16-bit KV |
| [docs/DETAILS.md](docs/DETAILS.md) | upstream's full guide: install, API, images, troubleshooting |
| [docs/STRATA_README.md](docs/STRATA_README.md) | upstream Strata's README |
| `tools/2x2080ti/` | our tools: dense GGUF, Swift quantizers (Q8_0, NVFP4, ModelOpt), RadixArk metadata |

## Credits and licenses

[Strata](https://github.com/Niko1221/Strata) by Niko1221 — the engine this is built on (MIT since 0.1.16).
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Model weights carry their own licenses: Qwen Community
License 1.0 for Qwen3.8-Flash-Next, NVIDIA Open Model License for NVIDIA's NVFP4 experts, Swift Open License v1.0 for
Swift 1.5.

---

## Support the project

Everything here — the engine, the quants, the measurements — is made and published for free. If it helped you run a
model on your own hardware, you can say thanks with a donation.

| | address | QR |
|---|---|---|
| **YooMoney** (roubles: a YooMoney wallet or any bank card) | `4100119356331418` · [send](https://yoomoney.ru/to/4100119356331418) | <img src="docs/donate/yoomoney.svg" width="130" alt="QR YooMoney"> |
| **USDT** (TRC-20, Tron network) | `TBvoJHi7uyonSpvdH9Y6RAAXGWYVR2jeqw` | <img src="docs/donate/usdt-trc20.svg" width="130" alt="QR USDT TRC-20"> |
| **Ethereum** (ETH and Ethereum-network tokens, ERC-20) | `0x66CA7c683fbaF030b2300c918A3751209eA30dEa` | <img src="docs/donate/eth.svg" width="130" alt="QR Ethereum"> |

> ⚠️ **The network matters.** Send only Tron-network assets (USDT TRC-20, TRX) to the Tron address and only
> Ethereum-network assets to the Ethereum address. Anything sent over the wrong network is lost.

Thank you! 🙏
