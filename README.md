# DACAN (Дацан)

🇬🇧 **English** | 🇷🇺 [Русский](README.ru.md) · [❤️ Support the project](#support-the-project)

**DACAN** is an inference engine for **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** — a 125B
mixture-of-experts model — on hardware it was never meant for: **two RTX 2080 Ti (22 GB, NVLink) and two Xeon sockets
with AVX-512**. It is our fork of [Strata](https://github.com/Niko1221/Strata), rebuilt for Turing cards (sm_75) and
for a machine with two cards and two CPU sockets. The binary and build targets are still called `strata`.

## How fast

Measured 03.10.2026 on our rig (NVFP4 experts, greedy, every request new text, speeds from the engine's own log):

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

## DACAN vs upstream Strata 0.1.39 (measured 04.10.2026)

In one week Strata's author released 30 versions (0.1.10 to 0.1.39): a model split by layers across two or three
GPUs, several requests at once, official RTX 20 support, the MIT license. We built 0.1.39 on our rig (sm_75,
CUDA 13.0) and put it next to DACAN on the same model and the same request.

| engine | prompt read (1 902 tokens) | answer (512 tokens) |
|---|---:|---:|
| **DACAN** (two cards, two sockets) | 586 tok/s | **67.4 tok/s** |
| upstream Strata 0.1.39 (its layer split across the two cards) | 605 tok/s | 17.7 tok/s |

Same for both: Swift 1.5 IQ3_XXS by UkisAI, the same pack, MTP draft head and expert profile, the prompt "hamsters —
honest physics" (1 902 tokens), greedy, 512 answer tokens, each engine through its own server. Prompt reading is a tie;
**DACAN answers 3.8 times faster** on this hardware. Our production model, Swift 1.5 Q8_0, runs on DACAN at 75 tok/s;
upstream 0.1.39 cannot load it (its per-layer table accepts IQ4_NL, Q5_0 or FP8, not Q8_0).

What we take from upstream next, one change at a time and measured: several requests at once (`parallel`), the
system-prompt checkpoint, the faster sampler (#197), the watchdog that restarts a silent engine.

DACAN is a joint project of Vadim and Claude — one team. On its own hardware it is faster than the original, and we
built it together.

## The model

One model family: Qwen3.8-Flash-Next, architecture `qwen4exp` (Hugging Face `Qwen4ExpForConditionalGeneration`) —
125B parameters with 6B active per token, plus a 51B n-gram embedding table and a 4B MTP layer; 48 layers of
3 × Gated DeltaNet + 1 × Qwen Sparse Attention, each followed by a MoE of 512 experts (10 routed + 1 shared); context
262,144 tokens. The loader checks this geometry and refuses anything else, so fine-tunes of the same shape work —
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next) by UkisAI runs unchanged — while other models and
pruned variants (REAP etc.) do not.

Weights packed for DACAN:

- **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)** — the original
  model: **NVFP4 + Q8 (DACAN)** with NVIDIA's NVFP4 experts repacked (68 GB, what our service runs), or our Q8_0
  experts from Qwen's FP8 checkpoint (128 GB); the dense GGUF, the n-gram table, the MTP layer.
- **[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN)** —
  Swift 1.5, our quants from its BF16: 8-bit (Q8_0) and **NVFP4 + Q8 (DACAN)** with our NVFP4 experts (uploaded 04.10.2026: 30 files, 321 GB).

**NVFP4 + Q8 (DACAN)** is our name for this mix. The routed experts — about 95 % of the 125B weights — are 4-bit NVFP4
(FP4 E2M1 values, an FP8 E4M3 scale per 16 values, an FP32 scale per tensor). The large matrices every token passes
through (attention and DeltaNet projections, the shared expert, embeddings, output head) and the n-gram table are 8-bit
Q8_0; the small sensitive ones (router, hyper-connections, the sparse-attention indexer, DeltaNet α/β, 1.4 GiB in all)
stay BF16.
- The [ISTA-DASLab GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) quants (2–3.5 bit)
  that upstream's installer downloads.

## What it does

- **Built for two cards and two sockets.** DACAN spreads the work over both GPUs, both CPU sockets and their memory in
  its own way; `--numa --second-card 1` turns the whole machine on.
- **CPU experts on AVX-512 VNNI**, bitwise equal to ggml's results.
- **MTP speculative decoding**: the model's own draft layer proposes 4 tokens, one pass checks them.
- **Fast prompt reading**: a fresh 19K-token prompt at ~800 tok/s, a 98K one at ~715.
- **The n-gram table** (51 GB) stays in RAM; **conversations are cached** and up to 16 other chats are parked in RAM,
  so a returning chat does not re-read its whole context.
- **Server**: OpenAI-compatible (`/v1/chat/completions`) and Anthropic-compatible (`/v1/messages`), a web app with
  Chat and a live Monitor of the cards, sockets and RAM, the answer speed and the prompt-read speed (fresh
  tokens per second, live while a prompt is read and per request); images through the model's mmproj encoder.

## Hardware

Ours: 2× RTX 2080 Ti 22 GB (NVLink, PCIe 3.0 x16 each), 2× Xeon Ice Lake (64 cores, AVX-512 VNNI), 256 GB DDR4-2666 on
16 channels. RAM needed: the experts (40 GiB for 3-bit, 63 GiB NVFP4, 120 GiB Q8_0) plus the n-gram table (27–51 GiB).

Other machines users have run it on:

- **One RTX 4090 48 GB**, Ryzen 7 9700X: 95 tok/s on a long answer at 75–85K of context after filling the card with experts
  — see [docs/ONE_BIG_CARD.md](docs/ONE_BIG_CARD.md) if your big card shows "8 000 experts cached".
- **RTX 4070 Ti SUPER + RTX 5060 Ti (16 GB each), 2× Xeon E5-2678 v3**: about 45 tok/s while reasoning; these Haswell
  CPUs have no AVX-512, so the experts run on ggml's AVX2 path and the CPUs set the pace. NVLink is not required.

## Install and run

- **One card, Windows or Linux** — upstream's installer: `START-HERE.bat` (Windows) or `./setup.sh` (Linux) asks for the
  model, size, context and images, downloads everything and starts the server; from 20 GB of VRAM it offers 16-bit KV.
  Details: [docs/DETAILS.md](docs/DETAILS.md).
- **Two cards and two sockets** (our setup): build with `-DCMAKE_CUDA_ARCHITECTURES=75` and run with `--numa
  --second-card 1` — the full command and the switches: [README-2x2080Ti.md](README-2x2080Ti.md).
  `tools/2x2080ti/run-fast.sh` writes the server config and starts it.

Once running: the web app at the server's address, `/v1` for OpenAI-compatible clients and coding agents,
`/v1/messages` for Anthropic's API.

## Documents

| | |
|---|---|
| [README-2x2080Ti.md](README-2x2080Ti.md) | build and run on two cards, switches, measurements |
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
