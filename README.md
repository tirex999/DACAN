# DACAN (Дацан)

🇬🇧 **English** | 🇷🇺 [Русский](README.ru.md) · [❤️ Support the project](#support-the-project)

**DACAN** is an inference engine for **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (125B
parameters, mixture of experts) and fine-tunes of the same shape such as
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next), on ordinary hardware: one or two consumer
NVIDIA cards and the PC's RAM. It comes with an installer, a server with OpenAI- and Anthropic-compatible APIs, and a
web chat.

**The engine is published ready-made** (`dacan` / `dacan.exe` in the [releases](https://github.com/tirex999/DACAN/releases),
for Windows x64 and Linux x64). Its source is not published; the installer and the server are open (MIT,
[LICENSE](LICENSE)).

## Our own engine

DACAN is our own engine. It started in September 2026 on the base of [Strata](https://github.com/Niko1221/Strata) (MIT)
and has been developed separately since then, with its own kernels: NVFP4 + Q8 on RTX 20 (Turing) cards, one or two
cards, one or two CPU sockets, a context up to 524K, vision and a live monitor. Strata's copyright notice is kept in
[LICENSE](LICENSE), as MIT requires.

## What you need

- **An NVIDIA card from the RTX 20 series up** (Turing, Ampere, Ada, Blackwell), **driver 580 or newer**:
  - RTX 20 (Turing): RTX 2060 / 2070 / 2080 / 2080 Ti (including the 22 GB 2080 Ti), Titan RTX, Quadro RTX;
  - the Turing and Ampere mining cards **CMP 30HX / 40HX / 50HX** (Turing) and **CMP 70HX / 90HX** (Ampere) — accepted
    by the installer like the cards above; we have not run them ourselves. Their PCIe link is cut down, so loading the
    model and moving experts to the card take longer;
  - RTX 30, RTX 40, RTX 50 and the workstation cards of those generations.

  Older cards (GTX 10, Volta), AMD and Intel are not supported.
- **Windows 10/11 x64** or **Linux x64 with glibc 2.35 or newer** (Ubuntu 22.04 and newer, Debian 12, Fedora 36 and
  newer, ...).
- **A CPU with AVX2.** RAM depends on the model: the installer says what fits and offers only that.

## How much VRAM

What a card needs depends on the model's quant. The numbers below are for **one card** with the draft layer
on (the installer's default Cyrillic subset), context 32K and an 8-bit KV cache. DACAN first places the model's always-used part
on the card; whatever VRAM is left becomes the **expert cache** (the experts kept on the card; the rest are read from
the PC's RAM).

| Free VRAM on the card | Swift 1.5 **IQ3_XXS** (GSQ-RCO, 3 bit) | Swift 1.5 **NVFP4 + Q8** (DACAN) |
|---|---|---|
| needed before the expert cache | **5.43 GiB** | **6.78 GiB** |
| 6 GB card | does not start (803 MiB short) | does not fit |
| 8 GB card | starts: 831 experts on the card (1.37 GiB) | does not start (211 MiB short) |
| 12 GB card | 3,140 experts on the card (5.12 GiB) | 1,445 experts on the card (3.72 GiB) |
| 16 GB card | not measured (more than at 12 GB) | 2,902 experts on the card (7.47 GiB) |

**The minimum:** IQ3_XXS from an **8 GB** card, NVFP4 + Q8 from a **12 GB** card.

- Of IQ3_XXS's 5.43 GiB: the dense weights 3.17 GiB, the draft layer 0.82 GiB, a 32K session with its KV cache
  0.56 GiB, the rest CUDA's own buffers.
- These are **free** gigabytes. A card that also drives the monitor gives 0.3-1 GB to the desktop: count without
  that, or plug the monitor into another card / the integrated graphics.
- Without the draft layer the engine needs about 0.8 GiB less (IQ3_XXS 4.61 GiB, NVFP4 + Q8 5.95 GiB): then IQ3_XXS
  started on a 6 GB card (327 experts on the card) and NVFP4 + Q8 on an 8 GB card (369). The installer always sets
  the draft layer up, so these two are for reference.
- If the start stops with "the draft head does not fit" or "no VRAM is left for the expert cache", the installer's
  tips may help: `--draft-vocab en` (a smaller draft head, ~60 MiB less) and a shorter context (a smaller KV
  cache).
- The 2-bit quants (Q2_0, IQ2_XS) and IQ3_S have not been measured yet.
- RAM is a separate matter: the experts that are not on the card are in RAM; the installer checks it.

How it was measured (11.10.2026): one RTX 2080 Ti 22 GB on Linux, the DACAN 0.1.42 engine from the release with its
server; the rest of the card was held by another process so that 5.59 / 7.59 / 11.34 / 15.09 GiB stayed free, as on
6 / 8 / 12 / 16 GB cards without a monitor. "Does not start" means the engine stopped with an out-of-memory error at
the start; the number of experts on the card is what the engine reported after the start.

## Install

**Windows**

1. Get the folder: `git clone https://github.com/tirex999/DACAN` or the
   [ZIP](https://github.com/tirex999/DACAN/archive/refs/heads/main.zip) (unpack it anywhere).
2. Run **`START-HERE.bat`**. It asks for the model, its size, the context and images, downloads everything (the DACAN
   engine from the release, the models from Hugging Face) and starts the server. Python is installed when missing.

**Linux**

```sh
git clone https://github.com/tirex999/DACAN
cd DACAN
./setup.sh
```

The same questions as on Windows; the engine `dacan-linux-x64.zip` comes from the release. Only the NVIDIA driver is
needed beforehand: Python 3.10+ with venv is installed through apt / dnf / pacman when missing (it asks for sudo), the
CUDA 13 libraries come as pip packages into the folder's own `.venv`. Later runs of `./setup.sh` just start the model.

**Our own model: Swift 1.5 NVFP4 + Q8 (DACAN)**

[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN):
NVIDIA RTX 20 or newer, ~80 GB of RAM. It is the last entry of the first question, "Which model?". Setup then asks
whether you already downloaded it: type that folder (the one with the two model files and `pack-nvfp4`) and the files
are used where they are - checked once (SHA-256, a few minutes), nothing is copied; press Enter and setup downloads
them (~132 GB). The same without questions:

```
START-HERE.bat --from-dir D:\models\DACAN        (Linux: ./setup.sh --from-dir ~/models/DACAN)
START-HERE.bat --dacan-nvfp4                      (download the files)
```

## Update

**`UPDATE.bat`** (Windows) or **`./update.sh`** (Linux): the newest files (`git pull`), a new DACAN engine when one is
out, the models' settings. The models are not downloaded again. Then start as usual (`START-HERE.bat` / `./setup.sh`).

An installation of an earlier DACAN is moved over by the update itself: the folder with the downloaded models is
renamed to `DACAN-data`, the models' configs to `dacan-<model>.json`, the old engine is replaced.

If your copy has no `UPDATE.bat` / `update.sh` yet, run `git pull` in its folder once (or unpack the ZIP over it), then
the update.

## Sampling settings

How the next word is chosen. The defaults are the model's recommended settings, and they are what DACAN uses:

| setting | default | what it does |
|---|---|---|
| `temperature` | **0.6** | how freely words are picked: 0 = always the most likely word (the same question gives the same answer), higher = more varied |
| `top_p` | **0.95** | only the most likely words that together make up 95 % of the probability are considered |
| `top_k` | **20** | at most this many of the most likely words are considered (DACAN takes 1 to 64) |
| `presence_penalty`, `repetition_penalty`, `frequency_penalty` | **off** | push the model away from words it already used; turn on only if answers repeat themselves |

- **The web chat** sends these by default (the ⚙ settings: Temperature, Top-p, Top-k). "Use for other apps too" makes
  the web chat's settings the defaults for every API client.
- **API clients** (Open WebUI, Claude Code, your own scripts) get the defaults for every field they do not send
  themselves: they are in the `"sampling"` block of the model's config `dacan-<model>.json` next to `setup.py`, and
  you can change them there (restart the model afterwards). A request's own fields always win.
- **Speed.** `temperature` 0 is the fastest: the engine guesses several tokens ahead and checks them in one step, and
  with the most likely word every time more of the guesses are kept. Sampling (temperature above 0) keeps fewer of
  them.

## Two cards and 8-bit models

With an 8-bit model (Q8_0) on **two cards**: the DACAN engine of build 4 (Linux 12.10.2026, Windows 13.10.2026;
`./update.sh` or `UPDATE.bat` installs it) sets this by itself. With an older engine add `"--pcie-frac", "0"` to the
`"args"` of the model's config `dacan-<model>.json` next to `setup.py`, then restart the model.

Measured on 12.10.2026: 2x RTX 2080 Ti 22 GB, 2x Xeon Ice Lake-SP with 32 cores each (engineering samples), the DACAN 0.1.42 engine, Swift 1.5 Q8_0.

| | code | Russian | English |
|---|---|---|---|
| `--pcie-frac 0` | **72.4 tok/s** | **58.6 tok/s** | **54.9 tok/s** |
| build 4, no flag (first / second request) | 59.3 / 65.6 tok/s | 56.7 / 58.5 tok/s | 53.9 / 56.0 tok/s |
| without it, `--peer-device 1` | 21.3 tok/s | 20.8 tok/s | 19.6 tok/s |
| without it, the default two-card split | 26.3 tok/s on a 25,132-token code answer (temperature 0.6) | | |

The first three rows: the same three requests, temperature 0, 1,024-token answers.

The other quants were measured with the defaults.

## Two cards: reading and answering with the NVFP4 + Q8 model

Measured on 13.10.2026: 2x RTX 2080 Ti 22 GB, 2x Xeon Ice Lake-SP with 32 cores each (engineering samples), the DACAN
engine of build 4 on both cards (layer split), Swift 1.5 in NVFP4 + Q8 (DACAN), `--max-context 524288`, vision on.
A Windows VM was working on the same machine at the time.

| prompt, new tokens | read, tok/s |
|---|---|
| 3,111 | 629 – 646 |
| 11,156 | 1,016 – 1,025 |
| 22,232 | 1,429 – 1,436 |
| 39,056 | 1,560 |
| 289,985 (a live conversation) | 1,398 |

The answer (the same three requests as above, temperature 0, 1,024-token answers): code 77.8 – 78.5 tok/s,
Russian 66.5 – 67.6 tok/s, English 64.5 – 66.4 tok/s. The engine's container with 72 of the machine's 128 CPU threads
gives the same figures as with all 128.

The Monitor tab shows the read speed while a prompt is read: `Reading prompt · 58% · 1,448 tok/s`.

## Weights

- **[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN)**:
  Swift 1.5 as **NVFP4 + Q8 (DACAN)** and 8-bit (Q8_0).
- **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)**: the original model
  in the same formats.
- GSQ-RCO quants (2-3.5 bits) by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) and
  [UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) are installed by the installer.

The draft vocabulary (the words the engine guesses ahead) is chosen by the installer: Cyrillic by default for
Russian-language chat; English and French are offered too.

## Documents

[docs/](docs/): [DETAILS](docs/DETAILS.md) (flags, API, images, how the engine works),
[MULTI_GPU](docs/MULTI_GPU.md) and [SECOND_GPU](docs/SECOND_GPU.md) (two cards), [BATCHING](docs/BATCHING.md)
(several requests at once), [UNSLOTH_Q4](docs/UNSLOTH_Q4.md), [MCP_SERVER](docs/MCP_SERVER.md) (DACAN as a tool for AI
assistants), [TROUBLESHOOTING](docs/TROUBLESHOOTING.md). Questions and problems:
[issues](https://github.com/tirex999/DACAN/issues).

## Licenses

The installer and the server: MIT ([LICENSE](LICENSE)). [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
(MIT) inside the engine and the image encoder. The weights have their own licenses: Qwen Community License 1.0 for
Qwen3.8-Flash-Next, NVIDIA Open Model License for NVIDIA's NVFP4 experts, Swift Open License v1.0 for Swift 1.5.

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
