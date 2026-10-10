# DACAN (Дацан)

🇬🇧 **English** | 🇷🇺 [Русский](README.ru.md) · [❤️ Support the project](#support-the-project)

**DACAN** is an inference engine for **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (125B
parameters, mixture of experts) and fine-tunes of the same shape such as
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next), on ordinary hardware. From this version on it
is built on [Strata](https://github.com/Niko1221/Strata) 0.1.42 and adds:

- **RTX 20 (Turing)**: faster prompt reading on these cards, no flags needed;
- **our NVFP4 + Q8 (DACAN) quants**: NVFP4 experts run both on the GPU and on the CPU (AVX-512);
- **`--numa`** for two-socket servers (Linux).

Everything else — the installer, the server, the web app, the OpenAI and Anthropic APIs, splitting the model across
several cards — is Strata 0.1.42's.

**The engine is published ready-made** (`strata.exe` in the [releases](https://github.com/tirex999/DACAN/releases));
the source of our changes is not published. The installer and the server are open, from Strata (MIT).

## Install (Windows)

You need an NVIDIA card from the RTX 20 series up, driver 580 or newer, Windows 10/11 x64 and a CPU with AVX2.
Python is installed by the installer when missing.

1. Get the folder: `git clone https://github.com/tirex999/DACAN` or the
   [ZIP](https://github.com/tirex999/DACAN/archive/refs/heads/main.zip) (unpack it anywhere).
2. Run **`START-HERE.bat`**. It asks for the model, its size, the context and images, downloads everything (the DACAN
   engine from the release, the models from Hugging Face) and starts the server.

## Update

**`UPDATE.bat`**: the newest files (`git pull`), a new DACAN engine when one is out, and the models' settings. The
models are not downloaded again. Then start as usual with `START-HERE.bat`.

**If you had the earlier DACAN** (no `UPDATE.bat` in it): run `git pull` once in its folder (or download the ZIP again
and unpack it next to it), then `UPDATE.bat`. The installer moves the downloaded models into the `Strata-data` folder
next to it and finds them there. An upstream engine installed there is replaced by the DACAN one.

## Linux

The ready-made DACAN engine for Linux comes next. Until then the earlier version runs on Linux: branch
[`dacan-1`](https://github.com/tirex999/DACAN/tree/dacan-1) (two RTX 2080 Ti, `--second-card`, `--park`, its README).

## Cards older than RTX 20, and AMD

DACAN builds its engine for NVIDIA RTX 20 and newer only. For Pascal / Volta (CUDA 12) and AMD the installer takes
Strata's ready-made engine of the same version.

## Weights

- **[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN)**:
  Swift 1.5 as **NVFP4 + Q8 (DACAN)** and 8-bit (Q8_0).
- **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)**: the original model
  in the same formats.
- GSQ-RCO quants (2-3.5 bits) by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) and
  [UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) are installed by the installer.

## Documents

Strata's guide (install, flags, API, images, troubleshooting) is in [docs/](docs/), mainly
[docs/DETAILS.md](docs/DETAILS.md).

## Thanks and licenses

[Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors (MIT, [LICENSE](LICENSE)): the engine's
base, the installer and the server. [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). The weights have
their own licenses: Qwen Community License 1.0 for Qwen3.8-Flash-Next, NVIDIA Open Model License for NVIDIA's NVFP4
experts, Swift Open License v1.0 for Swift 1.5.

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
