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
