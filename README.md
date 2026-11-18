# DACAN for RTX 2080 Ti on Linux

Optional text-generation engine for Linux x86-64 and NVIDIA Turing SM 7.5, tested on two 22 GB RTX 2080 Ti cards and two Xeon Gold 8358 processors with Qwen3.8-Flash-Next NVFP4 + Q8. It uses glibc 2.35 or newer and the CUDA 13 runtime and cuBLAS. Start with a working CUDA 13 installation of DACAN build 6 and an existing model pack. [Stable installer and packages](https://github.com/tirex999/DACAN/releases/tag/v0.1.42-build6)

## Install alongside build 6

Download both files into a new directory and verify the archive before extracting it. Run these commands from the DACAN installation directory.

```sh
mkdir -p engine-sm75-20261118
(
  cd engine-sm75-20261118 &&
  base=https://github.com/tirex999/DACAN/releases/download/v0.1.42-sm75.20261118 &&
  curl -fL -O "$base/dacan-linux-x64-sm75.zip" &&
  curl -fL -O "$base/SHA256SUMS.txt" &&
  sha256sum -c SHA256SUMS.txt &&
  unzip dacan-linux-x64-sm75.zip
)
```

Make a copy of the working model JSON in its original directory and name it `dacan-sm75.json`. In the copy, set the top-level `exe` to the absolute path of `engine-sm75-20261118/dacan`. The existing configuration provides the model, tokenizer and CUDA library paths. To start it on the same GPUs, stop the current server, then run this command from the DACAN installation directory in its Python environment.

```sh
DACAN_SM75_NVFP4_PREFILL=1 python -m serve.server --engine dacan --config dacan-sm75.json --host 127.0.0.1 --port 8080
```

The SM75 prompt optimization is off by default. To return to the stable engine, stop this instance and start DACAN with the original model configuration.

## Measured on 18 November 2026

| Workload on the reference server | Throughput |
| --- | ---: |
| One uncached prompt, 79,395 input tokens | 1,965.5 input tokens/s |
| Two uncached prompts submitted together, 79,381 tokens each | 1,841.2 input tokens/s total |
| Two clients generating 1,024 tokens each, 48 CPU workers | 74.87 output tokens/s total |
| Same workload, 64 CPU workers, two runs | 78.05 and 76.82 output tokens/s total |
| Same workload, 96 / 128 CPU workers | 77.29 / 64.60 output tokens/s total |

The single-prompt row uses the engine's prompt timing. The previous engine measured 1,914.9 input tokens/s on the same prompt, with one run per build. For the two long requests, 158,762 input tokens took 86.23 seconds from submission to both responses finishing. Both returned the expected answer. For generation, throughput is the 2,048 output tokens divided by the common wall-clock interval from first submission to last completion, including the wait for the first token. Those requests used short uncached prompts, temperature 0 and seed 9.
