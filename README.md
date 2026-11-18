# DACAN for RTX 2080 Ti on Linux

Optional text-generation engine for Linux x86-64 and NVIDIA Turing SM 7.5, tested on two 22 GB RTX 2080 Ti cards and two Xeon Gold 8358 processors with Qwen3.8-Flash-Next NVFP4 + Q8. It uses glibc 2.35 or newer and the CUDA 13 runtime and cuBLAS. Start with a working CUDA 13 installation of DACAN build 6 and an existing model pack. [Stable installer and packages](https://github.com/tirex999/DACAN/releases/tag/v0.1.42-build6)

## Prompt cache fixes

When a client disconnects during prefill, the engine retains the computed prefix for the retry. If there is too little host memory for another conversation snapshot, a failed save preserves the snapshots already held in memory.

| Retry on the two-slot reference server | Cached tokens | Tokens read on retry |
| --- | ---: | ---: |
| 71,461-token prompt interrupted during prefill | 71,454 | 7 |
| 71,449-token prompt interrupted after the reader yielded its slot | 16,384 | 55,065 |

Both retries returned the expected answer. In a separate memory-pressure regression, all 12 previous snapshots survived a rejected save and one restored a 296,622-token prefix. The retry tests used a server that forwards cancellations to the engine.

The ZIP contains the native engine and its license. It uses the server and web interface from your existing installation.

## Install alongside build 6

Download both files into a new directory and verify the archive before extracting it. Run these commands from the DACAN installation directory.

```sh
mkdir -p engine-sm75-20261119
(
  cd engine-sm75-20261119 &&
  base=https://github.com/tirex999/DACAN/releases/download/v0.1.42-sm75.20261119 &&
  curl -fL -O "$base/dacan-linux-x64-sm75.zip" &&
  curl -fL -O "$base/SHA256SUMS.txt" &&
  sha256sum -c SHA256SUMS.txt &&
  unzip dacan-linux-x64-sm75.zip
)
```

Make a copy of the working model JSON in its original directory and name it `dacan-sm75.json`. In the copy, set the top-level `exe` to the absolute path of `engine-sm75-20261119/dacan`. The existing configuration provides the model, tokenizer and CUDA library paths. To start it on the same GPUs, stop the current server, then run this command from the DACAN installation directory in its Python environment.

```sh
DACAN_SM75_NVFP4_PREFILL=1 python -m serve.server --engine dacan --config dacan-sm75.json --host 127.0.0.1 --port 8080
```

The SM75 prompt optimization is off by default. To return to the stable engine, stop this instance and start DACAN with the original model configuration.
