# DACAN 0.43 for Linux x86-64 and RTX 2080 Ti

Conversation snapshots now expire after 480 minutes (8 hours) by default. Set --conversation-cache-expire-min N to choose another lifetime. A value of 0 disables timed expiration. Snapshots use host memory and may be evicted sooner. An explicit setting in an existing profile takes precedence.

Requires SM75, CUDA runtime and cuBLAS 13, glibc 2.35 or newer, and a working server installation and model pack. The archive contains the dacan executable and its license. Running ./dacan --version prints dacan 0.43.

## Install alongside build 6

Download both files into a new directory and verify the archive before extracting it. Run these commands from the DACAN installation directory.

```sh
mkdir -p engine-sm75-0.43
(
  cd engine-sm75-0.43 &&
  base=https://github.com/tirex999/DACAN/releases/download/v0.43 &&
  curl -fL -O "$base/dacan-linux-x64-sm75-0.43.zip" &&
  curl -fL -O "$base/SHA256SUMS.txt" &&
  sha256sum -c SHA256SUMS.txt &&
  unzip dacan-linux-x64-sm75-0.43.zip
)
```

Make a copy of the working model JSON in its original directory and name it `dacan-sm75.json`. In the copy, set the top-level `exe` to the absolute path of `engine-sm75-0.43/dacan`. The existing configuration provides the model, tokenizer and CUDA library paths. To start it on the same GPUs, stop the current server, then run this command from the DACAN installation directory in its Python environment.

```sh
DACAN_SM75_NVFP4_PREFILL=1 python -m serve.server --engine dacan --config dacan-sm75.json --host 127.0.0.1 --port 8080
```

The SM75 prompt optimization is off by default. To return to the stable engine, stop this instance and start DACAN with the original model configuration.
