# Four-client measurements — build 6

Measured 8 November 2026 on Linux: two RTX 2080 Ti cards with 22 GiB each and NVLink,
two Xeon 8358 CPUs, 16 DDR4-2666 channels, 384 GB host RAM. Model: Flash-Next NVFP4 + Q8.
The GPU unlock was already applied on the machine; its separate contribution was not measured here.

| workload | simultaneous clients | output tokens per client | aggregate API tokens/s |
|---|---:|---:|---:|
| four different English requests, repeat 1 | 4 | 256 | 82.20 |
| same workload, repeat 2 | 4 | 256 | 81.59 |
| four different Russian requests, repeat 1 | 4 | 512 | 90.86 |
| same Russian workload, repeat 2 | 4 | 512 | 96.18 |

The Russian requests cover a Python LRU cache, database transactions, a story and a photo-archive plan.
Temperature 0.9, seed 9 on every request. Times include prompt admission. The Russian runs produced
2,048 output tokens in 22.54 and 21.29 seconds. Individual API rates were about 23–25 tokens/s.
These are **totals**, not 90–96 tokens/s for each client. The figures apply to these short-prompt workloads;
they do not promise the same rate at every prompt length. A 1M context limit is not a 1M-token speed test.

## Measured profile

Batch 4, two groups, GPU order 1,0, layer split 22; 64 CPU threads with SMT disabled, NUMA enabled.
Context limit 1,048,576, YaRN x4; int8 KV with 20,480 resident cells; prefill 8,192.
Conversation cache 40,960 MiB / 16 entries, retention 120 minutes.

On this machine the model configuration used `--pcie-frac 0` in its existing `args` list.
The `env` settings included:

```json
{
  "DACAN_NUMA_ARENA_FIRST": "1",
  "DACAN_BATCH_DIRECT_COPY": "1",
  "DACAN_BATCH_MTP": "1",
  "DACAN_BATCH_PIPE_MTP": "1"
}
```

Merge these into the existing model configuration, preserving model paths and other settings. Select four
slots in About / Model settings, or keep `--batch 4` in the model's args. An optional explicit
`--batch-groups 2` describes the measured split. Cache retention is set by
`--conversation-cache-expire-min 120`. Restart after current requests finish to apply engine settings.
This is a measured two-socket reference profile; other CPU/GPU combinations need their own comparison.

## Validation and scope

Controlled model checks compared 3,072 output tokens across solo/batched/MTP execution and repeats at
temperatures 0 and 0.7 with a fixed cache setup. Cancellation, slot reuse, output limits 1/2/3/17 and
prompt-cache reuse were checked separately. Sampling with a fixed seed does not guarantee identical
texts when execution/cache conditions change. Existing batch penalties/logit-bias support is not extended
by this profile.

Both portable binaries in build 6 are compiled from the same updated source snapshot for CUDA
75/80/86/89/120. Linux requires glibc 2.35+. The separately labelled reference archive contains the exact
Linux engine used in the measured installation and has stricter requirements in BUILD.json.
The reference model was exercised on Linux; a new full-model GPU benchmark of each portable platform
was not run. Engine implementation source remains in the private repository.
