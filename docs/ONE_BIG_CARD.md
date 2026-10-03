# One big card (24–48 GB): fill the VRAM with experts, 16-bit KV

**Symptom.** On a 24–48 GB card (a 4090 48 GB, a 3090 / 4090 24 GB, an A6000) the monitor shows half the VRAM empty
and **"8 000 experts cached on the main card"**, while the CPU sits at 100 %.

## Why

`--expert-cache auto` sizes the GPU's expert cache from the VRAM that is free after the weights and the KV cache, but
it never goes past the length of the expert profile's ranked list (`src/program/generate.cpp`:
`slots = min(slots, profile.size())`). The installer passes `data/expert-profile.bin`, which ranks **8,000** experts,
what a 12 GB card holds. On a bigger card the cache stops at 8,000 and the rest of the VRAM stays unused, while every
expert that is not on the GPU is computed by the CPU.

## The fix

`data/expert-profile-full.bin` ranks all 24,576 experts: the same first 8,000 as `data/expert-profile.bin`, in the same
order, then every other expert by routing count (`tools/extend_profile.py`). With it `auto` keeps filling the card
down the same ranking until the VRAM (minus the reserve) is used. A 12 GB card gets exactly the experts it got before.

**New install.** Update the repository and run the installer as usual. From 20 GB of VRAM it picks the full profile
by itself (`expert profile: all experts ranked` in its output).

**Existing install, no reinstall.** Open the config the installer wrote next to `setup.py`, `strata-<model>.json`, and
in `"args"` change

```
"--expert-profile", ".../data/expert-profile.bin",
```

to

```
"--expert-profile", ".../data/expert-profile-full.bin",
```

then restart with the same `run-<model>` script. To make the file yourself (another profile, or a routing table of
your own runs):

```
python tools/extend_profile.py data/expert-profile.bin data/expert-profile-full.bin [--usage data/usage_other_2609.bin] [--pairs N]
```

**Check.** The engine log (`strata-<model>.log`) says how many slots it took:

```
strata generate: expert cache auto: <free> GiB free, <reserve> MiB reserved -> <slots> slots
```

and the monitor's "experts cached" goes past 8,000. If the card is shared with something else (a desktop, a browser
with hardware acceleration), leave headroom with `--vram-reserve-mib` (default 700; we run 1024).

How much of the routing the cache then covers, counted over our own runs (`data/usage_other_2609.bin`, our workload,
not yours): the first 8,000 experts of `data/expert-profile.bin` 57 %, the first 16,000 of the full profile 97 %,
everything 100 %. A 48 GB card holds most of the 24,576 experts of the 2- and 3-bit quants.

## 16-bit KV

The engine takes `--kv fp16` (it is even its default); only the installer offered just `int8` and `q4_0`. It now offers
`fp16` too (`--kv fp16`, or answer 3 to "KV cache?"). In an existing `strata-<model>.json` replace `"--kv", "int8"`
with `"--kv", "fp16"`.

What it costs, per context token: 2,048 bytes per layer instead of 1,056 for `int8`, over 13 layers (12 attention
layers and the draft layer) - about **7 GB at 262K** instead of 3.6 GB. That VRAM comes out of the expert cache. With
KV streaming (`--kv-resident`, which the installer turns on from 64K of context when the RAM allows) the KV lives in
system RAM instead, so 16-bit needs those 7 GB of RAM.

## Measured on a 48 GB card (a user's report, 03.10.2026)

RTX 4090 48 GB, Ryzen 7 9700X (8 cores), 59 GB of RAM; the same model and settings before and after: ISTA's
Qwen3.8-Flash-Next IQ3_S, `--kv fp16`, 262K context. Figures from DACAN's Monitor:

| | before | with the full profile |
|---|---:|---:|
| experts cached on the card | 8,000 (22.8 of 48 GB of VRAM) | **17,304** (32.8 GB; the card at 47.6 of 48 GB) |
| answer speed | 50.6 tok/s (the Monitor's live figure, early in an answer) | **95.3 tok/s** over a whole 2,279-token answer, 75-85K of context |
| short answers at 75-85K of context | - | 112-176 tok/s |

The user's own word: "noticeably faster, almost twice". The CPU stays at 97 % and the card at ~81 %: every expert moved
onto the card is work taken off the 8 cores. The "before" figure is a live reading, not a whole answer, so "about twice"
is an estimate rather than an exact ratio. The requests carry the Monitor's ESP badge (the experimental projection
vector was on); by [DETAILS.md](DETAILS.md#experimental-speed-projection-experimental-off-by-default) it costs
0.2-0.4 % per token, so it is not where the gain comes from.

We run DACAN on two 22 GB cards ourselves (see [README-2x2080Ti.md](../README-2x2080Ti.md)). The cap and the fix follow
from the code above; the profile files were checked (the first 8,000 pairs identical, all 24,576 distinct and in range,
a valid frequency table).
