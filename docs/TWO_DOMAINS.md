# Two domains: socket + card + its memory, twice, NVLink between

The machine this fork is tuned on is two Xeon Ice Lake sockets (32 cores, 8 DDR4 channels, AVX-512 VNNI each) with
one RTX 2080 Ti 22 GB on each socket's PCIe root and an NVLink bridge between the two cards. Strata was written for
one card and one desktop CPU; here the engine is being rebuilt around two symmetric domains:

    domain A: socket 0 + card 4B:00 + node 0's 8 channels
    domain B: socket 1 + card 98:00 + node 1's 8 channels
    between:  NVLink (2 x 25.8 GB/s, peer access both ways) and UPI

## What is done (28.09.2026)

**The second card as the main card's coprocessor** (`Verifier::enable_helper`, `STRATA_HELPER`). Each card replays
its own CUDA graph of the verify window; they meet on flags in device memory (`raise_flag64` / `wait_flag64`,
`(epoch << 10) | step`, no reset between windows). The second card pulls a layer's input over NVLink and writes its
outputs straight into the main card's buffers. Parts: `z` (GDN alpha/beta and z), `q` (QSA q+gate), `h` (the second
half of the head's rows), `o` (half of each output projection - slower than keeping it whole, measured), `e` (the
second card's expert share from its own graph instead of host API calls; always on with the helper - host-issued work
on a card whose graph spins on a flag can queue behind it). Same kernels on the same inputs: bitwise the single-card
output (checked token for token against a control with the same experts on the second card:
`STRATA_HELPER_RESERVE_ONLY=1`). +7 % on the verify round; with `STRATA_COMMIT_OVERLAP=1` (the commit graph beside the
MTP draft) +8.5 %. Without `o` (1.3 GB of weight copies instead of 1.7, 145 more experts fit on the second card), the
plan copied by many blocks, a faster row gather and the router / combine in one launch for the group: 73.07 -> 79.49
tok/s against its control (+8.8 %, CPU pool 8.2 -> 4.4 ms a round, commit 0.89 -> 0.07); at ~24K tokens of context
69.82 -> 73.56. Both pairs identical token for token. `--spec-split` (two groups a window, card and CPUs overlapped):
58.02 tok/s against 73.07 - the CPUs compute experts shared by the groups twice; off.

**CPU side**: NVFP4 expert rows on AVX-512 VNNI (`nvfp4_avx512.cpp`: ggml-cpu has only AVX2 for NVFP4 and decodes a
row again for every token; bitwise ggml's result, 1.3x-2.9x per core); `pool_bench` / `nvfp4_parity`. SMT siblings as
pool workers (`STRATA_SMT=1`) measured slower here - the pool is bound by the cores' vector units, not by threads.

## Where the round goes (Nsight Systems, helper on, NVFP4, window of ~4 tokens)

Main card, per round: ~27 ms of kernels (hyper-connections 6.8, dense projections 5.0, experts 4.7, small bf16 1.7,
MTP layer 1.5, GDN recurrence 1.1, attention 0.9, per-token router/combine 1.2), ~3.9 ms waiting for the CPU pool and
~3.9 ms waiting for the second card (its expert rows ~30 us after the CPU's; half the output projection 26 us late).
The second card is busy ~35 % of the round.

## Next

1. Hyper-connection kernels (the largest block, and in any split both cards run them whole): `gr_down2` sits at 25 %
   occupancy (one 8-warp block per SM for its 40-60 KB activation tile) stalled on shared memory, 27 % of DRAM
   bandwidth. Tried and dropped: 3 rows a warp in one wave (v3), 32 warps a block on the same tile (v4) - neither
   faster at the window sizes that matter (T = 3-4). Next: Nsight Compute source-level stall sampling.
2. Tensor parallel by heads: GDN (36 layers) and QSA (12) split by heads between the domains - each card its half of
   qkv / z / the recurrent state / the KV cache, the output projection as partial sums, one exchange; then the experts
   by domain as partial sums, a second exchange; hyper-connections and the router replicated.
3. The CPU arena by domain (whole experts in their socket's memory), one planning thread per socket.
4. The prompt path on both domains: the second card's VRAM tier and PCIe link unused today; the CPU sockets computing
   non-resident experts (VNNI GEMM over tens of tokens per expert) instead of streaming them over one PCIe link.
   Today, at 256K of context: 309 tok/s at 8K, 301 at 24K, 285 at 105K, 267 at 195K (12 minutes for 195K tokens).
