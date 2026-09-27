#!/usr/bin/env python3
"""tools/nvfp4_experts.py - 27.09.2026: NVIDIA's NVFP4 experts (ModelOpt checkpoint) straight into Strata's arena form.

    python3 tools/nvfp4_experts.py <modelopt_dir> <experts.bin> <base_pack_dir> <out_pack_dir> [<experts.bin name>]

llama.cpp's converter keeps every repacked expert in memory until it writes the GGUF (~2 GB a layer here, ~100 GB for
the model), so this writes them one expert at a time into a raw file instead.  Only the experts change: the dense part
of the NVFP4 checkpoint is the same Q8_0 / BF16 / F32 as quant A's GGUF (checked bit for bit on 27.09), so Strata keeps
--native on quant A's file and takes each layer's experts from this file, named as a shard beside it
(native_experts.txt v4; make the name resolve in --native's directory, e.g. a symlink).

The block packing is llama.cpp's (convert_hf_to_gguf / conversion/base.py `_nvfp4_pack`): 64 values a block, the four
16-value groups' E4M3 scale bytes with the sign stripped (UE4M3) first, then 32 bytes of E2M1 nibbles - value j of a
group in the low nibble of byte j, value j + 8 in the high nibble.  The per-expert global scale (weight_scale_2)
stays a separate F32 per expert (llama.cpp's ".scale" tensor); Strata puts it in the blob's 16-byte tail.

File layout, per layer: gate [E][FF rows], up [E][FF rows], down [E][H rows] of packed rows, then the F32 global
scales of gate [E], up [E], down [E].
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

E, H, FF = 512, 2560, 640
NVFP4 = 40


def pack(w: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """ModelOpt rows (w: [out, in/2] uint8 nibble pairs, sc: [out, in/16] E4M3 bytes) -> [out, in/64 * 36] block_nvfp4."""
    out_f, nb = sc.shape
    w = w.reshape(out_f, nb, 8)
    vals = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(out_f, nb, 16)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).astype(np.uint8).reshape(out_f, nb // 4, 32)
    d = (sc & 0x7F).astype(np.uint8).reshape(out_f, nb // 4, 4)
    return np.concatenate([d, qs], axis=-1).reshape(out_f, nb // 4 * 36)


def main() -> int:
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    base, out = pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
    shard_name = sys.argv[5] if len(sys.argv) > 5 else dst.name
    wmap = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    n_layers = 1 + max(int(k.split(".")[3]) for k in wmap if k.startswith("model.language_model.layers."))
    gu_row, d_row = H // 64 * 36, FF // 64 * 36
    per = {"gate_proj": FF * gu_row, "up_proj": FF * gu_row, "down_proj": H * d_row}   # bytes per expert
    layer_bytes = E * (per["gate_proj"] + per["up_proj"] + per["down_proj"]) + 3 * E * 4
    w_bytes = per["gate_proj"] + per["up_proj"] + per["down_proj"]
    blob = w_bytes + 16
    handles = {}

    def get(name):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(str(src / f), "pt")
        return handles[f].get_tensor(name)

    total = n_layers * layer_bytes
    mode = "r+b" if dst.exists() and dst.stat().st_size == total else "w+b"
    lines, arena = [], 0
    t0 = time.time()
    with open(dst, mode) as fo:
        if mode == "w+b":
            fo.truncate(total)
        for l in range(n_layers):
            base_off = l * layer_bytes
            region = {"gate_proj": base_off, "up_proj": base_off + E * per["gate_proj"],
                      "down_proj": base_off + E * (per["gate_proj"] + per["up_proj"])}
            scale_off = base_off + E * w_bytes
            scales = {p: np.zeros(E, dtype=np.float32) for p in per}
            for p in ("gate_proj", "up_proj", "down_proj"):
                buf = bytearray()
                for e in range(E):
                    pre = "model.language_model.layers.%d.mlp.experts.%d.%s." % (l, e, p)
                    w = get(pre + "weight").numpy()
                    sc = get(pre + "weight_scale").view(torch.uint8).numpy()
                    scales[p][e] = float(get(pre + "weight_scale_2").float())
                    raw = pack(w, sc)
                    if raw.nbytes != per[p]:
                        print("layer %d expert %d %s: %d bytes, expected %d" % (l, e, p, raw.nbytes, per[p]))
                        return 1
                    buf += raw.tobytes()
                    if len(buf) >= (64 << 20) or e == E - 1:        # write in ~64 MB pieces
                        at = region[p] + (e + 1) * per[p] - len(buf)
                        fo.seek(at)
                        fo.write(buf)
                        buf = bytearray()
            fo.seek(scale_off)
            for p in ("gate_proj", "up_proj", "down_proj"):
                fo.write(scales[p].tobytes())
            lines.append("%d %d %d %d %d %d %d %d %s %d %d %d" % (
                l, NVFP4, NVFP4, arena, blob, region["gate_proj"], region["up_proj"], region["down_proj"], shard_name,
                scale_off, scale_off + 4 * E, scale_off + 8 * E))
            arena += blob * E
            el = time.time() - t0
            print("layer %2d done: %.1f GiB written, %.0f s, %.1f MB/s" % (
                l, (l + 1) * layer_bytes / 2**30, el, (l + 1) * layer_bytes / el / 1e6), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dense.bin", "index.txt", "tokenizer"):
        if not (out / name).exists():
            os.symlink(base / name, out / name)
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off shard "
                 "gate_scale_off up_scale_off down_scale_off (n_expert %d, total %d; NVFP4 experts of %s in the shard "
                 "%s beside --native; blob = weights %d + global scales 16)\n" % (E, arena, src.name, shard_name, w_bytes))
        fo.write("\n".join(lines) + "\n")
    print("pack %s: %d layers, blob %d B, arena %.2f GiB" % (out, n_layers, blob, arena / 2**30))
    return 0


if __name__ == "__main__":
    sys.exit(main())
