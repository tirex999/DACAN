#!/usr/bin/env python3
"""tools/q8_experts.py - Q8_0 routed experts for Strata, made from Qwen's FP8 checkpoint (27.09.2026).

Qwen3.8-Flash-Next-FP8 stores each expert matrix as float8_e4m3fn with one bf16 `weight_scale_inv` per 128x128
tile.  This dequantizes a matrix (fp8 * tile scale, in float32) and requantizes it to ggml's Q8_0 the way
ggml's quantize_row_q8_0_ref does (d = amax / 127 per 32 values, stored fp16; q = roundf(x / d) from the fp32 d),
then lays the experts out as a GGUF's ffn_{gate,up,down}_exps would be: per layer, gate of all 512 experts, then up,
then down.  The result is one raw file plus a pack directory whose native_experts.txt (v3) points at it:

  python3 tools/q8_experts.py --src /LLM/models/flash-next-fp8 --out /cloude/Qwen3.8-Flash-Next-Q8_0-experts.bin \
      --pack /q/packs/q8-a --base-pack /q/packs/ours-a --workers 16

The pack's dense.bin / index.txt / tokenizer are links into --base-pack; the shard name in native_experts.txt is the
output's file name, which the engine resolves beside --native (so link it there: ln -s <out> <dir of --native>/).
Q8_0 is 8.5 bits a weight: 5 222 400 bytes an expert, 128.3 GB for 48 x 512 - the arena must fit in RAM.

Each matrix's relative error against the FP8 original (||q - w|| / ||w||) is logged; --check-nvfp4 <file> <pack>
also measures an NVFP4 pack's experts against the same original, for a few experts of a few layers.
"""
import argparse
import json
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

H, FF, NE, NL = 2560, 640, 512, 48
QK = 32
BQ8 = 34                                   # fp16 d + 32 int8
GU_ROW, D_ROW = H // QK * BQ8, FF // QK * BQ8
PER = FF * GU_ROW                          # bytes of one expert's gate (= up = down)
assert PER == H * D_ROW
BLOB = 3 * PER
LAYER = BLOB * NE
Q8 = np.dtype([("d", "<f2"), ("qs", "i1", (QK,))])
assert Q8.itemsize == BQ8


def fp8_lut():
    """float8_e4m3fn -> float32 for all 256 codes (no infinities; 0x7F / 0xFF are NaN, never in weights)."""
    lut = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            v = float("nan")
        elif e == 0:
            v = m / 8.0 * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        lut[b] = -v if s else v
    return lut


LUT = fp8_lut()


class Shards:
    """safetensors headers, read once per file; tensors fetched with pread (no mapping, no page cache kept)."""

    def __init__(self, src):
        self.src = src
        self.wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
        self.hdr, self.fd = {}, {}

    def _open(self, fn):
        if fn not in self.fd:
            fd = os.open(os.path.join(self.src, fn), os.O_RDONLY)
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.hdr[fn] = (json.loads(os.pread(fd, n, 8)), 8 + n)
            self.fd[fn] = fd
        return self.fd[fn], self.hdr[fn]

    def raw(self, name):
        fn = self.wmap[name]
        fd, (h, base) = self._open(fn)
        t = h[name]
        a, b = t["data_offsets"]
        buf = os.pread(fd, b - a, base + a)
        if len(buf) != b - a:
            raise IOError(f"{name}: short read")
        try:
            os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        return t["dtype"], t["shape"], buf


def fp8_matrix(sh, name):
    dt, shape, buf = sh.raw(name + ".weight")
    assert dt == "F8_E4M3", (name, dt)
    r, c = shape
    w = LUT[np.frombuffer(buf, np.uint8)].reshape(r, c)
    dt, sshape, sbuf = sh.raw(name + ".weight_scale_inv")
    assert dt == "BF16" and sshape == [(r + 127) // 128, (c + 127) // 128], (name, dt, sshape)
    s = (np.frombuffer(sbuf, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(sshape)
    w = (w.reshape(r // 128, 128, c // 128, 128) * s[:, None, :, None]).reshape(r, c)
    return w


def q8_0(w):
    """ggml quantize_row_q8_0_ref on every row of w (row length a multiple of 32) -> Q8 records, and the dequant."""
    r, c = w.shape
    x = w.reshape(r, c // QK, QK)
    amax = np.abs(x).max(axis=2)
    d = amax / np.float32(127.0)
    idv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    v = x * idv[:, :, None]
    q = np.clip(np.trunc(v + np.copysign(np.float32(0.5), v)), -128, 127).astype(np.int8)   # roundf
    out = np.empty((r, c // QK), Q8)
    out["d"] = d.astype(np.float16)
    out["qs"] = q
    back = (q.astype(np.float32) * out["d"].astype(np.float32)[:, :, None]).reshape(r, c)
    return out, back


def rel(a, b):
    n = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / n if n > 0 else 0.0


def job(args):
    """one (layer, projection): 512 experts -> one contiguous region of the output file."""
    src, out, layer, p, prefix = args
    sh = Shards(src)
    fd = os.open(out, os.O_WRONLY)
    at = layer * LAYER + p * NE * PER
    proj = ("gate_proj", "up_proj", "down_proj")[p]
    errs = []
    chunk = []
    t0 = time.time()
    for e in range(NE):
        w = fp8_matrix(sh, f"{prefix}.layers.{layer}.mlp.experts.{e}.{proj}")
        q, back = q8_0(w)
        errs.append(rel(back, w))
        chunk.append(q.tobytes())
        if len(chunk) == 32 or e == NE - 1:
            b = b"".join(chunk)
            off = at + (e + 1 - len(chunk)) * PER
            n = os.pwrite(fd, b, off)
            if n != len(b):
                raise IOError("short write")
            chunk = []
    os.fdatasync(fd)
    try:
        os.posix_fadvise(fd, at, NE * PER, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    os.close(fd)
    return layer, p, float(np.mean(errs)), float(np.max(errs)), time.time() - t0


def write_pack(out, pack, base):
    os.makedirs(pack, exist_ok=True)
    for f in ("dense.bin", "index.txt", "tokenizer"):
        dst = os.path.join(pack, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(base, f), dst)
    shard = os.path.basename(out)
    lines = [f"# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off shard "
             f"(n_expert {NE}, total {NL * LAYER}; Q8_0 experts from the FP8 checkpoint by tools/q8_experts.py, "
             f"in the shard {shard} beside --native)"]
    for l in range(NL):
        o = l * LAYER
        lines.append(f"{l} 8 8 {o} {BLOB} {o} {o + NE * PER} {o + 2 * NE * PER} {shard}")
    open(os.path.join(pack, "native_experts.txt"), "w").write("\n".join(lines) + "\n")


def check_nvfp4(src, nv_file, nv_pack, layers, experts, prefix):
    """NVFP4 blob (native_experts.txt v4) vs the FP8 original, same experts - how far each quant is from it."""
    # ggml's kvalues_fp4 are E2M1 doubled; ue4m3() halves the scale to match (as the CUDA ue4m3_to_f32 does)
    kv = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)

    def ue4m3(b):
        b = b.astype(np.int32)
        e, m = (b >> 3) & 0xF, b & 7
        v = np.where(e == 0, np.ldexp(m.astype(np.float32), -9), np.ldexp(1.0 + m / 8.0, e - 7)).astype(np.float32)
        return np.where((b == 0) | (b == 0x7F), 0.0, v / 2).astype(np.float32)

    rows = {}
    for ln in open(os.path.join(nv_pack, "native_experts.txt")):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        rows[int(f[0])] = [int(x) for x in f[5:8]] + [int(x) for x in f[9:12]]
    sh = Shards(src)
    fd = os.open(nv_file, os.O_RDONLY)
    nvb = 36                                           # block_nvfp4: 4 ue4m3 scales + 32 bytes of 64 values
    for l in layers:
        go, uo, do, gso, uso, dso = rows[l]
        for e in experts:
            res = []
            for p, (off, so, r, c) in enumerate(((go, gso, FF, H), (uo, uso, FF, H), (do, dso, H, FF))):
                per = r * c // 64 * nvb
                raw = np.frombuffer(os.pread(fd, per, off + e * per), np.uint8).reshape(r * c // 64, nvb)
                gs = struct.unpack("<f", os.pread(fd, 4, so + 4 * e))[0]
                d = ue4m3(raw[:, :4])                                  # [blocks, 4] sub-block scales
                qs = raw[:, 4:].reshape(-1, 4, 8)
                lo, hi = kv[qs & 0xF], kv[qs >> 4]
                vals = np.concatenate([lo, hi], axis=2) * d[:, :, None] * gs   # [blocks, 4, 16]
                w_nv = vals.reshape(r, c)
                w = fp8_matrix(sh, f"{prefix}.layers.{l}.mlp.experts.{e}.{('gate_proj', 'up_proj', 'down_proj')[p]}")
                _, back = q8_0(w)
                res.append((rel(w_nv, w), rel(back, w)))
            print(f"layer {l:2d} expert {e:3d}: NVFP4 vs FP8 " + " ".join(f"{a:.4f}" for a, _ in res) +
                  " | Q8_0 vs FP8 " + " ".join(f"{b:.5f}" for _, b in res), flush=True)


def mini_gguf(raw, path, layers, n=8):
    """the first n experts of some layers of the raw file as blk.L.ffn_{gate,up,down}_exps (Q8_0) in a small GGUF v3 -
    the input build/native_expert_parity takes (float reference vs the CPU vs the GPU kernels)."""
    fd = os.open(raw, os.O_RDONLY)
    tensors = []
    for l in layers:
        for p, role in enumerate(("gate", "up", "down")):
            ne = (H, FF, n) if p < 2 else (FF, H, n)
            tensors.append((f"blk.{l}.ffn_{role}_exps.weight", ne, os.pread(fd, n * PER, l * LAYER + p * NE * PER)))
    hdr = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 0))
    off = 0
    for name, ne, data in tensors:
        nb = name.encode()
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        hdr += struct.pack("<IQ", 8, off)
        off += (len(data) + 31) // 32 * 32
    hdr += b"\0" * ((32 - len(hdr) % 32) % 32)
    with open(path, "wb") as f:
        f.write(hdr)
        for _, _, data in tensors:
            f.write(data + b"\0" * ((32 - len(data) % 32) % 32))
    print(f"{path}: {len(tensors)} tensors, layers {layers}, {n} experts each")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--pack")
    ap.add_argument("--base-pack")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--prefix", default="model.language_model")
    ap.add_argument("--check-nvfp4", nargs=2, metavar=("FILE", "PACK"))
    ap.add_argument("--mini-gguf", metavar="PATH", help="write a small GGUF of --out's layers (--layers) and exit")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    if a.mini_gguf:
        mini_gguf(a.out, a.mini_gguf, layers)
        return
    if a.check_nvfp4:
        check_nvfp4(a.src, a.check_nvfp4[0], a.check_nvfp4[1], layers, (0, 137, 511), a.prefix)
        return
    if not os.path.exists(a.out):
        with open(a.out, "wb") as f:
            f.truncate(NL * LAYER)
    elif os.path.getsize(a.out) != NL * LAYER:
        sys.exit(f"{a.out}: size {os.path.getsize(a.out)} != {NL * LAYER}")
    if a.pack:
        write_pack(a.out, a.pack, a.base_pack)
    jobs = [(a.src, a.out, l, p, a.prefix) for l in layers for p in range(3)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (l, p, me, mx, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] layer {l:2d} {('gate', 'up', 'down')[p]:4s}: rel err mean {me:.5f} max {mx:.5f}"
                  f" ({dt:.0f} s, {time.time() - t0:.0f} s total)", flush=True)
    print(f"ГОТОВО: {a.out}, {len(jobs)} regions in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
