#!/usr/bin/env python3
"""NVFP4 experts from BF16 (ModelOpt's weight recipe) into DACAN's arena form - 03.10.2026, for Swift 1.5 Flash-Next.

    python3 swift_nvfp4.py selftest <experts.bin> <native_experts.txt>      # re-encode NVIDIA's experts, compare bytes
    python3 swift_nvfp4.py run <bf16_safetensors_dir> <experts.bin> <base_pack_dir> <out_pack_dir> [shard name]

There is no NVIDIA NVFP4 checkpoint of Swift, so the experts are quantized here the way ModelOpt quantizes weights for
NVIDIA's checkpoint of the original model (modelopt NVFP4QTensor): a global scale per expert matrix,
weight_scale_2 = amax|W| / (6 * 448); a scale per 16 inputs, (amax_block / 6) / weight_scale_2 rounded to E4M3; each
value W / (scale * weight_scale_2) rounded to E2M1 {0, .5, 1, 1.5, 2, 3, 4, 6} (ties to the even code, as ModelOpt).
The packing and the file layout are tools/nvfp4_experts.py's own (`pack`), so DACAN reads the result exactly like
NVIDIA's experts: per layer gate [E][FF rows], up [E][FF rows], down [E][H rows], then F32 global scales of gate, up,
down [E] each.  Swift stores experts fused: experts.gate_up_proj [E, 2*FF, H] (gate = first FF rows, up = the rest,
as llama.cpp's converter splits it) and experts.down_proj [E, H, FF].
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # tools/nvfp4_experts.py
from nvfp4_experts import E, FF, H, NVFP4, pack  # noqa: E402

BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
TIE_UP = torch.tensor([False, True, False, True, False, True, False])
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
CHUNK = 32


def quant(w: torch.Tensor, g=None):
    """w [n, out, in] float32 -> (codes [n, out, in/2] uint8, E4M3 scale bytes [n, out, in/16] uint8, g [n] float32).
    g: the global scale per matrix; by default amax / (6 * 448)."""
    n, o, i = w.shape
    if g is None:
        g = w.abs().amax(dim=(1, 2)) / (6.0 * 448.0)
    g = torch.where(g > 0, g, torch.ones_like(g))
    wb = w.view(n, o, i // 16, 16)
    bs = wb.abs().amax(-1) / 6.0
    q = bs / g[:, None, None]
    q = torch.where(bs == 0, torch.ones_like(q), q)
    s8 = q.to(torch.float8_e4m3fn)
    eff = s8.float() * g[:, None, None]
    x = wb / eff[..., None]
    a = x.abs().contiguous()
    code = torch.bucketize(a, BOUNDS)                                   # boundaries[i-1] < a <= boundaries[i]
    code = code + ((a.unsqueeze(-1) == BOUNDS) & TIE_UP).any(-1).to(code.dtype)
    code = code.to(torch.uint8) | ((x < 0).to(torch.uint8) << 3)
    code = code.view(n, o, i)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous().numpy(), s8.view(torch.uint8).numpy(), g.numpy()


def unpack_rows(raw: np.ndarray, n_in: int) -> np.ndarray:
    """block_nvfp4 rows [out, n_in/64*36] -> (codes [out, n_in] uint8, UE4M3 bytes [out, n_in/16])."""
    out = raw.shape[0]
    b = raw.reshape(out, n_in // 64, 36)
    d = b[:, :, :4].reshape(out, n_in // 16)
    qs = b[:, :, 4:].reshape(out, n_in // 16, 8)
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)              # value j low nibble, j + 8 high nibble
    return vals.reshape(out, n_in), d


def decode(raw: np.ndarray, n_in: int, g: float) -> torch.Tensor:
    codes, d = unpack_rows(raw, n_in)
    c = torch.from_numpy(codes.astype(np.int64))
    v = E2M1[c & 7] * torch.where((c & 8) > 0, -1.0, 1.0)
    s = torch.from_numpy((d & 0x7F).astype(np.uint8)).view(torch.float8_e4m3fn).float()
    return (v.view(raw.shape[0], n_in // 16, 16) * s[..., None] * g).view(raw.shape[0], n_in)


def selftest(bin_path, ne_path):
    line = next(l for l in open(ne_path) if l.strip() and not l.startswith("#")).split()
    offs = {"gate": (int(line[5]), int(line[9]), FF, H), "up": (int(line[6]), int(line[10]), FF, H),
            "down": (int(line[7]), int(line[11]), H, FF)}
    with open(bin_path, "rb") as f:
        tot = same = 0
        for e in (0, 7, 311):
            mats = {}
            for p, (off, soff, rows, n_in) in offs.items():
                rb = n_in // 64 * 36
                f.seek(off + e * rows * rb)
                raw = np.frombuffer(f.read(rows * rb), dtype=np.uint8).reshape(rows, rb)
                f.seek(soff + 4 * e)
                g = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
                codes, d = unpack_rows(raw, n_in)
                smax = float(torch.from_numpy((d & 0x7F).astype(np.uint8)).view(torch.float8_e4m3fn).float().max())
                mats[p] = (raw, g, rows, n_in, smax)
            for p, (raw, g, rows, n_in, smax) in mats.items():
                w = decode(raw, n_in, g)
                cw, sc, _ = quant(w[None], torch.tensor([g]))       # NVIDIA's own global scale
                again = pack(cw[0], sc[0])
                eq = float((again == raw).mean())
                c1, d1 = unpack_rows(raw, n_in)
                c2, d2 = unpack_rows(again, n_in)
                z = lambda c: np.where((c & 7) == 0, 0, c)             # -0 (код 8) и +0 считать одним
                eq_codes = float((z(c1) == z(c2)).mean())
                eq_sc = float(((d1 & 0x7F) == (d2 & 0x7F)).mean())
                neg0 = float(((c1 & 15) == 8).mean())
                print(f"эксперт {e:3d} {p:4s}: глоб. {g:.4e}, наиб. масштаб блока {smax:6.1f}; байты {eq * 100:.2f} %, "
                      f"коды (−0=+0) {eq_codes * 100:.4f} %, масштабы блоков {eq_sc * 100:.4f} %, кодов −0 у NVIDIA {neg0 * 100:.2f} %")
                eq = min(eq_codes, eq_sc)
                tot += 1
                same += eq > 0.999
    print("САМОПРОВЕРКА:", "ПРОЙДЕНА" if same == tot else "НЕ ПРОЙДЕНА")
    return 0 if same == tot else 1


def run(src, dst, base, out, shard_name):
    wmap = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    n_layers = 1 + max(int(k.split(".")[3]) for k in wmap if k.startswith("model.language_model.layers."))
    gu_row, d_row = H // 64 * 36, FF // 64 * 36
    per = {"gate_proj": FF * gu_row, "up_proj": FF * gu_row, "down_proj": H * d_row}
    w_bytes = sum(per.values())
    layer_bytes = E * w_bytes + 3 * E * 4
    blob = w_bytes + 16
    handles = {}

    def get(name):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(str(src / f), "pt")
        return handles[f].get_tensor(name)

    total = n_layers * layer_bytes
    mode = "r+b" if dst.exists() and dst.stat().st_size == total else "w+b"
    lines, arena, t0 = [], 0, time.time()
    with open(dst, mode) as fo:
        if mode == "w+b":
            fo.truncate(total)
        for l in range(n_layers):
            base_off = l * layer_bytes
            region = {"gate_proj": base_off, "up_proj": base_off + E * per["gate_proj"],
                      "down_proj": base_off + E * (per["gate_proj"] + per["up_proj"])}
            scale_off = base_off + E * w_bytes
            scales = {p: np.zeros(E, dtype=np.float32) for p in per}
            pre = "model.language_model.layers.%d.mlp.experts." % l
            gu = get(pre + "gate_up_proj")                      # [E, 2FF, H] bf16
            dn = get(pre + "down_proj")                          # [E, H, FF] bf16
            assert tuple(gu.shape) == (E, 2 * FF, H) and tuple(dn.shape) == (E, H, FF), (gu.shape, dn.shape)
            errs = []
            for p, rows, n_in in (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF)):
                for e0 in range(0, E, CHUNK):
                    if p == "gate_proj":
                        w = gu[e0:e0 + CHUNK, :FF, :].float()
                    elif p == "up_proj":
                        w = gu[e0:e0 + CHUNK, FF:, :].float()
                    else:
                        w = dn[e0:e0 + CHUNK].float()
                    cw, sc, g = quant(w)
                    n = w.shape[0]
                    raw = pack(cw.reshape(n * rows, n_in // 2), sc.reshape(n * rows, n_in // 16))
                    assert raw.nbytes == n * per[p]
                    fo.seek(region[p] + e0 * per[p])
                    fo.write(raw.tobytes())
                    scales[p][e0:e0 + n] = g
                    if e0 == 0:                                   # spot check: expert 0 back from its bytes
                        w0 = decode(raw[:rows], n_in, float(g[0]))
                        errs.append(float((w0 - w[0]).norm() / w[0].norm()))
            fo.seek(scale_off)
            for p in ("gate_proj", "up_proj", "down_proj"):
                fo.write(scales[p].tobytes())
            lines.append("%d %d %d %d %d %d %d %d %s %d %d %d" % (
                l, NVFP4, NVFP4, arena, blob, region["gate_proj"], region["up_proj"], region["down_proj"], shard_name,
                scale_off, scale_off + 4 * E, scale_off + 8 * E))
            arena += blob * E
            el = time.time() - t0
            print("слой %2d: ошибка эксперта 0 gate/up/down %s %%, %.1f ГиБ, %.0f с" % (
                l, "/".join("%.2f" % (x * 100) for x in errs), (l + 1) * layer_bytes / 2**30, el), flush=True)
            del gu, dn
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dense.bin", "index.txt", "tokenizer"):
        if not (out / name).exists():
            os.symlink(base / name, out / name)
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off shard "
                 "gate_scale_off up_scale_off down_scale_off (n_expert %d, total %d; NVFP4 experts quantized from BF16 "
                 "of %s by swift_nvfp4.py (ModelOpt weight recipe) in the shard %s beside --native; blob = weights %d + "
                 "global scales 16)\n" % (E, arena, src.name, shard_name, w_bytes))
        fo.write("\n".join(lines) + "\n")
    print("пакет %s: %d слоёв, блоб %d Б, арена %.2f ГиБ" % (out, n_layers, blob, arena / 2**30))
    return 0


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NVFP4_THREADS", "32")))
    if sys.argv[1] == "selftest":
        sys.exit(selftest(sys.argv[2], sys.argv[3]))
    src, dst = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    base, out = pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])
    sys.exit(run(src, dst, base, out, sys.argv[6] if len(sys.argv) > 6 else dst.name))
