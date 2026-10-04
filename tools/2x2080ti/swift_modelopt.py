#!/usr/bin/env python3
"""Swift 1.5 Flash-Next as an NVFP4 checkpoint in RadixArk's form (ModelOpt safetensors) - 03.10.2026.

    RX_META=rx_meta python3 swift_modelopt.py run <swift_bf16_dir> <out_dir> [--layers 0,1] [--no-table] [--no-rest]
    RX_META=rx_meta python3 swift_modelopt.py verify <swift_bf16_dir> <out_dir> [--layers 0]

RadixArk/Qwen3.8-Flash-Next-NVFP4 is the NVFP4 checkpoint of the original model that served well on FreeToken
(06-08.09).  Its recipe, read from its hf_quant_config.json and safetensors headers (kept in /root/rx_meta; no weights
downloaded):
  * routed experts of the 48 layers in NVFP4, each expert's gate/up/down its own module:
    weight U8 [out, in/2] (value 2k in the low nibble), weight_scale F8_E4M3 [out, in/16], weight_scale_2 F32 [],
    input_scale F32 [];
  * weight_scale_2: gate and up share one value per expert, and the experts of a layer fall into a few groups with one
    value each (RadixArk converted the experts in chunks of 128 - its layer-*-experts-0000-0127.complete.json -
    so a layer has 4 values, or 3 where two chunks share their maximum); down has its own per expert;
  * the n-gram table in FP8 E4M3 (128 shards [2500012, 160]) with one BF16 ngram_embedding.weight_scale;
  * everything else BF16, as Swift stores it (attention, linear attention, routers, shared experts, hyper-connections,
    PLE projections, MTP with its fused BF16 experts, vision, embeddings, head).
Here the experts are quantized from Swift's BF16 with ModelOpt's weight math (swift_nvfp4.quant: block scale
(amax_16 / 6) / weight_scale_2 rounded to E4M3, values to E2M1, ties to the even code); weight_scale_2 =
amax / (6 * 448), the amax taken over the group RadixArk uses (gate+up of the experts in one group; down per expert).
The groups are read from RadixArk's own scales (radixark_meta.py's groups.json, or DACAN's copy of its experts).
input_scale is RadixArk's (calibrated on the original model): it only matters to W4A4 kernels; FreeToken computes
W4A16 and does not read it.  The table: weight_scale = amax / 448 (BF16), values x / weight_scale to E4M3 (nearest).
"""
import json
import os
import pathlib
import re
import shutil
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swift_nvfp4 import E2M1, quant  # noqa: E402

META = pathlib.Path(os.environ.get("RX_META", "rx_meta"))           # radixark_meta.py output
RX_BIN = os.environ.get("RX_EXPERTS_BIN", "/cloude/Qwen3.8-Flash-Next-NVFP4-experts.bin")
RX_NE = os.environ.get("RX_NATIVE_EXPERTS", "/q/packs/nvfp4-a/native_experts.txt")
E, FF, H = 512, 640, 2560
CHUNK = 32
PRE = "model.language_model.layers.%d."
TABLE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "U8": 1, "F8_E4M3": 1}
PER_TABLE_FILE = 8
REST_FILE_BYTES = 4 << 30


class Src:
    """Swift's safetensors, read as raw bytes by absolute offsets."""

    def __init__(self, d):
        self.d = d
        self.wmap = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        self.info = {}
        for fn in sorted(set(self.wmap.values())):
            with open(d / fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            for k, v in h.items():
                if k != "__metadata__":
                    self.info[k] = (fn, v["dtype"], tuple(v["shape"]), 8 + n + v["data_offsets"][0],
                                    v["data_offsets"][1] - v["data_offsets"][0])
        self.fh = {}

    def raw(self, name):
        fn, _, _, off, nb = self.info[name]
        if fn not in self.fh:
            self.fh[fn] = open(self.d / fn, "rb")
        f = self.fh[fn]
        f.seek(off)
        return f.read(nb)

    def bf16(self, name):
        fn, dt, shape, _, _ = self.info[name]
        assert dt == "BF16", (name, dt)
        return torch.frombuffer(bytearray(self.raw(name)), dtype=torch.bfloat16).view(shape)


class Writer:
    """One safetensors file: the header first (sizes are known up front), then the tensors in the same order."""

    def __init__(self, path, entries):
        self.f = open(path, "wb")
        self.order, header, off = [], {}, 0
        for name, dt, shape in entries:
            nb = int(np.prod(shape, dtype=np.int64)) * DT_BYTES[dt] if shape else DT_BYTES[dt]
            header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
            self.order.append((name, nb))
            off += nb
        header["__metadata__"] = {"format": "pt"}
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        self.f.write(struct.pack("<Q", len(hb)))
        self.f.write(hb)
        self.total, self.i = off, 0

    def put(self, name, data: bytes):
        want, nb = self.order[self.i]
        assert name == want and len(data) == nb, (name, want, len(data), nb)
        self.f.write(data)
        self.i += 1

    def close(self):
        assert self.i == len(self.order), (self.i, len(self.order))
        self.f.close()


def rx_groups():
    """RadixArk's gate/up global-scale groups: per layer, the group id of each expert (experts with one value).
    From radixark_meta.py's groups.json; else from DACAN's own copy of RadixArk's experts (RX_EXPERTS_BIN)."""
    if (META / "groups.json").exists():
        return {int(k): np.array(v) for k, v in json.load(open(META / "groups.json")).items()}
    out = {}
    with open(RX_BIN, "rb") as f:
        for line in open(RX_NE):
            if not line.strip() or line.startswith("#"):
                continue
            x = line.split()
            f.seek(int(x[9]))
            gate = np.frombuffer(f.read(4 * E), dtype=np.float32)
            f.seek(int(x[10]))
            up = np.frombuffer(f.read(4 * E), dtype=np.float32)
            assert np.array_equal(gate, up), "RadixArk: gate and up scales differ in layer %s" % x[0]
            _, gid = np.unique(gate, return_inverse=True)
            out[int(x[0])] = gid
    return out


def input_scales():
    p = META / "input_scales.json"
    assert p.exists(), "no %s: run radixark_meta.py first" % p
    return json.load(open(p))


def expert_entries(l):
    ent = []
    for e in range(E):
        for p, rows, n_in in (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF)):
            b = PRE % l + "mlp.experts.%d.%s." % (e, p)
            ent += [(b + "input_scale", "F32", ()), (b + "weight", "U8", (rows, n_in // 2)),
                    (b + "weight_scale", "F8_E4M3", (rows, n_in // 16)), (b + "weight_scale_2", "F32", ())]
    return ent


def quant_layer(src, l, gid):
    gu = src.bf16(PRE % l + "mlp.experts.gate_up_proj")            # [E, 2FF, H]
    dn = src.bf16(PRE % l + "mlp.experts.down_proj")               # [E, H, FF]
    assert tuple(gu.shape) == (E, 2 * FF, H) and tuple(dn.shape) == (E, H, FF), (gu.shape, dn.shape)
    amax_gu = torch.zeros(E)
    for e0 in range(0, E, CHUNK):
        amax_gu[e0:e0 + CHUNK] = gu[e0:e0 + CHUNK].float().abs().amax(dim=(1, 2))
    g_gu = torch.zeros(E)
    for k in np.unique(gid):
        m = torch.from_numpy(gid == k)
        g_gu[m] = amax_gu[m].max() / (6.0 * 448.0)
    res = {}
    for p, rows, n_in in (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF)):
        w8 = np.zeros((E, rows, n_in // 2), np.uint8)
        s8 = np.zeros((E, rows, n_in // 16), np.uint8)
        g = np.zeros(E, np.float32)
        for e0 in range(0, E, CHUNK):
            if p == "gate_proj":
                w, gg = gu[e0:e0 + CHUNK, :FF].float(), g_gu[e0:e0 + CHUNK]
            elif p == "up_proj":
                w, gg = gu[e0:e0 + CHUNK, FF:].float(), g_gu[e0:e0 + CHUNK]
            else:
                w, gg = dn[e0:e0 + CHUNK].float(), None                  # down: amax / (6 * 448) per expert
            cw, sc, gv = quant(w, gg)
            w8[e0:e0 + CHUNK], s8[e0:e0 + CHUNK], g[e0:e0 + CHUNK] = cw, sc, gv
        res[p] = (w8, s8, g)
    return res


def decode(w8, s8, g):
    """ModelOpt NVFP4 -> float32 [rows, in]: E2M1(code) * E4M3(block scale) * weight_scale_2."""
    codes = np.stack([w8 & 0x0F, w8 >> 4], axis=-1).reshape(w8.shape[0], -1).astype(np.int64)
    c = torch.from_numpy(codes)
    v = E2M1[c & 7] * torch.where((c & 8) > 0, -1.0, 1.0)
    s = torch.from_numpy(s8.copy()).view(torch.float8_e4m3fn).float()
    return (v.view(v.shape[0], -1, 16) * s[..., None] * float(g)).view(v.shape[0], -1)


def plan(src, layers, with_table, with_rest):
    """(file name, entries, filler) for every output file; filler(writer) writes its tensors in order."""
    rx = json.load(open(META / "tensors.json"))
    files, used = [], set()
    n_layers = 1 + max(int(k.split(".")[3]) for k in src.info if k.startswith("model.language_model.layers."))
    table = sorted((int(TABLE.search(k).group(1)), k) for k in src.info if TABLE.search(k))
    table_names = {k for _, k in table}
    for l in range(n_layers):
        if layers is not None and l not in layers:
            continue
        own = sorted(k for k in src.info if k.startswith(PRE % l) and ".mlp.experts." not in k and k not in table_names)
        ent = expert_entries(l) + [(k, src.info[k][1], src.info[k][2]) for k in own]
        files.append(("layer", l, ent, own))
        used.update(own)
    if with_table and table:
        scale_name = TABLE.sub(".ple.ple_embedding.ngram_embedding.weight_scale", table[0][1])
        assert scale_name in rx and rx[scale_name][0] == "BF16", scale_name
        for i in range(0, len(table), PER_TABLE_FILE):
            part = [k for _, k in table[i:i + PER_TABLE_FILE]]
            ent = ([(scale_name, "BF16", (1,))] if i == 0 else []) + [(k, "F8_E4M3", src.info[k][2]) for k in part]
            files.append(("table", i, ent, part))
    if with_rest:
        rest = sorted(k for k in src.info if k not in used and k not in table_names and
                      not re.search(r"^model\.language_model\.layers\.\d+\.", k))
        cur, size = [], 0
        for k in rest:
            if cur and size + src.info[k][4] > REST_FILE_BYTES:
                files.append(("rest", len(files), [(n, src.info[n][1], src.info[n][2]) for n in cur], cur))
                cur, size = [], 0
            cur.append(k)
            size += src.info[k][4]
        if cur:
            files.append(("rest", len(files), [(n, src.info[n][1], src.info[n][2]) for n in cur], cur))
    return files, table


def table_scale(src, table):
    p = META / "swift_table_amax.json"
    if p.exists():
        return json.load(open(p))["scale_bf16"]
    amax = 0.0
    for i, (_, k) in enumerate(table):
        amax = max(amax, float(src.bf16(k).float().abs().max()))
        print("  таблица, проход 1: шард %3d, amax %.6f" % (i, amax), flush=True)
    s = float(torch.tensor(amax / 448.0).to(torch.bfloat16))
    json.dump({"amax": amax, "scale_bf16": s}, open(p, "w"))
    return s


def run(src_dir, out, layers, with_table, with_rest):
    src = Src(src_dir)
    files, table = plan(src, layers, with_table, with_rest)
    n = len(files)
    out.mkdir(parents=True, exist_ok=True)
    gid = rx_groups()
    isc = input_scales()
    s_table = table_scale(src, table) if with_table and table else None
    wmap, total, t0 = {}, 0, time.time()
    for idx, (kind, key, ent, names) in enumerate(files):
        fn = "model-%05d-of-%05d.safetensors" % (idx + 1, n)
        w = Writer(out / fn, ent)
        if kind == "layer":
            res = quant_layer(src, key, gid[key])
            for e in range(E):
                for p in ("gate_proj", "up_proj", "down_proj"):
                    b = PRE % key + "mlp.experts.%d.%s." % (e, p)
                    w8, s8, g = res[p]
                    w.put(b + "input_scale", struct.pack("<f", isc[b + "input_scale"]))
                    w.put(b + "weight", w8[e].tobytes())
                    w.put(b + "weight_scale", s8[e].tobytes())
                    w.put(b + "weight_scale_2", struct.pack("<f", float(g[e])))
            for k in names:
                w.put(k, src.raw(k))
            print("слой %2d: gate/up масштабов %d, down %d; %.0f с" % (
                key, len(np.unique(res["gate_proj"][2])), len(np.unique(res["down_proj"][2])), time.time() - t0),
                flush=True)
        elif kind == "table":
            if key == 0:
                w.put(ent[0][0], torch.tensor([s_table], dtype=torch.bfloat16).view(torch.int16).numpy().tobytes())
            for k in names:
                x = (src.bf16(k).float() / s_table).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
                w.put(k, x.view(torch.uint8).numpy().tobytes())
            print("таблица: шарды %d-%d, %.0f с" % (key, key + len(names) - 1, time.time() - t0), flush=True)
        else:
            for k in names:
                w.put(k, src.raw(k))
        w.close()
        for name, _, _ in ent:
            wmap[name] = fn
        total += w.total
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wmap.items()))},
              open(out / "model.safetensors.index.json", "w"), indent=2)
    cfg = json.load(open(src_dir / "config.json"))
    rxc = json.load(open(META / "config.json"))
    cfg["text_config"]["ple_embedding_dtype"] = rxc["text_config"]["ple_embedding_dtype"]
    cfg["quantization_config"] = rxc["quantization_config"]
    json.dump(cfg, open(out / "config.json", "w"), indent=2, ensure_ascii=False)
    shutil.copy(META / "hf_quant_config.json", out / "hf_quant_config.json")
    for f in os.listdir(src_dir):
        if f.endswith(".safetensors") or f in ("config.json", "model.safetensors.index.json") or (src_dir / f).is_dir() \
                or f.startswith(".dl"):                     # .dl_list.txt: our downloader's list, not the model's
            continue
        shutil.copy(src_dir / f, out / f)
    print("ГОТОВО: %s, файлов весов %d, %.1f ГиБ, %.0f с" % (out, n, total / 2**30, time.time() - t0))
    return 0


def verify(src_dir, out, layers):
    src = Src(src_dir)
    rx = json.load(open(META / "tensors.json"))
    got = Src(out)
    bad = 0
    for k, (dt, shape, _) in rx.items():
        if layers is not None:
            m = re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.", k)
            if not m or int(m.group(1)) not in layers:
                continue
        if k not in got.info:
            bad += 1
            if bad <= 5:
                print("  нет у нас:", k)
            continue
        if got.info[k][1] != dt or got.info[k][2] != tuple(shape):
            bad += 1
            if bad <= 5:
                print("  расходится:", k, got.info[k][1:3], "у RadixArk", dt, shape)
    extra = [k for k in got.info if k not in rx]
    print("имена/типы/формы против RadixArk: расхождений %d, лишних у нас %d %s" % (bad, len(extra), extra[:3]))
    for l in (layers or [0, 21, 47]):
        gu = src.bf16(PRE % l + "mlp.experts.gate_up_proj")
        dn = src.bf16(PRE % l + "mlp.experts.down_proj")
        for e in (0, 7, 311):
            errs, gs = [], []
            for p, ref in (("gate_proj", gu[e, :FF]), ("up_proj", gu[e, FF:]), ("down_proj", dn[e])):
                b = PRE % l + "mlp.experts.%d.%s." % (e, p)
                w8 = np.frombuffer(got.raw(b + "weight"), np.uint8).reshape(got.info[b + "weight"][2])
                s8 = np.frombuffer(got.raw(b + "weight_scale"), np.uint8).reshape(got.info[b + "weight_scale"][2])
                g = struct.unpack("<f", got.raw(b + "weight_scale_2"))[0]
                d = decode(w8, s8, g)
                r = ref.float()
                errs.append(float((d - r).norm() / r.norm()))
                gs.append(g)
            print("слой %2d эксперт %3d: ошибка gate/up/down %s %%, weight_scale_2 %.4e/%.4e/%.4e (gate==up: %s)" % (
                l, e, "/".join("%.2f" % (x * 100) for x in errs), *gs, gs[0] == gs[1]))
    tk = [k for k in got.info if TABLE.search(k)]
    if tk:
        sk = [k for k in got.info if k.endswith("ngram_embedding.weight_scale")][0]
        s = float(torch.frombuffer(bytearray(got.raw(sk)), dtype=torch.bfloat16)[0])
        for k in (tk[0], tk[-1]):
            ref = src.bf16(k)[:20000].float()
            q = torch.frombuffer(bytearray(got.raw(k)), dtype=torch.uint8)[:20000 * 160].view(torch.float8_e4m3fn)
            d = q.float().view(-1, 160) * s
            print("таблица %s: масштаб %.4e, ошибка первых 20000 строк %.2f %%" % (
                k.split(".")[-2], s, float((d - ref).norm() / ref.norm()) * 100))
    for k in [k for k in got.info if k in src.info and got.info[k][1] == src.info[k][1]][:5]:
        print("байт в байт %s: %s" % (k, got.raw(k) == src.raw(k)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NVFP4_THREADS", "32")))
    a = sys.argv
    lay = [int(x) for x in a[a.index("--layers") + 1].split(",")] if "--layers" in a else None
    if a[1] == "run":
        sys.exit(run(pathlib.Path(a[2]), pathlib.Path(a[3]), lay, "--no-table" not in a, "--no-rest" not in a))
    sys.exit(verify(pathlib.Path(a[2]), pathlib.Path(a[3]), lay))
