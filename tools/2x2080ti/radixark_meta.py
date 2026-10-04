"""What swift_modelopt.py needs from RadixArk/Qwen3.8-Flash-Next-NVFP4, without its 126 GB of weights - 04.10.2026.

    python radixark_meta.py <out_dir>

Writes into <out_dir>: the repository's small files (config.json, hf_quant_config.json, ...); tensors.json (every
tensor's dtype, shape and file, from the safetensors headers); input_scales.json (every routed expert's gate/up/down
input_scale); groups.json (per layer, the group of every expert: the experts that share one gate/up weight_scale_2).
Everything is read by HTTP byte ranges from the public repository: one redirect per file, then the CDN.
"""
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
R = "https://huggingface.co/%s/resolve/main/" % REPO
E, L = 512, 48
PRE = "model.language_model.layers.%d.mlp.experts.%d.%s.%s"
s = requests.Session()


def signed(fn):
    r = s.get(R + fn, headers={"Range": "bytes=0-7"}, allow_redirects=False, timeout=60)
    return r.headers["Location"] if r.status_code in (301, 302, 307, 308) else R + fn


def rng(url, a, b):
    for _ in range(5):
        try:
            r = s.get(url, headers={"Range": "bytes=%d-%d" % (a, b)}, timeout=60)
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
        except requests.RequestException:
            pass
    raise RuntimeError("range %s %d-%d" % (url[:80], a, b))


def per_file(item):
    fn, names = item
    url = signed(fn)
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    h = json.loads(rng(url, 8, 7 + n))
    shapes = {k: [v["dtype"], v["shape"], fn] for k, v in h.items() if k != "__metadata__"}
    vals = {}
    for name in names:
        a, b = h[name]["data_offsets"]
        vals[name] = struct.unpack("<f", rng(url, 8 + n + a, 8 + n + b - 1))[0]
    return shapes, vals


def main(out):
    os.makedirs(out, exist_ok=True)
    files = [x["rfilename"] for x in s.get("https://huggingface.co/api/models/" + REPO, timeout=60).json()["siblings"]]
    for f in files:
        if not f.endswith(".safetensors"):
            open(os.path.join(out, f), "wb").write(s.get(R + f, timeout=120).content)
    wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    want = []
    for l in range(L):
        want += [PRE % (l, e, p, "input_scale") for e in (0, 255, 511) for p in ("gate_proj", "up_proj")]
        want += [PRE % (l, e, "down_proj", "input_scale") for e in range(E)]
        want += [PRE % (l, e, "gate_proj", "weight_scale_2") for e in range(E)]
        want += [PRE % (l, e, "up_proj", "weight_scale_2") for e in (0, 255, 511)]
    by = {fn: [] for fn in set(wm.values())}
    for name in want:
        by[wm[name]].append(name)
    tensors, got = {}, {}
    with ThreadPoolExecutor(16) as ex:
        for i, (shapes, vals) in enumerate(ex.map(per_file, sorted(by.items()))):
            tensors.update(shapes)
            got.update(vals)
            if i % 20 == 0:
                print("files %d of %d, values %d" % (i + 1, len(by), len(got)), flush=True)
    json.dump(tensors, open(os.path.join(out, "tensors.json"), "w"))
    isc, groups = {}, {}
    for l in range(L):
        v = {got[PRE % (l, e, p, "input_scale")] for e in (0, 255, 511) for p in ("gate_proj", "up_proj")}
        assert len(v) == 1, ("gate/up input_scale differs within layer", l, v)
        g_in = v.pop()
        gate = [got[PRE % (l, e, "gate_proj", "weight_scale_2")] for e in range(E)]
        for e in (0, 255, 511):
            assert got[PRE % (l, e, "up_proj", "weight_scale_2")] == gate[e], ("gate/up weight_scale_2 differ", l, e)
        uniq = sorted(set(gate))
        groups[l] = [uniq.index(x) for x in gate]
        for e in range(E):
            isc[PRE % (l, e, "gate_proj", "input_scale")] = g_in
            isc[PRE % (l, e, "up_proj", "input_scale")] = g_in
            isc[PRE % (l, e, "down_proj", "input_scale")] = got[PRE % (l, e, "down_proj", "input_scale")]
    json.dump(isc, open(os.path.join(out, "input_scales.json"), "w"))
    json.dump(groups, open(os.path.join(out, "groups.json"), "w"))
    print("done: %d tensors, %d input scales, groups per layer %s" % (
        len(tensors), len(isc), sorted({len(set(g)) for g in groups.values()})))


if __name__ == "__main__":
    main(sys.argv[1])
