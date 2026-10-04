#!/usr/bin/env python3
"""8-bit GGUF of Swift 1.5 Flash-Next with quant A's tensor layout - 03.10.2026.

    python3 swift_q8.py <bf16.gguf> <out.gguf> <layout_ref.gguf> [--limit N]

llama-quantize cannot give DACAN what it needs here: it copies ffn_gate_inp untouched (a built-in exclusion that
--tensor-type does not override), and the converter wrote the routers as F32, while DACAN takes them only as BF16.
So the types are set per tensor, following quant A's own file (layout_ref: the GSQ-RCO IQ4X-Q8A shard): every tensor
gets quant A's type for its name, except the routed experts and the n-gram table, which become Q8_0 (quant A keeps
4-bit experts and the table in another file).  Conversions: BF16 -> Q8_0 (gguf-py's Q8_0, the same rounding as
ggml's reference: d = amax / 127, roundf), F32 -> BF16 (round to nearest even), BF16 -> F16, same type -> copy.
Big tensors are quantized by row ranges in a process pool; the main process writes them in order.
"""
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", "/q/llama-up/gguf-py"))   # gguf-py of a llama.cpp that knows the architecture
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

SRC = DST = REF = None
ROWS_PER_JOB = 1 << 14


def pattern(name):
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def logical(t):
    """numpy-order element shape (last = ne0), whatever form the reader gives the data in."""
    return tuple(int(x) for x in reversed(t.shape.tolist()))


def to_f32(t, r0, r1):
    """rows [r0, r1) of a tensor (flattened to rows of ne0) as float32."""
    rows = int(np.prod(logical(t)[:-1]))
    a = t.data.reshape(rows, -1)[r0:r1]
    if t.tensor_type == Q.BF16:
        u16 = a.view(np.uint16) if a.dtype == np.uint8 else a.astype(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if t.tensor_type == Q.F32:
        return a.astype(np.float32, copy=False)
    if t.tensor_type == Q.F16:
        return a.astype(np.float32)
    raise ValueError(f"{t.name}: source type {t.tensor_type.name}")


_reader = None


def init_worker(src_path):
    """Python 3.14 starts pool workers by forkserver: globals set in main() do not reach them."""
    global SRC
    SRC = src_path


def job(args):
    idx, r0, r1, qt = args
    global _reader
    if _reader is None:
        _reader = gguf.GGUFReader(SRC, "r")
    t = _reader.tensors[idx]
    return gguf.quants.quantize(to_f32(t, r0, r1), qt)


def stream_write(w, parts, nbytes):
    """GGUFWriter.write_tensor_data for one tensor that arrives in row chunks: the same padding and bookkeeping."""
    fout = w.fout[0]
    name = next(iter(w.tensors[0]))
    ti = w.tensors[0].pop(name)
    assert ti.nbytes == nbytes, (name, ti.nbytes, nbytes)
    w.write_padding(fout, fout.tell())
    n = 0
    for part in parts:
        part.tofile(fout)
        n += part.nbytes
    assert n == nbytes, (name, n, nbytes)
    w.write_padding(fout, nbytes)
    w.state = gguf.gguf_writer.WriterState.WEIGHTS


def main():
    global SRC, DST, REF
    SRC, DST, REF = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    src = gguf.GGUFReader(SRC, "r")
    ref = gguf.GGUFReader(REF, "r")
    want = {}
    for t in ref.tensors:
        want[pattern(t.name)] = t.tensor_type
    plan = []
    for i, t in enumerate(src.tensors):
        p = pattern(t.name)
        if re.search(r"ffn_(gate|up|down)_exps\.weight$", t.name) or t.name == "per_layer_token_embd.weight":
            qt = Q.Q8_0
        elif p in want:
            qt = want[p]
        else:
            raise SystemExit(f"{t.name}: no type in the layout reference")
        if qt not in (Q.Q8_0, Q.BF16, Q.F16, Q.F32):
            raise SystemExit(f"{t.name}: reference type {qt.name} is not handled here")
        plan.append((i, t, qt))
    if limit:
        plan = plan[:limit]
    if "--only" in sys.argv:                                   # for a test: only the tensors whose name matches
        rx = re.compile(sys.argv[sys.argv.index("--only") + 1])
        plan = [x for x in plan if rx.search(x[1].name)]
    arch = src.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(DST, arch=arch, endianess=src.endianess)
    for field in src.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    w.add_string("general.quantized_by", "swift_q8.py: Q8_0 with quant A's layout (03.10.2026)")
    sizes = {}
    for i, t, qt in plan:
        shape = logical(t)
        if qt == t.tensor_type:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            sizes[t.name] = t.data.nbytes
        elif qt == Q.F16:
            nb = int(np.prod(shape)) * 2
            w.add_tensor_info(t.name, shape, np.dtype(np.float16), nb)
            sizes[t.name] = nb
        else:                                                                # Q8_0 / BF16: byte rows
            bshape = gguf.quants.quant_shape_to_byte_shape(shape, qt)
            nb = int(np.prod(bshape))
            w.add_tensor_info(t.name, bshape, np.dtype(np.uint8), nb, raw_dtype=qt)
            sizes[t.name] = nb
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0, done, total = time.time(), 0, sum(t.data.nbytes for _, t, _ in plan)
    counts = {}
    with mp.Pool(int(os.environ.get("Q8_PROCS", "16")), initializer=init_worker, initargs=(SRC,)) as pool:
        for i, t, qt in plan:
            counts[(t.tensor_type.name, qt.name)] = counts.get((t.tensor_type.name, qt.name), 0) + 1
            if qt == t.tensor_type:
                w.write_tensor_data(t.data, tensor_endianess=src.endianess)
            elif qt == Q.F16:
                w.write_tensor_data(to_f32(t, 0, int(np.prod(logical(t)[:-1]))).astype(np.float16).reshape(logical(t)))
            else:
                rows = int(np.prod(logical(t)[:-1]))
                jobs = [(i, r0, min(rows, r0 + ROWS_PER_JOB), qt) for r0 in range(0, rows, ROWS_PER_JOB)]
                stream_write(w, pool.imap(job, jobs, chunksize=1), sizes[t.name])
            done += t.data.nbytes
            if t.data.nbytes > (1 << 30) or i % 50 == 0:
                el = time.time() - t0
                print(f"{time.strftime('%H:%M:%S')} {t.name:40s} {t.tensor_type.name:>5s} -> {qt.name:5s} | "
                      f"{done / 2**30:7.1f} из {total / 2**30:.1f} ГиБ, {done / el / 1e6:6.1f} МБ/с", flush=True)
    w.close()
    print("ГОТОВО:", DST, "| переходы типов:", {f"{a}->{b}": n for (a, b), n in sorted(counts.items())})


if __name__ == "__main__":
    main()
