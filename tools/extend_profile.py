"""Extend an expert profile (STRP) to every expert, so `--expert-cache auto` can fill a big GPU.

The engine caps `--expert-cache auto` at the length of the profile's ranked list: `data/expert-profile.bin` ranks
8,000 (layer, expert) pairs, which is what a 12 GB card holds, so on a 24-48 GB card the cache stops at 8,000 experts
and the rest of the VRAM stays empty.  This tool keeps the profile's ranked list exactly as it is and appends every
other pair, most-routed first, so a small card loads the same experts as before and a big one keeps going down the
same ranking.

    python tools/extend_profile.py data/expert-profile.bin data/expert-profile-full.bin [--usage U.bin] [--pairs N]

The order of the appended pairs comes from the profile's own frequency table when it is valid; the frequency table
in `data/expert-profile.bin` is not (NaN and denormals), so then it comes from a routing-count table (`STRU`, default
`data/usage_other_2609.bin`: counts over our own runs) and that table, normalised, is written into the new file.
N defaults to every pair (48 x 512 = 24,576 for Qwen3.8-Flash-Next).

STRP: "STRP", 5 x u32 (version, layers, experts, slots, ranked), ranked x (u16 layer, u16 expert),
layers x experts x f32 frequencies.  STRU: "STRU", 2 x i32 (layers, experts), layers x experts x u64 counts.
"""
import argparse
import math
import os
import struct


def read_profile(path):
    b = open(path, "rb").read()
    if b[:4] != b"STRP":
        raise SystemExit(f"{path}: not an STRP profile")
    version, nl, ne, _slots, n_ranked = struct.unpack_from("<5I", b, 4)
    off = 24
    pairs = [struct.unpack_from("<2H", b, off + 4 * i) for i in range(n_ranked)]
    off += 4 * n_ranked
    freq = None
    if len(b) - off == 4 * nl * ne:
        freq = list(struct.unpack_from(f"<{nl * ne}f", b, off))
        if any(math.isnan(x) or x < 0 for x in freq) or not 0.5 < sum(freq) < 1.5:
            freq = None                                  # not a frequency table (data/expert-profile.bin)
    return version, nl, ne, pairs, freq


def read_usage(path, nl, ne):
    u = open(path, "rb").read()
    if u[:4] != b"STRU":
        raise SystemExit(f"{path}: not an STRU routing-count table")
    ul, ue = struct.unpack_from("<2i", u, 4)
    if (ul, ue) != (nl, ne):
        raise SystemExit(f"{path} is {ul}x{ue}, the profile is {nl}x{ne}")
    cnt = struct.unpack_from(f"<{nl * ne}Q", u, 12)
    tot = float(sum(cnt)) or 1.0
    return [c / tot for c in cnt]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--usage", default=os.path.join(here, "..", "data", "usage_other_2609.bin"))
    ap.add_argument("--pairs", type=int, default=0, help="length of the ranked list (default: every pair)")
    a = ap.parse_args()
    version, nl, ne, pairs, freq = read_profile(a.src)
    source = "the profile's own frequency table"
    if freq is None:
        freq = read_usage(a.usage, nl, ne)
        source = os.path.basename(a.usage)
    n = a.pairs if a.pairs > 0 else nl * ne
    seen = {l * ne + e for l, e in pairs}
    order = [l * ne + e for l, e in pairs] + sorted((i for i in range(nl * ne) if i not in seen),
                                                   key=lambda i: (-freq[i], i))
    order = order[:n]
    with open(a.dst, "wb") as f:
        f.write(b"STRP")
        f.write(struct.pack("<5I", version, nl, ne, len(order), len(order)))
        for i in order:
            f.write(struct.pack("<2H", i // ne, i % ne))
        f.write(struct.pack(f"<{nl * ne}f", *freq))
    cover = " / ".join(f"{100.0 * sum(freq[i] for i in order[:k]):.1f}%" for k in (len(pairs), 16000, len(order))
                       if k <= len(order))
    print(f"{a.dst}: {len(order)} ranked pairs, the first {min(len(pairs), len(order))} exactly as in {a.src}, the "
          f"rest by {source}; routing covered (by {source}) at {len(pairs)} / 16000 / all pairs: {cover}")


if __name__ == "__main__":
    main()
