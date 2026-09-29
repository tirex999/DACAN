#!/bin/bash
# Strata on two GPUs and two CPU sockets: writes strata-<TAG>.json for serve/server.py and starts the OpenAI server.
#
#   S1=<first GGUF shard (--native)> S2=<n-gram table shard> PACK=<pack dir> TAG=<name> MTP=<mtp/rt dir> \
#     [PROFILE=...] [USAGE=...] [CTX=131072] [KV=int8|fp16] [NUMA=1|0] [MAIN=1] [SECOND=0] [PORT=8080] \
#     [EXTRA="<more engine switches>"] tools/2x2080ti/run-fast.sh
#
# MAIN / SECOND: CUDA indices of the main card (dense part + hot experts) and of the second card (experts only).
# NUMA=1: --numa (arena split by node, host loop on the main card's node); NUMA=0: everything on node 1 via numactl.
# KV: int8 is what our numbers were measured with; fp16 is upstream's default.
# EXTRA: e.g. "--vram-reserve-mib 1024" for a Q8_0 output head.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
S1=${S1:?S1}; S2=${S2:?S2}; PACK=${PACK:?PACK}; TAG=${TAG:?TAG}; MTP=${MTP:-$ROOT/mtp/rt}
PROFILE=${PROFILE:-$ROOT/data/profile_other_2609.bin}; USAGE=${USAGE:-$ROOT/data/usage_other_2609.bin}
CTX=${CTX:-131072}; EXTRA=${EXTRA:-}; NUMA=${NUMA:-1}; KV=${KV:-int8}; MAIN=${MAIN:-1}; SECOND=${SECOND:-0}
PORT=${PORT:-8080}
if [ "$NUMA" = 1 ]; then RUNNER=""; EXTRA="--numa $EXTRA"; else RUNNER="numactl --cpunodebind=1 --membind=1"; fi
cat > "$ROOT/run-fast-$TAG.sh" <<EOW
#!/bin/sh
export CUDA_VISIBLE_DEVICES=$MAIN,$SECOND
exec $RUNNER "$ROOT/build/strata" "\$@"
EOW
chmod +x "$ROOT/run-fast-$TAG.sh"
"$PY" - "$ROOT" "$PACK" "$S1" "$S2" "$MTP" "$CTX" "$TAG" "$PROFILE" "$USAGE" "$EXTRA" "$KV" "$PORT" <<'PYEOF'
import json, sys
root, pack, s1, s2, mtp, ctx, tag, prof, usage, extra, kv, port = sys.argv[1:13]
args = ["--pack", pack, "--native", s1, "--ple-gguf", s2, "--ple-io", "mmap",
        "--expert-profile", prof, "--expert-cache", "auto", "--adapt-every", "0",
        "--second-card", "1", "--second-card-usage", usage, "--pcie-frac", "0",
        "--prefill", "8192", "--park", "16", "--park-mib", "16384", "--spec", "4", "--spec-min-p", "0.5", "--mtp", mtp, "--max-context", ctx, "--kv", kv] + extra.split()
cfg = {"exe": "%s/run-fast-%s.sh" % (root, tag), "args": args, "cwd": root, "tokenizer": pack + "/tokenizer",
       "model_name": "qwen3.8-flash-next-" + tag, "log": "%s/strata-%s.log" % (root, tag), "port": int(port)}
open("%s/strata-%s.json" % (root, tag), "w").write(json.dumps(cfg, indent=1))
print(" ".join(args))
PYEOF
cd "$ROOT"
exec "$PY" -u serve/server.py --engine strata --config "$ROOT/strata-$TAG.json" --port "$PORT" --host "${HOST:-127.0.0.1}"
