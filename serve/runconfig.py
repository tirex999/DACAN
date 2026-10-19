"""serve/runconfig.py - #564: the web page's Settings view of the run config (dacan-<model>.json).

A short list of documented keys can be read and changed from DACAN's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as dacan-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.
"""
from __future__ import annotations

import json
import mmap
import os
import shutil
from pathlib import Path

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", or ("sampling", check) for a key
# of the "sampling" block, ("arg", flag) for an engine option kept in "args".
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy, the default without a sampling block)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool", "Shorten a max_tokens that does not fit the context instead of answering 400"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    # DACAN 19.10.2026: the web page's CPU threads slider; a change starts the engine again (server.py)
    ("cpu_threads", ("threads", None),
     "CPU threads (logical processors) of the engine's expert pool, its own loop counted - up to every logical "
     "thread of the CPU; empty: the engine's own count; a change restarts the model"),
]
SPEC = {k: kind for k, kind, _ in EDITABLE}
_KNOWS: dict = {}


def engine_knows(exe, option: bytes) -> bool:
    """Whether the engine file names `option` (its usage text holds every option it takes); read once per file."""
    key = (str(exe), option)
    if key not in _KNOWS:
        try:
            with open(exe, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                _KNOWS[key] = m.find(option) >= 0
        except (OSError, ValueError, TypeError):
            _KNOWS[key] = False
    return _KNOWS[key]


def threads_flag(cfg: dict) -> str:
    """DACAN: the engine's option for cpu_threads - --threads N (all the pool's threads, the engine's loop counted;
    with --numa spread over the sockets) when the engine has it, else --pool-workers N-1 (the workers only: an
    engine 0.1.42 as released).  An option the engine does not know would stop it at the start."""
    return "--threads" if engine_knows(cfg.get("exe") or "", b"--threads-smt") else "--pool-workers"


def physical_cores() -> int:
    """The physical cores this process may run on, one per core: a core's second hardware thread made the engine
    slower, not faster (31 / 25 tok/s against 82 on 2 x 32 cores, 19.10.2026)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:                            # Windows
        cpus = []
    if cpus:
        seen = set()
        for c in cpus:
            base = f"/sys/devices/system/cpu/cpu{c}/topology/"
            try:
                with open(base + "physical_package_id") as a, open(base + "core_id") as b:
                    seen.add((a.read().strip(), b.read().strip()))
            except OSError:
                return len(cpus)
        return len(seen) or len(cpus)
    try:                                              # GetLogicalProcessorInformationEx, RelationProcessorCore (0)
        import ctypes
        k32 = ctypes.windll.kernel32
        n = ctypes.c_uint32(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))
        buf = (ctypes.c_byte * n.value)()
        count = 0
        if n.value and k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
            at = 0
            while at + 8 <= n.value:
                size = ctypes.c_uint32.from_buffer(buf, at + 4).value
                if size == 0:
                    break
                count += 1
                at += size
        if count:
            return count
    except (AttributeError, OSError, ValueError):
        pass
    return os.cpu_count() or 1


def cpu_threads_total() -> int:
    """Every CPU thread the engine may run on, SMT threads too (19.10.2026: the slider's maximum is how many threads
    the CPU has): a PC's all, a container's or a CPU set's own."""
    try:
        return len(os.sched_getaffinity(0)) or (os.cpu_count() or 1)
    except AttributeError:                            # Windows: every processor group
        try:
            import ctypes
            n = ctypes.windll.kernel32.GetActiveProcessorCount(0xFFFF)
            if n > 0:
                return int(n)
        except (AttributeError, OSError):
            pass
        return os.cpu_count() or 1


def cpu_threads_machine() -> int:
    """The machine's own CPU threads (a container may be given fewer: cpu_threads_total)."""
    try:
        n = len([d for d in os.listdir("/sys/devices/system/cpu") if d[3:].isdigit() and d.startswith("cpu")])
    except OSError:
        n = 0
    return max(n, cpu_threads_total())


def _arg(cfg: dict, flag: str):
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() else v
    return None


def value_of(cfg: dict, key: str):
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if isinstance(kind, tuple) and kind[0] == "arg":
        return _arg(cfg, kind[1])
    if isinstance(kind, tuple) and kind[0] == "threads":   # the engine's option, as threads in all
        flag = threads_flag(cfg)
        v = _arg(cfg, flag)
        if isinstance(v, int):
            return v + 1 if flag == "--pool-workers" else v
        env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
        e = str(env.get("DACAN_THREADS", "")) if flag == "--threads" else ""   # a service may set it there
        return int(e) if e.isdigit() and int(e) > 0 else None
    return cfg.get(key)


def view(cfg: dict, path: str | Path) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies)."""
    out = []
    for key, kind, help_ in EDITABLE:
        k = kind[0] if isinstance(kind, tuple) else kind
        out.append({"key": key, "value": value_of(cfg, key), "help": help_,
                    "kind": "number" if k in ("sampling", "arg", "int>=0", "num>=0", "threads") else k,
                    **({"choices": kind[1]} if k == "enum" else {}),
                    **({"min": 2, "max": cpu_threads_machine(), "usable": cpu_threads_total(),
                        "restart": True} if k == "threads" else {})})
    return {"file": Path(path).name, "keys": out,
            "note": "Saved to the run config; used from the next start of the model."}


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        return _number(key, v, whole=True)
    if k == "num>=0":
        return _number(key, v)
    if k == "arg":
        return _number(key, v, whole=True)
    if k == "threads":
        return _number(key, v, whole=True, lo=2, hi=cpu_threads_machine())
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """The config with `changes` ({key: value or None}) made, and the keys that changed.  Every value is checked
    first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        if value_of(new, key) == v:
            continue
        kind = SPEC[key]
        if isinstance(kind, tuple) and kind[0] == "sampling":
            s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
            sub = key.split(".", 1)[1]
            if v is None:
                s.pop(sub, None)
            else:
                s[sub] = v
            if s:
                new["sampling"] = s
            else:
                new.pop("sampling", None)
        elif isinstance(kind, tuple) and kind[0] == "arg":
            a = list(new.get("args") or [])
            flag = kind[1]
            if flag in a[:-1]:
                i = a.index(flag)
                if v is None:
                    del a[i:i + 2]
                else:
                    a[i + 1] = str(v)
            elif v is not None:
                a += [flag, str(v)]
            new["args"] = a
        elif isinstance(kind, tuple) and kind[0] == "threads":   # one thread option in "args", the engine's own
            old, a, i = list(new.get("args") or []), [], 0
            while i < len(old):
                if old[i] in ("--threads", "--pool-workers") and i + 1 < len(old):
                    i += 2
                    continue
                if old[i] != "--threads-smt":
                    a.append(old[i])
                i += 1
            if v is not None:
                flag = threads_flag(new)
                a += [flag, str(v - 1 if flag == "--pool-workers" else v)]
                if flag == "--threads" and v > physical_cores():   # past the cores: their second threads too
                    a.append("--threads-smt")
            new["args"] = a
        elif v is None:
            new.pop(key, None)
        else:
            new[key] = v
        changed.append(key)
    return new, changed


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = path.with_name(path.name + ".bak")
    shutil.copyfile(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return bak


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
