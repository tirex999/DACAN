"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).
28.09.2026 (DACAN): every GPU (`gpus`, with its PCI bus and NUMA node) and every CPU socket / NUMA node (`sockets`: load
of its CPUs, RAM of its node) besides the old single-card `gpu_*` and whole-machine `cpu` / `ram_*` keys.

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    _lib = None          # 28.09.2026 (DACAN): one NVML load and init for every card
    _tried = False

    class Pci(ctypes.Structure):     # nvmlPciInfo_t
        _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                    ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                    ("busId", ctypes.c_char * 32)]

    @classmethod
    def _load(cls):
        if cls._tried:
            return cls._lib
        cls._tried = True
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                lib = ctypes.CDLL(n)
            except OSError:
                continue
            try:
                init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
                if init() == 0:
                    cls._lib = lib
            except (AttributeError, OSError):
                pass
            break
        return cls._lib

    @classmethod
    def count(cls):
        lib = cls._load()
        if lib is None:
            return 0
        n = ctypes.c_uint()
        try:
            get = getattr(lib, "nvmlDeviceGetCount_v2", None) or lib.nvmlDeviceGetCount
            return n.value if get(ctypes.byref(n)) == 0 else 0
        except (AttributeError, OSError):
            return 0

    def __init__(self, index=0):
        self.lib = self.dev = None
        self.index = index
        lib = self._load()
        if lib is None:
            return
        try:
            h = ctypes.c_void_p()
            get = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None) or lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                return
            self.lib, self.dev = lib, h
        except (AttributeError, OSError):
            self.lib = None

    def pci(self):
        """-> (bus id like 0000:98:00.0, NUMA node or None)"""
        p = self.Pci()
        try:
            fn = getattr(self.lib, "nvmlDeviceGetPciInfo_v3", None) or getattr(self.lib, "nvmlDeviceGetPciInfo_v2", None) \
                or self.lib.nvmlDeviceGetPciInfo
            if fn(self.dev, ctypes.byref(p)) != 0:
                return None, None
        except (AttributeError, OSError):
            return None, None
        bus = (p.busId or p.busIdLegacy).decode(errors="replace").lower()
        if bus.count(":") == 2 and len(bus.split(":")[0]) == 8:      # NVML pads the domain to 8 digits
            bus = bus[4:]
        node = None
        try:
            node = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read().strip())
            node = node if node >= 0 else None
        except (OSError, ValueError):
            pass
        return bus, node

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            # 28.09.2026 (DACAN): only the first 8 fields - guest and guest_nice are already counted inside user and nice,
            # so summing them too counted a virtual machine's load twice (host with VM 105: 36 % shown as 56 %)
            return f[3] + f[4], sum(f[:8])
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


def _cpu_list(text):
    """'0-31,64-95' -> [0..31, 64..95]"""
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class _Sockets:
    """28.09.2026 (DACAN): each NUMA node (= a CPU socket here) on its own - the load of its logical CPUs from /proc/stat
    and the RAM of its memory controllers from /sys/devices/system/node/nodeN/meminfo. Linux only; empty elsewhere."""
    BASE = "/sys/devices/system/node"

    def __init__(self):
        self.nodes = []
        try:
            for d in os.listdir(self.BASE):
                if d.startswith("node") and d[4:].isdigit():
                    cpus = _cpu_list(open(f"{self.BASE}/{d}/cpulist").read())
                    if cpus:
                        self.nodes.append((int(d[4:]), cpus))
        except (OSError, ValueError):
            self.nodes = []
        self.nodes.sort()
        self.prev = self._times()

    @staticmethod
    def _times():
        t = {}
        try:
            for line in open("/proc/stat"):
                if line.startswith("cpu") and line[3:4].isdigit():
                    f = line.split()
                    v = [int(x) for x in f[1:]]
                    t[int(f[0][3:])] = (v[3] + v[4], sum(v[:8]))   # guest time is already inside user, see _times
        except (OSError, ValueError):
            pass
        return t

    def read(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        out = []
        for node, cpus in self.nodes:
            idle = total = 0
            for c in cpus:
                if c in cur and c in prev:
                    idle += cur[c][0] - prev[c][0]
                    total += cur[c][1] - prev[c][1]
            s = {"node": node, "cpu": max(0.0, min(100.0, 100.0 * (1 - idle / total))) if total > 0 else None}
            try:
                info = {}
                for line in open(f"{self.BASE}/node{node}/meminfo"):
                    f = line.split()        # "Node 0 MemTotal:  131713012 kB"
                    if len(f) >= 4:
                        info[f[2].rstrip(":")] = int(f[3]) * 1024
                s["ram_total"] = info.get("MemTotal")
                if "MemTotal" in info and "MemFree" in info:   # without the page cache, which the kernel gives back
                    s["ram_used"] = info["MemTotal"] - info["MemFree"] - info.get("FilePages", 0)
            except (OSError, ValueError):
                pass
            out.append(s)
        return out


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.gpu = _Nvml()
        # 28.09.2026 (DACAN): every card and every socket, not just card 0 and the whole machine
        self.gpus = [g for g in (_Nvml(i) for i in range(_Nvml.count())) if g.ok()]
        self.sockets = _Sockets()
        gstatic = []
        for g in self.gpus:
            bus, node = g.pci()
            gstatic.append({"index": g.index, "name": g.name(), "bus": bus, "numa": node})
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": self.gpu.name() if self.gpu.ok() else None,
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
            "gpus": gstatic,
            "sockets": [{"node": n, "threads": len(c)} for n, c in self.sockets.nodes],
            "engine_name": "DACAN",
        }
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        s["gpus"] = [g.read() for g in self.gpus]
        if s["gpus"] and self.gpus[0].index == 0:        # the old single-card keys from the same reading of card 0
            s.update({f"gpu_{k}": v for k, v in s["gpus"][0].items()})
        elif self.gpu.ok():
            g = self.gpu.read()
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["sockets"] = self.sockets.read()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
                for i, g in enumerate(s.get("gpus") or []):
                    for k in ("util", "mem_used", "power"):
                        v = g.get(k)
                        self.hist[f"gpu{i}_{k}"].append(round(v, 2) if isinstance(v, float) else v)
                for i, c in enumerate(s.get("sockets") or []):
                    v = c.get("cpu")
                    self.hist[f"socket{i}_cpu"].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
