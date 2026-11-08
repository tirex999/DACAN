"""Release checks for 1M settings and deferred restarts; no model or GPU is used.

Run from the repository root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import runconfig
from serve.server import DacanEngine, Service

spec = importlib.util.spec_from_file_location("dacan_setup", ROOT / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ReleaseContext(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.exe = Path(self.temp.name) / "engine"
        self.exe.write_bytes(b"--threads-smt --kv fp16|int8 --kv q4_0 --kv k8v4")
        self.cfg = {"exe": str(self.exe), "args": ["--pack", "model", "--max-context", "524288",
                    "--kv", "int8", "--rope-scaling", "yarn", "--rope-scale", "4", "--threads", "6"],
                    "sampling": {"temperature": 0.9}, "api_key": "keep-this", "before_load": ["keep-this-too"]}

    def test_install_and_mcp_offer_1m_and_setup_selects_yarn4(self):
        self.assertIn(1048576, setup.CONTEXTS)
        self.assertEqual(setup.resolve_rope(1048576, None, None), ("yarn", 4.0))
        from tools.dacan_mcp import FALLBACK_CONTEXTS, tool_defs
        self.assertIn(1048576, FALLBACK_CONTEXTS)
        install = next(t for t in tool_defs(setup.MODELS, setup.FAMILIES, setup.CONTEXTS) if t["name"] == "dacan_install")
        self.assertIn(1048576, install["inputSchema"]["properties"]["context"]["enum"])
        self.assertIn("v0.1.42-build6", setup.prebuilt_bases(setup.PREBUILT_URL, setup.PREBUILT_ASSET)[0])
        self.assertEqual(setup.DACAN_BUILD, 6)

    def test_1m_round_trip_preserves_unrelated_settings(self):
        before = copy.deepcopy(self.cfg)
        cfg, changed = runconfig.apply(self.cfg, {"max_context": 1048576})
        self.assertEqual(self.cfg, before)
        expected = copy.deepcopy(before)
        expected["args"][expected["args"].index("--max-context") + 1] = "1048576"
        self.assertEqual(cfg, expected)
        self.assertEqual(changed, ["max_context"])
        path = Path(self.temp.name) / "dacan.json"
        path.write_text(json.dumps(before), encoding="utf-8")
        backup = runconfig.save(path, cfg)
        self.assertEqual(runconfig.load(backup), before)
        loaded = runconfig.load(path)
        self.assertEqual(loaded, cfg)
        context = next(k for k in runconfig.view(loaded, path)["keys"] if k["key"] == "max_context")
        self.assertEqual(context["value"], 1048576)
        self.assertEqual(context["max"], 1048576)
        self.assertIn(1048576, context["choices"])
        self.assertEqual(runconfig.apply(loaded, {"max_context": 1048576}), (loaded, []))

    def test_invalid_context_is_atomic_with_other_valid_changes(self):
        for bad in [True, 4095, 1048577, 524288.5, "1048576"]:
            with self.subTest(value=bad):
                before = copy.deepcopy(self.cfg)
                with self.assertRaises(ValueError):
                    runconfig.apply(self.cfg, {"sampling.temperature": 0.3, "max_context": bad})
                self.assertEqual(self.cfg, before)

    def fake_service(self, alive=True, fail=False):
        engine = object.__new__(DacanEngine)
        engine.spawn = ("engine", list(self.cfg["args"]))
        engine.alive = lambda: alive
        calls = []
        done = threading.Event()
        def restart():
            calls.append(list(engine.spawn[1]))
            if fail and len(calls) == 1:
                raise RuntimeError("synthetic startup failure")
            done.set()
        engine.restart = restart
        svc = object.__new__(Service)
        svc.engine = engine
        svc.fifo = threading.Lock()
        return svc, calls, done

    def test_restart_waits_for_active_request_and_uses_1m(self):
        cfg, _ = runconfig.apply(self.cfg, {"max_context": 1048576, "kv_cache": "q4_0"})
        svc, calls, done = self.fake_service()
        svc.fifo.acquire()
        try:
            self.assertEqual(svc.restart_engine(cfg), "restarting")
            self.assertFalse(done.wait(0.05))
            self.assertEqual(calls, [])
        finally:
            svc.fifo.release()
        self.assertTrue(done.wait(2))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index("--max-context") + 1], "1048576")
        self.assertEqual(calls[0][calls[0].index("--kv") + 1], "q4_0")

    def test_restart_failure_restores_previous_command(self):
        cfg, _ = runconfig.apply(self.cfg, {"max_context": 1048576})
        svc, calls, done = self.fake_service(fail=True)
        self.assertEqual(svc.restart_engine(cfg), "restarting")
        self.assertTrue(done.wait(2))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], self.cfg["args"])

    def test_unloaded_model_saves_next_start_and_unchanged_does_not_restart(self):
        cfg, _ = runconfig.apply(self.cfg, {"max_context": 1048576})
        svc, calls, done = self.fake_service(alive=False)
        self.assertEqual(svc.restart_engine(cfg), "saved")
        self.assertEqual(svc.restart_engine(cfg), "unchanged")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
