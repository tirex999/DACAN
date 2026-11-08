"""Ownership transfer and native release acknowledgements, without a GPU or a model."""
import queue
import threading
import time
import unittest
from unittest import mock

from serve.server import DacanEngine


class SlotReleaseFence(unittest.TestCase):
    def engine(self):
        e = DacanEngine.__new__(DacanEngine)
        e.slot_cv = threading.Condition()
        e.slot_q = [queue.Queue()]
        e.slot_release_q = [queue.Queue()]
        e.slot_busy = [True]
        e.slot_held = [[11, 12]]
        e.slot_claims = {0}
        e.slot_releasing = set()
        e._slot_release_seq = 0
        e._orphan_seen = {}
        e.ORPHAN_SLOT_S = 0.0
        e.info = {"batch_release_ack": 1}
        return e

    def wait_free(self, e):
        with e.slot_cv:
            self.assertTrue(e.slot_cv.wait_for(lambda: not e.slot_busy[0], timeout=2))

    def test_stop_is_sent_after_atomic_ownership_transfer(self):
        e = self.engine()
        def send(line):
            self.assertEqual(line, "BSTOP 0")
            self.assertNotIn(0, e.slot_claims)
            self.assertIn(0, e.slot_releasing)
            e.slot_q[0].put("BDONE 0 1 cancel 1.0")
        e._send = send
        e._release_slot_when_done(0)
        self.wait_free(e)
        with e.slot_cv:
            e.slot_busy[0] = True
            e.slot_claims.add(0)
        self.assertEqual(e.slot_claims, {0}, "the old owner must not discard a new owner's claim")

    def test_old_completion_and_wrong_tag_do_not_free_orphan(self):
        e = self.engine()
        e.slot_claims.clear()
        sent = []
        e._send = sent.append
        e.slot_q[0].put("BDONE 0 999 stop 1.0")
        with e.slot_cv:
            self.assertEqual(e._reap_orphan_slots(), [0])
            self.assertTrue(e.slot_busy[0])
            self.assertEqual(e.slot_releasing, {0})
            self.assertEqual(e._reap_orphan_slots(), [])
        self.assertEqual(sent, ["BRELEASE 0 1"])
        e.slot_release_q[0].put("BRELEASED 0 999")
        self.assertTrue(e.slot_busy[0])
        e.slot_release_q[0].put("BRELEASED 0 1")
        self.wait_free(e)
        self.assertTrue(e.slot_q[0].empty(), "old tokens are drained only after the native fence")
        self.assertEqual(e.slot_held[0], [])

    def test_old_engine_eof_cannot_free_successor_slot(self):
        e = self.engine()
        e.slot_claims.clear()
        e._send = lambda line: None
        old_q, old_busy = e.slot_release_q[0], e.slot_busy
        with e.slot_cv:
            e._release_orphan_slot(0)
            e.slot_busy = [True]
            e.slot_claims = {0}
            e.slot_releasing = set()
            old_q.put(None)
            self.assertTrue(e.slot_cv.wait_for(lambda: not old_busy[0], timeout=2))
            self.assertTrue(e.slot_busy[0])
            self.assertEqual(e.slot_claims, {0})


if __name__ == "__main__":
    unittest.main()
