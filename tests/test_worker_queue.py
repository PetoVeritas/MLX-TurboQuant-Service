from __future__ import annotations

import unittest

from supervisor.main import validate_chat_request
from supervisor.worker_manager import WorkerManager


def config(max_depth: int) -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": 4029},
        "model": {"id": "mlx-test-model"},
        "worker": {
            "stubMode": True,
            "queue": {"maxDepth": max_depth},
            "idleUnload": {"enabled": False},
        },
        "governor": {"enabled": False},
    }


class WorkerQueueTests(unittest.TestCase):
    def make_manager(self, max_depth: int) -> WorkerManager:
        manager = WorkerManager(config(max_depth))
        self.addCleanup(manager.shutdown)
        with manager._state_lock:
            manager._loaded = True
            manager._set_state("busy", accepting_requests=False, loaded=True)
        return manager

    def test_zero_depth_preserves_worker_busy_rejection(self):
        manager = self.make_manager(0)

        self.assertFalse(manager.can_accept_requests())
        self.assertEqual(manager.rejection_reason(), "worker_busy")
        self.assertEqual(manager.begin_request(), (False, "worker_busy"))
        self.assertEqual(manager.stats_payload()["metrics"]["queue_full_rejections"], 0)

    def test_one_deep_queue_accepts_and_completes(self):
        manager = self.make_manager(1)

        self.assertTrue(manager.can_accept_requests())
        self.assertIsNone(manager.rejection_reason())
        self.assertEqual(manager.begin_request(), (True, "queued"))

        stats = manager.stats_payload()
        self.assertEqual(stats["worker"]["queue_depth"], 1)
        self.assertEqual(stats["worker"]["queue_max_depth"], 1)
        self.assertEqual(stats["metrics"]["queued_requests"], 1)

        manager.complete_request(success=True)
        stats = manager.stats_payload()
        self.assertEqual(stats["worker"]["queue_depth"], 0)
        self.assertEqual(stats["worker"]["state"], "ready")

    def test_one_deep_queue_rejects_when_full(self):
        manager = self.make_manager(1)
        with manager._state_lock:
            manager._queued_requests = 1

        self.assertFalse(manager.can_accept_requests())
        self.assertEqual(manager.rejection_reason(), "queue_full")
        self.assertEqual(manager.begin_request(), (False, "queue_full"))
        self.assertEqual(manager.stats_payload()["metrics"]["queue_full_rejections"], 1)

    def test_validation_preserves_queue_full_rejection(self):
        manager = self.make_manager(1)
        with manager._state_lock:
            manager._queued_requests = 1
        app = type("FakeApp", (), {"worker": manager})()
        payload = {
            "model": manager.model_id,
            "messages": [{"role": "user", "content": "hello"}],
        }

        self.assertEqual(
            validate_chat_request(payload, app),
            (409, "queue_full", "Worker queue is full (1/1)"),
        )


if __name__ == "__main__":
    unittest.main()
