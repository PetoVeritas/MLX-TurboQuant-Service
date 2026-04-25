from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from shared.governor import GovernorAdmissionError, MemoryGovernor


def cfg(state_dir: str, *, instance: str, priority: int, rss: float, ceiling: float, allow_lower: bool = False):
    return {
        "server": {"host": "127.0.0.1", "port": 4000},
        "governor": {
            "enabled": True,
            "instanceId": instance,
            "priority": priority,
            "rssEstimateLoadedGb": rss,
            "stateDir": state_dir,
            "ceilingGb": ceiling,
            "allowLowerPriorityToPreemptHigher": allow_lower,
            "staleAfterSeconds": 30,
        },
    }


class GovernorTests(unittest.TestCase):
    def test_admit_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            gov = MemoryGovernor(cfg(tmp, instance="a", priority=1, rss=10, ceiling=12))
            lease = gov.admit(pid=None)
            self.assertTrue(lease.admitted)
            self.assertIn('"a"', Path(tmp, "state.json").read_text())
            gov.release()
            self.assertNotIn('"a"', Path(tmp, "state.json").read_text())

    def test_refuses_when_over_ceiling_without_preemptable_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            MemoryGovernor(cfg(tmp, instance="a", priority=1, rss=10, ceiling=12)).admit(pid=None)
            with self.assertRaises(GovernorAdmissionError):
                MemoryGovernor(cfg(tmp, instance="b", priority=2, rss=5, ceiling=12)).admit(pid=None)

    def test_prunes_stale_rows_without_live_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp, "state.json")
            state_path.write_text(
                '{"version":1,"instances":{"old":{"instanceId":"old","priority":1,"rssEstimateLoadedGb":10,"updatedAt":1}}}\n'
            )
            gov = MemoryGovernor(cfg(tmp, instance="new", priority=1, rss=5, ceiling=8))
            gov.admit(pid=None)
            state = state_path.read_text()
            self.assertIn('"new"', state)
            self.assertNotIn('"old"', state)

    def test_does_not_prune_stale_row_with_live_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp, "state.json")
            state_path.write_text(
                '{"version":1,"instances":{"live":{"instanceId":"live","priority":1,"rssEstimateLoadedGb":10,"pid":%d,"updatedAt":1}}}\n'
                % os.getpid()
            )
            with self.assertRaises(GovernorAdmissionError):
                MemoryGovernor(cfg(tmp, instance="new", priority=2, rss=5, ceiling=12)).admit(pid=None)
            state = state_path.read_text()
            self.assertIn('"live"', state)


if __name__ == "__main__":
    unittest.main()
