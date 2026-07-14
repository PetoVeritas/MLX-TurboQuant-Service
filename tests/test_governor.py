from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from shared.governor import GovernorAdmissionError, MemoryGovernor, evaluate_lane_version_drift


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

    def test_lane_version_drift_flags_stale_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane_dir = Path(tmp, "lane-a")
            lane_dir.mkdir()
            marker = lane_dir / "DEPLOYED_COMMIT"
            marker.write_text("oldhash\n", encoding="utf-8")
            with self.assertLogs("mlx_turbo_gemma.shared.governor", level="WARNING") as logs:
                status = evaluate_lane_version_drift(
                    {
                        "governor": {
                            "driftCheck": {
                                "enabled": True,
                                "latestKnownGoodCommit": "newhash",
                                "lanes": [{"instanceId": "lane-a", "deployDir": str(lane_dir)}],
                            }
                        }
                    }
                )
            self.assertEqual(status["status"], "stale")
            self.assertEqual(status["summary"], {"current": 0, "stale": 1, "unknown": 0})
            self.assertTrue(status["lanes"][0]["stale"])
            self.assertEqual(status["lanes"][0]["warning"], "deployed_commit_behind_latest_known_good")
            self.assertEqual(status["source_of_truth"], "governor.driftCheck.latestKnownGoodCommit")
            self.assertIn("governor_lane_drift", logs.output[0])
            self.assertIn("lane-a:deployed_commit_behind_latest_known_good", logs.output[0])

    def test_lane_version_drift_missing_marker_is_unknown_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane_dir = Path(tmp, "lane-missing")
            lane_dir.mkdir()
            with self.assertLogs("mlx_turbo_gemma.shared.governor", level="WARNING") as logs:
                status = evaluate_lane_version_drift(
                    {
                        "governor": {
                            "driftCheck": {
                                "enabled": True,
                                "latestKnownGoodCommit": "newhash",
                                "lanes": [{"laneId": "lane-missing", "deployDir": str(lane_dir)}],
                            }
                        }
                    }
                )
            self.assertEqual(status["status"], "stale")
            self.assertEqual(status["summary"], {"current": 0, "stale": 0, "unknown": 1})
            self.assertTrue(status["lanes"][0]["stale"])
            self.assertEqual(status["lanes"][0]["status"], "unknown")
            self.assertIn("lane-missing:deployed_commit_unreadable", status["warnings"][0])
            self.assertIn("governor_lane_drift", logs.output[0])
            self.assertIn("lane-missing:deployed_commit_unreadable", logs.output[0])

    def test_lane_version_drift_all_current_is_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane_a = Path(tmp, "lane-a", "DEPLOYED_COMMIT")
            lane_b = Path(tmp, "lane-b", "DEPLOYED_COMMIT")
            lane_a.parent.mkdir()
            lane_b.parent.mkdir()
            lane_a.write_text("goodhash\n", encoding="utf-8")
            lane_b.write_text("goodhash\n", encoding="utf-8")
            with self.assertNoLogs("mlx_turbo_gemma.shared.governor", level="WARNING"):
                status = evaluate_lane_version_drift(
                    {
                        "governor": {
                            "driftCheck": {
                                "enabled": True,
                                "latestKnownGoodCommit": "goodhash",
                                "lanes": [
                                    {"instanceId": "lane-a", "deployedCommitPath": str(lane_a)},
                                    {"instanceId": "lane-b", "deployedCommitPath": str(lane_b)},
                                ],
                            }
                        }
                    }
                )
            self.assertEqual(status["status"], "current")
            self.assertEqual(status["summary"], {"current": 2, "stale": 0, "unknown": 0})
            self.assertEqual(status["warnings"], [])
            self.assertFalse(any(lane["stale"] for lane in status["lanes"]))


if __name__ == "__main__":
    unittest.main()
