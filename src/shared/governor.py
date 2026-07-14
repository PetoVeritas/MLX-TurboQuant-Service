"""Shared memory governor for sibling MLX TurboQuant supervisors.

The governor is intentionally small: one JSON state file protected by an
advisory file lock. Supervisors reserve their estimated resident memory before
cold-loading a worker and release that reservation when the worker unloads.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG = logging.getLogger("mlx_turbo_gemma.shared.governor")
DEPLOYED_COMMIT_FILENAME = "DEPLOYED_COMMIT"
DRIFT_CHECK_SOURCE_OF_TRUTH = "governor.driftCheck.latestKnownGoodCommit"
DRIFT_CHECK_UPDATED_BY = "deployment operator or rollout automation after an approved lane deployment"
_LAST_DRIFT_WARNING_FINGERPRINT: str | None = None


@dataclass(frozen=True)
class GovernorLease:
    instance_id: str
    admitted: bool
    reason: str | None = None


class GovernorAdmissionError(RuntimeError):
    """Raised when the governor refuses a cold-load admission."""


def evaluate_lane_version_drift(config: dict[str, Any]) -> dict[str, Any]:
    """Read configured lane commit markers and return a non-fatal drift status."""

    governor_cfg = config.get("governor", {}) if isinstance(config.get("governor", {}), dict) else {}
    drift_cfg = governor_cfg.get("driftCheck", {}) if isinstance(governor_cfg.get("driftCheck", {}), dict) else {}
    enabled = bool(drift_cfg.get("enabled", False))
    payload: dict[str, Any] = {
        "enabled": enabled,
        "status": "disabled",
        "source_of_truth": DRIFT_CHECK_SOURCE_OF_TRUTH,
        "updated_by": DRIFT_CHECK_UPDATED_BY,
        "latest_known_good_commit": str(drift_cfg.get("latestKnownGoodCommit", "")).strip(),
        "lanes": [],
        "summary": {"current": 0, "stale": 0, "unknown": 0},
        "warnings": [],
    }
    if not enabled:
        return payload

    latest = payload["latest_known_good_commit"]
    lanes_cfg = drift_cfg.get("lanes", [])
    if not latest:
        payload["status"] = "unknown"
        payload["summary"]["unknown"] = len(lanes_cfg) if isinstance(lanes_cfg, list) else 0
        payload["warnings"].append("missing_latest_known_good_commit")
        return _finalize_lane_version_drift(payload)
    if not isinstance(lanes_cfg, list) or not lanes_cfg:
        payload["status"] = "unknown"
        payload["warnings"].append("no_lanes_configured")
        return _finalize_lane_version_drift(payload)

    for index, lane_cfg in enumerate(lanes_cfg):
        lane: dict[str, Any] = lane_cfg if isinstance(lane_cfg, dict) else {}
        lane_id = str(lane.get("instanceId") or lane.get("laneId") or lane.get("id") or f"lane-{index + 1}").strip()
        marker_path = _deployed_commit_path(lane)
        row: dict[str, Any] = {
            "lane_id": lane_id,
            "marker_path": str(marker_path) if marker_path is not None else None,
            "deployed_commit": None,
            "status": "unknown",
            "stale": True,
            "warning": None,
        }
        if marker_path is None:
            row["warning"] = "missing_deployed_commit_path"
        else:
            try:
                deployed = marker_path.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, IndexError) as exc:
                row["warning"] = f"deployed_commit_unreadable:{exc.__class__.__name__}"
            else:
                row["deployed_commit"] = deployed
                if deployed == latest:
                    row["status"] = "current"
                    row["stale"] = False
                else:
                    row["status"] = "stale"
                    row["warning"] = "deployed_commit_behind_latest_known_good"

        payload["lanes"].append(row)
        payload["summary"][row["status"]] += 1
        if row["warning"]:
            payload["warnings"].append(f"{lane_id}:{row['warning']}")

    payload["status"] = "current" if payload["summary"]["stale"] == 0 and payload["summary"]["unknown"] == 0 else "stale"
    return _finalize_lane_version_drift(payload)


def _finalize_lane_version_drift(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["status"] not in {"stale", "unknown"}:
        return payload
    _emit_lane_version_drift_warning(payload)
    return payload


def _emit_lane_version_drift_warning(payload: dict[str, Any]) -> None:
    global _LAST_DRIFT_WARNING_FINGERPRINT
    fingerprint = json.dumps(
        {
            "latest_known_good_commit": payload["latest_known_good_commit"],
            "status": payload["status"],
            "summary": payload["summary"],
            "warnings": payload["warnings"],
        },
        sort_keys=True,
    )
    if fingerprint == _LAST_DRIFT_WARNING_FINGERPRINT:
        return
    _LAST_DRIFT_WARNING_FINGERPRINT = fingerprint
    LOG.warning(
        "governor_lane_drift %s",
        json.dumps(
            {
                "status": payload["status"],
                "source_of_truth": payload["source_of_truth"],
                "updated_by": payload["updated_by"],
                "latest_known_good_commit": payload["latest_known_good_commit"],
                "summary": payload["summary"],
                "warnings": payload["warnings"],
            },
            sort_keys=True,
        ),
    )


def _deployed_commit_path(lane_cfg: dict[str, Any]) -> Path | None:
    explicit = str(lane_cfg.get("deployedCommitPath", "")).strip()
    if explicit:
        return Path(os.path.expanduser(explicit))
    deploy_dir = str(lane_cfg.get("deployDir", "")).strip()
    if deploy_dir:
        return Path(os.path.expanduser(deploy_dir)) / DEPLOYED_COMMIT_FILENAME
    return None


class MemoryGovernor:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("governor", {}) if isinstance(config.get("governor", {}), dict) else {}
        server_cfg = config.get("server", {}) if isinstance(config.get("server", {}), dict) else {}
        self.enabled = bool(cfg.get("enabled", False))
        configured_instance = str(cfg.get("instanceId", "")).strip()
        port = str(server_cfg.get("port", "unknown"))
        self.instance_id = configured_instance or f"mlx-{port}"
        self.priority = int(cfg.get("priority", 100) or 100)
        self.rss_estimate_gb = float(cfg.get("rssEstimateLoadedGb", 0.0) or 0.0)
        self.ceiling_gb = float(cfg.get("ceilingGb", 0.0) or 0.0)
        self.allow_lower_priority_to_preempt_higher = bool(cfg.get("allowLowerPriorityToPreemptHigher", False))
        self.stale_after_s = max(30, int(cfg.get("staleAfterSeconds", 900) or 900))
        state_dir = Path(os.path.expanduser(str(cfg.get("stateDir", "~/Library/Application Support/MLX-TurboQuant-Governor"))))
        self.state_dir = state_dir
        self.state_path = state_dir / "state.json"
        self.lock_path = state_dir / "state.lock"
        host = str(server_cfg.get("host", "127.0.0.1"))
        self.admin_base_url = str(cfg.get("adminBaseUrl") or f"http://{host}:{port}").rstrip("/")

    @contextlib.contextmanager
    def _locked_state(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield self._read_state(), lock_handle
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"version": 1, "instances": {}}
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and isinstance(data.get("instances"), dict):
                return data
        except Exception as exc:
            LOG.warning("governor_state_read_failed %s", exc)
        return {"version": 1, "instances": {}}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(self.state_path)

    def _prune_stale(self, state: dict[str, Any], now: float) -> list[str]:
        instances = state.setdefault("instances", {})
        removed: list[str] = []
        for instance_id, row in list(instances.items()):
            heartbeat = float(row.get("updatedAt", row.get("admittedAt", 0)) or 0)
            pid = row.get("pid")
            pid_alive = True
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, 0)
                except OSError:
                    pid_alive = False
            stale_without_live_pid = not isinstance(pid, int) and (now - heartbeat) > self.stale_after_s
            if stale_without_live_pid or not pid_alive:
                removed.append(str(instance_id))
                instances.pop(instance_id, None)
        return removed

    @staticmethod
    def _priority_allows_preemption(requester_priority: int, target_priority: int, allow_lower_to_higher: bool) -> bool:
        # Lower numeric priority is more protected. By default, a requester may
        # only preempt strictly lower-priority rows (for example, 26B priority 1
        # can preempt E4B priority 2, but not another priority-1 service).
        if requester_priority < target_priority:
            return True
        if requester_priority > target_priority:
            return allow_lower_to_higher
        return False

    def _preempt_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            url = str(row.get("adminBaseUrl", "")).rstrip("/")
            if not url:
                continue
            endpoint = f"{url}/admin/worker/unload"
            try:
                req = urllib.request.Request(endpoint, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    response.read()
                LOG.info("governor_preempted %s", json.dumps({"target": row.get("instanceId"), "url": endpoint}, sort_keys=True))
            except (OSError, urllib.error.URLError) as exc:
                raise GovernorAdmissionError(f"governor_preempt_failed:{row.get('instanceId')}:{exc}") from exc

    def admit(self, *, pid: int | None = None) -> GovernorLease:
        if not self.enabled:
            return GovernorLease(self.instance_id, admitted=True)
        if self.ceiling_gb <= 0 or self.rss_estimate_gb <= 0:
            raise GovernorAdmissionError("governor_invalid_config:ceilingGb and rssEstimateLoadedGb must be positive")

        now = time.time()
        with self._locked_state() as (state, _lock_handle):
            removed = self._prune_stale(state, now)
            instances = state.setdefault("instances", {})
            current = instances.get(self.instance_id)
            current_gb = sum(float(row.get("rssEstimateLoadedGb", 0.0) or 0.0) for row in instances.values())
            if current:
                current_gb -= float(current.get("rssEstimateLoadedGb", 0.0) or 0.0)
            needed_gb = current_gb + self.rss_estimate_gb
            preemptions: list[dict[str, Any]] = []
            if needed_gb > self.ceiling_gb:
                candidates = [
                    row
                    for row in instances.values()
                    if row.get("instanceId") != self.instance_id
                    and self._priority_allows_preemption(self.priority, int(row.get("priority", 100) or 100), self.allow_lower_priority_to_preempt_higher)
                ]
                candidates.sort(key=lambda row: (int(row.get("priority", 100) or 100), float(row.get("rssEstimateLoadedGb", 0.0) or 0.0)), reverse=True)
                projected = needed_gb
                for row in candidates:
                    preemptions.append(dict(row))
                    projected -= float(row.get("rssEstimateLoadedGb", 0.0) or 0.0)
                    if projected <= self.ceiling_gb:
                        break
                if projected > self.ceiling_gb:
                    reason = f"governor_refused:need={needed_gb:.1f}gb ceiling={self.ceiling_gb:.1f}gb"
                    self._write_state(state)
                    raise GovernorAdmissionError(reason)
            self._write_state(state)

        if preemptions:
            self._preempt_rows(preemptions)

        # Re-lock after preemptions and commit our own row.
        now = time.time()
        with self._locked_state() as (state, _lock_handle):
            removed = self._prune_stale(state, now)
            instances = state.setdefault("instances", {})
            current = instances.get(self.instance_id)
            current_gb = sum(float(row.get("rssEstimateLoadedGb", 0.0) or 0.0) for row in instances.values())
            if current:
                current_gb -= float(current.get("rssEstimateLoadedGb", 0.0) or 0.0)
            if current_gb + self.rss_estimate_gb > self.ceiling_gb:
                reason = f"governor_refused:need={current_gb + self.rss_estimate_gb:.1f}gb ceiling={self.ceiling_gb:.1f}gb"
                self._write_state(state)
                raise GovernorAdmissionError(reason)
            instances[self.instance_id] = {
                "instanceId": self.instance_id,
                "priority": self.priority,
                "rssEstimateLoadedGb": self.rss_estimate_gb,
                "adminBaseUrl": self.admin_base_url,
                "pid": pid or os.getpid(),
                "admittedAt": now,
                "updatedAt": now,
            }
            self._write_state(state)
        if removed:
            LOG.info("governor_pruned_stale %s", json.dumps({"removed": removed}, sort_keys=True))
        LOG.info("governor_admitted %s", json.dumps({"instance_id": self.instance_id, "priority": self.priority, "rss_gb": self.rss_estimate_gb}, sort_keys=True))
        return GovernorLease(self.instance_id, admitted=True)

    def release(self) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._locked_state() as (state, _lock_handle):
            self._prune_stale(state, now)
            instances = state.setdefault("instances", {})
            if instances.pop(self.instance_id, None) is not None:
                self._write_state(state)
                LOG.info("governor_released %s", json.dumps({"instance_id": self.instance_id}, sort_keys=True))
            else:
                self._write_state(state)
