"""Supervisor-side worker manager with a real subprocess boundary."""

from __future__ import annotations

import itertools
import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG = logging.getLogger("mlx_turbo_gemma.supervisor.worker_manager")

FAILURE_STARTUP = "startup_failure"
FAILURE_TIMEOUT = "timeout"
FAILURE_CRASH = "worker_crash"
FAILURE_PROTOCOL = "protocol_error"
FAILURE_BACKEND = "backend_error"
FAILURE_UNLOAD = "unload_failure"
FAILURE_CONFIG = "invalid_config"
FAILURE_COOLDOWN = "cooldown_active"


@dataclass
class CompletionResult:
    content: str
    finish_reason: str
    usage: dict[str, int]
    metrics: dict[str, int | None]
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class CompletionChunk:
    content: str


class WorkerManager:
    # How often the background idle-unload thread wakes up to check.
    _IDLE_CHECK_INTERVAL_SECONDS = 5

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        # Two-lock split:
        #   _state_lock: cheap metadata (metrics, state, loaded flag, timestamps).
        #     Held only for very short critical sections so readers like /ready,
        #     /admin/stats, /v1/models never block on a long generation.
        #   _worker_lock: serializes actual worker I/O (_send + _read_message
        #     pairs, spawn, shutdown). Held across the full duration of a
        #     streaming generation, but the state_lock is released before the
        #     first yield, so readers remain responsive.
        self._state_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._metrics = {
            "successful_requests": 0,
            "failed_requests": 0,
            "worker_starts": 0,
            "worker_restarts": 0,
            "worker_unloads": 0,
            "oom_events": 0,
            "forced_kills": 0,
            "startup_timeouts": 0,
            "request_timeouts": 0,
            "probe_timeouts": 0,
            "shutdown_timeouts": 0,
        }
        self._loaded = False
        self._accepting_requests = False
        self._cold_load_expected = False
        self._state = "not_loaded"
        self._last_error: str | None = None
        self._last_failure_kind: str | None = None
        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        self._process: subprocess.Popen[str] | None = None
        self._stderr_handle: Any | None = None
        self._root_dir = Path(__file__).resolve().parents[2]
        self._request_counter = itertools.count(1)
        self._worker_start_timeout = max(1, int(self._config.get("worker", {}).get("startupTimeoutMs", 120000)) // 1000)
        self._worker_request_timeout = max(1, int(self._config.get("worker", {}).get("requestTimeoutMs", 180000)) // 1000)
        self._worker_probe_timeout = max(1, int(self._config.get("worker", {}).get("probeTimeoutMs", 1000) or 1000) // 1000)
        recycle_cfg = self._config.get("worker", {}).get("recycle", {})
        self._max_consecutive_errors = max(1, int(recycle_cfg.get("maxConsecutiveErrors", 3) or 3))
        self._cooldown_seconds = max(1, int(recycle_cfg.get("cooldownMs", 15000) or 15000) // 1000)
        idle_cfg = self._config.get("worker", {}).get("idleUnload", {})
        self._idle_unload_enabled = bool(idle_cfg.get("enabled", True))
        self._idle_unload_seconds = max(1, int(idle_cfg.get("idleMs", 300000) or 300000) // 1000)
        self._last_activity_at = time.time()
        self._shutdown_event = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._validate_initial_state()
        self._start_idle_thread()

    def _set_state(self, state: str, *, accepting_requests: bool | None = None, loaded: bool | None = None, error: str | None = None) -> None:
        previous_state = self._state
        self._state = state
        if accepting_requests is not None:
            self._accepting_requests = accepting_requests
        if loaded is not None:
            self._loaded = loaded
        if error is not None:
            self._last_error = error
        if previous_state != state:
            LOG.info(
                "worker_state_transition %s",
                json.dumps(
                    {
                        "from": previous_state,
                        "to": state,
                        "accepting_requests": self._accepting_requests,
                        "loaded": self._loaded,
                        "pid": self._worker_pid(),
                        "last_error": self._last_error,
                        "last_failure_kind": self._last_failure_kind,
                        "consecutive_failures": self._consecutive_failures,
                        "cooldown_remaining_s": self._cooldown_remaining_seconds_locked(),
                    },
                    sort_keys=True,
                ),
            )

    def _cooldown_remaining_seconds_locked(self) -> int:
        remaining = int(self._cooldown_until - time.time())
        return max(0, remaining)

    def _cooldown_active_locked(self) -> bool:
        return self._cooldown_remaining_seconds_locked() > 0

    def _clear_cooldown_locked(self) -> None:
        self._cooldown_until = 0.0

    def _record_success_locked(self) -> None:
        self._consecutive_failures = 0
        self._last_failure_kind = None
        self._clear_cooldown_locked()
        self._last_activity_at = time.time()

    def _activate_cooldown_locked(self, *, error: str) -> None:
        self._cooldown_until = time.time() + self._cooldown_seconds
        self._set_state("failed", accepting_requests=False, loaded=False, error=error)

    def _record_failure_locked(self, failure_kind: str, error: str, *, loaded: bool | None = None, crash: bool = False) -> None:
        # Failure taxonomy (see audit: wedge bug fix):
        #   * Terminal (CONFIG/UNLOAD): the lane cannot recover without admin
        #     intervention — configuration is broken or a clean shutdown
        #     failed in a way that leaves the process-lifecycle model unsafe.
        #     Stay "failed" with admission closed.
        #   * Recoverable-with-respawn (TIMEOUT/CRASH/PROTOCOL/STARTUP, and
        #     any explicit ``crash=True``): the worker process is gone or
        #     cannot be trusted. Previously these transitioned to "failed"
        #     with ``accepting_requests=False`` and wedged the lane: a
        #     single hang turned into permanent 503s because
        #     ``_maybe_release_cooldown_locked`` only runs when cooldown is
        #     active, and cooldown only activates at
        #     ``_max_consecutive_errors``. Now we tear down any residual
        #     process handle and call ``_set_not_loaded_state`` so admission
        #     reopens (under lazy-load) and the next request triggers a
        #     clean cold reload.
        #   * Soft / worker-reported (BACKEND, plus any unclassified kind):
        #     the worker is probably still healthy for the next request.
        #     Mark "degraded" but keep admitting so repeated failures
        #     naturally escalate to cooldown.
        self._consecutive_failures += 1
        self._last_failure_kind = failure_kind
        if self._consecutive_failures >= self._max_consecutive_errors:
            self._activate_cooldown_locked(error=error)
            return
        if failure_kind in {FAILURE_CONFIG, FAILURE_UNLOAD}:
            self._set_state("failed", accepting_requests=False, loaded=loaded, error=error)
            return
        if crash or failure_kind in {FAILURE_STARTUP, FAILURE_TIMEOUT, FAILURE_PROTOCOL, FAILURE_CRASH}:
            if self._process is not None:
                try:
                    self._terminate_process_locked(force=True)
                except Exception:
                    pass
            self._last_error = error
            self._set_not_loaded_state(error=error)
            return
        self._set_state("degraded", accepting_requests=True, loaded=loaded, error=error)

    def _model_config_valid(self) -> bool:
        worker_cfg = self._config.get("worker", {})
        stub_mode = bool(worker_cfg.get("stubMode", False))
        model_cfg = self._config.get("model", {})
        model_path = str(model_cfg.get("path", "")).strip()
        model_id = str(model_cfg.get("id", "")).strip()
        return bool(model_id) and (stub_mode or bool(model_path))

    def _lazy_load_enabled(self) -> bool:
        return bool(self._config.get("worker", {}).get("lazyLoad", True))

    def _set_not_loaded_state(self, *, error: str | None = None) -> None:
        valid_model = self._model_config_valid()
        lazy_load = self._lazy_load_enabled()
        self._cold_load_expected = lazy_load and valid_model
        if self._cooldown_active_locked():
            self._set_state("failed", accepting_requests=False, loaded=False, error=error or self._last_error)
            return
        if valid_model:
            self._set_state("not_loaded", accepting_requests=lazy_load, loaded=False, error=error)
            return
        self._last_failure_kind = FAILURE_CONFIG
        self._set_state(
            "failed",
            accepting_requests=False,
            loaded=False,
            error="Model configuration is not valid yet. Set model.path or enable worker.stubMode.",
        )

    def _validate_initial_state(self) -> None:
        if self._model_config_valid() and self._lazy_load_enabled():
            self._set_not_loaded_state(error=None)
        elif self._model_config_valid():
            self._cold_load_expected = False
            self._set_state("starting", accepting_requests=False, loaded=False, error=None)
        else:
            self._cold_load_expected = False
            self._last_failure_kind = FAILURE_CONFIG
            self._set_state(
                "failed",
                accepting_requests=False,
                loaded=False,
                error="Model configuration is not valid yet. Set model.path or enable worker.stubMode.",
            )

    @property
    def model_id(self) -> str:
        return str(self._config.get("model", {}).get("id", ""))

    def _worker_pid(self) -> int | None:
        if self._process and self._process.poll() is None:
            return self._process.pid
        return None

    def _close_stderr_handle(self) -> None:
        if self._stderr_handle:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None

    def _reset_to_not_loaded(self) -> None:
        self._process = None
        self._close_stderr_handle()
        self._set_not_loaded_state(error=None)

    def _mark_failed(self, error: str, *, failure_kind: str = FAILURE_BACKEND) -> None:
        self._record_failure_locked(failure_kind, error, loaded=False)

    def _mark_degraded(self, error: str, *, failure_kind: str = FAILURE_BACKEND, loaded: bool | None = None) -> None:
        self._record_failure_locked(failure_kind, error, loaded=loaded)

    def _maybe_release_cooldown_locked(self) -> None:
        if self._cooldown_until and not self._cooldown_active_locked():
            self._clear_cooldown_locked()
            self._last_error = None
            self._last_failure_kind = None
            if self._model_config_valid() and not self._loaded:
                self._set_not_loaded_state(error=None)

    def _idle_seconds_locked(self) -> int:
        return max(0, int(time.time() - self._last_activity_at))

    def _idle_unload_due_locked(self) -> bool:
        """Cheap read-only check: should an idle-unload happen right now?

        Must be called under _state_lock. Never performs I/O or state changes.
        """

        if not self._idle_unload_enabled:
            return False
        if not self._loaded or not self._process or self._process.poll() is not None:
            return False
        if self._state != "ready":
            return False
        if self._cooldown_active_locked():
            return False
        return self._idle_seconds_locked() >= self._idle_unload_seconds

    def _detect_crash_locked(self) -> None:
        """State-lock-only crash detection. No unloads, no I/O, no blocking.

        Called from reader paths (/ready, /admin/stats, can_accept_requests,
        rejection_reason, begin_request). The previous _ensure_process_state
        also triggered idle-unload here, which made readers block for seconds
        and could kill the worker itself — see audit C2. That responsibility
        now lives in _idle_unload_thread_loop.
        """

        self._maybe_release_cooldown_locked()
        if self._process and self._process.poll() is not None:
            code = self._process.returncode
            self._process = None
            self._close_stderr_handle()
            self._record_failure_locked(
                FAILURE_CRASH,
                f"Worker exited unexpectedly with code {code}",
                loaded=False,
                crash=True,
            )

    # Kept as an alias for internal call sites that want cheap crash detection.
    _ensure_process_state = _detect_crash_locked

    def _probe_worker_locked(self) -> tuple[bool, str | None]:
        if not self._process or self._process.poll() is not None:
            return False, "worker_process_not_running"
        if not self._loaded:
            return False, "worker_not_loaded"
        try:
            self._send({"command": "ping"})
            response = self._read_message(timeout_seconds=self._worker_probe_timeout, timeout_metric="probe_timeouts")
        except TimeoutError as exc:
            self._terminate_process_locked(force=True)
            return False, f"worker_probe_timeout:{exc}"
        except Exception as exc:
            self._terminate_process_locked(force=True)
            return False, f"worker_probe_failed:{exc}"
        if response.get("type") != "pong":
            self._terminate_process_locked(force=True)
            return False, f"worker_probe_unexpected_response:{response}"
        return True, None

    def can_accept_requests(self) -> bool:
        with self._state_lock:
            self._ensure_process_state()
            return self._accepting_requests and not self._cooldown_active_locked()

    def rejection_reason(self) -> str | None:
        with self._state_lock:
            self._ensure_process_state()
            if self._cooldown_active_locked():
                return FAILURE_COOLDOWN
            if self._state == "busy":
                return "worker_busy"
            if not self._accepting_requests:
                return "worker_not_ready"
            return None

    def ready_payload(self) -> dict[str, Any]:
        with self._state_lock:
            self._ensure_process_state()
            actively_ready = self._loaded and self._accepting_requests and self._state == "ready" and not self._cooldown_active_locked()
            cold_load_acceptable = bool(self._accepting_requests and not self._loaded and self._cold_load_expected and not self._cooldown_active_locked())
            return {
                "ok": actively_ready,
                "service": "mlx-turbo-gemma-service",
                "accepting_requests": self._accepting_requests and not self._cooldown_active_locked(),
                "actively_ready": actively_ready,
                "cold_load_expected": self._cold_load_expected,
                "cold_load_acceptable": cold_load_acceptable,
                "worker": {
                    "state": self._state,
                    "loaded": self._loaded,
                    "model_id": self.model_id,
                    "pid": self._worker_pid(),
                    "last_error": self._last_error,
                    "last_failure_kind": self._last_failure_kind,
                    "cooldown_remaining_s": self._cooldown_remaining_seconds_locked(),
                    "consecutive_failures": self._consecutive_failures,
                    "stub_mode": bool(self._config.get("worker", {}).get("stubMode", False)),
                    "idle_unload_enabled": self._idle_unload_enabled,
                    "idle_unload_threshold_s": self._idle_unload_seconds,
                    "idle_seconds": self._idle_seconds_locked(),
                },
            }

    def models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": self.model_id, "object": "model", "owned_by": "local"}],
        }

    def stats_payload(self) -> dict[str, Any]:
        with self._state_lock:
            self._ensure_process_state()
            return {
                "worker": {
                    "state": self._state,
                    "loaded": self._loaded,
                    "model_id": self.model_id,
                    "pid": self._worker_pid(),
                    "last_error": self._last_error,
                    "last_failure_kind": self._last_failure_kind,
                    "cooldown_remaining_s": self._cooldown_remaining_seconds_locked(),
                    "consecutive_failures": self._consecutive_failures,
                    "stub_mode": bool(self._config.get("worker", {}).get("stubMode", False)),
                },
                "metrics": dict(self._metrics),
                "config": {
                    "startup_timeout_s": self._worker_start_timeout,
                    "request_timeout_s": self._worker_request_timeout,
                    "probe_timeout_s": self._worker_probe_timeout,
                    "lazy_load": bool(self._config.get("worker", {}).get("lazyLoad", True)),
                    "cooldown_s": self._cooldown_seconds,
                    "max_consecutive_errors": self._max_consecutive_errors,
                    "idle_unload_enabled": self._idle_unload_enabled,
                    "idle_unload_s": self._idle_unload_seconds,
                },
            }

    def begin_request(self) -> tuple[bool, str | None]:
        with self._state_lock:
            self._ensure_process_state()
            if self._cooldown_active_locked():
                return False, FAILURE_COOLDOWN
            if not self._accepting_requests:
                return False, "worker_not_ready"
            if self._state == "busy":
                return False, "worker_busy"
            if not self._loaded and self._cold_load_expected:
                self._set_state("starting", accepting_requests=False)
            else:
                self._set_state("busy", accepting_requests=False)
            return True, None

    def _spawn_worker_if_needed(self) -> None:
        """Spawn the worker subprocess if it is not already running.

        Caller must hold _worker_lock. We enter _state_lock only for the
        short metadata transitions so that readers (/ready, /admin/stats)
        stay responsive even while the worker is coming up.
        """

        with self._state_lock:
            self._detect_crash_locked()
            if self._process and self._process.poll() is None:
                return
            self._set_state("starting", accepting_requests=False, loaded=False)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self._root_dir / "src")
            stderr_path = self._root_dir / "tmp" / "worker-stderr.log"
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_handle = stderr_path.open("a", encoding="utf-8")
            configured_python = str(self._config.get("worker", {}).get("pythonExecutable", "")).strip()
            worker_python = os.path.expanduser(configured_python) if configured_python else sys.executable
            # NOTE: binary mode (no text=True, no bufsize=1) is deliberate.
            # Text-mode Popen with line-buffering raced on back-to-back
            # streaming requests — worker's next stdin read would block even
            # though supervisor's write+flush returned cleanly. Binary I/O
            # bypasses TextIOWrapper entirely on both ends, keeping the JSON+
            # newline framing intact. See _send / _read_message and
            # worker/main.py for the matching binary access pattern.
            self._process = subprocess.Popen(
                [worker_python, "-m", "worker.main"],
                cwd=str(self._root_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_handle,
                env=env,
            )
        # Block for the bootstrap message OUTSIDE _state_lock so readers stay
        # responsive even during a slow cold-load.
        try:
            started = self._read_message(timeout_seconds=self._worker_start_timeout, timeout_metric="startup_timeouts")
        except TimeoutError as exc:
            with self._state_lock:
                self._terminate_process_locked(force=True)
            raise RuntimeError(f"{FAILURE_STARTUP}:Worker startup timed out after {self._worker_start_timeout}s") from exc
        if started.get("type") != "worker_started":
            with self._state_lock:
                self._terminate_process_locked(force=True)
            raise RuntimeError(f"{FAILURE_PROTOCOL}:Unexpected worker bootstrap message: {started}")
        with self._state_lock:
            self._loaded = True
            self._last_error = None
            self._last_activity_at = time.time()
            self._metrics["worker_starts"] += 1
            pid = self._worker_pid()
            start_count = self._metrics["worker_starts"]
            restart_count = self._metrics["worker_restarts"]
        LOG.info(
            "worker_started %s",
            json.dumps(
                {
                    "pid": pid,
                    "backend": started.get("backend"),
                    "start_count": start_count,
                    "restart_count": restart_count,
                },
                sort_keys=True,
            ),
        )

    def _read_message(self, timeout_seconds: int, *, timeout_metric: str | None = None) -> dict[str, Any]:
        # Snapshot the process pointer so we're not racing against admin
        # preempt/unload while waiting on select.
        proc = self._process
        stdout = proc.stdout if proc is not None else None
        if proc is None or stdout is None:
            raise RuntimeError("Worker stdout is not available")
        ready, _, _ = select.select([stdout], [], [], timeout_seconds)
        if not ready:
            if timeout_metric is not None:
                with self._state_lock:
                    self._metrics[timeout_metric] += 1
            raise TimeoutError(f"Timed out waiting for worker response after {timeout_seconds}s")
        line = stdout.readline()
        if not line:
            raise RuntimeError("Worker exited unexpectedly before sending a response")
        return json.loads(line)

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._process
        stdin = proc.stdin if proc is not None else None
        if proc is None or stdin is None:
            raise RuntimeError("Worker stdin is not available")
        stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        stdin.flush()

    def _terminate_process_locked(self, force: bool = False) -> None:
        if not self._process:
            self._close_stderr_handle()
            return
        proc = self._process
        prior_pid = proc.pid
        try:
            if proc.poll() is None:
                if force:
                    proc.kill()
                    self._metrics["forced_kills"] += 1
                else:
                    proc.terminate()
                    proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                self._metrics["forced_kills"] += 1
            except Exception:
                pass
        finally:
            self._process = None
            self._loaded = False
            self._close_stderr_handle()
            LOG.info("worker_terminated %s", json.dumps({"pid": prior_pid, "force": force}, sort_keys=True))

    def _preempt_worker_for_admin(self) -> None:
        """Kill the worker process directly (SIGKILL) so any in-flight generation
        holding _worker_lock fails fast and releases the lock. This lets admin
        unload/restart recover a wedged worker without waiting for a potentially
        long generation to finish naturally.

        We also wait briefly for the process to be reaped before returning so
        that callers which inspect ``proc.poll()`` immediately after (like
        ``unload_worker``'s already-dead fast path) see the expected non-None
        return code instead of racing the kernel.
        """

        proc: subprocess.Popen[str] | None = None
        with self._state_lock:
            if self._process and self._process.poll() is None:
                proc = self._process
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            return
        # Reap so poll() returns non-None for subsequent checks.
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        with self._state_lock:
            self._metrics["forced_kills"] += 1

    def unload_worker(self) -> dict[str, Any]:
        # Preempt any in-flight generation so this call doesn't block behind a
        # long SSE stream. Reading the pid happens under _state_lock only; the
        # actual kill happens outside so we never block readers.
        with self._state_lock:
            preempt_candidate_pid = self._worker_pid()
        self._preempt_worker_for_admin()
        with self._worker_lock:
            with self._state_lock:
                if not self._process or self._process.poll() is not None:
                    # Fast path after preempt: the worker is already dead.
                    # Count this as a genuine unload if the caller saw a live
                    # worker when they made the request.
                    if preempt_candidate_pid is not None:
                        self._metrics["worker_unloads"] += 1
                        LOG.info(
                            "worker_unloaded %s",
                            json.dumps(
                                {"pid": preempt_candidate_pid, "reason": "admin_preempt"},
                                sort_keys=True,
                            ),
                        )
                    self._reset_to_not_loaded()
                    return {"ok": True, "action": "unload", "state": self._state, "pid": None}
                prior_pid = self._worker_pid()
                self._set_state("unloading", accepting_requests=False)
            try:
                self._send({"command": "shutdown"})
                self._read_message(timeout_seconds=2, timeout_metric="shutdown_timeouts")
                with self._state_lock:
                    if self._process is not None:
                        self._process.wait(timeout=2)
                    self._metrics["worker_unloads"] += 1
                    LOG.info("worker_unloaded %s", json.dumps({"pid": prior_pid}, sort_keys=True))
                    self._reset_to_not_loaded()
                    return {"ok": True, "action": "unload", "state": self._state, "pid": None}
            except Exception as exc:
                with self._state_lock:
                    self._terminate_process_locked(force=True)
                    self._mark_failed(f"worker_unload_failed:{exc}", failure_kind=FAILURE_UNLOAD)
                raise RuntimeError(f"worker_unload_failed:{exc}") from exc

    def restart_worker(self) -> dict[str, Any]:
        # Same preemption rationale as unload_worker: SIGKILL any live worker
        # first so we don't wait on an in-flight generation. After preempt the
        # worker is already reaped, so there's no need to run the full
        # unload_worker path (which would try to talk to a dead process).
        # Capture old_pid BEFORE preempt so we can still report it in the
        # response and bump the worker_restarts metric.
        with self._state_lock:
            old_pid = self._worker_pid()
        self._preempt_worker_for_admin()
        with self._worker_lock:
            with self._state_lock:
                # Reset any residual process state from the preempt / prior
                # failure so the new spawn starts from a clean slate.
                if self._process is not None:
                    self._terminate_process_locked(force=False)
                self._clear_cooldown_locked()
                self._consecutive_failures = 0
                self._last_error = None
                self._last_failure_kind = None
            started_at = time.perf_counter()
            try:
                self._spawn_worker_if_needed()
                probe_ok, probe_error = self._probe_worker_locked()
            except Exception as exc:
                with self._state_lock:
                    self._record_failure_locked(FAILURE_PROTOCOL, str(exc), loaded=False, crash=True)
                raise
            if not probe_ok:
                with self._state_lock:
                    self._record_failure_locked(
                        FAILURE_PROTOCOL,
                        probe_error or "worker_probe_failed",
                        loaded=False,
                        crash=True,
                    )
                raise RuntimeError(probe_error or "worker_probe_failed")
            with self._state_lock:
                if old_pid is not None:
                    self._metrics["worker_restarts"] += 1
                # Re-clear failure bookkeeping in case probe/unload paths
                # briefly set it during the restart sequence.
                self._consecutive_failures = 0
                self._last_failure_kind = None
                self._set_state("ready", accepting_requests=True, loaded=True, error=None)
                new_pid = self._worker_pid()
            return {
                "ok": True,
                "action": "restart",
                "old_pid": old_pid,
                "new_pid": new_pid,
                "startup_ms": int((time.perf_counter() - started_at) * 1000),
            }

    def complete_request(self, success: bool, error: str | None = None, failure_kind: str | None = None) -> None:
        with self._state_lock:
            if success:
                self._metrics["successful_requests"] += 1
                self._record_success_locked()
                self._set_state("ready", accepting_requests=True, loaded=True, error=None)
            else:
                self._metrics["failed_requests"] += 1
                resolved_kind = failure_kind or self._last_failure_kind or FAILURE_BACKEND
                resolved_error = error or self._last_error or "worker_failed"
                if self._state == "failed" and self._cooldown_active_locked():
                    self._last_failure_kind = resolved_kind
                    self._last_error = resolved_error
                    return
                self._record_failure_locked(resolved_kind, resolved_error, loaded=self._loaded)

    def _begin_worker_request_locked(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool,
    ) -> str:
        """Spawn worker if needed, transition to 'busy', and send the command.

        Must be called holding _worker_lock. State transitions happen under
        _state_lock in short critical sections so readers (/ready, /admin/stats)
        stay responsive.
        """

        self._spawn_worker_if_needed()
        with self._state_lock:
            if self._state != "busy":
                self._set_state("busy", accepting_requests=False, loaded=True, error=None)
            request_id = f"req_{next(self._request_counter)}_{uuid.uuid4().hex[:8]}"
            pid = self._worker_pid()
            state_snapshot = self._state
        LOG.info(
            "worker_request_start %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "pid": pid,
                    "state": state_snapshot,
                    "message_count": len(messages),
                    "max_tokens": max_tokens,
                    "tool_count": len(tools or []),
                    "stream": stream,
                },
                sort_keys=True,
            ),
        )
        self._send(
            {
                "command": "generate_stream" if stream else "generate",
                "request_id": request_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "tools": tools,
            }
        )
        return request_id

    def generate_completion(self, messages: list[dict[str, str]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> CompletionResult:
        # Acquire _worker_lock (serializes against other generations, admin
        # unload/restart, and idle-unload) but NOT _state_lock across the full
        # request. Readers like /ready, /admin/stats, /v1/models only need
        # _state_lock and so remain responsive during generation.
        with self._worker_lock:
            try:
                self._begin_worker_request_locked(messages, max_tokens, tools, stream=False)
                message = self._read_message(timeout_seconds=self._worker_request_timeout, timeout_metric="request_timeouts")
            except TimeoutError:
                with self._state_lock:
                    self._terminate_process_locked(force=True)
                raise
            except RuntimeError:
                raise
            except Exception as exc:
                with self._state_lock:
                    self._terminate_process_locked(force=True)
                raise RuntimeError(f"{FAILURE_CRASH}:Worker request crashed:{exc}") from exc

        if message.get("type") == "error":
            raise RuntimeError(f"{FAILURE_BACKEND}:{str(message.get('error', 'worker_error'))}")
        if message.get("type") != "completion_result":
            raise RuntimeError(f"{FAILURE_PROTOCOL}:Unexpected worker response: {message}")

        with self._state_lock:
            pid = self._worker_pid()
        LOG.info(
            "worker_request_complete %s",
            json.dumps(
                {
                    "request_id": message.get("request_id"),
                    "pid": pid,
                    "usage": message.get("usage", {}),
                    "metrics": message.get("metrics", {}),
                    "finish_reason": message.get("finish_reason", "stop"),
                },
                sort_keys=True,
            ),
        )

        return CompletionResult(
            content=str(message["content"]),
            finish_reason=str(message.get("finish_reason", "stop")),
            usage=dict(message.get("usage", {})),
            metrics=dict(message.get("metrics", {})),
            tool_calls=message.get("tool_calls"),
        )

    def generate_completion_stream(self, messages: list[dict[str, str]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None):
        # Note: _worker_lock is held across the full generation (including
        # yields) so concurrent worker I/O is serialized, but _state_lock is
        # released between each read, so readers stay responsive.
        with self._worker_lock:
            try:
                self._begin_worker_request_locked(messages, max_tokens, tools, stream=True)
                while True:
                    message = self._read_message(timeout_seconds=self._worker_request_timeout, timeout_metric="request_timeouts")
                    if message.get("type") == "completion_chunk":
                        yield CompletionChunk(content=str(message.get("content", "")))
                        continue
                    if message.get("type") == "error":
                        raise RuntimeError(f"{FAILURE_BACKEND}:{str(message.get('error', 'worker_error'))}")
                    if message.get("type") != "completion_result":
                        raise RuntimeError(f"{FAILURE_PROTOCOL}:Unexpected worker response: {message}")

                    with self._state_lock:
                        pid = self._worker_pid()
                    LOG.info(
                        "worker_request_complete %s",
                        json.dumps(
                            {
                                "request_id": message.get("request_id"),
                                "pid": pid,
                                "usage": message.get("usage", {}),
                                "metrics": message.get("metrics", {}),
                                "finish_reason": message.get("finish_reason", "stop"),
                                "stream": True,
                            },
                            sort_keys=True,
                        ),
                    )

                    yield CompletionResult(
                        content=str(message["content"]),
                        finish_reason=str(message.get("finish_reason", "stop")),
                        usage=dict(message.get("usage", {})),
                        metrics=dict(message.get("metrics", {})),
                        tool_calls=message.get("tool_calls"),
                    )
                    return
            except TimeoutError:
                with self._state_lock:
                    self._terminate_process_locked(force=True)
                raise
            except RuntimeError:
                raise
            except Exception as exc:
                with self._state_lock:
                    self._terminate_process_locked(force=True)
                raise RuntimeError(f"{FAILURE_CRASH}:Worker request crashed:{exc}") from exc

    def _start_idle_thread(self) -> None:
        """Start the background idle-unload thread.

        Separated from ``__init__`` so tests can construct a ``WorkerManager``
        without a live daemon if needed (by skipping this call). The thread is
        a daemon so it never prevents supervisor shutdown on its own.
        """

        if self._idle_thread is not None:
            return
        thread = threading.Thread(
            target=self._idle_thread_loop,
            name="mlx-turbo-gemma-idle-unload",
            daemon=True,
        )
        self._idle_thread = thread
        thread.start()

    def _idle_thread_loop(self) -> None:
        """Background loop that unloads the worker after it has been idle.

        Key properties (see audit C1/C2):
          * Never blocks readers. We only acquire ``_worker_lock`` via
            ``acquire(blocking=False)`` so if a generation or admin op is
            running, we skip this tick and try again later.
          * Never triggers unload from reader paths (``/ready``,
            ``/admin/stats``, ``can_accept_requests``). Those paths only
            observe state; the unload decision lives here.
          * Uses ``_shutdown_event.wait`` so supervisor shutdown wakes us
            immediately instead of waiting out the interval.
        """

        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(self._IDLE_CHECK_INTERVAL_SECONDS):
                return
            # Cheap read-only peek first to avoid ever grabbing _worker_lock
            # when there's no reason to unload.
            with self._state_lock:
                due = self._idle_unload_due_locked()
            if not due:
                continue
            # Non-blocking: if a generation or admin op is in flight, skip.
            if not self._worker_lock.acquire(blocking=False):
                continue
            try:
                # Re-check under both locks in case state changed while we
                # were waiting to acquire _worker_lock.
                with self._state_lock:
                    if not self._idle_unload_due_locked():
                        continue
                    prior_pid = self._worker_pid()
                    self._set_state("unloading", accepting_requests=False)
                try:
                    self._send({"command": "shutdown"})
                    self._read_message(timeout_seconds=2, timeout_metric="shutdown_timeouts")
                    with self._state_lock:
                        if self._process is not None:
                            try:
                                self._process.wait(timeout=2)
                            except Exception:
                                pass
                        self._metrics["worker_unloads"] += 1
                        LOG.info(
                            "worker_unloaded %s",
                            json.dumps({"pid": prior_pid, "reason": "idle"}, sort_keys=True),
                        )
                        self._reset_to_not_loaded()
                except Exception as exc:
                    with self._state_lock:
                        self._terminate_process_locked(force=True)
                        self._mark_failed(
                            f"idle_unload_failed:{exc}", failure_kind=FAILURE_UNLOAD
                        )
            finally:
                self._worker_lock.release()

    def shutdown(self) -> None:
        # Signal the idle thread first so it doesn't race our worker I/O or
        # try to unload underneath us while we're already shutting down.
        self._shutdown_event.set()
        idle_thread = self._idle_thread
        if idle_thread is not None and idle_thread.is_alive():
            idle_thread.join(timeout=self._IDLE_CHECK_INTERVAL_SECONDS + 1)
        # Serialize worker I/O against any in-flight generation / admin op.
        with self._worker_lock:
            with self._state_lock:
                if not self._process or self._process.poll() is not None:
                    self._reset_to_not_loaded()
                    return
                self._set_state("unloading", accepting_requests=False)
            try:
                self._send({"command": "shutdown"})
                self._read_message(timeout_seconds=2, timeout_metric="shutdown_timeouts")
                with self._state_lock:
                    if self._process is not None:
                        try:
                            self._process.wait(timeout=2)
                        except Exception:
                            pass
                    self._metrics["worker_unloads"] += 1
            except Exception:
                with self._state_lock:
                    self._terminate_process_locked(force=True)
            finally:
                with self._state_lock:
                    self._reset_to_not_loaded()
