"""Supervisor metadata store for short-lived SI Drone sessions."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_SESSION_POLICY = {
    "ttl_s": 300,
    "max_turns": 16,
    "max_context_tokens": 20_000,
    "audio_seconds_per_turn": 45,
    "on_overflow": "reject",
}
SESSION_POLICY_CEILINGS = {
    "ttl_s": 300,
    "max_turns": 16,
    "max_context_tokens": 20_000,
}


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: float
    last_active_at: float
    expires_at: float
    turn_count: int
    state: str
    policy: dict[str, Any]
    worker_binding: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "expires_at": self.expires_at,
            "turn_count": self.turn_count,
            "state": self.state,
            "policy": dict(self.policy),
            "worker_binding": self.worker_binding,
        }


class SessionStore:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        on_count_change: Callable[[int], None] | None = None,
        on_expire: Callable[[str], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self._on_count_change = on_count_change
        self._on_expire = on_expire
        self._shutdown = threading.Event()
        self._policy = self._load_policy(config)
        self._reaper_interval_s = max(1, int(config.get("sessions", {}).get("reaperIntervalS", 5) or 5))
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            name="mlx-turbo-gemma-session-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    @staticmethod
    def _load_policy(config: dict[str, Any]) -> dict[str, Any]:
        raw_policy = config.get("sessions", {})
        raw_policy = raw_policy if isinstance(raw_policy, dict) else {}
        policy = dict(DEFAULT_SESSION_POLICY)
        for key, ceiling in SESSION_POLICY_CEILINGS.items():
            raw_value = raw_policy.get(key, policy[key])
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                value = int(policy[key])
            policy[key] = max(1, min(value, ceiling))
        try:
            audio_seconds = int(raw_policy.get("audio_seconds_per_turn", policy["audio_seconds_per_turn"]))
        except (TypeError, ValueError):
            audio_seconds = int(policy["audio_seconds_per_turn"])
        policy["audio_seconds_per_turn"] = max(1, audio_seconds)
        on_overflow = raw_policy.get("onOverflow", raw_policy.get("on_overflow", policy["on_overflow"]))
        policy["on_overflow"] = "reject" if on_overflow != "reject" else "reject"
        return policy

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self._policy)

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create(self) -> SessionRecord:
        now = time.time()
        session_id = f"sidr_{uuid.uuid4().hex[:24]}"
        record = SessionRecord(
            session_id=session_id,
            created_at=now,
            last_active_at=now,
            expires_at=now + int(self._policy["ttl_s"]),
            turn_count=0,
            state="active",
            policy=dict(self._policy),
        )
        with self._lock:
            self._sessions[session_id] = record
            count = len(self._sessions)
        self._notify_count(count)
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        self.reap_expired()
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> list[SessionRecord]:
        self.reap_expired()
        with self._lock:
            return sorted(self._sessions.values(), key=lambda record: record.created_at)

    def delete(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            record = self._sessions.pop(session_id, None)
            count = len(self._sessions)
        if record is not None:
            self._notify_count(count)
        return record

    def begin_turn(self, session_id: str) -> SessionRecord:
        now = time.time()
        expired: SessionRecord | None = None
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise KeyError("session_not_found")
            if record.expires_at <= now:
                expired = self._sessions.pop(session_id)
                count = len(self._sessions)
            elif record.turn_count >= int(record.policy["max_turns"]):
                raise RuntimeError("max_turns_exceeded")
            else:
                updated = SessionRecord(
                    session_id=record.session_id,
                    created_at=record.created_at,
                    last_active_at=now,
                    expires_at=now + int(record.policy["ttl_s"]),
                    turn_count=record.turn_count + 1,
                    state=record.state,
                    policy=dict(record.policy),
                    worker_binding=record.worker_binding,
                )
                self._sessions[session_id] = updated
                return updated
        if expired is not None:
            self._notify_count(count)
            self._notify_expire(expired.session_id)
            raise KeyError("session_expired")
        raise KeyError("session_not_found")

    def bind_worker(self, session_id: str, worker_pid: int | None) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return
            self._sessions[session_id] = SessionRecord(
                session_id=record.session_id,
                created_at=record.created_at,
                last_active_at=record.last_active_at,
                expires_at=record.expires_at,
                turn_count=record.turn_count,
                state=record.state,
                policy=dict(record.policy),
                worker_binding=worker_pid,
            )

    def reap_expired(self) -> list[str]:
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for session_id, record in list(self._sessions.items()):
                if record.expires_at <= now:
                    expired.append(session_id)
                    del self._sessions[session_id]
            count = len(self._sessions)
        if expired:
            self._notify_count(count)
            for session_id in expired:
                self._notify_expire(session_id)
        return expired

    def shutdown(self) -> None:
        self._shutdown.set()
        self._reaper_thread.join(timeout=self._reaper_interval_s + 1)

    def _reaper_loop(self) -> None:
        while not self._shutdown.wait(self._reaper_interval_s):
            self.reap_expired()

    def _notify_count(self, count: int) -> None:
        if self._on_count_change is None:
            return
        try:
            self._on_count_change(count)
        except Exception:
            pass

    def _notify_expire(self, session_id: str) -> None:
        if self._on_expire is None:
            return
        try:
            self._on_expire(session_id)
        except Exception:
            pass
