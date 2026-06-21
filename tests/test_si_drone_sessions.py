from __future__ import annotations

import base64
import io
import unittest
import wave

from supervisor.main import normalize_session_parts
from supervisor.session_store import SessionStore
from supervisor.worker_manager import WorkerManager
from worker.backends import MlxVlmTurboQuantBackend, StubBackend, strip_channel_markup
from worker.main import handle_session_generate, handle_session_teardown


def config() -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": 4029},
        "model": {"id": "mlx-test-model"},
        "worker": {
            "stubMode": True,
            "queue": {"maxDepth": 0},
            "idleUnload": {"enabled": True, "idleMs": 1000},
        },
        "governor": {"enabled": False},
        "sessions": {
            "ttl_s": 120,
            "max_turns": 8,
            "max_context_tokens": 16000,
            "audio_seconds_per_turn": 8,
            "reaperIntervalS": 60,
        },
    }


def wav_b64(seconds: float = 0.05, rate: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * max(1, int(seconds * rate)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SiDroneSessionTests(unittest.TestCase):
    def test_session_store_caps_policy_and_counts_live_sessions(self):
        counts: list[int] = []
        cfg = config()
        cfg["sessions"]["ttl_s"] = 999
        cfg["sessions"]["max_turns"] = 99
        store = SessionStore(cfg, on_count_change=counts.append)
        self.addCleanup(store.shutdown)

        record = store.create()

        self.assertEqual(record.policy["ttl_s"], 300)
        self.assertEqual(record.policy["max_turns"], 16)
        self.assertEqual(counts[-1], 1)
        self.assertEqual(store.begin_turn(record.session_id).turn_count, 1)
        self.assertIsNotNone(store.delete(record.session_id))
        self.assertEqual(counts[-1], 0)

    def test_normalize_session_parts_accepts_text_and_wav_audio(self):
        policy = {"audio_seconds_per_turn": 8}
        parts, error = normalize_session_parts(
            {
                "parts": [
                    {"type": "text", "text": "remember this"},
                    {"type": "audio", "audio": {"format": "wav", "data": wav_b64()}},
                ]
            },
            policy,
        )

        self.assertIsNone(error)
        self.assertEqual(parts[0], {"type": "text", "text": "remember this"})
        self.assertEqual(parts[1]["type"], "audio")
        self.assertEqual(parts[1]["mime_type"], "audio/wav")
        self.assertTrue(parts[1]["data_url"].startswith("data:audio/wav;base64,"))

    def test_normalize_session_parts_rejects_future_modalities_for_v1(self):
        parts, error = normalize_session_parts({"parts": [{"type": "image", "image": {}}]}, {"audio_seconds_per_turn": 8})

        self.assertIsNone(parts)
        self.assertEqual(error, (415, "unsupported_part_type", "SI Drone v1 does not support part type: image"))

    def test_worker_idle_unload_is_blocked_while_sessions_are_live(self):
        manager = WorkerManager(config())
        self.addCleanup(manager.shutdown)
        with manager._state_lock:
            manager._loaded = True
            manager._process = type("Proc", (), {"poll": lambda self: None, "pid": 123})()
            manager._set_state("ready", accepting_requests=True, loaded=True)
            manager._last_activity_at = 0
            self.assertTrue(manager._idle_unload_due_locked())
            manager._live_session_count = 1
            self.assertFalse(manager._idle_unload_due_locked())

    def test_worker_session_generate_dispatches_to_backend(self):
        backend = StubBackend()
        result = handle_session_generate(
            backend,
            {
                "request_id": "r1",
                "session_id": "sidr_test",
                "parts": [{"type": "text", "text": "hello"}],
                "max_tokens": 8,
                "policy": {"max_turns": 8},
                "turn_index": 1,
            },
        )

        self.assertEqual(result["type"], "session_result")
        self.assertEqual(result["request_id"], "r1")
        self.assertIn("sidr_test", result["content"])
        self.assertEqual(result["metrics"]["turn_index"], 1)

    def test_stub_session_generate_carries_text_between_turns(self):
        backend = StubBackend()
        first = handle_session_generate(
            backend,
            {
                "request_id": "r1",
                "session_id": "sidr_text",
                "parts": [{"type": "text", "text": "remember alpha"}],
                "policy": {},
                "turn_index": 1,
            },
        )
        second = handle_session_generate(
            backend,
            {
                "request_id": "r2",
                "session_id": "sidr_text",
                "parts": [{"type": "text", "text": "what was it"}],
                "policy": {},
                "turn_index": 2,
            },
        )

        self.assertEqual(first["type"], "session_result")
        self.assertIn("remember alpha", second["content"])
        self.assertIn("what was it", second["content"])

    def test_stub_session_generate_tracks_audio_between_turns(self):
        backend = StubBackend()
        first = handle_session_generate(
            backend,
            {
                "request_id": "r1",
                "session_id": "sidr_audio",
                "parts": [{"type": "audio", "data_url": f"data:audio/wav;base64,{wav_b64()}", "mime_type": "audio/wav"}],
                "policy": {},
                "turn_index": 1,
            },
        )
        second = handle_session_generate(
            backend,
            {
                "request_id": "r2",
                "session_id": "sidr_audio",
                "parts": [{"type": "text", "text": "was audio processed?"}],
                "policy": {},
                "turn_index": 2,
            },
        )

        self.assertEqual(first["metrics"]["audio_token_count"], 1)
        self.assertIn("audio_count: 1", second["content"])

    def test_turboquant_rejects_prior_turn_when_worker_cache_is_missing(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._session_caches = {}
        backend._max_output_tokens = 8

        with self.assertRaisesRegex(RuntimeError, "session_lost"):
            backend.session_generate(
                "sidr_missing",
                [{"type": "text", "text": "second turn"}],
                max_tokens=8,
                policy={"max_context_tokens": 16000},
                turn_index=2,
            )

    def test_turboquant_session_parts_wrap_first_turn_with_gemma4_markers(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = type("Model", (), {"config": type("Config", (), {"model_type": "gemma4"})()})()
        prepared = backend._prepare_session_parts(
            [
                {"type": "text", "text": "before"},
                {"type": "audio", "data_url": f"data:audio/wav;base64,{wav_b64()}", "mime_type": "audio/wav", "byte_length": 1},
                {"type": "text", "text": "after"},
            ],
            turn_index=1,
        )
        self.addCleanup(prepared.cleanup)

        self.assertEqual(prepared.prompt, "<start_of_turn>user\nbefore\n<|audio|>\nafter<end_of_turn>\n<start_of_turn>model\n")
        self.assertTrue(prepared.add_special_tokens)
        self.assertEqual(len(prepared.audio_paths), 1)

    def test_turboquant_session_parts_append_followup_turn_without_special_tokens(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = type("Model", (), {"config": type("Config", (), {"model_type": "gemma4"})()})()
        prepared = backend._prepare_session_parts([{"type": "text", "text": "second turn"}], turn_index=2)

        self.assertEqual(prepared.prompt, "<end_of_turn>\n<start_of_turn>user\nsecond turn<end_of_turn>\n<start_of_turn>model\n")
        self.assertFalse(prepared.add_special_tokens)

    def test_worker_session_teardown_calls_backend_hook(self):
        class Backend:
            def __init__(self) -> None:
                self.deleted: list[str] = []

            def teardown_session(self, session_id: str) -> None:
                self.deleted.append(session_id)

        backend = Backend()
        result = handle_session_teardown(backend, {"session_id": "sidr_test"})

        self.assertEqual(result["type"], "session_teardown_ack")
        self.assertEqual(backend.deleted, ["sidr_test"])

    def test_backend_without_session_hook_rejects_cleanly(self):
        class PlainBackend:
            pass

        result = handle_session_generate(PlainBackend(), {"request_id": "r1", "session_id": "sidr_test", "parts": []})

        self.assertEqual(result["type"], "error")
        self.assertTrue(result["error"].startswith("unsupported_backend:"))

    def test_strip_channel_markup_removes_gemma_turn_markers(self):
        self.assertEqual(strip_channel_markup("4<end_of_turn>\n<start_of_turn>model"), "4")


if __name__ == "__main__":
    unittest.main()
