from __future__ import annotations

import unittest

from shared.parts import extract_message_parts
from supervisor.main import normalize_messages, validate_chat_request
from supervisor.worker_manager import WorkerManager
from worker.backends import StubBackend
from worker.main import handle_generate


IMAGE_DATA_URL = "data:image/png;base64,YQ=="


def config(*, image_enabled: bool = False, image_max_inputs: int = 4) -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": 4029},
        "model": {"id": "mlx-test-model"},
        "worker": {
            "stubMode": True,
            "queue": {"maxDepth": 0},
            "idleUnload": {"enabled": False},
        },
        "governor": {"enabled": False},
        "modalities": {
            "text": {"enabled": True},
            "image": {
                "enabled": image_enabled,
                "maxInputs": image_max_inputs,
                "maxBytesMb": 1,
                "allowedMimeTypes": ["image/png"],
                "transport": ["data_url"],
            },
            "video": {"enabled": False, "maxInputs": 1, "maxBytesMb": 1, "allowedMimeTypes": ["video/mp4"], "transport": ["data_url"]},
            "audio": {"enabled": False, "maxInputs": 1, "maxBytesMb": 1, "allowedMimeTypes": ["audio/wav"], "transport": ["data_url"]},
            "document": {"enabled": False, "maxInputs": 1, "maxBytesMb": 1, "allowedMimeTypes": ["application/pdf"], "transport": ["data_url"]},
            "strictCapabilityCheck": True,
        },
    }


class MessagePartTests(unittest.TestCase):
    def make_app(self, cfg: dict):
        manager = WorkerManager(cfg)
        self.addCleanup(manager.shutdown)
        return type("FakeApp", (), {"config": cfg, "worker": manager})()

    def test_text_only_request_remains_valid_and_normalizes_to_content(self):
        cfg = config()
        app = self.make_app(cfg)
        payload = {
            "model": "mlx-test-model",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }

        self.assertIsNone(validate_chat_request(payload, app))
        self.assertEqual(normalize_messages(payload["messages"], config=cfg), [{"role": "user", "content": "hello"}])

    def test_disabled_image_rejects_at_supervisor_layer(self):
        cfg = config(image_enabled=False)
        app = self.make_app(cfg)
        payload = {
            "model": "mlx-test-model",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}}]}],
        }

        self.assertEqual(validate_chat_request(payload, app), (422, "unsupported_modality", "Modality is disabled: image"))

    def test_enabled_image_survives_normalization_as_typed_part(self):
        cfg = config(image_enabled=True)
        app = self.make_app(cfg)
        payload = {
            "model": "mlx-test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}},
                    ],
                }
            ],
        }

        self.assertIsNone(validate_chat_request(payload, app))
        normalized = normalize_messages(payload["messages"], config=cfg)
        self.assertEqual(normalized[0]["content"], "describe this")
        self.assertEqual(normalized[0]["parts"][0]["type"], "text")
        self.assertEqual(normalized[0]["parts"][1]["type"], "image")
        self.assertEqual(normalized[0]["parts"][1]["mime_type"], "image/png")
        self.assertEqual(normalized[0]["parts"][1]["byte_length"], 1)

    def test_malformed_data_url_rejects_cleanly(self):
        cfg = config(image_enabled=True)
        app = self.make_app(cfg)
        payload = {
            "model": "mlx-test-model",
            "messages": [{"role": "user", "content": [{"type": "input_image", "data_url": "not-a-data-url"}]}],
        }

        self.assertEqual(validate_chat_request(payload, app), (400, "bad_request", "image parts must use data_url transport"))

    def test_max_inputs_rejects_cleanly(self):
        cfg = config(image_enabled=True, image_max_inputs=1)
        app = self.make_app(cfg)
        payload = {
            "model": "mlx-test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "data_url": IMAGE_DATA_URL},
                        {"type": "input_image", "data_url": IMAGE_DATA_URL},
                    ],
                }
            ],
        }

        self.assertEqual(validate_chat_request(payload, app), (400, "bad_request", "image input count exceeds configured limit"))

    def test_worker_backend_rejects_enabled_but_unsupported_modality(self):
        cfg = config(image_enabled=True)
        messages = normalize_messages(
            [{"role": "user", "content": [{"type": "input_image", "data_url": IMAGE_DATA_URL}]}],
            config=cfg,
        )

        result = handle_generate(StubBackend(), {"request_id": "r1", "messages": messages})

        self.assertEqual(result["type"], "error")
        self.assertEqual(result["request_id"], "r1")
        self.assertEqual(result["error"], "unsupported_modality:Backend does not support requested modality: image")

    def test_status_surfaces_configured_backend_and_effective_modalities(self):
        cfg = config(image_enabled=True)
        manager = WorkerManager(cfg)
        self.addCleanup(manager.shutdown)

        model = manager.models_payload()["data"][0]
        ready = manager.ready_payload()
        stats = manager.stats_payload()

        self.assertEqual(model["modalities"]["configured"], ["image", "text"])
        self.assertEqual(model["modalities"]["backend_supported"], ["text"])
        self.assertEqual(model["modalities"]["effective"], ["text"])
        self.assertTrue(model["modalities"]["vision"]["configured"])
        self.assertFalse(model["modalities"]["vision"]["effective"])
        self.assertEqual(ready["modalities"], model["modalities"])
        self.assertEqual(stats["modalities"], model["modalities"])

    def test_stats_surfaces_cached_backend_memory_only_on_admin_payload(self):
        cfg = config()
        manager = WorkerManager(cfg)
        self.addCleanup(manager.shutdown)

        with manager._state_lock:
            manager._update_backend_stats_locked({"memory": {"units": "GiB", "samples": {"after_load": {"weights_gb": 1.25}}}})

        stats = manager.stats_payload()

        self.assertEqual(stats["backend"]["memory"]["units"], "GiB")
        self.assertEqual(stats["backend"]["memory"]["samples"]["after_load"]["weights_gb"], 1.25)
        self.assertNotIn("backend", manager.ready_payload())

    def test_capability_rejection_does_not_degrade_worker(self):
        cfg = config(image_enabled=True)
        manager = WorkerManager(cfg)
        self.addCleanup(manager.shutdown)
        with manager._state_lock:
            manager._loaded = True
            manager._set_state("busy", accepting_requests=False, loaded=True)

        manager.complete_request_rejected("Backend does not support requested modality: image")
        stats = manager.stats_payload()

        self.assertEqual(stats["worker"]["state"], "ready")
        self.assertEqual(stats["worker"]["consecutive_failures"], 0)
        self.assertEqual(stats["metrics"]["failed_requests"], 1)

    def test_extract_message_parts_accepts_string_as_text_part(self):
        parts = extract_message_parts("hello")

        self.assertIsNotNone(parts)
        self.assertEqual(parts[0].type, "text")
        self.assertEqual(parts[0].text, "hello")


if __name__ == "__main__":
    unittest.main()
