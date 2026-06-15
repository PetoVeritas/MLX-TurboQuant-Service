from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from shared.backend_adapters import backend_descriptor, backend_supported_modalities, configured_backend_id
from shared.parts import modalities_status
from worker.backends import BackendResult, MlxVlmTurboQuantBackend, StubBackend
from worker.main import handle_generate


IMAGE_DATA_URL = "data:image/png;base64,YQ=="
AUDIO_DATA_URL = "data:audio/wav;base64,Yg=="


def config(*, backend: str | None = None, stub_mode: bool = False, image: bool = False, audio: bool = False) -> dict:
    worker = {"stubMode": stub_mode}
    if backend is not None:
        worker["backend"] = backend
    return {
        "model": {"id": "test-model", "path": "/tmp/model"},
        "worker": worker,
        "modalities": {
            "text": {"enabled": True},
            "image": {"enabled": image},
            "video": {"enabled": False},
            "audio": {"enabled": audio},
            "document": {"enabled": False},
            "strictCapabilityCheck": True,
        },
    }


class BackendAdapterTests(unittest.TestCase):
    def test_default_backend_is_mlx_vlm_turboquant(self):
        cfg = config()

        self.assertEqual(configured_backend_id(cfg), "mlx_vlm_turboquant")
        self.assertEqual(backend_descriptor(cfg).backend_id, "mlx_vlm_turboquant")
        self.assertEqual(backend_supported_modalities(cfg), {"text", "image", "audio"})

    def test_stub_mode_overrides_backend_selection(self):
        cfg = config(backend="mlx_vlm_turboquant", stub_mode=True)

        self.assertEqual(configured_backend_id(cfg), "stub")
        self.assertEqual(backend_descriptor(cfg).backend_id, "stub")
        self.assertEqual(backend_supported_modalities(cfg), {"text"})

    def test_backend_classes_declare_capabilities(self):
        self.assertEqual(StubBackend.supported_modalities_for_config(), {"text"})
        self.assertEqual(MlxVlmTurboQuantBackend.supported_modalities_for_config(), {"text", "image", "audio"})

    def test_turboquant_capabilities_follow_model_config_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gemma4",
                        "vision_config": {},
                        "audio_config": None,
                    }
                )
            )
            cfg = config(backend="mlx_vlm_turboquant")
            cfg["model"]["path"] = str(model_dir)

            self.assertEqual(backend_supported_modalities(cfg), {"text", "image"})

    def test_vlm_turboquant_status_intersects_with_lane_policy(self):
        cfg = config(backend="mlx_vlm_turboquant", image=True, audio=True)

        status = modalities_status(cfg)

        self.assertEqual(status["configured"], ["audio", "image", "text"])
        self.assertEqual(status["backend_supported"], ["audio", "image", "text"])
        self.assertEqual(status["effective"], ["audio", "image", "text"])
        self.assertTrue(status["vision"]["configured"])
        self.assertTrue(status["vision"]["effective"])

    def test_vlm_turboquant_does_not_enable_disabled_policy_modalities(self):
        cfg = config(backend="mlx_vlm_turboquant", image=True, audio=False)

        status = modalities_status(cfg)

        self.assertEqual(status["configured"], ["image", "text"])
        self.assertEqual(status["backend_supported"], ["audio", "image", "text"])
        self.assertEqual(status["effective"], ["image", "text"])

    def test_unknown_backend_rejects_cleanly(self):
        with self.assertRaisesRegex(ValueError, "unknown_backend_adapter:nope"):
            backend_descriptor(config(backend="nope"))

    def test_worker_generate_allows_media_when_backend_supports_it(self):
        class FakeUnifiedBackend:
            def supported_modalities(self) -> set[str]:
                return {"text", "image", "audio"}

            def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools=None) -> BackendResult:
                self.messages = messages
                self.max_tokens = max_tokens
                return BackendResult(
                    content="ok",
                    finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    metrics={"queue_wait_ms": 0, "load_ms": 0, "prefill_ms": None, "generation_ms": None, "total_ms": 0},
                )

        backend = FakeUnifiedBackend()
        payload = {
            "request_id": "r1",
            "messages": [
                {
                    "role": "user",
                    "content": "describe this",
                    "parts": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image", "data_url": IMAGE_DATA_URL, "mime_type": "image/png", "byte_length": 1},
                        {"type": "audio", "data_url": AUDIO_DATA_URL, "mime_type": "audio/wav", "byte_length": 1},
                    ],
                }
            ],
            "max_tokens": 8,
        }

        result = handle_generate(backend, payload)

        self.assertEqual(result["type"], "completion_result")
        self.assertEqual(result["content"], "ok")
        self.assertEqual(backend.max_tokens, 8)

    def test_vlm_turboquant_materializes_image_and_audio_parts_for_generate(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 32
        backend._sampling_kwargs = {"temperature": 0.1, "top_p": 0.9}
        backend._load_ms = 5
        backend._load_consumed = False
        captured: dict[str, Any] = {}

        class FakeResponse:
            text = "heard and saw it"
            finish_reason = "stop"
            prompt_tokens = 4
            generation_tokens = 3

        def fake_generate(model, processor, prompt, **kwargs):
            captured["model"] = model
            captured["processor"] = processor
            captured["prompt"] = prompt
            captured["image"] = kwargs.get("image")
            captured["audio"] = kwargs.get("audio")
            captured["max_tokens"] = kwargs.get("max_tokens")
            captured["temperature"] = kwargs.get("temperature")
            captured["top_p"] = kwargs.get("top_p")
            captured["image_bytes"] = [Path(path).read_bytes() for path in captured["image"]]
            captured["audio_bytes"] = [Path(path).read_bytes() for path in captured["audio"]]
            return FakeResponse()

        backend._generate = fake_generate
        messages = [
            {
                "role": "user",
                "content": "describe this",
                "parts": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "data_url": IMAGE_DATA_URL, "mime_type": "image/png", "byte_length": 1},
                    {"type": "audio", "data_url": AUDIO_DATA_URL, "mime_type": "audio/wav", "byte_length": 1},
                ],
            }
        ]

        result = backend.generate(messages, max_tokens=12)

        self.assertEqual(result.content, "heard and saw it")
        self.assertEqual(result.usage, {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        self.assertEqual(result.metrics["load_ms"], 5)
        self.assertEqual(captured["prompt"], "<|image|>\ndescribe this\n<|audio|>")
        self.assertEqual(captured["image_bytes"], [b"a"])
        self.assertEqual(captured["audio_bytes"], [b"b"])
        self.assertEqual(captured["max_tokens"], 12)
        self.assertEqual(captured["temperature"], 0.1)
        self.assertEqual(captured["top_p"], 0.9)

    def test_vlm_turboquant_handles_plain_string_generate_response(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 32
        backend._sampling_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        backend._generate = lambda *args, **kwargs: "plain string answer"

        result = backend.generate([{"role": "user", "content": "say something"}], max_tokens=12)

        self.assertEqual(result.content, "plain string answer")
        self.assertEqual(result.usage, {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(result.finish_reason, "stop")

    def test_vlm_turboquant_passes_tools_to_chat_template_and_decodes_prior_tool_args(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        captured: dict[str, Any] = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                captured["tools"] = kwargs.get("tools")
                captured["tokenize"] = kwargs.get("tokenize")
                captured["add_generation_prompt"] = kwargs.get("add_generation_prompt")
                return "templated prompt"

        backend._processor = type("Processor", (), {"tokenizer": FakeTokenizer()})()
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
        messages = [
            {"role": "user", "content": "lookup dc"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"city\":\"dc\"}"},
                    }
                ],
            },
        ]

        prompt = backend._build_prompt(messages, tools=tools)

        self.assertEqual(prompt, "templated prompt")
        self.assertIs(captured["tools"], tools)
        self.assertFalse(captured["tokenize"])
        self.assertTrue(captured["add_generation_prompt"])
        self.assertEqual(captured["messages"][1]["tool_calls"][0]["function"]["arguments"], {"city": "dc"})

    def test_vlm_turboquant_returns_openai_tool_calls_from_gemma_call_output(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 32
        backend._sampling_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        backend._generate = lambda *args, **kwargs: 'call:lookup{city:<|"|>dc<|"|>}'
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        result = backend.generate([{"role": "user", "content": "lookup dc"}], max_tokens=12, tools=tools)

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.content, "")
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0]["type"], "function")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "lookup")
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"city": "dc"})

    def test_vlm_turboquant_filters_undeclared_tool_calls(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 32
        backend._sampling_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        backend._generate = lambda *args, **kwargs: 'call:not_allowed{city:<|"|>dc<|"|>}'
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]

        result = backend.generate([{"role": "user", "content": "lookup dc"}], max_tokens=12, tools=tools)

        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNone(result.tool_calls)
        self.assertIn("not_allowed", result.content)


if __name__ == "__main__":
    unittest.main()
