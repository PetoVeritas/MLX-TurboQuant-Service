from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from shared.backend_adapters import backend_descriptor, backend_supported_modalities, configured_backend_id
from shared.parts import modalities_status
from worker.backends import BackendResult, MlxVlmDiffusionGemmaBackend, MlxVlmTurboQuantBackend, StubBackend
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

    def test_diffusion_gemma_backend_selects_new_descriptor_only(self):
        cfg = config(backend="mlx_vlm_diffusion_gemma", image=True, audio=True)

        descriptor = backend_descriptor(cfg)
        status = modalities_status(cfg)

        self.assertEqual(configured_backend_id(cfg), "mlx_vlm_diffusion_gemma")
        self.assertEqual(descriptor.backend_id, "mlx_vlm_diffusion_gemma")
        self.assertEqual(descriptor.display_name, "MLX-VLM DiffusionGemma backend")
        self.assertEqual(backend_supported_modalities(cfg), {"text", "image"})
        self.assertEqual(status["configured"], ["audio", "image", "text"])
        self.assertEqual(status["backend_supported"], ["image", "text"])
        self.assertEqual(status["effective"], ["image", "text"])

    def test_stub_mode_overrides_backend_selection(self):
        cfg = config(backend="mlx_vlm_turboquant", stub_mode=True)

        self.assertEqual(configured_backend_id(cfg), "stub")
        self.assertEqual(backend_descriptor(cfg).backend_id, "stub")
        self.assertEqual(backend_supported_modalities(cfg), {"text"})

    def test_backend_classes_declare_capabilities(self):
        self.assertEqual(StubBackend.supported_modalities_for_config(), {"text"})
        self.assertEqual(MlxVlmTurboQuantBackend.supported_modalities_for_config(), {"text", "image", "audio"})
        self.assertEqual(MlxVlmDiffusionGemmaBackend.supported_modalities_for_config(), {"text", "image"})

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

    def test_vlm_diffusion_gemma_maps_diffusion_knobs_and_materializes_images(self):
        backend = MlxVlmDiffusionGemmaBackend.__new__(MlxVlmDiffusionGemmaBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 64
        backend._sampling_kwargs = {"temperature": 0.0, "top_p": 0.95}
        backend._diffusion_kwargs = {
            "max_denoising_steps": 48,
            "diffusion_sampler": "confidence-threshold",
            "diffusion_threshold": 0.9,
            "stability_steps": 2,
        }
        backend._load_ms = 7
        backend._load_consumed = False
        captured: dict[str, Any] = {}

        class FakeResponse:
            text = "<|channel|>final diffusion answer"
            finish_reason = "stop"
            prompt_tokens = 5
            generation_tokens = 2
            peak_memory = 18.5
            diffusion_canvas_tokens = 16
            diffusion_denoising_steps = 48
            diffusion_work_tokens = 768
            diffusion_canvas_tps = 36.0
            diffusion_work_tps = 1728.0

        def fake_generate(model, processor, prompt, **kwargs):
            captured["model"] = model
            captured["processor"] = processor
            captured["prompt"] = prompt
            captured.update(kwargs)
            captured["image_bytes"] = [Path(path).read_bytes() for path in captured["image"]]
            return FakeResponse()

        backend._generate = fake_generate
        messages = [
            {
                "role": "user",
                "content": "describe this",
                "parts": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "data_url": IMAGE_DATA_URL, "mime_type": "image/png", "byte_length": 1},
                ],
            }
        ]

        result = backend.generate(messages, max_tokens=12)

        self.assertEqual(result.content, "diffusion answer")
        self.assertEqual(result.usage, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        self.assertEqual(result.metrics["load_ms"], 7)
        self.assertIsNone(result.metrics["prefill_ms"])
        self.assertIsNone(result.metrics["generation_ms"])
        self.assertEqual(result.metrics["first_visible_output_ms"], result.metrics["total_ms"])
        self.assertEqual(result.metrics["peak_memory_gb"], 18.5)
        self.assertEqual(result.metrics["diffusion_canvas_tokens"], 16)
        self.assertEqual(result.metrics["diffusion_denoising_steps"], 48)
        self.assertEqual(result.metrics["diffusion_work_tokens"], 768)
        self.assertEqual(result.metrics["diffusion_canvas_tps"], 36.0)
        self.assertEqual(result.metrics["diffusion_work_tps"], 1728.0)
        self.assertEqual(result.metrics["diffusion_tool_retry_count"], 0)
        self.assertEqual(captured["prompt"], "<|image|>\ndescribe this")
        self.assertEqual(captured["image_bytes"], [b"a"])
        self.assertEqual(len(captured["image"]), 1)
        self.assertEqual(captured["max_tokens"], 12)
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["top_p"], 0.95)
        self.assertEqual(captured["max_denoising_steps"], 48)
        self.assertEqual(captured["diffusion_sampler"], "confidence-threshold")
        self.assertEqual(captured["diffusion_threshold"], 0.9)
        self.assertEqual(captured["stability_steps"], 2)

    def test_vlm_diffusion_gemma_config_merges_diffusion_kwargs(self):
        backend = MlxVlmDiffusionGemmaBackend.__new__(MlxVlmDiffusionGemmaBackend)
        backend._config = {
            "model": {
                "diffusion": {
                    "max_denoising_steps": 48,
                    "diffusion_full_canvas": False,
                    "diffusion_sampler": "entropy-bound",
                    "threshold": None,
                    "block_length": 64,
                    "unknown": "ignored",
                }
            },
            "worker": {
                "diffusion": {
                    "diffusion_sampler": "confidence-threshold",
                    "threshold": 0.9,
                    "stability_steps": 2,
                }
            },
        }

        self.assertEqual(
            backend._generation_diffusion_kwargs(),
            {
                "max_denoising_steps": 48,
                "diffusion_full_canvas": False,
                "diffusion_sampler": "confidence-threshold",
                "diffusion_threshold": 0.9,
                "block_length": 64,
                "stability_steps": 2,
            },
        )

    def test_vlm_diffusion_gemma_passes_tools_to_chat_template_and_decodes_prior_args(self):
        backend = MlxVlmDiffusionGemmaBackend.__new__(MlxVlmDiffusionGemmaBackend)
        captured: dict[str, Any] = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                captured["tools"] = kwargs.get("tools")
                captured["tokenize"] = kwargs.get("tokenize")
                captured["add_generation_prompt"] = kwargs.get("add_generation_prompt")
                return "diffusion templated prompt"

        backend._processor = type("Processor", (), {"tokenizer": FakeTokenizer()})()
        tools = [{"type": "function", "function": {"name": "search_notes", "parameters": {"type": "object", "properties": {}}}}]
        messages = [
            {"role": "user", "content": "search notes"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_notes", "arguments": "{\"query\":\"DiffusionGemma OptIQ memory numbers\"}"},
                    }
                ],
            },
        ]

        prompt = backend._build_prompt(messages, tools=tools)

        self.assertEqual(prompt, "diffusion templated prompt")
        self.assertIs(captured["tools"], tools)
        self.assertFalse(captured["tokenize"])
        self.assertTrue(captured["add_generation_prompt"])
        self.assertEqual(
            captured["messages"][1]["tool_calls"][0]["function"]["arguments"],
            {"query": "DiffusionGemma OptIQ memory numbers"},
        )

    def test_vlm_diffusion_gemma_returns_openai_tool_calls_from_gemma_output(self):
        backend = MlxVlmDiffusionGemmaBackend.__new__(MlxVlmDiffusionGemmaBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 64
        backend._sampling_kwargs = {}
        backend._diffusion_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        backend._generate = lambda *args, **kwargs: '<|tool_call>call:search_notes{query:<|"|>DiffusionGemma OptIQ memory numbers<|"|>}<tool_call|>'
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_notes",
                    "description": "Search local notes",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        result = backend.generate([{"role": "user", "content": "search notes"}], max_tokens=16, tools=tools)

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.content, "")
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "search_notes")
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"query": "DiffusionGemma OptIQ memory numbers"})

    def test_vlm_diffusion_gemma_retries_malformed_tool_call_once(self):
        backend = MlxVlmDiffusionGemmaBackend.__new__(MlxVlmDiffusionGemmaBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 64
        backend._sampling_kwargs = {}
        backend._diffusion_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        prompts: list[str] = []
        responses = iter(
            [
                "<|tool_call>call:search_notes{query:",
                'call:search_notes{query:<|"|>DiffusionGemma OptIQ memory numbers<|"|>}',
            ]
        )

        def fake_generate(_model, _processor, prompt, **_kwargs):
            prompts.append(prompt)
            return next(responses)

        backend._generate = fake_generate
        tools = [{"type": "function", "function": {"name": "search_notes", "parameters": {"type": "object", "properties": {}}}}]

        result = backend.generate([{"role": "user", "content": "search notes"}], max_tokens=16, tools=tools)

        self.assertEqual(len(prompts), 2)
        self.assertIn("previous tool-call output was malformed", prompts[1])
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.metrics["diffusion_tool_retry_count"], 1)
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"query": "DiffusionGemma OptIQ memory numbers"})

    def test_vlm_diffusion_gemma_streams_finalized_content_only(self):
        backend = MlxVlmDiffusionGemmaBackend.__new__(MlxVlmDiffusionGemmaBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 64
        backend._sampling_kwargs = {}
        backend._diffusion_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        backend._generate = lambda *args, **kwargs: "final diffusion answer"

        events = list(backend.stream_generate([{"role": "user", "content": "hello"}], max_tokens=8))

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].text, "final diffusion answer")
        self.assertEqual(events[1].content, "final diffusion answer")

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
