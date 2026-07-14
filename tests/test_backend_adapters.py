from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any

from shared.backend_adapters import backend_descriptor, backend_supported_modalities, configured_backend_id
from shared.parts import modalities_status
from supervisor.main import validate_speech_request
from worker.backends import (
    BackendResult,
    BackendStreamChunk,
    MlxVlmDiffusionGemmaBackend,
    MlxVlmTurboQuantBackend,
    Qwen3TtsBackend,
    StepLoopResult,
    StepLoopToken,
    StubBackend,
    _DeferredExplicitTurboQuantKVCache,
    _memory_stats,
    _parse_kv_cache_config,
    _sum_mlx_array_nbytes,
    build_backend,
    split_reasoning_markup,
    strip_channel_markup,
)
from worker.main import handle_generate, handle_speech_generate


IMAGE_DATA_URL = "data:image/png;base64,YQ=="
AUDIO_DATA_URL = "data:audio/wav;base64,Yg=="


def write_wav(path: Path, *, sample_rate: int = 24_000, frames: int = 2_400, leading_silence: int = 0, trailing_silence: int = 0, active: bool = False) -> None:
    payload = bytearray()
    payload.extend(b"\0\0" * leading_silence)
    if active:
        for index in range(frames):
            sample = 9_000 if index % 2 == 0 else -9_000
            payload.extend(sample.to_bytes(2, byteorder="little", signed=True))
    else:
        payload.extend(b"\0\0" * frames)
    payload.extend(b"\0\0" * trailing_silence)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(payload))


class FakeMlxArray:
    __module__ = "mlx.core"

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class FakeTensor:
    def __init__(self, tokens: int) -> None:
        self.shape = (1, 1, tokens, 8)


class FakeKVCache:
    def __init__(self) -> None:
        self.offset = 0
        self.keys = None
        self.values = None
        self.state_reads = 0

    def update_and_fetch(self, keys: FakeTensor, values: FakeTensor) -> tuple[str, str]:
        self.offset += keys.shape[2]
        self.keys = f"k{self.offset}"
        self.values = f"v{self.offset}"
        return self.keys, self.values

    @property
    def state(self) -> tuple[str | None, str | None]:
        self.state_reads += 1
        return self.keys, self.values

    @state.setter
    def state(self, value: tuple[str | None, str | None]) -> None:
        self.keys, self.values = value
        self.offset = 0 if self.keys is None else int(str(self.keys)[1:])

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        return self.offset * 10


class FakeRotatingKVCache(FakeKVCache):
    pass


class FakeCacheList:
    def __init__(self, *caches: Any) -> None:
        self.caches = tuple(caches)


class FakeExplicitTurboQuantKVCache(FakeKVCache):
    conversions: list[dict[str, Any]] = []

    def __init__(
        self,
        *,
        key_bits: int,
        value_bits: int,
        group_size: int = 64,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.offset = offset
        self.keys = None if offset == 0 else f"k{offset}"
        self.values = None if offset == 0 else f"v{offset}"

    @classmethod
    def from_cache(
        cls,
        cache: FakeKVCache,
        *,
        key_bits: int,
        value_bits: int,
        group_size: int = 64,
    ) -> "FakeExplicitTurboQuantKVCache":
        cls.conversions.append(
            {
                "offset": cache.offset,
                "key_bits": key_bits,
                "value_bits": value_bits,
                "group_size": group_size,
            }
        )
        return cls(
            key_bits=key_bits,
            value_bits=value_bits,
            group_size=group_size,
            offset=cache.offset,
        )


class FakeVlmCacheModule:
    KVCache = FakeKVCache
    RotatingKVCache = FakeRotatingKVCache
    CacheList = FakeCacheList

    def __init__(self, prompt_cache: list[Any]) -> None:
        self.prompt_cache = prompt_cache

    def make_prompt_cache(self, _model: Any) -> list[Any]:
        return self.prompt_cache


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
    def test_memory_accounting_sums_mlx_arrays_only(self):
        payload = {
            "a": FakeMlxArray(128),
            "b": [FakeMlxArray(256), {"plain": object(), "nested": FakeMlxArray(512)}],
            "not_mlx": type("FakeNumpyArray", (), {"nbytes": 1024})(),
        }

        self.assertEqual(_sum_mlx_array_nbytes(payload), 896)

    def test_memory_stats_declares_binary_units_and_sampling_points(self):
        stats = _memory_stats({"after_load": {"sampling_point": "after_load", "weights_gb": 1.5}})

        self.assertEqual(stats["units"], "GiB")
        self.assertIn("binary GiB", stats["unit_note"])
        self.assertIn("prefill_peak", stats["sampling_points"])
        self.assertEqual(stats["samples"]["after_load"]["weights_gb"], 1.5)

    def test_kv_cache_config_defaults_disabled_and_declares_scope(self):
        parsed = _parse_kv_cache_config({"model": {}})

        self.assertFalse(parsed.enabled)
        self.assertEqual(parsed.bits, 4)
        self.assertIsNone(parsed.key_bits)
        self.assertIsNone(parsed.value_bits)
        self.assertEqual(parsed.effective_key_bits, 4)
        self.assertEqual(parsed.effective_value_bits, 4)
        self.assertEqual(parsed.group_size, 64)
        self.assertEqual(parsed.quantized_kv_start, 2048)
        self.assertEqual(parsed.quant_scheme, "turboquant")
        self.assertEqual(parsed.generation_kwargs(), {})
        self.assertEqual(parsed.to_dict()["excluded"], ["si_drone_session_caches"])
        self.assertTrue(parsed.to_dict()["asymmetricSupported"])

    def test_kv_cache_config_accepts_symmetric_integer_bits(self):
        parsed = _parse_kv_cache_config(
            {
                "model": {
                    "kvCache": {
                        "enabled": True,
                        "bits": 3,
                        "groupSize": 64,
                        "quantizedKvStart": 1024,
                        "quantScheme": "turboquant",
                    }
                }
            }
        )

        self.assertTrue(parsed.enabled)
        self.assertEqual(parsed.effective_key_bits, 3)
        self.assertEqual(parsed.effective_value_bits, 3)
        self.assertEqual(
            parsed.generation_kwargs(),
            {"kv_bits": 3, "kv_group_size": 64, "kv_quant_scheme": "turboquant", "quantized_kv_start": 1024},
        )

    def test_kv_cache_config_accepts_explicit_asymmetric_bits(self):
        parsed = _parse_kv_cache_config(
            {
                "model": {
                    "kvCache": {
                        "enabled": True,
                        "bits": 4,
                        "keyBits": 3,
                        "valueBits": 2,
                        "groupSize": 64,
                        "quantizedKvStart": 2048,
                        "quantScheme": "turboquant",
                    }
                }
            }
        )

        self.assertTrue(parsed.enabled)
        self.assertEqual(parsed.bits, 4)
        self.assertEqual(parsed.key_bits, 3)
        self.assertEqual(parsed.value_bits, 2)
        self.assertEqual(parsed.effective_key_bits, 3)
        self.assertEqual(parsed.effective_value_bits, 2)
        self.assertEqual(parsed.to_dict()["keyBits"], 3)
        self.assertEqual(parsed.to_dict()["valueBits"], 2)
        self.assertTrue(parsed.to_dict()["asymmetricSupported"])

    def test_kv_cache_config_accepts_higher_explicit_asymmetric_bits(self):
        parsed = _parse_kv_cache_config(
            {"model": {"kvCache": {"enabled": True, "keyBits": 8, "valueBits": 3}}}
        )

        self.assertEqual(parsed.effective_key_bits, 8)
        self.assertEqual(parsed.effective_value_bits, 3)

    def test_kv_cache_config_defaults_enabled_asymmetric_gemma_recipe_without_bits_shorthand(self):
        parsed = _parse_kv_cache_config({"model": {"kvCache": {"enabled": True}}})

        self.assertEqual(parsed.bits, 4)
        self.assertEqual(parsed.key_bits, 3)
        self.assertEqual(parsed.value_bits, 2)
        self.assertEqual(parsed.effective_key_bits, 3)
        self.assertEqual(parsed.effective_value_bits, 2)
        self.assertEqual(parsed.quantized_kv_start, 2048)

    def test_kv_cache_config_rejects_invalid_bits_uniform_and_unknown_fields(self):
        with self.assertRaisesRegex(RuntimeError, "bits_must_be_integer"):
            _parse_kv_cache_config({"model": {"kvCache": {"enabled": True, "bits": 3.5}}})
        with self.assertRaisesRegex(RuntimeError, "keyBits_below_min_3"):
            _parse_kv_cache_config({"model": {"kvCache": {"enabled": True, "keyBits": 2, "valueBits": 2}}})
        with self.assertRaisesRegex(RuntimeError, "valueBits_below_min_2"):
            _parse_kv_cache_config({"model": {"kvCache": {"enabled": True, "keyBits": 3, "valueBits": 1}}})
        with self.assertRaisesRegex(RuntimeError, "quantizedKvStart_must_be_positive"):
            _parse_kv_cache_config({"model": {"kvCache": {"enabled": True, "quantizedKvStart": 0}}})
        with self.assertRaisesRegex(RuntimeError, "quantScheme_must_be_turboquant"):
            _parse_kv_cache_config({"model": {"kvCache": {"enabled": True, "quantScheme": "uniform"}}})
        with self.assertRaisesRegex(RuntimeError, "unsupported_field:kBits"):
            _parse_kv_cache_config({"model": {"kvCache": {"enabled": True, "kBits": 8, "valueBits": 3}}})

    def test_turboquant_session_cache_is_explicitly_uncompressed(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._kv_cache_config = _parse_kv_cache_config({"model": {"kvCache": {"enabled": True}}})
        sentinel = object()
        backend._vlm_cache = type(
            "FakeVlmCache",
            (),
            {"make_prompt_cache": staticmethod(lambda _model: [sentinel, sentinel])},
        )()
        language_model = type("FakeLanguageModel", (), {"layers": [object(), object(), object()]})()
        backend._model = type("FakeModel", (), {"language_model": language_model})()

        self.assertEqual(backend._make_session_prompt_cache(), [sentinel, sentinel])
        self.assertEqual(backend._make_request_prompt_cache(), [sentinel, sentinel])
        self.assertTrue(backend._should_use_step_generate_path())

    def test_turboquant_request_cache_wraps_gemma_kv_only_and_skips_rotating_and_last_layer(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._kv_cache_config = _parse_kv_cache_config(
            {"model": {"kvCache": {"enabled": True, "keyBits": 3, "valueBits": 2}}}
        )
        backend._explicit_turboquant_cache_cls = FakeExplicitTurboQuantKVCache
        first = FakeKVCache()
        rotating = FakeRotatingKVCache()
        nested = FakeCacheList(FakeKVCache(), FakeRotatingKVCache())
        last = FakeKVCache()
        backend._vlm_cache = FakeVlmCacheModule([first, rotating, nested, last])
        language_model = type(
            "FakeLanguageModel",
            (),
            {"layers": [object(), object(), object(), object()]},
        )()
        backend._model = type(
            "FakeModel",
            (),
            {
                "config": type("FakeConfig", (), {"model_type": "gemma4"})(),
                "language_model": language_model,
            },
        )()

        prompt_cache = backend._make_request_prompt_cache()

        self.assertIsInstance(prompt_cache[0], _DeferredExplicitTurboQuantKVCache)
        self.assertIsInstance(prompt_cache[0], FakeExplicitTurboQuantKVCache)
        self.assertEqual(prompt_cache[0]._group_size, 64)
        self.assertIs(prompt_cache[1], rotating)
        self.assertIsInstance(prompt_cache[2], FakeCacheList)
        self.assertIsInstance(prompt_cache[2].caches[0], _DeferredExplicitTurboQuantKVCache)
        self.assertIs(prompt_cache[2].caches[1], nested.caches[1])
        self.assertIs(prompt_cache[3], last)
        self.assertEqual(backend._kv_generation_kwargs(), {})

    def test_turboquant_request_cache_keeps_non_gemma_lanes_on_symmetric_path(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._kv_cache_config = _parse_kv_cache_config(
            {"model": {"kvCache": {"enabled": True, "bits": 3, "keyBits": 8, "valueBits": 3}}}
        )
        first = FakeKVCache()
        backend._vlm_cache = FakeVlmCacheModule([first])
        backend._model = type(
            "FakeModel",
            (),
            {
                "config": type("FakeConfig", (), {"model_type": "qwen3"})(),
                "language_model": type(
                    "FakeLanguageModel",
                    (),
                    {"layers": [object()]},
                )(),
            },
        )()

        self.assertEqual(backend._make_request_prompt_cache(), [first])
        self.assertEqual(
            backend._kv_generation_kwargs(),
            {
                "kv_bits": 3,
                "kv_group_size": 64,
                "kv_quant_scheme": "turboquant",
                "quantized_kv_start": 2048,
            },
        )

    def test_deferred_explicit_turboquant_cache_converts_whole_cache_after_threshold(self):
        FakeExplicitTurboQuantKVCache.conversions = []
        base = FakeKVCache()
        cache = _DeferredExplicitTurboQuantKVCache(
            base,
            explicit_cache_cls=FakeExplicitTurboQuantKVCache,
            key_bits=3,
            value_bits=2,
            group_size=64,
            quantized_kv_start=4,
        )

        self.assertEqual(cache.update_and_fetch(FakeTensor(3), FakeTensor(3)), ("k3", "v3"))
        self.assertFalse(FakeExplicitTurboQuantKVCache.conversions)
        self.assertEqual(cache.update_and_fetch(FakeTensor(1), FakeTensor(1)), ("k4", "v4"))
        self.assertFalse(FakeExplicitTurboQuantKVCache.conversions)

        self.assertEqual(cache.state, ("k4", "v4"))
        self.assertEqual(
            FakeExplicitTurboQuantKVCache.conversions,
            [{"offset": 4, "key_bits": 3, "value_bits": 2, "group_size": 64}],
        )
        self.assertEqual(cache.update_and_fetch(FakeTensor(1), FakeTensor(1)), ("k5", "v5"))

    def _fake_streaming_backend(self, decoded_by_generated: dict[tuple[int, ...], str]) -> MlxVlmTurboQuantBackend:
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._max_output_tokens = 64
        backend._load_ms = 5
        backend._load_consumed = False
        backend._kv_cache_config = _parse_kv_cache_config({"model": {"kvCache": {"enabled": True}}})
        backend._session_caches = {"sid": ["session-cache"]}
        backend._prepare_request = lambda _messages, tools=None: type("Prepared", (), {"cleanup": lambda self: None})()
        backend._make_request_prompt_cache = lambda: ["request-cache"]
        backend._cache_offsets = lambda _cache: [42]
        backend._max_cache_offset = lambda _cache: 42
        backend._decode_generated_tokens = lambda generated: decoded_by_generated[tuple(generated)]

        def fake_iter_step_loop(_prepared, prompt_cache, _max_tokens, *, kv_generation_kwargs=None, failure_code="mlx_vlm_generation_failed"):
            self.assertEqual(prompt_cache, ["request-cache"])
            for token_id in range(1, len(decoded_by_generated) + 1):
                yield StepLoopToken(token_id=token_id, generated=tuple(range(1, token_id + 1)))
            final_generated = tuple(range(1, len(decoded_by_generated) + 1))
            yield StepLoopResult(
                generated=final_generated,
                raw_text=decoded_by_generated[final_generated],
                finish_reason="length",
                prompt_tokens=11,
                input_ids=None,
                first_token_ms=12,
                total_ms=34,
                peak_memory_gb=4.5,
                prompt_cache=["request-cache"],
            )

        backend._iter_step_loop = fake_iter_step_loop
        return backend

    def test_turboquant_stream_generate_emits_text_chunks_without_touching_sessions(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        decoded = {tuple(range(1, index + 1)): text[:index] for index in range(1, len(text) + 1)}
        backend = self._fake_streaming_backend(decoded)

        events = list(backend.stream_generate([{"role": "user", "content": "spell"}], max_tokens=64))

        chunks = [event.text for event in events if isinstance(event, BackendStreamChunk)]
        final = events[-1]
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertIsInstance(final, BackendResult)
        self.assertEqual(final.content, text)
        self.assertEqual(final.metrics["prefill_ms"], 12)
        self.assertEqual(final.metrics["generation_ms"], 34)
        self.assertEqual(final.metrics["stream_sentinel_set"][0], "call:")
        self.assertEqual(backend._session_caches, {"sid": ["session-cache"]})

    def test_turboquant_stream_generate_buffers_tool_calls_until_final(self):
        raw = "call:record_answer{answer:ok,city:dc}"
        decoded = {tuple(range(1, index + 1)): raw[:index] for index in range(1, len(raw) + 1)}
        backend = self._fake_streaming_backend(decoded)
        tools = [{"type": "function", "function": {"name": "record_answer", "parameters": {"type": "object", "properties": {}}}}]

        events = list(backend.stream_generate([{"role": "user", "content": "record"}], max_tokens=64, tools=tools))

        self.assertFalse([event for event in events if isinstance(event, BackendStreamChunk)])
        final = events[-1]
        self.assertIsInstance(final, BackendResult)
        self.assertEqual(final.content, "")
        self.assertEqual(final.finish_reason, "tool_calls")
        self.assertEqual(final.tool_calls[0]["function"]["name"], "record_answer")
        self.assertEqual(json.loads(final.tool_calls[0]["function"]["arguments"]), {"answer": "ok", "city": "dc"})

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

    def test_qwen3_tts_backend_selects_speech_descriptor(self):
        cfg = config(backend="mlx_audio_qwen3_tts")

        descriptor = backend_descriptor(cfg)

        self.assertEqual(configured_backend_id(cfg), "mlx_audio_qwen3_tts")
        self.assertEqual(descriptor.backend_id, "mlx_audio_qwen3_tts")
        self.assertEqual(descriptor.display_name, "MLX-Audio Qwen3-TTS backend")
        self.assertEqual(backend_supported_modalities(cfg), {"text"})

    def test_stub_mode_overrides_backend_selection(self):
        cfg = config(backend="mlx_vlm_turboquant", stub_mode=True)

        self.assertEqual(configured_backend_id(cfg), "stub")
        self.assertEqual(backend_descriptor(cfg).backend_id, "stub")
        self.assertEqual(backend_supported_modalities(cfg), {"text"})

    def test_backend_classes_declare_capabilities(self):
        self.assertEqual(StubBackend.supported_modalities_for_config(), {"text"})
        self.assertEqual(MlxVlmTurboQuantBackend.supported_modalities_for_config(), {"text", "image", "audio"})
        self.assertEqual(MlxVlmDiffusionGemmaBackend.supported_modalities_for_config(), {"text", "image"})
        self.assertEqual(Qwen3TtsBackend.supported_modalities_for_config(), {"text"})

    def test_build_backend_can_create_qwen3_tts_contract_backend(self):
        backend = build_backend(config(backend="mlx_audio_qwen3_tts"))

        self.assertIsInstance(backend, Qwen3TtsBackend)

    def test_qwen3_tts_rejects_misrouted_chat_generate_cleanly(self):
        backend = Qwen3TtsBackend(config(backend="mlx_audio_qwen3_tts"))

        with self.assertRaisesRegex(RuntimeError, "unsupported_request_family:Qwen3-TTS backend only supports speech.generate"):
            backend.generate([{"role": "user", "content": "say this"}], max_tokens=8)

    def test_strip_channel_markup_removes_partial_gemma_turn_markers(self):
        self.assertEqual(strip_channel_markup("Amber 7\n<end_of_turn<end"), "Amber 7")
        self.assertEqual(strip_channel_markup("cobalt lantern 42<end"), "cobalt lantern 42")
        self.assertEqual(strip_channel_markup("cobalt lantern 42<"), "cobalt lantern 42")
        self.assertEqual(strip_channel_markup("visit <endpoint> now"), "visit <endpoint> now")
        self.assertEqual(strip_channel_markup("4<end_of_turn>\n<start_of_turn>user\nWhat was asked?"), "4\n\nWhat was asked?")

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

    def test_qwen3_tts_model_info_reads_supported_speakers(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "talker_config": {
                            "spk_id": {"aiden": 2861, "vivian": 3065},
                        },
                        "tts_model_type": "custom_voice",
                    }
                )
            )
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {"modelPath": str(model_dir), "maxInputChars": 120}
            backend = Qwen3TtsBackend(cfg)

            info = backend.model_info()

            self.assertEqual(info["backendId"], "mlx_audio_qwen3_tts")
            self.assertEqual(info["family"], "tts")
            self.assertEqual(info["modes"], ["speech.generate"])
            self.assertEqual(info["sampleRateHz"], 24000)
            self.assertEqual(info["outputFormats"], ["wav"])
            self.assertTrue(Path(info["outputDir"]).is_absolute())
            self.assertEqual(info["supportedSpeakers"], ["aiden", "vivian"])
            self.assertEqual(info["defaultSpeaker"], "aiden")
            self.assertEqual(info["maxInputChars"], 120)
            self.assertTrue(info["modelExists"])
            self.assertGreater(info["modelSizeBytes"], 0)

    def test_qwen3_tts_model_info_supports_legacy_codec_speaker_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "codec_config": {
                            "spk_id": {"aiden": 2861, "serena": 3066},
                        },
                    }
                )
            )
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {"modelPath": str(model_dir)}
            backend = Qwen3TtsBackend(cfg)

            self.assertEqual(backend.model_info()["supportedSpeakers"], ["aiden", "serena"])

    def test_qwen3_tts_relative_paths_resolve_against_configured_service_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            service_root = Path(tmp)
            model_dir = service_root / "models/qwen"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"aiden": 2861}}}))
            runtime_python = service_root / "runtime/qwen3-tts-smoke/.venv/bin/python"

            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {
                "serviceRoot": str(service_root),
                "modelPath": "models/qwen",
                "pythonExecutable": "runtime/qwen3-tts-smoke/.venv/bin/python",
                "outputDir": "var/../qwen3-tts-output",
            }
            backend = Qwen3TtsBackend(cfg)

            info = backend.model_info()

            self.assertEqual(Path(info["serviceRoot"]), service_root)
            self.assertEqual(Path(info["modelPath"]), model_dir)
            self.assertEqual(Path(info["pythonExecutable"]), runtime_python)
            self.assertEqual(Path(info["outputDir"]), service_root / "qwen3-tts-output")
            self.assertTrue(info["modelExists"])

    def test_qwen3_tts_service_root_can_come_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("MLX_GEMMA_SERVICE_ROOT")
            os.environ["MLX_GEMMA_SERVICE_ROOT"] = tmp
            try:
                backend = Qwen3TtsBackend(config(backend="mlx_audio_qwen3_tts"))
            finally:
                if previous is None:
                    os.environ.pop("MLX_GEMMA_SERVICE_ROOT", None)
                else:
                    os.environ["MLX_GEMMA_SERVICE_ROOT"] = previous

            info = backend.model_info()

            self.assertEqual(Path(info["serviceRoot"]), Path(tmp))
            self.assertEqual(Path(info["outputDir"]), Path(tmp) / "tmp/qwen3-tts-output")
            self.assertEqual(Path(info["pythonExecutable"]), Path(tmp) / "runtime/qwen3-tts-smoke/.venv/bin/python")

    def test_qwen3_tts_estimate_memory_shape_uses_phase1_baseline(self):
        backend = Qwen3TtsBackend(config(backend="mlx_audio_qwen3_tts"))

        estimate = backend.estimate_memory({"input": "hello", "speaker": "aiden"})

        self.assertEqual(estimate["backendId"], "mlx_audio_qwen3_tts")
        self.assertEqual(estimate["baselinePeakMemoryGb"], 5.23)
        self.assertGreaterEqual(estimate["estimatedPeakMemoryGb"], 6.0)
        self.assertGreater(estimate["recommendedMinFreeMemoryGb"], estimate["estimatedPeakMemoryGb"])
        self.assertEqual(estimate["confidence"], "low")
        self.assertEqual(estimate["request"]["inputChars"], 5)
        self.assertIn("decodedAudioBuffers", estimate["components"])

    def test_qwen3_tts_structured_speech_error_shapes(self):
        cfg = config(backend="mlx_audio_qwen3_tts")
        cfg["speech"] = {"maxInputChars": 5}
        backend = Qwen3TtsBackend(cfg)

        missing = backend.validate_speech_request({})
        too_long = backend.validate_speech_request({"input": "too long"})
        bad_speaker = backend.validate_speech_request({"input": "hello", "speaker": "Ethan"})
        bad_format = backend.validate_speech_request({"input": "hello", "speaker": "aiden", "format": "mp3"})
        missing_ref = backend.validate_speech_request({"input": "hello", "speaker": "aiden", "referenceAudioPath": "/tmp/missing-reference.wav"})
        bad_ref_text = backend.validate_speech_request({"input": "hello", "speaker": "aiden", "referenceText": 42})
        bad_speed = backend.validate_speech_request({"input": "hello", "speaker": "aiden", "postprocessSpeed": 0.1})
        timeout = backend.speech_timeout_error(120)

        self.assertEqual(missing["code"], "bad_request")
        self.assertEqual(missing["status"], 400)
        self.assertEqual(too_long["code"], "input_too_long")
        self.assertEqual(too_long["status"], 413)
        self.assertEqual(too_long["details"]["maxInputChars"], 5)
        self.assertEqual(bad_speaker["code"], "unsupported_speaker")
        self.assertEqual(bad_speaker["status"], 422)
        self.assertIn("supportedSpeakers", bad_speaker["details"])
        self.assertEqual(bad_format["code"], "unsupported_format")
        self.assertEqual(bad_format["status"], 415)
        self.assertEqual(missing_ref["code"], "reference_audio_not_found")
        self.assertEqual(missing_ref["status"], 400)
        self.assertEqual(bad_ref_text["code"], "bad_request")
        self.assertEqual(bad_ref_text["status"], 400)
        self.assertEqual(bad_speed["code"], "bad_request")
        self.assertEqual(bad_speed["details"]["field"], "postprocessSpeed")
        self.assertEqual(timeout["code"], "timeout")
        self.assertEqual(timeout["status"], 504)
        self.assertEqual(timeout["details"]["timeoutSeconds"], 120)

    def test_qwen3_tts_speech_generate_returns_load_failed_when_model_missing(self):
        cfg = config(backend="mlx_audio_qwen3_tts")
        cfg["speech"] = {"modelPath": "/tmp/definitely-missing-qwen3-tts-model"}
        backend = Qwen3TtsBackend(cfg)

        result = backend.speech_generate({"input": "hello", "speaker": "aiden"})

        self.assertEqual(result["type"], "speech_error")
        self.assertEqual(result["error"]["code"], "load_failed")
        self.assertEqual(result["error"]["status"], 503)

    def test_qwen3_tts_speech_generate_returns_local_wav_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"aiden": 2861}}}))
            output_dir = root / "out"
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {
                "modelPath": str(model_dir),
                "pythonExecutable": sys.executable,
                "outputDir": str(output_dir),
                "retentionMaxFiles": 2,
            }
            backend = Qwen3TtsBackend(cfg)
            old_keep = output_dir / "speech_old_keep.wav"
            old_delete = output_dir / "speech_old_delete.wav"
            output_dir.mkdir(parents=True)
            write_wav(old_keep)
            write_wav(old_delete)
            os.utime(old_keep, (20, 20))
            os.utime(old_delete, (10, 10))

            def fake_run(command: list[str], timeout_seconds: int | float) -> subprocess.CompletedProcess[str]:
                out_path = Path(command[command.index("--output_path") + 1])
                prefix = command[command.index("--file_prefix") + 1]
                voice = command[command.index("--voice") + 1]
                out_path.mkdir(parents=True, exist_ok=True)
                write_wav(out_path / f"{prefix}-{voice}_000.wav")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            backend._run_mlx_audio = fake_run  # type: ignore[method-assign]

            result = backend.speech_generate({"input": "hello", "speaker": "aiden", "timeoutSeconds": 5})

            self.assertEqual(result["type"], "speech_result")
            self.assertEqual(result["object"], "audio.speech")
            self.assertEqual(result["backend"], "mlx_audio_qwen3_tts")
            self.assertEqual(result["format"], "wav")
            self.assertEqual(result["sampleRateHz"], 24000)
            self.assertEqual(result["durationSeconds"], 0.1)
            self.assertTrue(Path(result["audioPath"]).exists())
            self.assertGreater(result["fileSizeBytes"], 0)
            self.assertEqual(result["metrics"]["generatedAudioSeconds"], 0.1)
            self.assertTrue(old_keep.exists())
            self.assertFalse(old_delete.exists())

    def test_qwen3_tts_speech_generate_passes_style_instruction_to_mlx_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"aiden": 2861}}}))
            output_dir = root / "out"
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {
                "modelPath": str(model_dir),
                "pythonExecutable": sys.executable,
                "outputDir": str(output_dir),
            }
            backend = Qwen3TtsBackend(cfg)
            captured: dict[str, Any] = {}

            def fake_run(command: list[str], timeout_seconds: int | float) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                out_path = Path(command[command.index("--output_path") + 1])
                prefix = command[command.index("--file_prefix") + 1]
                out_path.mkdir(parents=True, exist_ok=True)
                write_wav(out_path / f"{prefix}_000.wav")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            backend._run_mlx_audio = fake_run  # type: ignore[method-assign]

            result = backend.speech_generate(
                {
                    "input": "Wait, that cannot be right.",
                    "speaker": "aiden",
                    "instruct": "Speak in an incredulous tone, with a hint of panic.",
                    "timeoutSeconds": 5,
                }
            )

            self.assertEqual(result["type"], "speech_result")
            command = captured["command"]
            self.assertIn("--instruct", command)
            self.assertEqual(command[command.index("--instruct") + 1], "Speak in an incredulous tone, with a hint of panic.")

    def test_qwen3_tts_speech_generate_accepts_nested_voice_style_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"aiden": 2861}}}))
            output_dir = root / "out"
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {
                "modelPath": str(model_dir),
                "pythonExecutable": sys.executable,
                "outputDir": str(output_dir),
            }
            backend = Qwen3TtsBackend(cfg)
            captured: dict[str, Any] = {}

            def fake_run(command: list[str], timeout_seconds: int | float) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                out_path = Path(command[command.index("--output_path") + 1])
                prefix = command[command.index("--file_prefix") + 1]
                out_path.mkdir(parents=True, exist_ok=True)
                write_wav(out_path / f"{prefix}_000.wav")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            backend._run_mlx_audio = fake_run  # type: ignore[method-assign]

            result = backend.speech_generate(
                {
                    "input": "This should sound warmer.",
                    "speaker": "aiden",
                    "voice": {"styleInstruction": "Speak warmly and gently."},
                    "timeoutSeconds": 5,
                }
            )

            self.assertEqual(result["type"], "speech_result")
            command = captured["command"]
            self.assertIn("--instruct", command)
            self.assertEqual(command[command.index("--instruct") + 1], "Speak warmly and gently.")

    def test_qwen3_tts_speech_generate_passes_reference_audio_to_mlx_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"aiden": 2861}}}))
            output_dir = root / "out"
            reference_audio = root / "reference.wav"
            write_wav(reference_audio)
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {
                "modelPath": str(model_dir),
                "pythonExecutable": sys.executable,
                "outputDir": str(output_dir),
            }
            backend = Qwen3TtsBackend(cfg)
            captured: dict[str, Any] = {}

            def fake_run(command: list[str], timeout_seconds: int | float) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                out_path = Path(command[command.index("--output_path") + 1])
                prefix = command[command.index("--file_prefix") + 1]
                out_path.mkdir(parents=True, exist_ok=True)
                write_wav(out_path / f"{prefix}_000.wav")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            backend._run_mlx_audio = fake_run  # type: ignore[method-assign]

            result = backend.speech_generate(
                {
                    "input": "Use the local reference audio style.",
                    "speaker": "aiden",
                    "referenceAudioPath": str(reference_audio),
                    "referenceText": "This is the reference transcript.",
                    "timeoutSeconds": 5,
                }
            )

            self.assertEqual(result["type"], "speech_result")
            command = captured["command"]
            self.assertIn("--ref_audio", command)
            self.assertEqual(Path(command[command.index("--ref_audio") + 1]), reference_audio)
            self.assertIn("--ref_text", command)
            self.assertEqual(command[command.index("--ref_text") + 1], "This is the reference transcript.")

    def test_qwen3_tts_base_model_omits_voice_and_forwards_generation_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "base-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}))
            output_dir = root / "out"
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["speech"] = {
                "modelPath": str(model_dir),
                "pythonExecutable": sys.executable,
                "outputDir": str(output_dir),
            }
            backend = Qwen3TtsBackend(cfg)
            captured: dict[str, Any] = {}

            def fake_run(command: list[str], timeout_seconds: int | float) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                out_path = Path(command[command.index("--output_path") + 1])
                prefix = command[command.index("--file_prefix") + 1]
                out_path.mkdir(parents=True, exist_ok=True)
                write_wav(out_path / f"{prefix}_000.wav", leading_silence=240, frames=2_400, trailing_silence=21_600, active=True)
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            backend._run_mlx_audio = fake_run  # type: ignore[method-assign]

            result = backend.speech_generate(
                {
                    "input": "Use the reference voice.",
                    "postprocessSpeed": 1.25,
                    "genDuration": 22,
                    "durationMultiplier": 1.1,
                    "maxTokens": 4096,
                    "timeoutSeconds": 5,
                }
            )

            self.assertEqual(result["type"], "speech_result")
            command = captured["command"]
            self.assertNotIn("--voice", command)
            self.assertNotIn("--speed", command)
            self.assertEqual(command[command.index("--gen_duration") + 1], "22")
            self.assertEqual(command[command.index("--duration_multiplier") + 1], "1.1")
            self.assertEqual(command[command.index("--max_tokens") + 1], "4096")
            postprocess = result["metrics"]["postprocessSpeed"]
            self.assertTrue(postprocess["enabled"])
            self.assertEqual(postprocess["speed"], 1.25)
            self.assertAlmostEqual(postprocess["expectedActiveDurationSeconds"], 0.08, places=3)
            self.assertAlmostEqual(postprocess["outputActiveDurationSeconds"], 0.08, places=3)
            self.assertAlmostEqual(result["durationSeconds"], 0.808, places=3)
            self.assertGreater(postprocess["trimmedTrailingSeconds"], 0.8)
            self.assertTrue(postprocess["activeDurationWithinTolerance"])
            self.assertFalse(postprocess["clippingIntroduced"])

    def test_qwen3_tts_speech_generate_rejects_timeout_above_worker_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"aiden": 2861}}}))
            cfg = config(backend="mlx_audio_qwen3_tts")
            cfg["worker"]["requestTimeoutMs"] = 10_000
            cfg["speech"] = {
                "modelPath": str(model_dir),
                "pythonExecutable": sys.executable,
            }
            backend = Qwen3TtsBackend(cfg)

            result = backend.speech_generate({"input": "hello", "speaker": "aiden", "timeoutSeconds": 30})

            self.assertEqual(result["type"], "speech_error")
            self.assertEqual(result["error"]["code"], "timeout_too_high")
            self.assertEqual(result["error"]["status"], 400)
            self.assertLessEqual(result["error"]["details"]["maxTimeoutSeconds"], 9.0)

    def test_worker_speech_generate_dispatches_to_backend(self):
        class FakeSpeechBackend:
            def speech_generate(self, request: dict[str, Any]) -> dict[str, Any]:
                return {"type": "speech_result", "id": "speech_test", "input": request["input"]}

        result = handle_speech_generate(FakeSpeechBackend(), {"request_id": "r1", "request": {"input": "hello"}})

        self.assertEqual(result["type"], "speech_result")
        self.assertEqual(result["request_id"], "r1")
        self.assertEqual(result["id"], "speech_test")

    def test_validate_speech_request_rejects_streaming_and_unknown_fields(self):
        self.assertIsNone(validate_speech_request({"input": "hello", "postprocessSpeed": 1.1}))
        self.assertEqual(validate_speech_request({"input": "hello", "stream": True}), (400, "unsupported_streaming", "Qwen3-TTS streaming is not wired yet"))
        self.assertEqual(validate_speech_request({"input": "hello", "postprocessSpeed": "fast"}), (400, "bad_request", "Field 'postprocessSpeed' must be numeric when provided"))
        unknown = validate_speech_request({"input": "hello", "nonsense": True})
        self.assertEqual(unknown, (400, "bad_request", "Unsupported field(s): nonsense"))

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

    def test_worker_generate_carries_options_and_reasoning_content(self):
        class FakeReasoningBackend:
            def supported_modalities(self) -> set[str]:
                return {"text"}

            def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools=None, options=None) -> BackendResult:
                self.options = options
                return BackendResult(
                    content="final",
                    reasoning_content="private",
                    finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    metrics={"think_control": {"level": "medium"}},
                )

        backend = FakeReasoningBackend()
        result = handle_generate(
            backend,
            {
                "request_id": "r1",
                "messages": [{"role": "user", "content": "hello"}],
                "options": {"reasoning_effort": "medium"},
            },
        )

        self.assertEqual(backend.options, {"reasoning_effort": "medium"})
        self.assertEqual(result["content"], "final")
        self.assertEqual(result["reasoning_content"], "private")

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
                    "prefill_step_size": 2048,
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
                "prefill_step_size": 2048,
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

    def test_vlm_turboquant_maps_generic_think_level_to_gemma_template_kwargs(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._config = {"model": {"thinking": {"defaultLevel": "off", "budgets": {"high": 4096}}}}
        backend._model = type("Model", (), {"config": type("Cfg", (), {"model_type": "gemma4"})()})()
        captured: dict[str, Any] = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured.update(kwargs)
                return "templated prompt"

        backend._processor = type("Processor", (), {"tokenizer": FakeTokenizer()})()
        control = backend._resolve_think_control({"thinkLevel": "high"})

        prompt = backend._build_prompt([{"role": "user", "content": "solve"}], think_control=control)

        self.assertEqual(prompt, "templated prompt")
        self.assertTrue(captured["enable_thinking"])
        self.assertEqual(captured["thinking_budget"], 4096)
        self.assertEqual(control.level, "high")
        self.assertEqual(control.mechanism, "apply_chat_template(enable_thinking, thinking_budget)")

    def test_vlm_turboquant_maps_qwen_think_levels_to_prompt_directives(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._config = {"model": {"thinking": {"defaultLevel": "off"}}}
        backend._model = type("Model", (), {"config": type("Cfg", (), {"model_type": "qwen3"})()})()
        captured: dict[str, Any] = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                return "templated prompt"

        backend._processor = type("Processor", (), {"tokenizer": FakeTokenizer()})()
        control = backend._resolve_think_control({"thinkLevel": "low"})

        backend._build_prompt([{"role": "user", "content": "solve"}], think_control=control)

        self.assertEqual(control.family, "qwen")
        self.assertEqual(control.mechanism, "prompt_directive(/think|/no_think)")
        self.assertTrue(captured["messages"][0]["content"].endswith("/think"))

    def test_reasoning_markup_split_preserves_openclaw_reasoning_field(self):
        visible, reasoning = split_reasoning_markup("<|channel|>thought\nprivate notes<channel|>final answer")

        self.assertEqual(visible, "final answer")
        self.assertEqual(reasoning, "private notes")

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

    def test_vlm_turboquant_accepts_bare_string_tool_arg_values(self):
        backend = MlxVlmTurboQuantBackend.__new__(MlxVlmTurboQuantBackend)
        backend._model = object()
        backend._processor = object()
        backend._max_output_tokens = 32
        backend._sampling_kwargs = {}
        backend._load_ms = 0
        backend._load_consumed = False
        backend._generate = lambda *args, **kwargs: "call:record_answer{answer:ok,city:dc}"
        tools = [{"type": "function", "function": {"name": "record_answer", "parameters": {"type": "object", "properties": {}}}}]

        result = backend.generate([{"role": "user", "content": "record"}], max_tokens=12, tools=tools)

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.content, "")
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "record_answer")
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"answer": "ok", "city": "dc"})

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
