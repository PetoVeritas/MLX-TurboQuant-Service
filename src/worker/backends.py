"""Worker backend contract and Phase 1 backends."""

from __future__ import annotations

import base64
import binascii
import json
import os
import resource
import re
import subprocess
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.backend_adapters import BACKEND_ADAPTERS, backend_descriptor, configured_backend_id, turboquant_supported_modalities


@dataclass
class BackendResult:
    content: str
    finish_reason: str
    usage: dict[str, int]
    metrics: dict[str, Any]
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


@dataclass
class BackendStreamChunk:
    text: str
    reasoning_content: str | None = None


@dataclass(frozen=True)
class StepLoopToken:
    token_id: int
    generated: tuple[int, ...]


@dataclass(frozen=True)
class StepLoopResult:
    generated: tuple[int, ...]
    raw_text: str
    finish_reason: str
    prompt_tokens: int
    input_ids: Any | None
    first_token_ms: int | None
    total_ms: int
    peak_memory_gb: float
    prompt_cache: list[Any]


@dataclass(frozen=True)
class SpeechError:
    code: str
    message: str
    status: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "status": self.status,
            "details": self.details,
        }


@dataclass
class VlmPreparedRequest:
    prompt: str
    image_paths: list[str]
    audio_paths: list[str]
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    add_special_tokens: bool | None = None
    think_control: dict[str, Any] | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


@dataclass(frozen=True)
class KvCacheConfig:
    enabled: bool = False
    bits: int = 4
    key_bits: int | None = None
    value_bits: int | None = None
    group_size: int = 64
    quantized_kv_start: int = 2048
    quant_scheme: str = "turboquant"

    @property
    def effective_key_bits(self) -> int:
        return self.key_bits if self.key_bits is not None else self.bits

    @property
    def effective_value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def generation_kwargs(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "kv_bits": self.bits,
            "kv_group_size": self.group_size,
            "kv_quant_scheme": self.quant_scheme,
            "quantized_kv_start": self.quantized_kv_start,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "bits": self.bits,
            "keyBits": self.effective_key_bits,
            "valueBits": self.effective_value_bits,
            "groupSize": self.group_size,
            "quantizedKvStart": self.quantized_kv_start,
            "quantScheme": self.quant_scheme,
            "appliesTo": ["blocking_generate", "future_streaming_generate"] if self.enabled else [],
            "excluded": ["si_drone_session_caches"],
            "asymmetricSupported": True,
        }


_KV_CACHE_CONFIG_KEYS = frozenset(
    {"enabled", "bits", "keyBits", "valueBits", "groupSize", "quantizedKvStart", "quantScheme"}
)


def _parse_kv_cache_config(config: dict[str, Any]) -> KvCacheConfig:
    raw = config.get("model", {}).get("kvCache", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError("invalid_kv_cache_config:kvCache_must_be_object")

    unknown = sorted(set(raw) - _KV_CACHE_CONFIG_KEYS)
    if unknown:
        raise RuntimeError("invalid_kv_cache_config:unsupported_field:" + ",".join(unknown))

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise RuntimeError("invalid_kv_cache_config:enabled_must_be_boolean")
    bits_present = "bits" in raw
    bits = raw.get("bits", 4)
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise RuntimeError("invalid_kv_cache_config:bits_must_be_integer")
    if bits not in {3, 4, 8}:
        raise RuntimeError("invalid_kv_cache_config:bits_must_be_3_4_or_8")

    key_bits = raw.get("keyBits")
    if key_bits is not None:
        if isinstance(key_bits, bool) or not isinstance(key_bits, int):
            raise RuntimeError("invalid_kv_cache_config:keyBits_must_be_integer")
        if key_bits < 3:
            raise RuntimeError("invalid_kv_cache_config:keyBits_below_min_3")
        if key_bits not in {3, 4, 8}:
            raise RuntimeError("invalid_kv_cache_config:keyBits_must_be_3_4_or_8")

    value_bits = raw.get("valueBits")
    if value_bits is not None:
        if isinstance(value_bits, bool) or not isinstance(value_bits, int):
            raise RuntimeError("invalid_kv_cache_config:valueBits_must_be_integer")
        if value_bits < 2:
            raise RuntimeError("invalid_kv_cache_config:valueBits_below_min_2")
        if value_bits not in {2, 3, 4, 8}:
            raise RuntimeError("invalid_kv_cache_config:valueBits_must_be_2_3_4_or_8")

    if enabled and key_bits is None and value_bits is None and not bits_present:
        key_bits = 3
        value_bits = 2

    group_size = raw.get("groupSize", 64)
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise RuntimeError("invalid_kv_cache_config:groupSize_must_be_integer")
    if group_size <= 0 or group_size % 32 != 0:
        raise RuntimeError("invalid_kv_cache_config:groupSize_must_be_positive_multiple_of_32")

    quantized_kv_start = raw.get("quantizedKvStart", 2048)
    if isinstance(quantized_kv_start, bool) or not isinstance(quantized_kv_start, int):
        raise RuntimeError("invalid_kv_cache_config:quantizedKvStart_must_be_integer")
    if quantized_kv_start <= 0:
        raise RuntimeError("invalid_kv_cache_config:quantizedKvStart_must_be_positive")

    quant_scheme = raw.get("quantScheme", "turboquant")
    if quant_scheme != "turboquant":
        raise RuntimeError("invalid_kv_cache_config:quantScheme_must_be_turboquant")

    return KvCacheConfig(
        enabled=enabled,
        bits=bits,
        key_bits=key_bits,
        value_bits=value_bits,
        group_size=group_size,
        quantized_kv_start=quantized_kv_start,
        quant_scheme=quant_scheme,
    )


def _ensure_kv_cache_config(backend: Any) -> KvCacheConfig:
    config = getattr(backend, "_kv_cache_config", None)
    if isinstance(config, KvCacheConfig):
        return config
    config = KvCacheConfig()
    setattr(backend, "_kv_cache_config", config)
    return config


class _DeferredExplicitTurboQuantKVCache:
    def __init__(
        self,
        cache: Any,
        *,
        explicit_cache_cls: Any,
        key_bits: int,
        value_bits: int,
        group_size: int,
        quantized_kv_start: int,
        seed: int | None = None,
    ) -> None:
        self._cache = cache
        self._explicit_cache_cls = explicit_cache_cls
        self._key_bits = key_bits
        self._value_bits = value_bits
        self._group_size = group_size
        self._quantized_kv_start = quantized_kv_start
        self._seed = seed
        self._pending_quantize = False

    def _is_explicit(self) -> bool:
        return isinstance(self._cache, self._explicit_cache_cls)

    def _quantize_if_ready(self) -> None:
        if self._is_explicit():
            self._pending_quantize = False
            return
        if int(getattr(self._cache, "offset", 0) or 0) < self._quantized_kv_start:
            return
        kwargs = {
            "key_bits": self._key_bits,
            "value_bits": self._value_bits,
            "group_size": self._group_size,
        }
        if self._seed is not None:
            kwargs["seed"] = self._seed
        self._cache = self._explicit_cache_cls.from_cache(self._cache, **kwargs)
        self._pending_quantize = False

    def update_and_fetch(self, keys: Any, values: Any) -> Any:
        if self._pending_quantize:
            self._quantize_if_ready()
        result = self._cache.update_and_fetch(keys, values)
        if (
            not self._is_explicit()
            and int(getattr(self._cache, "offset", 0) or 0) >= self._quantized_kv_start
        ):
            # Defer conversion until after the current model forward has used
            # the fp16 cache. This is deferred-onset quantization, not a
            # persistent fp16 attention sink.
            self._pending_quantize = True
        return result

    @property
    def offset(self) -> Any:
        return getattr(self._cache, "offset", 0)

    @property
    def keys(self) -> Any:
        return getattr(self._cache, "keys", None)

    @property
    def values(self) -> Any:
        return getattr(self._cache, "values", None)

    @property
    def state(self) -> Any:
        if self._pending_quantize:
            self._quantize_if_ready()
        return self._cache.state

    @state.setter
    def state(self, value: Any) -> None:
        self._cache.state = value
        self._pending_quantize = False

    @property
    def nbytes(self) -> int:
        return int(getattr(self._cache, "nbytes", 0) or 0)

    def size(self) -> Any:
        size = getattr(self._cache, "size", None)
        return size() if callable(size) else self.offset

    def empty(self) -> bool:
        empty = getattr(self._cache, "empty", None)
        return bool(empty()) if callable(empty) else self.keys is None

    def is_trimmable(self) -> bool:
        is_trimmable = getattr(self._cache, "is_trimmable", None)
        return bool(is_trimmable()) if callable(is_trimmable) else False

    def trim(self, n: int) -> Any:
        trim = getattr(self._cache, "trim")
        return trim(n)

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._cache.make_mask(*args, **kwargs)

    @staticmethod
    def _attention_arg(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        index: int,
        name: str,
        default: Any = None,
    ) -> Any:
        if name in kwargs:
            return kwargs[name]
        if len(args) > index:
            return args[index]
        return default

    def _fp16_attention(self, *args: Any, **kwargs: Any) -> Any:
        queries = self._attention_arg(args, kwargs, 0, "queries")
        keys_state = self._attention_arg(args, kwargs, 1, "keys_state")
        values_state = self._attention_arg(args, kwargs, 2, "values_state")
        scale = self._attention_arg(args, kwargs, 3, "scale", 1.0)
        mask = self._attention_arg(args, kwargs, 4, "mask")
        if queries is None or keys_state is None or values_state is None:
            return None
        import mlx.core as mx  # type: ignore

        return mx.fast.scaled_dot_product_attention(
            queries,
            keys_state.astype(queries.dtype),
            values_state.astype(queries.dtype),
            scale=scale,
            mask=mask,
        )

    def decode_attention(self, *args: Any, **kwargs: Any) -> Any:
        if self._pending_quantize:
            self._quantize_if_ready()
        if self._is_explicit():
            return self._cache.decode_attention(*args, **kwargs)
        return self._fp16_attention(*args, **kwargs)

    def prefill_attention(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_explicit():
            return self._cache.prefill_attention(*args, **kwargs)
        return self._fp16_attention(*args, **kwargs)

    def dequantize(self, keys_state: Any = None, values_state: Any = None) -> Any:
        if self._is_explicit():
            return self._cache.dequantize(keys_state, values_state)
        return keys_state, values_state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


MEMORY_UNITS = "GiB"
MEMORY_METAL_RECONCILE_TOLERANCE_GB = 0.25


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / float(1024**3), 6)


def _current_rss_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return _bytes_to_gib(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports ru_maxrss in bytes; Linux reports KiB. This service
            # runs on macOS, but keep the fallback sane if tests run elsewhere.
            if usage > 10 * 1024**3:
                return _bytes_to_gib(usage)
            return _bytes_to_gib(usage * 1024)
        except Exception:
            return None


def _mlx_metal_bytes(function_name: str) -> int | None:
    try:
        import mlx.core as mx  # type: ignore

        fn = getattr(mx, function_name, None)
        metal = getattr(mx, "metal", None)
        if fn is None and metal is not None:
            fn = getattr(metal, function_name, None)
        if callable(fn):
            return int(fn())
    except Exception:
        return None
    return None


def _reset_metal_peak_memory() -> None:
    try:
        import mlx.core as mx  # type: ignore

        fn = getattr(mx, "reset_peak_memory", None)
        metal = getattr(mx, "metal", None)
        if fn is None and metal is not None:
            fn = getattr(metal, "reset_peak_memory", None)
        if callable(fn):
            fn()
    except Exception:
        return


def _array_nbytes(value: Any) -> int:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return 0
    nbytes = getattr(value, "nbytes", None)
    if not isinstance(nbytes, int):
        return 0
    module = str(getattr(type(value), "__module__", ""))
    if module.startswith("mlx"):
        return nbytes
    return 0


def _sum_mlx_array_nbytes(value: Any, *, _seen: set[int] | None = None) -> int:
    if value is None:
        return 0
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return 0
    _seen.add(value_id)

    own_nbytes = _array_nbytes(value)
    if own_nbytes:
        return own_nbytes

    if isinstance(value, dict):
        return sum(_sum_mlx_array_nbytes(item, _seen=_seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_sum_mlx_array_nbytes(item, _seen=_seen) for item in value)

    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return sum(_sum_mlx_array_nbytes(item, _seen=_seen) for item in data.values())
    return 0


def _model_weights_nbytes(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return _sum_mlx_array_nbytes(parameters())
        except Exception:
            pass
    return _sum_mlx_array_nbytes(model)


def _memory_sample(
    sampling_point: str,
    *,
    model: Any | None = None,
    prompt_cache: Any | None = None,
) -> dict[str, Any]:
    weights_gb = _bytes_to_gib(_model_weights_nbytes(model)) if model is not None else None
    kv_gb = _bytes_to_gib(_sum_mlx_array_nbytes(prompt_cache)) if prompt_cache is not None else None
    active_gb = _bytes_to_gib(_mlx_metal_bytes("get_active_memory"))
    peak_gb = _bytes_to_gib(_mlx_metal_bytes("get_peak_memory"))
    cache_gb = _bytes_to_gib(_mlx_metal_bytes("get_cache_memory"))
    other_metal_gb = None
    if active_gb is not None:
        other_metal_gb = round(active_gb - float(weights_gb or 0.0) - float(kv_gb or 0.0), 6)
    return {
        "sampling_point": sampling_point,
        "units": MEMORY_UNITS,
        "rss_gb": _current_rss_gb(),
        "metal_active_gb": active_gb,
        "metal_peak_gb": peak_gb,
        "metal_cache_gb": cache_gb,
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
        "other_metal_gb": other_metal_gb,
    }


def _memory_stats(samples: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    return {
        "units": MEMORY_UNITS,
        "unit_note": "Fields ending in _gb are binary GiB (1024^3 bytes).",
        "metal_reconcile_tolerance_gb": MEMORY_METAL_RECONCILE_TOLERANCE_GB,
        "sampling_points": ["before_load", "after_load", "request_start", "prefill_peak", "post_generation", "idle", "after_unload"],
        "samples": {key: value for key, value in samples.items() if value is not None},
    }


def _ensure_memory_samples(backend: Any) -> dict[str, dict[str, Any] | None]:
    samples = getattr(backend, "_memory_samples", None)
    if not isinstance(samples, dict):
        samples = {
            "before_load": None,
            "after_load": None,
            "request_start": None,
            "prefill_peak": None,
            "post_generation": None,
            "idle": None,
            "after_unload": None,
        }
        setattr(backend, "_memory_samples", samples)
    return samples


_TOOL_SENTINEL_RE = re.compile(r"</?\|?tool_call\|?>")
_QUOTE_SENTINEL_RE = re.compile(r"<\|\"\|>")
_QUOTE_SENTINEL_PAIR_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_BARE_KEY_RE = re.compile(r'([\{,]\s*)([A-Za-z_]\w*)(\s*:)')
_BARE_STRING_VALUE_RE = re.compile(r'(:\s*)([A-Za-z_][\w.-]*)(\s*[,}])')
_PLACEHOLDER_RE = re.compile(r"\u0000MLXTG_STR_(\d+)\u0000")
STREAM_MARKER_SENTINELS = (
    "call:",
    "<|channel|>",
    "<|constrain|>",
    "<|tool_call|>",
    "<tool_call>",
    "</tool_call>",
    "<start_of_turn",
    "<end_of_turn",
)
STREAM_SENTINEL_HOLDBACK_CHARS = max(len(sentinel) for sentinel in STREAM_MARKER_SENTINELS)

# Harmony / Gemma-style channel markers. The model uses these internally to
# separate reasoning ("thought" / "thinking" / "analysis") from the
# user-visible answer ("final"). They're not registered as special tokens in
# every tokenizer, so they can leak into the decoded text. We strip them here
# and keep only the ``final`` channel's content when present.
#
# The regex deliberately restricts the channel name to a KNOWN set so we don't
# mis-match "The" in rendered text like ``<channel|>The forecast`` (a display
# artifact where a leading ``|`` got eaten). Unknown / unexpected names are
# left alone and later swept up by the defensive ``_CHANNEL_MARKER_RE`` pass.
_CHANNEL_NAMES = r"thought|thinking|analysis|final|commentary|reflection"
# Both pipes are optional on either side of the channel name. Gemma
# occasionally drops one of the pipes (seen in the wild: ``<|channel>thought``,
# ``<channel|>final``), and a strict ``<\|?channel\|>`` missed those cases
# and leaked raw markup to the client. Making both sides tolerant catches the
# full variant too without false positives on real text.
_CHANNEL_MARKER_RE = re.compile(rf"<\|?channel\|?>\s*({_CHANNEL_NAMES})\b", re.IGNORECASE)
# Liberal matcher for detection — catches unknown / partial / mangled names too.
_CHANNEL_MARKER_ANY_RE = re.compile(r"<\|?channel\|?>\s*[A-Za-z_][\w-]*\b")
# Defensive stripping regex: removes ONLY the marker characters, not any
# following word. Prevents the strict-unknown path (e.g. a display artifact
# like "<channel|>The forecast") from eating the first word of real content.
_CHANNEL_MARKER_BARE_RE = re.compile(r"<\|?channel\|?>")
_STRAY_CONTROL_RE = re.compile(
    r"<\|?(?:message|start|end|return)\|?>|"
    r"<end(?:_of_+turn)?(?:>|(?=<)|$)|"
    r"<start_of_turn>(?:user|model)?|"
    r"<$"
)
_HIDDEN_CHANNELS = frozenset({"thought", "thinking", "analysis"})
_THINK_LEVELS = frozenset({"off", "low", "medium", "high"})
_THINK_ALIASES = {
    "none": "off",
    "false": "off",
    "disabled": "off",
    "minimal": "low",
    "xhigh": "high",
    "max": "high",
}
_DEFAULT_THINK_BUDGETS = {"low": 512, "medium": 2048, "high": 8192}
_REASONING_TAG_BLOCK_RE = re.compile(r"<\s*(?:think|thinking|thought|reasoning)\b[^>]*>(.*?)<\s*/\s*(?:think|thinking|thought|reasoning)\s*>", re.IGNORECASE | re.DOTALL)
_HIDDEN_CHANNEL_BLOCK_RE = re.compile(r"<\|?channel\|?>\s*(thought|thinking|analysis)\b(.*?)(<\|?channel\|?>)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ThinkControl:
    level: str
    include_reasoning: bool
    family: str
    mechanism: str
    budget_tokens: int | None
    source: str
    requested_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "includeReasoning": self.include_reasoning,
            "family": self.family,
            "mechanism": self.mechanism,
            "budgetTokens": self.budget_tokens,
            "source": self.source,
            "requestedLevel": self.requested_level,
        }


def _extract_balanced_json(text: str, start: int) -> tuple[str | None, int]:
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
    return None, start


def _normalize_tool_args(raw_args: str) -> str:
    """Convert Gemma 4's loose JSON-ish tool-arg syntax into real JSON.

    Two-pass approach that mirrors upstream ``mlx_lm.tool_parsers.gemma4``:
      1. Replace every ``<|"|>...<|"|>`` segment with an opaque placeholder
         so bare-key quoting and arbitrary string contents (including literal
         ``"``, ``:``, ``{``, ``}``, newlines, etc.) cannot collide.
      2. Quote bare keys on the stripped skeleton.
      3. Reinsert each captured segment as a properly json-escaped string
         literal so it round-trips through json.loads unchanged.
    """

    captured: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        captured.append(match.group(1))
        return f"\u0000MLXTG_STR_{len(captured) - 1}\u0000"

    skeleton = _QUOTE_SENTINEL_PAIR_RE.sub(_capture, raw_args)
    # Fallback: any lone <|"|> sentinels outside a matched pair become plain quotes.
    skeleton = _QUOTE_SENTINEL_RE.sub('"', skeleton)
    skeleton = _BARE_KEY_RE.sub(r'\1"\2"\3', skeleton)

    def _quote_bare_value(match: re.Match[str]) -> str:
        value = match.group(2)
        if value in {"true", "false", "null"}:
            return match.group(0)
        return f"{match.group(1)}{json.dumps(value)}{match.group(3)}"

    skeleton = _BARE_STRING_VALUE_RE.sub(_quote_bare_value, skeleton)

    def _restore(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 0 <= index < len(captured):
            return json.dumps(captured[index])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_restore, skeleton)


def _parse_single_tool_call(
    normalized: str,
    match: re.Match[str],
) -> tuple[dict[str, Any] | None, int | None]:
    """Return (tool_call_dict, end_index) or (None, None) on parse failure."""

    function_name = match.group(1)
    brace_index = match.end(1)
    # Allow optional whitespace between name and opening brace.
    while brace_index < len(normalized) and normalized[brace_index].isspace():
        brace_index += 1
    if brace_index >= len(normalized) or normalized[brace_index] != "{":
        return None, None

    raw_args, end_index = _extract_balanced_json(normalized, brace_index)
    if raw_args is None:
        return None, None
    try:
        parsed_args = json.loads(_normalize_tool_args(raw_args))
    except (json.JSONDecodeError, ValueError):
        return None, None

    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(parsed_args, separators=(",", ":")),
        },
    }
    return tool_call, end_index


_CALL_TOKEN_RE = re.compile(r"call:([A-Za-z_][\w\-]*)")


def parse_tool_calls(content: str) -> tuple[str, str, list[dict[str, Any]] | None]:
    """Parse any ``call:<name>{...}`` occurrences anywhere in ``content``.

    The model is allowed to emit arbitrary preamble/epilogue text before or
    between tool calls; we strip those spans out of the returned content and
    return whatever remains as assistant text.
    """

    stripped = content.strip()
    if not stripped:
        return "", "stop", None

    # Remove <|tool_call|> sentinels but keep everything else so offsets stay meaningful.
    normalized = _TOOL_SENTINEL_RE.sub(" ", stripped)

    tool_calls: list[dict[str, Any]] = []
    keep_spans: list[tuple[int, int]] = []
    cursor = 0
    length = len(normalized)
    match = _CALL_TOKEN_RE.search(normalized, cursor)
    while match is not None:
        tool_call, end_index = _parse_single_tool_call(normalized, match)
        if tool_call is None:
            # Not a real tool call — treat as text. Keep scanning past this token.
            match = _CALL_TOKEN_RE.search(normalized, match.end())
            continue

        keep_spans.append((cursor, match.start()))
        tool_calls.append(tool_call)
        cursor = end_index if end_index is not None else length
        match = _CALL_TOKEN_RE.search(normalized, cursor)

    if not tool_calls:
        return stripped, "stop", None

    keep_spans.append((cursor, length))
    leftover = "".join(normalized[start:end] for start, end in keep_spans).strip()
    return leftover, "tool_calls", tool_calls


def _looks_like_tool_call_prefix(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return True
    if stripped.startswith("call:"):
        return True
    return "<|tool_call|>".startswith(stripped) or "<|tool_call>".startswith(stripped) or stripped.startswith("<|tool_call")


def _contains_tool_call_marker(text: str) -> bool:
    """Has the model already emitted a ``call:<name>`` or ``<|tool_call...`` marker?"""

    if not text:
        return False
    if _CALL_TOKEN_RE.search(text):
        return True
    return "<|tool_call" in text


def _contains_channel_marker(text: str) -> bool:
    """Has the model emitted a ``<|channel|>NAME`` or stray control token?

    Uses the liberal marker regex so we catch unknown / misspelled channel
    names too — anything that looks like channel markup should trigger the
    streaming buffer to avoid leaking raw tokens to the client.
    """

    if not text:
        return False
    return bool(_CHANNEL_MARKER_ANY_RE.search(text)) or bool(_STRAY_CONTROL_RE.search(text))


def _normalize_think_level(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _THINK_ALIASES.get(normalized, normalized)
    if normalized in _THINK_LEVELS:
        return normalized
    return None


def _thinking_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    raw = model_cfg.get("thinking", model_cfg.get("think"))
    return raw if isinstance(raw, dict) else {}


def _thinking_budget_for_level(config: dict[str, Any], level: str) -> int | None:
    if level == "off":
        return None
    thinking_cfg = _thinking_config(config)
    budgets = thinking_cfg.get("budgets", thinking_cfg.get("thinkingBudgets"))
    if isinstance(budgets, dict):
        value = budgets.get(level)
        if isinstance(value, int) and value > 0:
            return value
    return _DEFAULT_THINK_BUDGETS.get(level)


def _include_reasoning_requested(options: dict[str, Any], level: str) -> bool:
    include = options.get("includeReasoning")
    if isinstance(include, bool):
        return include
    return level != "off" and isinstance(options.get("reasoning_effort"), str)


def split_reasoning_markup(content: str) -> tuple[str, str | None]:
    if not content:
        return content, None

    reasoning_parts: list[str] = []

    def _collect_tag(match: re.Match[str]) -> str:
        reasoning_parts.append(match.group(1))
        return ""

    visible = _REASONING_TAG_BLOCK_RE.sub(_collect_tag, content)

    def _collect_channel(match: re.Match[str]) -> str:
        reasoning_parts.append(match.group(2))
        return ""

    visible = _HIDDEN_CHANNEL_BLOCK_RE.sub(_collect_channel, visible)
    cleaned = strip_channel_markup(visible)
    reasoning = strip_channel_markup("\n".join(part.strip() for part in reasoning_parts if part.strip())).strip()
    return cleaned, reasoning or None


def _looks_like_channel_prefix(text: str) -> bool:
    """A partial ``<|channel|>`` opener that might still be arriving."""

    stripped = text.lstrip()
    if not stripped:
        return False
    return "<|channel|>".startswith(stripped) or stripped.startswith("<|channel") or stripped.startswith("<|ch")


def strip_channel_markup(content: str) -> str:
    """Strip Harmony/Gemma-style channel markup from model output.

    The model may emit::

        <|channel|>thought[<|message|>]...<|channel|>final[<|message|>]...

    We keep only the ``final`` channel's content when present; otherwise we
    drop hidden channels (thought/thinking/analysis) and keep the rest with
    markers removed. Stray ``<|message|>`` / ``<|start|>`` / ``<|end|>`` /
    ``<|return|>`` control tokens are always removed.
    """

    if not content:
        return content

    matches = list(_CHANNEL_MARKER_RE.finditer(content))
    if matches:
        segments: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            channel_name = match.group(1).lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            segments.append((channel_name, content[start:end]))

        final_parts = [seg for name, seg in segments if name == "final"]
        if final_parts:
            # Explicit ``final`` channel present — keep only its content and
            # drop everything else (including hidden-reasoning channels).
            result = "".join(final_parts)
        else:
            # No ``final`` channel — keep every segment. Returning empty
            # content when the model only used a ``thought`` channel would
            # hide the entire answer from the user, which is worse than
            # leaking a little reasoning. Strip the markers and keep content.
            result = "".join(seg for _name, seg in segments)
    else:
        result = content

    # Strip stray control tokens and any leftover channel markers (defensive:
    # catches partial tokens and any markup that slipped past the strict
    # matcher above). We use the *bare* marker regex here — matching only the
    # marker characters themselves, never the following word — so that a
    # mangled sequence like ``<channel|>The forecast`` doesn't lose the word
    # "The" by having it mistaken for a channel name.
    result = _STRAY_CONTROL_RE.sub("", result)
    result = _CHANNEL_MARKER_BARE_RE.sub("", result)
    return result.strip()


def filter_hallucinated_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Drop tool_calls whose function.name is not in the provided tools list.

    Returns (kept_tool_calls_or_None, hallucinated_names). An empty `tools`
    list is treated as "no tools available" — every call is hallucinated.
    `tools is None` disables the guard entirely (no tools declared by the
    request means the caller hasn't opted into structured tool-use validation).
    """

    if not tool_calls:
        return tool_calls, []
    if tools is None:
        return tool_calls, []

    allowed: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
        name = (fn or tool).get("name") if isinstance((fn or tool).get("name", None), str) else None
        if isinstance(name, str) and name:
            allowed.add(name)

    kept: list[dict[str, Any]] = []
    hallucinated: list[str] = []
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn.get("name"), str) else ""
        if name in allowed:
            kept.append(call)
        else:
            hallucinated.append(name or "<unknown>")

    if not kept:
        return None, hallucinated
    return kept, hallucinated


def _build_hallucination_retry_messages(
    tool_calls: list[dict[str, Any]],
    error_text: str,
) -> list[dict[str, Any]]:
    """Build a synthetic ``assistant(tool_calls=...)`` + ``tool(...)`` pair.

    When every call the model emitted was hallucinated (name not in
    advertised tools[]), the containment layer replaces the tool_calls
    with a plain-text ``[tool-call error]`` message and ``finish_reason=stop``.
    That protects against infinite loops but also leaves the user staring at
    the error instead of a real answer.

    This helper builds the messages needed to feed the containment error
    back into the model as a tool-result, so it can recover and answer the
    user's underlying question directly on a bounded retry.
    """

    assistant_tool_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        assistant_tool_calls.append(
            {
                "id": call.get("id") or f"hallucinated_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": (fn.get("name") if isinstance(fn.get("name"), str) else "") or "unknown",
                    "arguments": (fn.get("arguments") if isinstance(fn.get("arguments"), str) else "") or "{}",
                },
            }
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": assistant_tool_calls,
        }
    ]
    for call in assistant_tool_calls:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": error_text,
            }
        )
    return messages


def _decode_assistant_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow-copied messages list where every assistant.tool_calls[i]
    .function.arguments that is a JSON string has been decoded into a dict.

    This mirrors what ``mlx_lm.server.process_message_content`` does before
    ``apply_chat_template`` — Gemma's template expects structured arguments,
    not stringified JSON.
    """

    decoded: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            decoded.append(message)
            continue
        if message.get("role") != "assistant" or not isinstance(message.get("tool_calls"), list):
            decoded.append(message)
            continue
        new_message = dict(message)
        new_calls: list[dict[str, Any]] = []
        for call in message["tool_calls"]:
            if not isinstance(call, dict):
                new_calls.append(call)
                continue
            new_call = dict(call)
            fn = new_call.get("function")
            if isinstance(fn, dict):
                new_fn = dict(fn)
                args = new_fn.get("arguments")
                if isinstance(args, str):
                    try:
                        new_fn["arguments"] = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        # Keep the raw string rather than crashing template rendering;
                        # the template will render it as-is, which is still better
                        # than a hard error.
                        pass
                new_call["function"] = new_fn
            new_calls.append(new_call)
        new_message["tool_calls"] = new_calls
        decoded.append(new_message)
    return decoded


class WorkerBackend:
    name = "base"
    backend_id = "base"
    supported_modality_set = frozenset({"text"})

    @classmethod
    def supported_modalities_for_config(cls, _config: dict[str, Any] | None = None) -> set[str]:
        return set(cls.supported_modality_set)

    def supported_modalities(self) -> set[str]:
        return self.supported_modalities_for_config(getattr(self, "_config", None))

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> BackendResult:
        raise NotImplementedError

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None):
        result = self.generate(messages, max_tokens, tools=tools, options=options)
        if result.content:
            yield BackendStreamChunk(text=result.content)
        yield result

    def teardown_session(self, _session_id: str) -> None:
        return


class StubBackend(WorkerBackend):
    name = "stub"
    backend_id = "stub"
    supported_modality_set = BACKEND_ADAPTERS["stub"].supported_modalities

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> BackendResult:
        if os.getenv("MLX_GEMMA_STUB_FAIL") == "1":
            raise RuntimeError("stub_forced_failure")

        delay_ms = int(os.getenv("MLX_GEMMA_STUB_DELAY_MS", "20"))
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = user_messages[-1] if user_messages else ""

        start = time.time()
        time.sleep(max(0, delay_ms) / 1000)
        content = (
            "[stub backend] worker subprocess and backend contract are wired, "
            f"but real MLX generation is not implemented yet. Last user message: {last_user[:160]}"
        )
        elapsed_ms = int((time.time() - start) * 1000)

        prompt_tokens = max(1, sum(len(str(m.get("content", "")).split()) for m in messages))
        completion_tokens = max(1, len(content.split()))
        if isinstance(max_tokens, int) and max_tokens > 0:
            completion_tokens = min(completion_tokens, max_tokens)

        return BackendResult(
            content=content,
            finish_reason="stop",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            metrics={
                "queue_wait_ms": 0,
                "load_ms": 0,
                "prefill_ms": None,
                "generation_ms": None,
                "total_ms": elapsed_ms,
            },
            tool_calls=None,
        )

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None):
        result = self.generate(messages, max_tokens, tools=tools, options=options)
        words = result.content.split()
        if not words:
            yield result
            return
        for index, word in enumerate(words):
            text = word if index == 0 else f" {word}"
            yield BackendStreamChunk(text=text)
        yield result

    def session_generate(
        self,
        session_id: str,
        parts: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        policy: dict[str, Any],
        turn_index: int = 0,
    ) -> BackendResult:
        text = " ".join(str(part.get("text", "")) for part in parts if part.get("type") == "text").strip()
        audio_count = sum(1 for part in parts if part.get("type") == "audio")
        memories = getattr(self, "_session_memories", None)
        if not isinstance(memories, dict):
            memories = {}
            self._session_memories = memories
        memory = memories.setdefault(session_id, {"texts": [], "audio_count": 0})
        if text:
            memory["texts"].append(text)
        memory["audio_count"] += audio_count
        content = f"[stub backend] si drone {session_id} turn accepted"
        if text:
            content += f": {text[:120]}"
        cached_text = " ".join(str(item) for item in memory["texts"])
        if cached_text:
            content += f" | cached_text: {cached_text[:160]}"
        if memory["audio_count"]:
            content += f" | audio_count: {memory['audio_count']}"
        return BackendResult(
            content=content,
            finish_reason="stop",
            usage={
                "prompt_tokens": max(1, len(text.split()) + audio_count),
                "completion_tokens": len(content.split()),
                "total_tokens": max(1, len(text.split()) + audio_count) + len(content.split()),
            },
            metrics={
                "prompt_tokens_new": max(1, len(text.split()) + audio_count),
                "cached_tokens": 0,
                "audio_token_count": audio_count,
                "prefill_ms": 0,
                "generation_ms": 0,
                "context_tokens_total": max(1, len(text.split()) + audio_count),
                "peak_memory_gb": None,
                "session_policy_max_turns": int(policy.get("max_turns", 0) or 0),
                "turn_index": int(turn_index or 0),
            },
        )

    def teardown_session(self, session_id: str) -> None:
        memories = getattr(self, "_session_memories", None)
        if isinstance(memories, dict):
            memories.pop(session_id, None)


class Qwen3TtsBackend(WorkerBackend):
    name = "mlx_audio_qwen3_tts"
    backend_id = "mlx_audio_qwen3_tts"
    supported_modality_set = BACKEND_ADAPTERS["mlx_audio_qwen3_tts"].supported_modalities
    default_model_id = "qwen3-tts-local-1.7b-customvoice-4bit"
    default_backend_id = "mlx_audio_qwen3_tts"
    default_sample_rate_hz = 24_000
    default_output_format = "wav"
    default_speaker = "aiden"
    observed_baseline_peak_memory_gb = 5.23
    postprocess_speed_min = 0.25
    postprocess_speed_max = 4.0
    postprocess_duration_tolerance_ratio = 0.05
    _SUBPROCESS_ENV_ALLOWLIST = frozenset(
        {
            "HOME",
            "PATH",
            "TMPDIR",
            "TEMP",
            "TMP",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "NO_COLOR",
        }
    )

    @staticmethod
    def _absolute_path(raw_path: str | Path, *, base_dir: Path) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return Path(os.path.abspath(os.fspath(path)))

    @classmethod
    def _configured_service_root(cls, config: dict[str, Any], speech_cfg: dict[str, Any]) -> Path:
        paths_cfg = config.get("paths", {})
        paths_cfg = paths_cfg if isinstance(paths_cfg, dict) else {}
        raw_root = (
            speech_cfg.get("serviceRoot")
            or speech_cfg.get("service_root")
            or paths_cfg.get("serviceRoot")
            or paths_cfg.get("service_root")
            or os.getenv("MLX_GEMMA_SERVICE_ROOT")
            or os.getcwd()
        )
        return cls._absolute_path(str(raw_root), base_dir=Path.cwd())

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        speech_cfg = config.get("speech", {})
        speech_cfg = speech_cfg if isinstance(speech_cfg, dict) else {}
        self._service_root = self._configured_service_root(config, speech_cfg)
        raw_model_path = str(
            speech_cfg.get("modelPath")
            or speech_cfg.get("model_path")
            or config.get("model", {}).get("path", "")
        ).strip()
        self._model_path = str(self._absolute_path(raw_model_path, base_dir=self._service_root)) if raw_model_path else ""
        self._model_id = str(speech_cfg.get("modelId") or speech_cfg.get("model_id") or self.default_model_id)
        self._max_input_chars = int(speech_cfg.get("maxInputChars") or speech_cfg.get("max_input_chars") or 4_000)
        raw_output_dir = str(speech_cfg.get("outputDir") or speech_cfg.get("output_dir") or "tmp/qwen3-tts-output")
        self._output_dir = str(self._absolute_path(raw_output_dir, base_dir=self._service_root))
        raw_python = str(
            speech_cfg.get("pythonExecutable")
            or speech_cfg.get("python_executable")
            or "runtime/qwen3-tts-smoke/.venv/bin/python"
        )
        # Do not resolve symlinks here: venv bin/python is often a symlink, and
        # resolving it loses the virtualenv package context for subprocess runs.
        self._python_executable = str(self._absolute_path(raw_python, base_dir=self._service_root))
        worker_timeout_seconds = max(1.0, float(config.get("worker", {}).get("requestTimeoutMs", 180_000) or 180_000) / 1000.0)
        raw_max_timeout = speech_cfg.get("maxTimeoutSeconds") or speech_cfg.get("max_timeout_seconds")
        configured_max_timeout = float(raw_max_timeout) if raw_max_timeout is not None else max(1.0, worker_timeout_seconds - 5.0)
        self._max_timeout_seconds = max(1.0, min(configured_max_timeout, max(1.0, worker_timeout_seconds - 1.0)))
        self._default_timeout_seconds = min(120.0, self._max_timeout_seconds)
        self._retention_max_files = max(1, int(speech_cfg.get("retentionMaxFiles") or speech_cfg.get("retention_max_files") or 50))

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> BackendResult:
        raise RuntimeError("unsupported_request_family:Qwen3-TTS backend only supports speech.generate")

    def _model_config(self) -> dict[str, Any]:
        if not self._model_path:
            return {}
        config_path = Path(self._model_path) / "config.json"
        if not config_path.exists():
            return {}
        try:
            loaded = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _supported_speakers(self) -> list[str]:
        config = self._model_config()
        talker_config = config.get("talker_config")
        if isinstance(talker_config, dict) and isinstance(talker_config.get("spk_id"), dict):
            return sorted(str(name) for name in talker_config["spk_id"].keys())
        codec_config = config.get("codec_config")
        if isinstance(codec_config, dict) and isinstance(codec_config.get("spk_id"), dict):
            return sorted(str(name) for name in codec_config["spk_id"].keys())
        return []

    def _model_size_bytes(self) -> int | None:
        if not self._model_path:
            return None
        model_dir = Path(self._model_path)
        if not model_dir.exists():
            return None
        try:
            return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
        except OSError:
            return None

    def model_info(self) -> dict[str, Any]:
        return {
            "backendId": self.default_backend_id,
            "modelId": self._model_id,
            "modelPath": self._model_path or None,
            "modelExists": bool(self._model_path and Path(self._model_path).exists()),
            "modelSizeBytes": self._model_size_bytes(),
            "serviceRoot": str(self._service_root),
            "family": "tts",
            "modes": ["speech.generate"],
            "sampleRateHz": self.default_sample_rate_hz,
            "outputFormats": [self.default_output_format],
            "outputDir": self._output_dir,
            "pythonExecutable": self._python_executable,
            "defaultSpeaker": self.default_speaker,
            "supportedSpeakers": self._supported_speakers(),
            "maxInputChars": self._max_input_chars,
            "defaultTimeoutSeconds": self._default_timeout_seconds,
            "maxTimeoutSeconds": self._max_timeout_seconds,
            "retentionMaxFiles": self._retention_max_files,
            "streaming": {"supported": False, "planned": True},
            "referenceAudio": {"supported": True, "transport": "local_path", "referenceText": "optional"},
            "postprocessSpeed": {
                "supported": True,
                "field": "postprocessSpeed",
                "aliases": ["postprocess_speed", "speed"],
                "default": 1.0,
                "min": self.postprocess_speed_min,
                "max": self.postprocess_speed_max,
                "durationToleranceRatio": self.postprocess_duration_tolerance_ratio,
            },
        }

    def estimate_memory(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request if isinstance(request, dict) else {}
        input_text = request.get("input") if isinstance(request.get("input"), str) else ""
        input_chars = len(input_text)
        # Keep admission conservative until Phase 2 captures cold/warm API measurements.
        estimated_peak_gb = max(6.0, round(self.observed_baseline_peak_memory_gb + min(input_chars / 8_000, 1.0), 2))
        return {
            "backendId": self.default_backend_id,
            "modelId": self._model_id,
            "baselinePeakMemoryGb": self.observed_baseline_peak_memory_gb,
            "estimatedPeakMemoryGb": estimated_peak_gb,
            "recommendedMinFreeMemoryGb": round(estimated_peak_gb + 1.0, 2),
            "confidence": "low",
            "basis": "phase1_cli_smoke",
            "components": {
                "modelWeights": "included_in_baseline",
                "speechCodec": "included_in_baseline",
                "decodedAudioBuffers": "request_dependent",
            },
            "request": {
                "inputChars": input_chars,
                "speaker": request.get("speaker") or self.default_speaker,
                "format": request.get("format") or self.default_output_format,
            },
        }

    def speech_error(self, code: str, message: str, *, status: int, **details: Any) -> dict[str, Any]:
        return SpeechError(code=code, message=message, status=status, details=details).to_dict()

    def speech_timeout_error(self, timeout_seconds: int | float | None) -> dict[str, Any]:
        return self.speech_error(
            "timeout",
            "Qwen3-TTS speech generation timed out",
            status=504,
            timeoutSeconds=timeout_seconds,
        )

    def validate_speech_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        input_text = request.get("input")
        if not isinstance(input_text, str) or not input_text.strip():
            return self.speech_error("bad_request", "Field 'input' must be a non-empty string", status=400)
        if len(input_text) > self._max_input_chars:
            return self.speech_error(
                "input_too_long",
                "Speech input exceeds maxInputChars",
                status=413,
                maxInputChars=self._max_input_chars,
                inputChars=len(input_text),
            )
        output_format = str(request.get("format") or self.default_output_format).lower()
        if output_format != self.default_output_format:
            return self.speech_error(
                "unsupported_format",
                "Qwen3-TTS Phase 2 only supports wav output",
                status=415,
                format=output_format,
                supportedFormats=[self.default_output_format],
            )
        supported = self._supported_speakers()
        requested_speaker = request.get("speaker")
        speaker = str(requested_speaker or self.default_speaker) if supported else str(requested_speaker or "")
        if supported and speaker not in supported:
            return self.speech_error(
                "unsupported_speaker",
                "Requested speaker is not supported by this model",
                status=422,
                speaker=speaker,
                supportedSpeakers=supported,
            )
        model_exists = bool(self._model_path and Path(self._model_path).exists())
        if not supported and requested_speaker not in (None, "") and (model_exists or str(requested_speaker) != self.default_speaker):
            return self.speech_error(
                "unsupported_speaker",
                "This Qwen3-TTS model does not advertise named speakers; omit speaker/voice and use reference audio instead",
                status=422,
                speaker=str(requested_speaker),
                supportedSpeakers=[],
            )
        reference_audio = (
            request.get("referenceAudioPath")
            or request.get("reference_audio_path")
            or request.get("ref_audio")
        )
        if reference_audio is not None:
            if not isinstance(reference_audio, str) or not reference_audio.strip():
                return self.speech_error(
                    "bad_request",
                    "Reference audio path must be a non-empty string",
                    status=400,
                )
            reference_path = self._absolute_path(reference_audio.strip(), base_dir=self._service_root)
            if not reference_path.exists():
                return self.speech_error(
                    "reference_audio_not_found",
                    "Reference audio path does not exist",
                    status=400,
                    referenceAudioPath=str(reference_path),
                )
        reference_text = request.get("referenceText") or request.get("reference_text") or request.get("ref_text")
        if reference_text is not None and not isinstance(reference_text, str):
            return self.speech_error("bad_request", "Reference text must be a string when provided", status=400)
        speed_error = self._validate_postprocess_speed(request)
        if speed_error is not None:
            return speed_error
        return None

    def _request_postprocess_speed(self, request: dict[str, Any]) -> float:
        for key in ("postprocessSpeed", "postprocess_speed", "speed"):
            value = request.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return 1.0

    def _validate_postprocess_speed(self, request: dict[str, Any]) -> dict[str, Any] | None:
        present = [key for key in ("postprocessSpeed", "postprocess_speed", "speed") if key in request]
        if not present:
            return None
        value = request.get(present[0])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return self.speech_error(
                "bad_request",
                "postprocessSpeed must be a numeric multiplier",
                status=400,
                field=present[0],
            )
        speed = float(value)
        if not (self.postprocess_speed_min <= speed <= self.postprocess_speed_max):
            return self.speech_error(
                "bad_request",
                "postprocessSpeed is outside the supported range",
                status=400,
                field=present[0],
                min=self.postprocess_speed_min,
                max=self.postprocess_speed_max,
            )
        return None

    def _wav_metadata(self, path: Path) -> dict[str, Any]:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            duration = frames / float(rate) if rate > 0 else None
        return {
            "sampleRateHz": rate,
            "channels": channels,
            "sampleWidthBytes": sample_width,
            "durationSeconds": duration,
        }

    def _sample_bounds(self, sample_width: int) -> tuple[int, int]:
        if sample_width == 1:
            return -128, 127
        bits = sample_width * 8
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    def _read_pcm_sample(self, payload: bytes, offset: int, sample_width: int) -> int:
        raw = payload[offset : offset + sample_width]
        if sample_width == 1:
            return raw[0] - 128 if raw else 0
        return int.from_bytes(raw, byteorder="little", signed=True)

    def _write_pcm_sample(self, value: float, sample_width: int) -> bytes:
        minimum, maximum = self._sample_bounds(sample_width)
        clipped = max(minimum, min(maximum, int(round(value))))
        if sample_width == 1:
            return bytes([clipped + 128])
        return clipped.to_bytes(sample_width, byteorder="little", signed=True)

    def _wav_peak_abs(self, payload: bytes, *, sample_width: int) -> int:
        if not payload:
            return 0
        peak = 0
        for offset in range(0, len(payload) - sample_width + 1, sample_width):
            peak = max(peak, abs(self._read_pcm_sample(payload, offset, sample_width)))
        return peak

    def _pcm_active_frame_span(self, payload: bytes, *, sample_width: int, channels: int) -> tuple[int, int] | None:
        bytes_per_frame = sample_width * channels
        if bytes_per_frame <= 0:
            return None
        frame_count = len(payload) // bytes_per_frame
        if frame_count <= 0:
            return None
        sample_min, sample_max = self._sample_bounds(sample_width)
        threshold = max(1, int(max(abs(sample_min), abs(sample_max)) * 0.005))

        def frame_is_active(frame_index: int) -> bool:
            frame_offset = frame_index * bytes_per_frame
            for channel in range(channels):
                sample_offset = frame_offset + (channel * sample_width)
                if abs(self._read_pcm_sample(payload, sample_offset, sample_width)) >= threshold:
                    return True
            return False

        start = 0
        while start < frame_count and not frame_is_active(start):
            start += 1
        if start >= frame_count:
            return None
        end = frame_count
        while end > start and not frame_is_active(end - 1):
            end -= 1
        return start, end

    def _resample_pcm_speed(self, payload: bytes, *, sample_width: int, channels: int, speed: float) -> bytes:
        bytes_per_frame = sample_width * channels
        if bytes_per_frame <= 0:
            return payload
        input_frames = len(payload) // bytes_per_frame
        if input_frames <= 1:
            return payload
        output_frames = max(1, int(round(input_frames / speed)))
        out = bytearray(output_frames * bytes_per_frame)
        for out_frame in range(output_frames):
            source_position = min((input_frames - 1), out_frame * speed)
            left = int(source_position)
            right = min(left + 1, input_frames - 1)
            fraction = source_position - left
            for channel in range(channels):
                left_offset = (left * channels + channel) * sample_width
                right_offset = (right * channels + channel) * sample_width
                left_value = self._read_pcm_sample(payload, left_offset, sample_width)
                right_value = self._read_pcm_sample(payload, right_offset, sample_width)
                interpolated = left_value + ((right_value - left_value) * fraction)
                out_offset = (out_frame * channels + channel) * sample_width
                out[out_offset : out_offset + sample_width] = self._write_pcm_sample(interpolated, sample_width)
        return bytes(out)

    def _postprocess_wav_speed(self, path: Path, speed: float) -> dict[str, Any]:
        if abs(speed - 1.0) < 0.0001:
            return {
                "enabled": False,
                "speed": 1.0,
                "durationToleranceRatio": self.postprocess_duration_tolerance_ratio,
            }
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)

        source_duration = frame_count / float(frame_rate) if frame_rate > 0 else None
        source_peak = self._wav_peak_abs(raw, sample_width=sample_width)
        source_active_span = self._pcm_active_frame_span(raw, sample_width=sample_width, channels=channels)
        processed = self._resample_pcm_speed(raw, sample_width=sample_width, channels=channels, speed=speed)

        tmp_path = path.with_suffix(path.suffix + ".speed.tmp")
        with wave.open(str(tmp_path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(sample_width)
            handle.setframerate(frame_rate)
            handle.writeframes(processed)
        tmp_path.replace(path)

        output_frames = len(processed) // max(1, channels * sample_width)
        output_duration = output_frames / float(frame_rate) if frame_rate > 0 else None
        output_active_span = self._pcm_active_frame_span(processed, sample_width=sample_width, channels=channels)
        expected_duration = (source_duration / speed) if source_duration is not None else None
        duration_error_ratio = None
        if expected_duration and output_duration is not None:
            duration_error_ratio = abs(output_duration - expected_duration) / expected_duration
        source_active_duration = None
        output_active_duration = None
        expected_active_duration = None
        active_duration_error_ratio = None
        active_duration_within_tolerance = None
        trimmed_leading = None
        trimmed_trailing = None
        if source_active_span is not None and output_active_span is not None and frame_rate > 0:
            source_start, source_end = source_active_span
            output_start, output_end = output_active_span
            source_active_duration = (source_end - source_start) / float(frame_rate)
            output_active_duration = (output_end - output_start) / float(frame_rate)
            expected_active_duration = source_active_duration / speed
            if expected_active_duration > 0:
                active_duration_error_ratio = abs(output_active_duration - expected_active_duration) / expected_active_duration
                active_duration_within_tolerance = active_duration_error_ratio <= self.postprocess_duration_tolerance_ratio
            trimmed_leading = source_start / float(frame_rate)
            trimmed_trailing = (frame_count - source_end) / float(frame_rate)
        output_peak = self._wav_peak_abs(processed, sample_width=sample_width)
        sample_min, sample_max = self._sample_bounds(sample_width)
        full_scale_peak = max(abs(sample_min), abs(sample_max))
        return {
            "enabled": True,
            "speed": speed,
            "originalDurationSeconds": source_duration,
            "expectedDurationSeconds": expected_duration,
            "outputDurationSeconds": output_duration,
            "durationErrorRatio": duration_error_ratio,
            "durationToleranceRatio": self.postprocess_duration_tolerance_ratio,
            "durationWithinTolerance": bool(duration_error_ratio is not None and duration_error_ratio <= self.postprocess_duration_tolerance_ratio),
            "activeDurationSeconds": source_active_duration,
            "expectedActiveDurationSeconds": expected_active_duration,
            "outputActiveDurationSeconds": output_active_duration,
            "activeDurationErrorRatio": active_duration_error_ratio,
            "activeDurationWithinTolerance": active_duration_within_tolerance,
            "trimmedLeadingSeconds": trimmed_leading,
            "trimmedTrailingSeconds": trimmed_trailing,
            "sampleRateHz": frame_rate,
            "channels": channels,
            "sampleWidthBytes": sample_width,
            "peakSampleBefore": source_peak,
            "peakSampleAfter": output_peak,
            "fullScalePeak": full_scale_peak,
            "clippingIntroduced": output_peak >= full_scale_peak and source_peak < full_scale_peak,
        }

    def _run_mlx_audio(self, command: list[str], timeout_seconds: int | float) -> subprocess.CompletedProcess[str]:
        env = {key: value for key, value in os.environ.items() if key in self._SUBPROCESS_ENV_ALLOWLIST}
        env["PYTHONNOUSERSITE"] = "1"
        return subprocess.run(
            command,
            cwd=str(self._service_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )

    def _process_text_tail(self, value: str | None, *, limit: int = 500) -> str:
        if not value:
            return ""
        sanitized = value.replace("\x00", "")
        return sanitized[-limit:]

    def _cleanup_output_dir(self, output_dir: Path, *, keep: Path) -> None:
        try:
            candidates = sorted(
                (path for path in output_dir.glob("speech_*.wav") if path != keep),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for stale in candidates[max(0, self._retention_max_files - 1) :]:
            try:
                stale.unlink()
            except OSError:
                pass

    def speech_generate(self, request: dict[str, Any]) -> dict[str, Any]:
        validation_error = self.validate_speech_request(request)
        if validation_error is not None:
            return {"type": "speech_error", "error": validation_error}
        if not self._model_path or not Path(self._model_path).exists():
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "load_failed",
                    "Qwen3-TTS model path is not configured or does not exist",
                    status=503,
                    modelPath=self._model_path or None,
                ),
            }
        if not Path(self._python_executable).exists():
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "load_failed",
                    "Qwen3-TTS Python runtime is not configured or does not exist",
                    status=503,
                    pythonExecutable=self._python_executable,
                ),
            }
        timeout_seconds = request.get("timeoutSeconds", request.get("timeout_seconds", self._default_timeout_seconds))
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError):
            timeout_seconds = self._default_timeout_seconds
        timeout_seconds = max(1.0, timeout_seconds)
        if timeout_seconds > self._max_timeout_seconds:
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "timeout_too_high",
                    "Speech request timeout exceeds backend maxTimeoutSeconds",
                    status=400,
                    timeoutSeconds=timeout_seconds,
                    maxTimeoutSeconds=self._max_timeout_seconds,
                ),
            }

        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        speech_id = f"speech_{uuid.uuid4().hex[:12]}"
        command = [
            self._python_executable,
            "-m",
            "mlx_audio.tts.generate",
            "--model",
            self._model_path,
            "--text",
            str(request["input"]),
            "--output_path",
            str(output_dir),
            "--file_prefix",
            speech_id,
            "--audio_format",
            self.default_output_format,
        ]
        supported = self._supported_speakers()
        speaker = str(request.get("speaker") or self.default_speaker) if supported else ""
        if speaker:
            command.extend(["--voice", speaker])
        voice = request.get("voice")
        style_instruction = None
        if isinstance(voice, dict):
            style_instruction = voice.get("styleInstruction") or voice.get("style_instruction")
        style_instruction = style_instruction or request.get("styleInstruction") or request.get("style_instruction") or request.get("instruct")
        if isinstance(style_instruction, str) and style_instruction.strip():
            command.extend(["--instruct", style_instruction.strip()])
        reference_audio = request.get("referenceAudioPath") or request.get("reference_audio_path") or request.get("ref_audio")
        if isinstance(reference_audio, str) and reference_audio.strip():
            command.extend(["--ref_audio", str(self._absolute_path(reference_audio.strip(), base_dir=self._service_root))])
        reference_text = request.get("referenceText") or request.get("reference_text") or request.get("ref_text")
        if isinstance(reference_text, str) and reference_text.strip():
            command.extend(["--ref_text", reference_text.strip()])
        gen_duration = request.get("genDuration", request.get("gen_duration"))
        if isinstance(gen_duration, (int, float)) and gen_duration > 0:
            command.extend(["--gen_duration", str(gen_duration)])
        duration_multiplier = request.get("durationMultiplier", request.get("duration_multiplier"))
        if isinstance(duration_multiplier, (int, float)) and duration_multiplier > 0:
            command.extend(["--duration_multiplier", str(duration_multiplier)])
        max_tokens = request.get("maxTokens", request.get("max_tokens"))
        if isinstance(max_tokens, int) and max_tokens > 0:
            command.extend(["--max_tokens", str(max_tokens)])

        started = time.perf_counter()
        try:
            completed = self._run_mlx_audio(command, timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"type": "speech_error", "error": self.speech_timeout_error(timeout_seconds)}
        total_ms = int((time.perf_counter() - started) * 1000)
        if completed.returncode != 0:
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "generation_failed",
                    "Qwen3-TTS generation command failed",
                    status=500,
                    returncode=completed.returncode,
                    stderrTail=self._process_text_tail(completed.stderr),
                    stdoutTail=self._process_text_tail(completed.stdout),
                ),
            }

        candidates = sorted(output_dir.glob(f"{speech_id}*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "generation_failed",
                    "Qwen3-TTS generation completed without producing a WAV file",
                    status=500,
                    outputDir=str(output_dir),
                    stdoutTail=self._process_text_tail(completed.stdout),
                    stderrTail=self._process_text_tail(completed.stderr),
                ),
            }
        audio_path = candidates[0]
        postprocess_speed = self._request_postprocess_speed(request)
        try:
            postprocess = self._postprocess_wav_speed(audio_path, postprocess_speed)
        except RuntimeError as exc:
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "generation_failed",
                    "Qwen3-TTS output WAV speed postprocess failed",
                    status=500,
                    audioPath=str(audio_path),
                    reason=str(exc),
                ),
            }
        try:
            wav = self._wav_metadata(audio_path)
        except wave.Error as exc:
            return {
                "type": "speech_error",
                "error": self.speech_error(
                    "generation_failed",
                    "Qwen3-TTS output WAV metadata could not be read",
                    status=500,
                    audioPath=str(audio_path),
                    reason=str(exc),
                ),
            }
        self._cleanup_output_dir(output_dir, keep=audio_path)
        return {
            "type": "speech_result",
            "id": speech_id,
            "object": "audio.speech",
            "model": self._model_id,
            "backend": self.default_backend_id,
            "format": self.default_output_format,
            "sampleRateHz": wav["sampleRateHz"],
            "durationSeconds": wav["durationSeconds"],
            "audioPath": str(audio_path),
            "fileSizeBytes": audio_path.stat().st_size,
            "metrics": {
                "loadMs": None,
                "firstAudioMs": None,
                "totalMs": total_ms,
                "rssGb": None,
                "metalActiveGb": None,
                "metalPeakGb": None,
                "generatedAudioSeconds": wav["durationSeconds"],
                "postprocessSpeed": postprocess,
            },
        }


class MlxVlmTurboQuantBackend(WorkerBackend):
    name = "mlx_vlm_turboquant"
    backend_id = "mlx_vlm_turboquant"
    supported_modality_set = BACKEND_ADAPTERS["mlx_vlm_turboquant"].supported_modalities

    @classmethod
    def supported_modalities_for_config(cls, config: dict[str, Any] | None = None) -> set[str]:
        if config is None:
            return set(cls.supported_modality_set)
        return turboquant_supported_modalities(config)

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        raw_model_path = str(self._config.get("model", {}).get("path", "")).strip()
        self._model_path = str(Path(raw_model_path).expanduser()) if raw_model_path else ""
        if not self._model_path:
            raise RuntimeError("mlx_vlm_model_path_not_configured")
        if not Path(self._model_path).exists():
            raise RuntimeError(f"mlx_vlm_model_path_missing:{self._model_path}")

        try:
            from mlx_vlm import generate, load  # type: ignore
            from mlx_vlm.generate.ar import generate_step  # type: ignore
            from mlx_vlm.models import cache as vlm_cache  # type: ignore
            from mlx_vlm.utils import prepare_inputs  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_runtime_import_failed:{exc}") from exc

        self._generate = generate
        self._generate_step = generate_step
        self._vlm_cache = vlm_cache
        self._prepare_inputs = prepare_inputs
        self._session_caches: dict[str, list[Any]] = {}
        self._max_output_tokens = int(self._config.get("model", {}).get("maxOutputTokens", 1024) or 1024)
        sampling_cfg = self._config.get("model", {}).get("sampling", {})
        self._sampling_kwargs = self._generation_sampling_kwargs(sampling_cfg if isinstance(sampling_cfg, dict) else {})
        self._kv_cache_config = _parse_kv_cache_config(self._config)
        self._load_ms = 0
        self._load_consumed = False
        self._memory_samples: dict[str, dict[str, Any] | None] = {
            "before_load": _memory_sample("before_load"),
            "after_load": None,
            "request_start": None,
            "prefill_peak": None,
            "post_generation": None,
            "idle": None,
            "after_unload": None,
        }

        load_started = time.perf_counter()
        try:
            self._model, self._processor = load(self._model_path, lazy=False)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_model_load_failed:{exc}") from exc
        self._load_ms = int((time.perf_counter() - load_started) * 1000)
        self._memory_samples["after_load"] = _memory_sample("after_load", model=self._model)

        model_type = str(getattr(getattr(self._model, "config", None), "model_type", "") or "")
        if model_type not in {"gemma4", "gemma4_unified"}:
            raise RuntimeError(f"mlx_vlm_unsupported_model_type:{model_type or 'unknown'}")

    def stats(self) -> dict[str, Any]:
        samples = _ensure_memory_samples(self)
        samples["idle"] = _memory_sample("idle", model=getattr(self, "_model", None))
        return {"memory": _memory_stats(samples), "kvCache": _ensure_kv_cache_config(self).to_dict()}

    def supported_modalities(self) -> set[str]:
        config = getattr(getattr(self._model, "config", None), "__dict__", None)
        if not isinstance(config, dict):
            return self.supported_modalities_for_config(getattr(self, "_config", None))
        supported = {"text"}
        if config.get("vision_config") is not None:
            supported.add("image")
        if config.get("audio_config") is not None:
            supported.add("audio")
        return supported & set(self.supported_modality_set)

    def _model_family(self) -> str:
        model_type = str(getattr(getattr(getattr(self, "_model", None), "config", None), "model_type", "") or "").lower()
        if model_type.startswith("gemma"):
            return "gemma"
        if model_type.startswith("qwen"):
            return "qwen"
        if model_type in {"gpt", "openai"}:
            return "openai"
        return "unknown"

    def _resolve_think_control(self, options: dict[str, Any] | None) -> ThinkControl:
        options = options if isinstance(options, dict) else {}
        thinking_cfg = _thinking_config(getattr(self, "_config", {}))
        family = self._model_family()
        requested = options.get("thinkLevel")
        source = "request.thinkLevel"
        if requested is None and options.get("reasoning_effort") is not None:
            requested = options.get("reasoning_effort")
            source = "request.reasoning_effort"
        default_level = thinking_cfg.get("defaultLevel", thinking_cfg.get("default_level", "off"))
        if requested is None:
            requested = default_level
            source = "model.thinking.defaultLevel"
        level = _normalize_think_level(requested)
        if level is None:
            level = "off"
            source = "unknown_family_or_invalid_level"
        budget = _thinking_budget_for_level(getattr(self, "_config", {}), level)
        mechanism = {
            "gemma": "apply_chat_template(enable_thinking, thinking_budget)",
            "qwen": "prompt_directive(/think|/no_think)",
            "openai": "reasoning_effort",
        }.get(family, "unsupported_noop")
        if family == "unknown":
            level = "off"
            budget = None
        return ThinkControl(
            level=level,
            include_reasoning=_include_reasoning_requested(options, level),
            family=family,
            mechanism=mechanism,
            budget_tokens=budget,
            source=source,
            requested_level=str(requested) if requested is not None else None,
        )

    @staticmethod
    def _with_qwen_think_directive(messages: list[dict[str, Any]], control: ThinkControl) -> list[dict[str, Any]]:
        directive = "/no_think" if control.level == "off" else "/think"
        patched = [dict(message) for message in messages]
        for message in reversed(patched):
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                if directive not in content:
                    message["content"] = f"{content}\n{directive}".strip()
                return patched
        patched.append({"role": "user", "content": directive})
        return patched

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think_control: ThinkControl | None = None,
        *,
        apply_think_control: bool = True,
    ) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        template_messages = _decode_assistant_tool_call_arguments(messages) if tools else messages
        control = think_control or (self._resolve_think_control(None) if apply_think_control else None)
        if control is not None and control.family == "qwen":
            template_messages = self._with_qwen_think_directive(template_messages, control)
        if callable(apply_chat_template):
            kwargs: dict[str, Any] = {}
            if control is not None and control.family == "gemma":
                kwargs["enable_thinking"] = control.level != "off"
                if control.budget_tokens is not None:
                    kwargs["thinking_budget"] = control.budget_tokens
            try:
                return apply_chat_template(template_messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
            except TypeError:
                if "thinking_budget" in kwargs:
                    kwargs.pop("thinking_budget", None)
                    try:
                        return apply_chat_template(template_messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
                    except Exception:
                        pass
            except Exception:
                pass
        if not tools and self._uses_gemma4_manual_chat_template():
            if control is not None and control.family == "gemma" and control.level != "off":
                template_messages = [{"role": "system", "content": "<|think|>"}, *template_messages]
            return self._gemma4_manual_chat_prompt(template_messages, add_generation_prompt=True)

        rendered: list[str] = []
        for message in template_messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system":
                rendered.append(content)
            elif role == "assistant":
                rendered.append(f"Assistant: {content}")
            else:
                rendered.append(content)
        return "\n\n".join(part for part in rendered if part).strip()

    def _uses_gemma4_manual_chat_template(self) -> bool:
        model = getattr(self, "_model", None)
        model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
        return model_type in {"gemma4", "gemma4_unified"}

    def _has_chat_template(self) -> bool:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return callable(getattr(tokenizer, "apply_chat_template", None)) and bool(getattr(tokenizer, "chat_template", None))

    @staticmethod
    def _message_text_content(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            return "\n".join(text_parts)
        return str(content)

    @classmethod
    def _gemma4_manual_chat_prompt(cls, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = cls._message_text_content(message)
            if role == "assistant":
                turn_role = "model"
            else:
                turn_role = "user"
            rendered.append(f"<start_of_turn>{turn_role}\n{content}<end_of_turn>\n")
        if add_generation_prompt:
            rendered.append("<start_of_turn>model\n")
        return "".join(rendered)

    @staticmethod
    def _media_suffix(mime_type: str, modality: str) -> str:
        suffixes = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
        }
        return suffixes.get(mime_type, f".{modality}")

    @staticmethod
    def _decode_data_url(data_url: str) -> bytes:
        if not data_url.startswith("data:") or "," not in data_url:
            raise RuntimeError("mlx_vlm_invalid_data_url")
        header, encoded = data_url[5:].split(",", 1)
        if "base64" not in {part.lower() for part in header.split(";")[1:]}:
            raise RuntimeError("mlx_vlm_data_url_must_be_base64")
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("mlx_vlm_invalid_data_url_payload") from exc

    @staticmethod
    def _generation_sampling_kwargs(sampling_cfg: dict[str, Any]) -> dict[str, float]:
        kwargs: dict[str, float] = {}
        temperature = sampling_cfg.get("temperature")
        if isinstance(temperature, (int, float)):
            kwargs["temperature"] = float(temperature)
        top_p = sampling_cfg.get("topP", sampling_cfg.get("top_p"))
        if isinstance(top_p, (int, float)):
            kwargs["top_p"] = float(top_p)
        return kwargs

    def _explicit_turboquant_cache_class(self) -> Any:
        cached = getattr(self, "_explicit_turboquant_cache_cls", None)
        if cached is not None:
            return cached
        try:
            import mlx.core as mx  # type: ignore
            import mlx_vlm.turboquant as tq  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_turboquant_import_failed:{exc}") from exc

        class ExplicitTurboQuantKVCache(tq.TurboQuantKVCache):  # type: ignore[misc]
            def __init__(
                self,
                key_bits: int,
                value_bits: int,
                group_size: int = 64,
                seed: int = tq.DEFAULT_TURBOQUANT_SEED,
            ) -> None:
                super().__init__(bits=float(key_bits), seed=seed)
                self.key_bits = int(key_bits)
                self.value_bits = int(value_bits)
                # MLX's shared quantized-attention helpers expect quantized
                # cache objects to expose group_size. TurboQuant does not use
                # affine group-size math, but mlx-vlm's own batch TurboQuant
                # cache publishes this same compatibility field.
                self.group_size = int(group_size)

            @classmethod
            def from_cache(
                cls,
                cache: Any,
                key_bits: int,
                value_bits: int,
                group_size: int = 64,
                seed: int = tq.DEFAULT_TURBOQUANT_SEED,
            ) -> Any:
                turbo_cache = cls(
                    key_bits=key_bits,
                    value_bits=value_bits,
                    group_size=group_size,
                    seed=seed,
                )
                keys, values = cache.state
                if keys is not None:
                    turbo_cache.update_and_fetch(keys, values)
                return turbo_cache

            def _ensure_codecs(self, keys: mx.array, values: mx.array) -> None:
                if self.key_codec is None:
                    self.key_codec = tq._build_codec(
                        keys,
                        float(self.key_bits),
                        mode="mse",
                        seed=self.seed,
                    )
                if self.value_codec is None:
                    self.value_codec = tq._build_codec(
                        values,
                        float(self.value_bits),
                        mode="mse",
                        seed=self.seed + 1,
                    )

        self._explicit_turboquant_cache_cls = ExplicitTurboQuantKVCache
        return ExplicitTurboQuantKVCache

    def _deferred_explicit_turboquant_cache_class(self) -> Any:
        cached = getattr(self, "_deferred_explicit_turboquant_cache_cls", None)
        if cached is not None:
            return cached
        explicit_cache_cls = self._explicit_turboquant_cache_class()

        class DeferredExplicitTurboQuantKVCache(
            _DeferredExplicitTurboQuantKVCache,
            explicit_cache_cls,  # type: ignore[valid-type, misc]
        ):
            pass

        self._deferred_explicit_turboquant_cache_cls = DeferredExplicitTurboQuantKVCache
        return DeferredExplicitTurboQuantKVCache

    def _uses_explicit_kv_request_cache(self) -> bool:
        return _ensure_kv_cache_config(self).enabled and self._model_family() == "gemma"

    def _wrap_explicit_kv_request_cache_entry(
        self,
        cache_entry: Any,
        *,
        layer_index: int,
        last_index: int,
    ) -> Any:
        if layer_index == last_index:
            return cache_entry
        vlm_cache = self._vlm_cache
        rotating_cls = getattr(vlm_cache, "RotatingKVCache", None)
        if rotating_cls is not None and isinstance(cache_entry, rotating_cls):
            return cache_entry
        cache_list_cls = getattr(vlm_cache, "CacheList", None)
        if cache_list_cls is not None and isinstance(cache_entry, cache_list_cls):
            cache_entry.caches = tuple(
                self._wrap_explicit_kv_nested_cache(sub_entry)
                for sub_entry in cache_entry.caches
            )
            return cache_entry
        if isinstance(cache_entry, list):
            for index, sub_entry in enumerate(cache_entry):
                cache_entry[index] = self._wrap_explicit_kv_nested_cache(sub_entry)
            return cache_entry
        if isinstance(cache_entry, tuple):
            return tuple(
                self._wrap_explicit_kv_nested_cache(sub_entry)
                for sub_entry in cache_entry
            )
        return self._wrap_explicit_kv_nested_cache(cache_entry)

    def _wrap_explicit_kv_nested_cache(self, cache_entry: Any) -> Any:
        vlm_cache = self._vlm_cache
        rotating_cls = getattr(vlm_cache, "RotatingKVCache", None)
        if rotating_cls is not None and isinstance(cache_entry, rotating_cls):
            return cache_entry
        cache_list_cls = getattr(vlm_cache, "CacheList", None)
        if cache_list_cls is not None and isinstance(cache_entry, cache_list_cls):
            cache_entry.caches = tuple(
                self._wrap_explicit_kv_nested_cache(sub_entry)
                for sub_entry in cache_entry.caches
            )
            return cache_entry
        if isinstance(cache_entry, list):
            for index, sub_entry in enumerate(cache_entry):
                cache_entry[index] = self._wrap_explicit_kv_nested_cache(sub_entry)
            return cache_entry
        if isinstance(cache_entry, tuple):
            return tuple(
                self._wrap_explicit_kv_nested_cache(sub_entry)
                for sub_entry in cache_entry
            )
        kv_cache_cls = getattr(vlm_cache, "KVCache", None)
        if kv_cache_cls is None or not isinstance(cache_entry, kv_cache_cls):
            return cache_entry
        kv_config = _ensure_kv_cache_config(self)
        return self._deferred_explicit_turboquant_cache_class()(
            cache_entry,
            explicit_cache_cls=self._explicit_turboquant_cache_class(),
            key_bits=kv_config.effective_key_bits,
            value_bits=kv_config.effective_value_bits,
            group_size=kv_config.group_size,
            quantized_kv_start=kv_config.quantized_kv_start,
        )

    def _make_request_prompt_cache(self) -> list[Any]:
        prompt_cache = self._vlm_cache.make_prompt_cache(self._model.language_model)
        if not self._uses_explicit_kv_request_cache():
            return prompt_cache
        last_index = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        return [
            self._wrap_explicit_kv_request_cache_entry(
                layer_cache,
                layer_index=index,
                last_index=last_index,
            )
            for index, layer_cache in enumerate(prompt_cache)
        ]

    def _make_session_prompt_cache(self) -> list[Any]:
        # Phase 3 deliberately excludes SI Drone/session caches from KV
        # compression so multi-turn behavior is not silently mutated.
        return self._vlm_cache.make_prompt_cache(self._model.language_model)

    def _kv_generation_kwargs(self) -> dict[str, Any]:
        if self._uses_explicit_kv_request_cache():
            return {}
        return _ensure_kv_cache_config(self).generation_kwargs()

    def _should_use_step_generate_path(self) -> bool:
        return _ensure_kv_cache_config(self).enabled

    def _decode_generated_tokens(self, generated: list[int]) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        try:
            return tokenizer.decode(generated, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(generated)

    def _prepare_request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> VlmPreparedRequest:
        prompt_messages: list[dict[str, Any]] = []
        image_paths: list[str] = []
        audio_paths: list[str] = []
        temp_dir: tempfile.TemporaryDirectory[str] | None = None

        for message in messages:
            next_message = dict(message)
            parts = message.get("parts")
            if isinstance(parts, list):
                text_parts: list[str] = []
                image_placeholders: list[str] = []
                audio_placeholders: list[str] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type", ""))
                    if part_type == "text":
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                        continue
                    if part_type not in {"image", "audio"}:
                        continue
                    data_url = part.get("data_url")
                    mime_type = str(part.get("mime_type", ""))
                    if not isinstance(data_url, str):
                        raise RuntimeError(f"mlx_vlm_{part_type}_missing_data_url")
                    if temp_dir is None:
                        temp_dir = tempfile.TemporaryDirectory(prefix="mlx-vlm-input-")
                    suffix = self._media_suffix(mime_type, part_type)
                    payload = self._decode_data_url(data_url)
                    media_path = Path(temp_dir.name) / f"{part_type}-{len(image_paths) + len(audio_paths)}{suffix}"
                    media_path.write_bytes(payload)
                    if part_type == "image":
                        image_paths.append(str(media_path))
                        image_placeholders.append("<|image|>")
                    else:
                        audio_paths.append(str(media_path))
                        audio_placeholders.append("<|audio|>")
                content_parts = [*image_placeholders, *text_parts, *audio_placeholders]
                next_message["content"] = "\n".join(content_parts) if content_parts else str(message.get("content", ""))
            prompt_messages.append(next_message)

        think_control = self._resolve_think_control(options)
        return VlmPreparedRequest(
            prompt=self._build_prompt(prompt_messages, tools=tools, think_control=think_control),
            image_paths=image_paths,
            audio_paths=audio_paths,
            temp_dir=temp_dir,
            add_special_tokens=self._uses_gemma4_manual_chat_template() and not tools,
            think_control=think_control.to_dict(),
        )

    def _prepare_session_parts(self, parts: list[dict[str, Any]], *, turn_index: int = 0) -> VlmPreparedRequest:
        prompt_parts: list[str] = []
        audio_paths: list[str] = []
        temp_dir: tempfile.TemporaryDirectory[str] | None = None

        for part in parts:
            part_type = str(part.get("type", ""))
            if part_type == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    prompt_parts.append(text)
                continue
            if part_type != "audio":
                continue
            data_url = part.get("data_url")
            mime_type = str(part.get("mime_type", "audio/wav"))
            if not isinstance(data_url, str):
                raise RuntimeError("mlx_vlm_session_audio_missing_data_url")
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory(prefix="mlx-vlm-session-input-")
            suffix = self._media_suffix(mime_type, "audio")
            payload = self._decode_data_url(data_url)
            media_path = Path(temp_dir.name) / f"audio-{len(audio_paths)}{suffix}"
            media_path.write_bytes(payload)
            audio_paths.append(str(media_path))
            prompt_parts.append("<|audio|>")

        prompt_content = "\n".join(prompt_parts).strip()
        if self._uses_gemma4_manual_chat_template():
            if self._has_chat_template():
                prompt = self._build_prompt([{"role": "user", "content": prompt_content}], tools=None, apply_think_control=False)
                add_special_tokens = False
                if int(turn_index or 0) > 1:
                    if prompt.startswith("<bos>"):
                        prompt = prompt[len("<bos>") :]
                    prompt = f"<turn|>\n{prompt}"
            elif int(turn_index or 0) <= 1:
                prompt = self._gemma4_manual_chat_prompt([{"role": "user", "content": prompt_content}], add_generation_prompt=True)
                add_special_tokens = True
            else:
                prompt = f"<end_of_turn>\n<start_of_turn>user\n{prompt_content}<end_of_turn>\n<start_of_turn>model\n"
                add_special_tokens = False
        else:
            prompt = prompt_content
            add_special_tokens = None

        return VlmPreparedRequest(
            prompt=prompt,
            image_paths=[],
            audio_paths=audio_paths,
            temp_dir=temp_dir,
            add_special_tokens=add_special_tokens,
        )

    def _cache_offsets(self, prompt_cache: list[Any]) -> list[int | None]:
        offsets: list[int | None] = []
        for cache_entry in prompt_cache:
            offset = getattr(cache_entry, "offset", None)
            if offset is None and getattr(cache_entry, "keys", None) is not None:
                try:
                    offset = cache_entry.keys.shape[2]
                except Exception:
                    offset = None
            try:
                offsets.append(int(offset) if offset is not None else None)
            except Exception:
                offsets.append(None)
        return offsets

    def _max_cache_offset(self, prompt_cache: list[Any]) -> int:
        offsets = [offset for offset in self._cache_offsets(prompt_cache) if isinstance(offset, int)]
        return max(offsets) if offsets else 0

    def _eos_ids(self) -> set[int]:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        ids: set[int] = set()
        for source in (getattr(getattr(self._model, "config", None), "eos_token_id", None), getattr(tokenizer, "eos_token_id", None)):
            if isinstance(source, int):
                ids.add(source)
            elif isinstance(source, (list, tuple)):
                ids.update(int(value) for value in source if isinstance(value, int))
        return ids

    def _step_inputs(self, prepared: VlmPreparedRequest) -> dict[str, Any]:
        model_type = str(getattr(getattr(self._model, "config", None), "model_type", "") or "")
        add_special_tokens = bool(prepared.add_special_tokens)
        if prepared.add_special_tokens is None and not prepared.image_paths and not prepared.audio_paths and model_type not in {"gemma3", "gemma3n", "gemma4", "gemma4_unified"}:
            add_special_tokens = True
        return self._prepare_inputs(
            self._processor,
            prompts=prepared.prompt,
            images=prepared.image_paths or None,
            audio=prepared.audio_paths or None,
            add_special_tokens=add_special_tokens,
            return_tensors="mlx",
            padding=True,
        )

    def _iter_step_loop(
        self,
        prepared: VlmPreparedRequest,
        prompt_cache: list[Any],
        effective_max_tokens: int,
        *,
        kv_generation_kwargs: dict[str, Any] | None = None,
        failure_code: str = "mlx_vlm_generation_failed",
    ):
        generation_started = time.perf_counter()
        first_token_ms: int | None = None
        generated: list[int] = []
        finish_reason = "length"
        prompt_tokens = 0
        raw_text = ""
        peak_memory_gb = 0.0
        closed = False
        try:
            import mlx.core as mx  # type: ignore

            inputs = self._step_inputs(prepared)
            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values")
            mask = inputs.get("attention_mask")
            extra_kwargs = {
                key: value
                for key, value in inputs.items()
                if key not in {"input_ids", "pixel_values", "attention_mask"}
            }
            prompt_tokens = int(getattr(input_ids, "size", 0) or 0)
            _reset_metal_peak_memory()
            samples = _ensure_memory_samples(self)
            samples["request_start"] = _memory_sample("request_start", model=self._model, prompt_cache=prompt_cache)
            stop_ids = self._eos_ids()
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            gen = self._generate_step(
                input_ids,
                self._model,
                pixel_values,
                mask,
                prompt_cache=prompt_cache,
                max_tokens=effective_max_tokens,
                temperature=float(self._sampling_kwargs.get("temperature", 0.0)),
                top_p=float(self._sampling_kwargs.get("top_p", 1.0)),
                **(kv_generation_kwargs or {}),
                **extra_kwargs,
            )
            for token, _logprobs in gen:
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - generation_started) * 1000)
                token_id = int(token.item() if hasattr(token, "item") else token)
                generated.append(token_id)
                if token_id in stop_ids:
                    finish_reason = "stop"
                    break
                try:
                    decoded_so_far = tokenizer.decode(generated)
                except Exception:
                    decoded_so_far = ""
                if "<end_of_turn" in decoded_so_far or "<start_of_turn" in decoded_so_far:
                    finish_reason = "stop"
                    break
                yield StepLoopToken(token_id=token_id, generated=tuple(generated))
            samples["prefill_peak"] = _memory_sample("prefill_peak", model=self._model, prompt_cache=prompt_cache)
            raw_text = self._decode_generated_tokens(generated)
            peak_memory_gb = float(_bytes_to_gib(mx.get_peak_memory()) or 0.0)
            samples["post_generation"] = _memory_sample("post_generation", model=self._model, prompt_cache=prompt_cache)
            mx.clear_cache()
            samples["idle"] = _memory_sample("idle", model=self._model, prompt_cache=prompt_cache)
        except GeneratorExit:
            closed = True
            raise
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"{failure_code}:{exc}") from exc
        finally:
            prepared.cleanup()
        if closed:
            return
        yield StepLoopResult(
            generated=tuple(generated),
            raw_text=raw_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            input_ids=input_ids if "input_ids" in locals() else None,
            first_token_ms=first_token_ms,
            total_ms=int((time.perf_counter() - generation_started) * 1000),
            peak_memory_gb=peak_memory_gb,
            prompt_cache=prompt_cache,
        )

    def session_generate(
        self,
        session_id: str,
        parts: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        policy: dict[str, Any],
        turn_index: int = 0,
    ) -> BackendResult:
        for part in parts:
            if part.get("type") not in {"text", "audio"}:
                raise RuntimeError(f"unsupported_part_type:SI Drone v1 does not support part type: {part.get('type')}")

        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        if session_id not in self._session_caches and int(turn_index or 0) > 1:
            raise RuntimeError("session_lost")
        if session_id not in self._session_caches:
            self._session_caches[session_id] = self._make_session_prompt_cache()
        prompt_cache = self._session_caches[session_id]
        cached_tokens_before = self._max_cache_offset(prompt_cache)
        max_context_tokens = int(policy.get("max_context_tokens", 0) or 0)
        if max_context_tokens > 0 and cached_tokens_before >= max_context_tokens:
            raise RuntimeError("max_context_tokens_exceeded")
        prepared = self._prepare_session_parts(parts, turn_index=turn_index)
        final: StepLoopResult | None = None
        for event in self._iter_step_loop(
            prepared,
            prompt_cache,
            effective_max_tokens,
            failure_code="mlx_vlm_session_generation_failed",
        ):
            if isinstance(event, StepLoopResult):
                final = event
        if final is None:
            raise RuntimeError("mlx_vlm_session_generation_failed:step_loop_returned_no_result")

        audio_token_id = getattr(self._processor, "audio_token_id", None)
        audio_token_count = 0
        if audio_token_id is not None and final.input_ids is not None:
            try:
                import mlx.core as mx  # type: ignore

                audio_token_count = int(mx.sum(final.input_ids == int(audio_token_id)).item())
            except Exception:
                audio_token_count = 0

        content = strip_channel_markup(str(final.raw_text or "")).strip()
        content, finish_reason, tool_calls = parse_tool_calls(content)
        if not tool_calls and "\n\n" in content:
            content = content.split("\n\n", 1)[0].strip()
        if tool_calls:
            finish_reason = "tool_calls"
        completion_tokens = max(1, len(final.generated))
        context_tokens_total = self._max_cache_offset(prompt_cache)
        if max_context_tokens > 0 and context_tokens_total > max_context_tokens:
            self.teardown_session(session_id)
            raise RuntimeError("max_context_tokens_exceeded")
        return BackendResult(
            content=content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": final.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": final.prompt_tokens + completion_tokens,
            },
            metrics={
                "prompt_tokens_new": final.prompt_tokens,
                "cached_tokens": cached_tokens_before,
                "audio_token_count": audio_token_count,
                "prefill_ms": final.first_token_ms if final.first_token_ms is not None else final.total_ms,
                "generation_ms": final.total_ms,
                "context_tokens_total": context_tokens_total,
                "cache_offsets": self._cache_offsets(prompt_cache),
                "peak_memory_gb": final.peak_memory_gb,
                "kv_cache": {**_ensure_kv_cache_config(self).to_dict(), "enabled": False, "excludedReason": "si_drone_session_cache"},
                "memory": _memory_stats(_ensure_memory_samples(self)),
                "session_policy_max_context_tokens": max_context_tokens,
                "turn_index": int(turn_index or 0),
            },
            tool_calls=tool_calls,
        )

    def teardown_session(self, session_id: str) -> None:
        self._session_caches.pop(session_id, None)

    def _generate_with_step_loop(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> BackendResult:
        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        prepared = self._prepare_request(messages, tools=tools, options=options) if options is not None else self._prepare_request(messages, tools=tools)
        prompt_cache = self._make_request_prompt_cache()
        final: StepLoopResult | None = None
        for event in self._iter_step_loop(
            prepared,
            prompt_cache,
            effective_max_tokens,
            kv_generation_kwargs=self._kv_generation_kwargs(),
        ):
            if isinstance(event, StepLoopResult):
                final = event
        if final is None:
            raise RuntimeError("mlx_vlm_generation_failed:step_loop_returned_no_result")

        load_ms = self._load_ms if not self._load_consumed else 0
        self._load_consumed = True
        finish_reason = final.finish_reason
        content, reasoning_content = split_reasoning_markup(str(final.raw_text or ""))
        content = content.strip()
        prepared_think_control = getattr(prepared, "think_control", None)
        if not _include_reasoning_requested(options or {}, prepared_think_control.get("level", "off") if isinstance(prepared_think_control, dict) else "off"):
            reasoning_content = None
        content, parsed_finish_reason, tool_calls = parse_tool_calls(content)
        if tool_calls:
            tool_calls, hallucinated_names = filter_hallucinated_tool_calls(tool_calls, tools)
            if not tool_calls and hallucinated_names:
                content = f"[tool-call error] Model attempted unavailable tool(s): {', '.join(hallucinated_names)}"
                finish_reason = "stop"
            elif tool_calls:
                finish_reason = "tool_calls"
        else:
            finish_reason = parsed_finish_reason if finish_reason == "stop" else finish_reason
        completion_tokens = max(1, len(final.generated))
        return BackendResult(
            content=content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": final.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": final.prompt_tokens + completion_tokens,
            },
            metrics={
                "queue_wait_ms": 0,
                "load_ms": load_ms,
                "prefill_ms": final.first_token_ms if final.first_token_ms is not None else final.total_ms,
                "generation_ms": final.total_ms,
                "peak_memory_gb": final.peak_memory_gb,
                "total_ms": final.total_ms,
                "context_tokens_total": self._max_cache_offset(prompt_cache),
                "cache_offsets": self._cache_offsets(prompt_cache),
                "kv_cache": _ensure_kv_cache_config(self).to_dict(),
                "think_control": getattr(prepared, "think_control", None),
                "memory": _memory_stats(_ensure_memory_samples(self)),
            },
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None):
        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        prepared = self._prepare_request(messages, tools=tools, options=options) if options is not None else self._prepare_request(messages, tools=tools)
        # Streaming requests are intentionally stateless: never touch
        # self._session_caches or the SI Drone lifecycle cache.
        prompt_cache = self._make_request_prompt_cache()
        emitted_len = 0
        marker_buffering = False
        final: StepLoopResult | None = None
        for event in self._iter_step_loop(
            prepared,
            prompt_cache,
            effective_max_tokens,
            kv_generation_kwargs=self._kv_generation_kwargs(),
        ):
            if isinstance(event, StepLoopResult):
                final = event
                continue
            raw_so_far = self._decode_generated_tokens(list(event.generated))
            if any(sentinel in raw_so_far for sentinel in STREAM_MARKER_SENTINELS):
                marker_buffering = True
            if marker_buffering:
                continue
            visible_so_far, _reasoning_so_far = split_reasoning_markup(raw_so_far)
            safe_len = max(0, len(visible_so_far) - STREAM_SENTINEL_HOLDBACK_CHARS)
            if safe_len > emitted_len:
                chunk = visible_so_far[emitted_len:safe_len]
                if chunk:
                    yield BackendStreamChunk(text=chunk)
                emitted_len = safe_len
        if final is None:
            raise RuntimeError("mlx_vlm_generation_failed:step_loop_returned_no_result")

        load_ms = self._load_ms if not self._load_consumed else 0
        self._load_consumed = True
        finish_reason = final.finish_reason
        content, reasoning_content = split_reasoning_markup(str(final.raw_text or ""))
        content = content.strip()
        prepared_think_control = getattr(prepared, "think_control", None)
        if not _include_reasoning_requested(options or {}, prepared_think_control.get("level", "off") if isinstance(prepared_think_control, dict) else "off"):
            reasoning_content = None
        content, parsed_finish_reason, tool_calls = parse_tool_calls(content)
        if tool_calls:
            tool_calls, hallucinated_names = filter_hallucinated_tool_calls(tool_calls, tools)
            if not tool_calls and hallucinated_names:
                content = f"[tool-call error] Model attempted unavailable tool(s): {', '.join(hallucinated_names)}"
                finish_reason = "stop"
            elif tool_calls:
                content = ""
                finish_reason = "tool_calls"
        else:
            finish_reason = parsed_finish_reason if finish_reason == "stop" else finish_reason
        if not tool_calls and content:
            remainder = content[emitted_len:]
            if remainder:
                yield BackendStreamChunk(text=remainder)
        completion_tokens = max(1, len(final.generated))
        yield BackendResult(
            content=content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": final.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": final.prompt_tokens + completion_tokens,
            },
            metrics={
                "queue_wait_ms": 0,
                "load_ms": load_ms,
                "prefill_ms": final.first_token_ms if final.first_token_ms is not None else final.total_ms,
                "generation_ms": final.total_ms,
                "peak_memory_gb": final.peak_memory_gb,
                "total_ms": final.total_ms,
                "stream_sentinel_holdback_chars": STREAM_SENTINEL_HOLDBACK_CHARS,
                "stream_sentinel_set": list(STREAM_MARKER_SENTINELS),
                "context_tokens_total": self._max_cache_offset(prompt_cache),
                "cache_offsets": self._cache_offsets(prompt_cache),
                "kv_cache": _ensure_kv_cache_config(self).to_dict(),
                "think_control": getattr(prepared, "think_control", None),
                "memory": _memory_stats(_ensure_memory_samples(self)),
            },
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> BackendResult:
        if self._should_use_step_generate_path():
            return self._generate_with_step_loop(messages, max_tokens, tools=tools, options=options)

        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        prepared = self._prepare_request(messages, tools=tools, options=options) if options is not None else self._prepare_request(messages, tools=tools)
        generation_started = time.perf_counter()
        try:
            _reset_metal_peak_memory()
            samples = _ensure_memory_samples(self)
            samples["request_start"] = _memory_sample("request_start", model=self._model)
            response = self._generate(
                self._model,
                self._processor,
                prepared.prompt,
                image=prepared.image_paths or None,
                audio=prepared.audio_paths or None,
                max_tokens=effective_max_tokens,
                verbose=False,
                **self._sampling_kwargs,
            )
            samples["post_generation"] = _memory_sample("post_generation", model=self._model)
            samples["idle"] = _memory_sample("idle", model=self._model)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_generation_failed:{exc}") from exc
        finally:
            prepared.cleanup()

        raw_content = response if isinstance(response, str) else getattr(response, "text", "")
        content, reasoning_content = split_reasoning_markup(str(raw_content or ""))
        content = content.strip()
        prepared_think_control = getattr(prepared, "think_control", None)
        if not _include_reasoning_requested(options or {}, prepared_think_control.get("level", "off") if isinstance(prepared_think_control, dict) else "off"):
            reasoning_content = None
        content, finish_reason, tool_calls = parse_tool_calls(content)
        if tool_calls:
            tool_calls, hallucinated_names = filter_hallucinated_tool_calls(tool_calls, tools)
            if not tool_calls and hallucinated_names:
                content = f"[tool-call error] Model attempted unavailable tool(s): {', '.join(hallucinated_names)}"
                finish_reason = "stop"
            elif tool_calls:
                finish_reason = "tool_calls"
        prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response, "generation_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, sum(len(str(m.get("content", "")).split()) for m in messages))
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content.split()))

        total_ms = int((time.perf_counter() - generation_started) * 1000)
        load_ms = self._load_ms if not self._load_consumed else 0
        self._load_consumed = True
        peak_memory = getattr(response, "peak_memory", None)
        return BackendResult(
            content=content,
            finish_reason=finish_reason if tool_calls else str(getattr(response, "finish_reason", finish_reason) or finish_reason),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            metrics={
                "queue_wait_ms": 0,
                "load_ms": load_ms,
                "prefill_ms": None,
                "generation_ms": None,
                "peak_memory_gb": float(peak_memory) if isinstance(peak_memory, (int, float)) else None,
                "kv_cache": _ensure_kv_cache_config(self).to_dict(),
                "think_control": getattr(prepared, "think_control", None),
                "memory": _memory_stats(_ensure_memory_samples(self)),
                "total_ms": total_ms,
            },
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )


_DIFFUSION_KWARG_KEYS = (
    "max_denoising_steps",
    "diffusion_full_canvas",
    "diffusion_min_canvas_length",
    "diffusion_max_canvas_length",
    "diffusion_sampler",
    "prefill_step_size",
    "threshold",
    "min_threshold",
    "block_length",
    "num_to_transfer",
    "max_transfer_per_step",
    "editing_threshold",
    "max_post_steps",
    "stability_steps",
)


class MlxVlmDiffusionGemmaBackend(WorkerBackend):
    name = "mlx_vlm_diffusion_gemma"
    backend_id = "mlx_vlm_diffusion_gemma"
    supported_modality_set = BACKEND_ADAPTERS["mlx_vlm_diffusion_gemma"].supported_modalities

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        raw_model_path = str(self._config.get("model", {}).get("path", "")).strip()
        self._model_path = str(Path(raw_model_path).expanduser()) if raw_model_path else ""
        if not self._model_path:
            raise RuntimeError("mlx_vlm_diffusion_model_path_not_configured")
        if not Path(self._model_path).exists():
            raise RuntimeError(f"mlx_vlm_diffusion_model_path_missing:{self._model_path}")

        try:
            from mlx_vlm import generate, load  # type: ignore
            from mlx_vlm.generate.diffusion import is_diffusion_model  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_diffusion_runtime_import_failed:{exc}") from exc

        self._generate = generate
        self._is_diffusion_model = is_diffusion_model
        self._max_output_tokens = int(self._config.get("model", {}).get("maxOutputTokens", 1024) or 1024)
        sampling_cfg = self._config.get("model", {}).get("sampling", {})
        self._sampling_kwargs = MlxVlmTurboQuantBackend._generation_sampling_kwargs(sampling_cfg if isinstance(sampling_cfg, dict) else {})
        self._diffusion_kwargs = self._generation_diffusion_kwargs()
        self._load_ms = 0
        self._load_consumed = False
        self._memory_samples: dict[str, dict[str, Any] | None] = {
            "before_load": _memory_sample("before_load"),
            "after_load": None,
            "request_start": None,
            "prefill_peak": None,
            "post_generation": None,
            "idle": None,
            "after_unload": None,
        }

        load_started = time.perf_counter()
        try:
            self._model, self._processor = load(self._model_path, lazy=False)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_diffusion_model_load_failed:{exc}") from exc
        self._load_ms = int((time.perf_counter() - load_started) * 1000)
        self._memory_samples["after_load"] = _memory_sample("after_load", model=self._model)

        model_type = str(getattr(getattr(self._model, "config", None), "model_type", "") or "")
        if model_type != "diffusion_gemma":
            raise RuntimeError(f"mlx_vlm_diffusion_unsupported_model_type:{model_type or 'unknown'}")
        if not self._is_diffusion_model(self._model):
            raise RuntimeError("mlx_vlm_diffusion_missing_canvas_config")

    def stats(self) -> dict[str, Any]:
        samples = _ensure_memory_samples(self)
        samples["idle"] = _memory_sample("idle", model=getattr(self, "_model", None))
        return {"memory": _memory_stats(samples)}

    def _generation_diffusion_kwargs(self) -> dict[str, Any]:
        diffusion_cfg: dict[str, Any] = {}
        model_diffusion = self._config.get("model", {}).get("diffusion", {})
        worker_diffusion = self._config.get("worker", {}).get("diffusion", {})
        if isinstance(model_diffusion, dict):
            diffusion_cfg.update(model_diffusion)
        if isinstance(worker_diffusion, dict):
            diffusion_cfg.update(worker_diffusion)

        kwargs: dict[str, Any] = {}
        for key in _DIFFUSION_KWARG_KEYS:
            if key not in diffusion_cfg:
                continue
            value = diffusion_cfg[key]
            if value is None:
                continue
            if key == "diffusion_full_canvas":
                kwargs[key] = bool(value)
            elif key == "threshold":
                kwargs["diffusion_threshold"] = value
            else:
                kwargs[key] = value
        return kwargs

    def supported_modalities(self) -> set[str]:
        config = getattr(getattr(self._model, "config", None), "__dict__", None)
        if not isinstance(config, dict):
            return set(self.supported_modality_set)
        supported = {"text"}
        if config.get("vision_config") is not None:
            supported.add("image")
        return supported & set(self.supported_modality_set)

    def _build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        template_messages = _decode_assistant_tool_call_arguments(messages) if tools else messages
        if callable(apply_chat_template):
            try:
                return apply_chat_template(template_messages, tools=tools, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        rendered: list[str] = []
        for message in template_messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system":
                rendered.append(content)
            elif role == "assistant":
                rendered.append(f"Assistant: {content}")
            else:
                rendered.append(content)
        return "\n\n".join(part for part in rendered if part).strip()

    def _prepare_request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> VlmPreparedRequest:
        prompt_messages: list[dict[str, Any]] = []
        image_paths: list[str] = []
        temp_dir: tempfile.TemporaryDirectory[str] | None = None

        for message in messages:
            next_message = dict(message)
            parts = message.get("parts")
            if isinstance(parts, list):
                text_parts: list[str] = []
                image_placeholders: list[str] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type", ""))
                    if part_type == "text":
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                        continue
                    if part_type != "image":
                        continue
                    data_url = part.get("data_url")
                    mime_type = str(part.get("mime_type", ""))
                    if not isinstance(data_url, str):
                        raise RuntimeError("mlx_vlm_diffusion_image_missing_data_url")
                    if temp_dir is None:
                        temp_dir = tempfile.TemporaryDirectory(prefix="mlx-vlm-diffusion-input-")
                    suffix = MlxVlmTurboQuantBackend._media_suffix(mime_type, "image")
                    payload = MlxVlmTurboQuantBackend._decode_data_url(data_url)
                    media_path = Path(temp_dir.name) / f"image-{len(image_paths)}{suffix}"
                    media_path.write_bytes(payload)
                    image_paths.append(str(media_path))
                    image_placeholders.append("<|image|>")
                content_parts = [*image_placeholders, *text_parts]
                next_message["content"] = "\n".join(content_parts) if content_parts else str(message.get("content", ""))
            prompt_messages.append(next_message)

        return VlmPreparedRequest(
            prompt=self._build_prompt(prompt_messages, tools=tools),
            image_paths=image_paths,
            audio_paths=[],
            temp_dir=temp_dir,
        )

    @staticmethod
    def _tool_retry_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retry_instruction = (
            "Your previous tool-call output was malformed. Return exactly one valid Gemma tool call "
            "using the declared tools. Do not answer in prose."
        )
        if messages and messages[0].get("role") in {"system", "developer"}:
            patched = [dict(messages[0])]
            patched[0]["content"] = f"{messages[0].get('content', '')}\n\n{retry_instruction}".strip()
            patched.extend(messages[1:])
            return patched
        return [{"role": "system", "content": retry_instruction}, *messages]

    def _generate_once(
        self,
        messages: list[dict[str, Any]],
        effective_max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        prepared = self._prepare_request(messages, tools=tools)
        try:
            return self._generate(
                self._model,
                self._processor,
                prepared.prompt,
                image=prepared.image_paths or None,
                max_tokens=effective_max_tokens,
                verbose=False,
                **self._sampling_kwargs,
                **self._diffusion_kwargs,
            )
        finally:
            prepared.cleanup()

    def _build_result(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        started: float,
        tools: list[dict[str, Any]] | None,
        retry_count: int,
    ) -> BackendResult:
        raw_content = response if isinstance(response, str) else getattr(response, "text", "")
        content = strip_channel_markup(str(raw_content or "")).strip()
        content, finish_reason, tool_calls = parse_tool_calls(content)
        malformed_tool_output = bool(tools and not tool_calls and _contains_tool_call_marker(str(raw_content or "")))
        if malformed_tool_output:
            raise RuntimeError("mlx_vlm_diffusion_malformed_tool_call")
        if tool_calls:
            tool_calls, hallucinated_names = filter_hallucinated_tool_calls(tool_calls, tools)
            if not tool_calls and hallucinated_names:
                content = f"[tool-call error] Model attempted unavailable tool(s): {', '.join(hallucinated_names)}"
                finish_reason = "stop"
            elif tool_calls:
                finish_reason = "tool_calls"

        prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response, "generation_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, sum(len(str(m.get("content", "")).split()) for m in messages))
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content.split()))

        total_ms = int((time.perf_counter() - started) * 1000)
        load_ms = self._load_ms if not self._load_consumed else 0
        self._load_consumed = True
        peak_memory = getattr(response, "peak_memory", None)
        first_visible_output_ms = total_ms if content or tool_calls else None
        samples = _ensure_memory_samples(self)
        samples["post_generation"] = _memory_sample("post_generation", model=self._model)
        samples["idle"] = _memory_sample("idle", model=self._model)
        return BackendResult(
            content=content,
            finish_reason=finish_reason if tool_calls else str(getattr(response, "finish_reason", finish_reason) or finish_reason),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            metrics={
                "queue_wait_ms": 0,
                "load_ms": load_ms,
                "prefill_ms": None,
                "generation_ms": None,
                "first_visible_output_ms": first_visible_output_ms,
                "peak_memory_gb": float(peak_memory) if isinstance(peak_memory, (int, float)) else None,
                "total_ms": total_ms,
                "diffusion_canvas_tokens": int(getattr(response, "diffusion_canvas_tokens", 0) or 0),
                "diffusion_denoising_steps": int(getattr(response, "diffusion_denoising_steps", 0) or 0),
                "diffusion_work_tokens": int(getattr(response, "diffusion_work_tokens", 0) or 0),
                "diffusion_canvas_tps": float(getattr(response, "diffusion_canvas_tps", 0.0) or 0.0),
                "diffusion_work_tps": float(getattr(response, "diffusion_work_tps", 0.0) or 0.0),
                "diffusion_tool_retry_count": retry_count,
                "memory": _memory_stats(samples),
            },
            tool_calls=tool_calls,
        )

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None) -> BackendResult:
        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        generation_started = time.perf_counter()
        try:
            _reset_metal_peak_memory()
            samples = _ensure_memory_samples(self)
            samples["request_start"] = _memory_sample("request_start", model=self._model)
            response = self._generate_once(messages, effective_max_tokens, tools)
            try:
                return self._build_result(response, messages, generation_started, tools, retry_count=0)
            except RuntimeError as exc:
                if str(exc) != "mlx_vlm_diffusion_malformed_tool_call" or not tools:
                    raise
                retry_response = self._generate_once(self._tool_retry_messages(messages), effective_max_tokens, tools)
                return self._build_result(retry_response, messages, generation_started, tools, retry_count=1)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            if isinstance(exc, RuntimeError) and str(exc).startswith("mlx_vlm_diffusion_"):
                raise
            raise RuntimeError(f"mlx_vlm_diffusion_generation_failed:{exc}") from exc

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None):
        result = self.generate(messages, max_tokens, tools=tools, options=options)
        if result.content and not result.tool_calls:
            yield BackendStreamChunk(text=result.content)
        yield result


MlxBackend = MlxVlmTurboQuantBackend


def build_backend(config: dict[str, Any]) -> WorkerBackend:
    worker_cfg = config.get("worker", {})
    if bool(worker_cfg.get("stubMode", False)):
        return StubBackend()
    descriptor = backend_descriptor(config)
    if descriptor.backend_id == "mlx_vlm_turboquant":
        return MlxVlmTurboQuantBackend(config)
    if descriptor.backend_id == "mlx_vlm_diffusion_gemma":
        return MlxVlmDiffusionGemmaBackend(config)
    if descriptor.backend_id == "mlx_audio_qwen3_tts":
        return Qwen3TtsBackend(config)
    raise RuntimeError(f"unknown_backend_adapter:{configured_backend_id(config)}")
