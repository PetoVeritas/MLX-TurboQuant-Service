"""Worker backend contract and Phase 1 backends."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.backend_adapters import BACKEND_ADAPTERS, backend_descriptor, configured_backend_id, turboquant_supported_modalities


@dataclass
class BackendResult:
    content: str
    finish_reason: str
    usage: dict[str, int]
    metrics: dict[str, int | float | None]
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class BackendStreamChunk:
    text: str


@dataclass
class VlmPreparedRequest:
    prompt: str
    image_paths: list[str]
    audio_paths: list[str]
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    add_special_tokens: bool | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


_TOOL_SENTINEL_RE = re.compile(r"</?\|?tool_call\|?>")
_QUOTE_SENTINEL_RE = re.compile(r"<\|\"\|>")
_QUOTE_SENTINEL_PAIR_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_BARE_KEY_RE = re.compile(r'([\{,]\s*)([A-Za-z_]\w*)(\s*:)')
_PLACEHOLDER_RE = re.compile(r"\u0000MLXTG_STR_(\d+)\u0000")

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

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> BackendResult:
        raise NotImplementedError

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None):
        result = self.generate(messages, max_tokens, tools=tools)
        if result.content:
            yield BackendStreamChunk(text=result.content)
        yield result

    def teardown_session(self, _session_id: str) -> None:
        return


class StubBackend(WorkerBackend):
    name = "stub"
    backend_id = "stub"
    supported_modality_set = BACKEND_ADAPTERS["stub"].supported_modalities

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> BackendResult:
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

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None):
        result = self.generate(messages, max_tokens, tools=tools)
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
        self._load_ms = 0
        self._load_consumed = False

        load_started = time.perf_counter()
        try:
            self._model, self._processor = load(self._model_path, lazy=False)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_model_load_failed:{exc}") from exc
        self._load_ms = int((time.perf_counter() - load_started) * 1000)

        model_type = str(getattr(getattr(self._model, "config", None), "model_type", "") or "")
        if model_type not in {"gemma4", "gemma4_unified"}:
            raise RuntimeError(f"mlx_vlm_unsupported_model_type:{model_type or 'unknown'}")

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

    def _build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        template_messages = _decode_assistant_tool_call_arguments(messages) if tools else messages
        if callable(apply_chat_template):
            try:
                return apply_chat_template(template_messages, tools=tools, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        if not tools and self._uses_gemma4_manual_chat_template():
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

    def _prepare_request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> VlmPreparedRequest:
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

        return VlmPreparedRequest(
            prompt=self._build_prompt(prompt_messages, tools=tools),
            image_paths=image_paths,
            audio_paths=audio_paths,
            temp_dir=temp_dir,
            add_special_tokens=self._uses_gemma4_manual_chat_template() and not tools,
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
                prompt = self._build_prompt([{"role": "user", "content": prompt_content}], tools=None)
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
            self._session_caches[session_id] = self._vlm_cache.make_prompt_cache(self._model.language_model)
        prompt_cache = self._session_caches[session_id]
        cached_tokens_before = self._max_cache_offset(prompt_cache)
        max_context_tokens = int(policy.get("max_context_tokens", 0) or 0)
        if max_context_tokens > 0 and cached_tokens_before >= max_context_tokens:
            raise RuntimeError("max_context_tokens_exceeded")
        prepared = self._prepare_session_parts(parts, turn_index=turn_index)
        generation_started = time.perf_counter()
        first_token_ms: int | None = None
        generated: list[int] = []
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
            prompt_tokens_new = int(getattr(input_ids, "size", 0) or 0)
            audio_token_id = getattr(self._processor, "audio_token_id", None)
            audio_token_count = 0
            if audio_token_id is not None:
                try:
                    audio_token_count = int(mx.sum(input_ids == int(audio_token_id)).item())
                except Exception:
                    audio_token_count = 0
            mx.reset_peak_memory()
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
                **extra_kwargs,
            )
            for token_index, (token, _logprobs) in enumerate(gen):
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - generation_started) * 1000)
                token_id = int(token.item() if hasattr(token, "item") else token)
                generated.append(token_id)
                if token_id in stop_ids:
                    break
                try:
                    decoded_so_far = tokenizer.decode(generated)
                except Exception:
                    decoded_so_far = ""
                if "<end_of_turn" in decoded_so_far or "<start_of_turn" in decoded_so_far:
                    break
            try:
                raw_text = tokenizer.decode(generated, skip_special_tokens=True)
            except TypeError:
                raw_text = tokenizer.decode(generated)
            peak_memory_gb = float(mx.get_peak_memory() / 1e9)
            mx.clear_cache()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_session_generation_failed:{exc}") from exc
        finally:
            prepared.cleanup()

        total_ms = int((time.perf_counter() - generation_started) * 1000)
        content = strip_channel_markup(str(raw_text or "")).strip()
        content, finish_reason, tool_calls = parse_tool_calls(content)
        if not tool_calls and "\n\n" in content:
            content = content.split("\n\n", 1)[0].strip()
        if tool_calls:
            finish_reason = "tool_calls"
        completion_tokens = max(1, len(generated))
        context_tokens_total = self._max_cache_offset(prompt_cache)
        if max_context_tokens > 0 and context_tokens_total > max_context_tokens:
            self.teardown_session(session_id)
            raise RuntimeError("max_context_tokens_exceeded")
        return BackendResult(
            content=content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": prompt_tokens_new,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens_new + completion_tokens,
            },
            metrics={
                "prompt_tokens_new": prompt_tokens_new,
                "cached_tokens": cached_tokens_before,
                "audio_token_count": audio_token_count,
                "prefill_ms": first_token_ms if first_token_ms is not None else total_ms,
                "generation_ms": total_ms,
                "context_tokens_total": context_tokens_total,
                "cache_offsets": self._cache_offsets(prompt_cache),
                "peak_memory_gb": peak_memory_gb,
                "session_policy_max_context_tokens": max_context_tokens,
                "turn_index": int(turn_index or 0),
            },
            tool_calls=tool_calls,
        )

    def teardown_session(self, session_id: str) -> None:
        self._session_caches.pop(session_id, None)

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> BackendResult:
        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        prepared = self._prepare_request(messages, tools=tools)
        generation_started = time.perf_counter()
        try:
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
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_generation_failed:{exc}") from exc
        finally:
            prepared.cleanup()

        raw_content = response if isinstance(response, str) else getattr(response, "text", "")
        content = strip_channel_markup(str(raw_content or "")).strip()
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
                "total_ms": total_ms,
            },
            tool_calls=tool_calls,
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

        load_started = time.perf_counter()
        try:
            self._model, self._processor = load(self._model_path, lazy=False)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_vlm_diffusion_model_load_failed:{exc}") from exc
        self._load_ms = int((time.perf_counter() - load_started) * 1000)

        model_type = str(getattr(getattr(self._model, "config", None), "model_type", "") or "")
        if model_type != "diffusion_gemma":
            raise RuntimeError(f"mlx_vlm_diffusion_unsupported_model_type:{model_type or 'unknown'}")
        if not self._is_diffusion_model(self._model):
            raise RuntimeError("mlx_vlm_diffusion_missing_canvas_config")

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
            },
            tool_calls=tool_calls,
        )

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> BackendResult:
        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        generation_started = time.perf_counter()
        try:
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

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None):
        result = self.generate(messages, max_tokens, tools=tools)
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
    raise RuntimeError(f"unknown_backend_adapter:{configured_backend_id(config)}")
