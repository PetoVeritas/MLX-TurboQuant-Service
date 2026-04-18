"""Worker backend contract and Phase 1 backends."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendResult:
    content: str
    finish_reason: str
    usage: dict[str, int]
    metrics: dict[str, int | None]
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class BackendStreamChunk:
    text: str


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
_CHANNEL_MARKER_RE = re.compile(rf"<\|?channel\|>\s*({_CHANNEL_NAMES})\b", re.IGNORECASE)
# Liberal matcher for detection — catches unknown / partial / mangled names too.
_CHANNEL_MARKER_ANY_RE = re.compile(r"<\|?channel\|>\s*[A-Za-z_][\w-]*\b")
# Defensive stripping regex: removes ONLY the marker characters, not any
# following word. Prevents the strict-unknown path (e.g. a display artifact
# like "<channel|>The forecast") from eating the first word of real content.
_CHANNEL_MARKER_BARE_RE = re.compile(r"<\|?channel\|>")
_STRAY_CONTROL_RE = re.compile(r"<\|(?:message|start|end|return)\|>")
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

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> BackendResult:
        raise NotImplementedError

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None):
        result = self.generate(messages, max_tokens, tools=tools)
        if result.content:
            yield BackendStreamChunk(text=result.content)
        yield result


class StubBackend(WorkerBackend):
    name = "stub"

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


class MlxBackend(WorkerBackend):
    name = "mlx"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        raw_model_path = str(self._config.get("model", {}).get("path", "")).strip()
        self._model_path = str(Path(raw_model_path).expanduser()) if raw_model_path else ""
        if not self._model_path:
            raise RuntimeError("mlx_model_path_not_configured")
        if not Path(self._model_path).exists():
            raise RuntimeError(f"mlx_model_path_missing:{self._model_path}")

        try:
            from mlx_lm import load, stream_generate  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_runtime_import_failed:{exc}") from exc

        self._stream_generate = stream_generate
        self._max_output_tokens = int(self._config.get("model", {}).get("maxOutputTokens", 1024) or 1024)
        self._load_ms = 0
        self._load_consumed = False

        # Default sampler. Greedy decoding on a tool-trained model tends to
        # lock onto a tool-call opener for prompts that pattern-match "function
        # call" (weather, time, search, etc.). A mild non-greedy sampler keeps
        # ordinary prompts answering in prose. If the installed mlx_lm version
        # doesn't expose make_sampler, fall back to greedy so we stay
        # backward-compatible instead of hard-failing the worker.
        sampling_cfg = self._config.get("model", {}).get("sampling") or {}
        try:
            temperature = float(sampling_cfg.get("temperature", 0.7))
        except (TypeError, ValueError):
            temperature = 0.7
        try:
            top_p = float(sampling_cfg.get("topP", 0.9))
        except (TypeError, ValueError):
            top_p = 0.9
        self._sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore

            self._sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:  # pragma: no cover - depends on local runtime
            self._sampler = None

        load_started = time.perf_counter()
        try:
            self._model, self._tokenizer = load(self._model_path, lazy=False)
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_model_load_failed:{exc}") from exc
        self._load_ms = int((time.perf_counter() - load_started) * 1000)

    def _build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            template_kwargs["tools"] = tools
        # Gemma's chat template expects structured tool_call arguments (dicts),
        # not OpenAI-style stringified JSON. Decode before templating so the
        # model sees its own prior tool call correctly rendered on replay.
        template_messages = _decode_assistant_tool_call_arguments(messages)
        try:
            return self._tokenizer.apply_chat_template(template_messages, **template_kwargs)
        except Exception:
            rendered: list[str] = []
            if tools:
                rendered.append("[TOOLS]\n" + json.dumps(tools, indent=2))
            for message in template_messages:
                role = str(message.get("role", "user")).upper()
                content = str(message.get("content", ""))
                rendered.append(f"[{role}]\n{content}")
            rendered.append("[ASSISTANT]\n")
            return "\n\n".join(rendered)

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None) -> BackendResult:
        final_result: BackendResult | None = None
        for event in self.stream_generate(messages, max_tokens, tools=tools):
            if isinstance(event, BackendResult):
                final_result = event
        if final_result is None:
            raise RuntimeError("mlx_generation_returned_no_result")
        return final_result

    def stream_generate(self, messages: list[dict[str, Any]], max_tokens: int | None, tools: list[dict[str, Any]] | None = None):
        prompt = self._build_prompt(messages, tools=tools)
        effective_max_tokens = self._max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            effective_max_tokens = min(max_tokens, self._max_output_tokens)

        generation_started = time.perf_counter()
        final_response = None
        chunks: list[str] = []
        buffered_chunks: list[str] = []
        streaming_mode: str | None = None
        stream_kwargs: dict[str, Any] = {"max_tokens": effective_max_tokens}
        if self._sampler is not None:
            stream_kwargs["sampler"] = self._sampler
        try:
            for response in self._stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                **stream_kwargs,
            ):
                final_response = response
                if response.text:
                    chunks.append(response.text)
                    if streaming_mode == "tool":
                        # Keep accumulating in buffered_chunks too — if the
                        # final parse fails, we need to flush these as text
                        # rather than silently dropping them.
                        buffered_chunks.append(response.text)
                        continue
                    if streaming_mode == "text":
                        # If the model started out looking like plain text but
                        # now emits tool-call or channel markup, pivot into
                        # "tool" (buffered) mode so we stop streaming raw
                        # markup to the client. Anything already streamed is
                        # already gone, but we contain further leakage and let
                        # the final parse decide.
                        if (
                            _contains_tool_call_marker(response.text)
                            or response.text.lstrip().startswith("<|tool_call")
                            or _contains_channel_marker(response.text)
                        ):
                            streaming_mode = "tool"
                            buffered_chunks.append(response.text)
                            continue
                        yield BackendStreamChunk(text=response.text)
                        continue
                    buffered_chunks.append(response.text)
                    buffered_text = "".join(buffered_chunks)
                    if (
                        buffered_text.lstrip().startswith("call:")
                        or buffered_text.lstrip().startswith("<|tool_call")
                        or _contains_tool_call_marker(buffered_text)
                        or _contains_channel_marker(buffered_text)
                    ):
                        streaming_mode = "tool"
                        continue
                    if _looks_like_tool_call_prefix(buffered_text) or _looks_like_channel_prefix(buffered_text):
                        continue
                    streaming_mode = "text"
                    for chunk_text in buffered_chunks:
                        yield BackendStreamChunk(text=chunk_text)
                    buffered_chunks.clear()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"mlx_generation_failed:{exc}") from exc

        if final_response is None:
            raise RuntimeError("mlx_generation_returned_no_response")

        content = "".join(chunks).strip()
        if not content:
            content = str(getattr(final_response, "text", "")).strip()
        # If we buffered anything in "tool" mode (or never committed) and the
        # final content does NOT parse as a tool call, flush those buffered
        # chunks as plain text so the SSE client actually sees what the model
        # emitted instead of an empty assistant turn. We run the flush through
        # strip_channel_markup so raw ``<|channel|>...`` / ``<|message|>``
        # tokens never leak to the client — callers only ever see the
        # ``final`` channel content when channel markup was present.
        if streaming_mode != "text" and buffered_chunks:
            _leftover, _reason, parsed_tool_calls = parse_tool_calls(content)
            if parsed_tool_calls is None:
                flushed_text = strip_channel_markup("".join(buffered_chunks))
                if flushed_text:
                    yield BackendStreamChunk(text=flushed_text)
                buffered_chunks.clear()
        yield self._build_result(messages, final_response, content, generation_started, tools=tools)

    def _build_result(
        self,
        messages: list[dict[str, Any]],
        final_response: Any,
        content: str,
        generation_started: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> BackendResult:
        prompt_tokens = int(getattr(final_response, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(final_response, "generation_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, sum(len(str(m.get("content", "")).split()) for m in messages))
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content.split()))

        prefill_ms = None
        prompt_tps = getattr(final_response, "prompt_tps", None)
        if isinstance(prompt_tps, (int, float)) and prompt_tps > 0 and prompt_tokens > 0:
            prefill_ms = int((prompt_tokens / float(prompt_tps)) * 1000)

        generation_ms = None
        generation_tps = getattr(final_response, "generation_tps", None)
        if isinstance(generation_tps, (int, float)) and generation_tps > 0 and completion_tokens > 0:
            generation_ms = int((completion_tokens / float(generation_tps)) * 1000)

        total_ms = int((time.perf_counter() - generation_started) * 1000)
        load_ms = self._load_ms if not self._load_consumed else 0
        self._load_consumed = True
        raw_finish_reason = str(getattr(final_response, "finish_reason", "stop") or "stop")
        # Strip Harmony/Gemma channel markup before tool-call parsing so we
        # don't match ``call:...`` tokens that live inside a hidden-reasoning
        # segment, and so the final content handed to the client is clean.
        content = strip_channel_markup(content)
        parsed_content, parsed_finish_reason, parsed_tool_calls = parse_tool_calls(content)

        # If we truncated mid-tool-call (length-limited) and the parser failed,
        # don't leak half-emitted `call:...{` markup to the client. Return an
        # explicit `length` finish with empty content so the caller can treat
        # it as a truncated response rather than as textual leakage.
        if (
            parsed_tool_calls is None
            and raw_finish_reason == "length"
            and _contains_tool_call_marker(parsed_content)
        ):
            parsed_content = ""

        # Guard against hallucinated tool names — the model sometimes calls a
        # tool that was not in the request's tools[] list. If every call is
        # hallucinated, surface a clear textual error with finish_reason=stop
        # so the caller does NOT keep replaying and re-calling the same
        # nonexistent tool in an infinite loop. This is the "break-glass" fix
        # for the known "Gemma loops on a nonexistent weather tool" bug.
        kept_tool_calls = parsed_tool_calls
        hallucinated: list[str] = []
        if parsed_tool_calls is not None:
            kept_tool_calls, hallucinated = filter_hallucinated_tool_calls(parsed_tool_calls, tools)
            if kept_tool_calls is None and hallucinated:
                names = ", ".join(sorted({name for name in hallucinated if name}))
                message = (
                    "[tool-call error] The model attempted to call a tool that is not available"
                )
                if names:
                    message += f" (requested: {names})."
                else:
                    message += "."
                message += " No tool was executed. Please answer without invoking that tool."
                parsed_content = message
                parsed_finish_reason = "stop"
                kept_tool_calls = None

        if kept_tool_calls:
            finish_reason = "tool_calls"
        elif parsed_tool_calls is not None and kept_tool_calls is None and hallucinated:
            finish_reason = "stop"
        else:
            finish_reason = parsed_finish_reason if kept_tool_calls else raw_finish_reason

        return BackendResult(
            content=parsed_content,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            metrics={
                "queue_wait_ms": 0,
                "load_ms": load_ms,
                "prefill_ms": prefill_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
            tool_calls=kept_tool_calls,
        )


def build_backend(config: dict[str, Any]) -> WorkerBackend:
    worker_cfg = config.get("worker", {})
    if bool(worker_cfg.get("stubMode", False)):
        return StubBackend()
    return MlxBackend(config)
