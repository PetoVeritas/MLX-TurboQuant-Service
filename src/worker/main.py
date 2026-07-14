"""Worker process for the MLX Turbo Gemma service."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from shared.parts import unsupported_backend_modalities
from worker.backends import BackendResult, BackendStreamChunk, build_backend
from worker.config import load_config


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.buffer.write((json.dumps(payload) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def ensure_backend_supports_messages(backend: Any, messages: list[dict[str, Any]]) -> None:
    supported = backend.supported_modalities() if hasattr(backend, "supported_modalities") else {"text"}
    unsupported = unsupported_backend_modalities(messages, set(supported))
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise RuntimeError(f"unsupported_modality:Backend does not support requested modality: {names}")


def handle_generate(backend: Any, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id", "unknown")
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens")
    tools = payload.get("tools")
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}

    try:
        ensure_backend_supports_messages(backend, messages)
        if options:
            result = backend.generate(messages, max_tokens, tools=tools, options=options)
        else:
            result = backend.generate(messages, max_tokens, tools=tools)
    except RuntimeError as exc:
        return {"type": "error", "request_id": request_id, "error": str(exc)}

    return {
        "type": "completion_result",
        "request_id": request_id,
        "content": result.content,
        "finish_reason": result.finish_reason,
        "usage": result.usage,
        "metrics": result.metrics,
        "tool_calls": result.tool_calls,
        "reasoning_content": result.reasoning_content,
    }


def handle_generate_stream(backend: Any, payload: dict[str, Any]) -> None:
    request_id = payload.get("request_id", "unknown")
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens")
    tools = payload.get("tools")
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}

    try:
        ensure_backend_supports_messages(backend, messages)
        stream = backend.stream_generate(messages, max_tokens, tools=tools, options=options) if options else backend.stream_generate(messages, max_tokens, tools=tools)
        for event in stream:
            if isinstance(event, BackendStreamChunk):
                emit({"type": "completion_chunk", "request_id": request_id, "content": event.text, "reasoning_content": event.reasoning_content})
            elif isinstance(event, BackendResult):
                emit(
                    {
                        "type": "completion_result",
                        "request_id": request_id,
                        "content": event.content,
                        "finish_reason": event.finish_reason,
                        "usage": event.usage,
                        "metrics": event.metrics,
                        "tool_calls": event.tool_calls,
                        "reasoning_content": event.reasoning_content,
                    }
                )
                return
    except RuntimeError as exc:
        emit({"type": "error", "request_id": request_id, "error": str(exc)})
        return

    emit({"type": "error", "request_id": request_id, "error": "stream_generate_returned_no_result"})


def handle_session_generate(backend: Any, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id", "unknown")
    session_id = payload.get("session_id")
    parts = payload.get("parts", [])
    max_tokens = payload.get("max_tokens")
    policy = payload.get("policy", {})
    turn_index = payload.get("turn_index", 0)
    if not isinstance(session_id, str) or not session_id:
        return {"type": "error", "request_id": request_id, "error": "bad_request:missing_session_id"}
    if not hasattr(backend, "session_generate"):
        return {"type": "error", "request_id": request_id, "error": "unsupported_backend:Backend does not support SI Drone sessions"}
    try:
        result = backend.session_generate(session_id, parts, max_tokens=max_tokens, policy=policy, turn_index=turn_index)
    except RuntimeError as exc:
        return {"type": "error", "request_id": request_id, "error": str(exc)}
    return {
        "type": "session_result",
        "request_id": request_id,
        "session_id": session_id,
        "content": result.content,
        "finish_reason": result.finish_reason,
        "usage": result.usage,
        "metrics": result.metrics,
    }


def handle_session_teardown(backend: Any, payload: dict[str, Any]) -> dict[str, Any]:
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id and hasattr(backend, "teardown_session"):
        backend.teardown_session(session_id)
    return {"type": "session_teardown_ack", "session_id": session_id}


def handle_speech_generate(backend: Any, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id", "unknown")
    request = payload.get("request", {})
    if not hasattr(backend, "speech_generate"):
        return {"type": "speech_error", "request_id": request_id, "error": {"code": "unsupported_backend", "message": "Backend does not support speech.generate", "status": 503, "details": {}}}
    if not isinstance(request, dict):
        return {"type": "speech_error", "request_id": request_id, "error": {"code": "bad_request", "message": "Speech request must be an object", "status": 400, "details": {}}}
    try:
        result = backend.speech_generate(request)
    except RuntimeError as exc:
        return {"type": "speech_error", "request_id": request_id, "error": {"code": "generation_failed", "message": str(exc), "status": 500, "details": {}}}
    result["request_id"] = request_id
    return result


def backend_stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "stats", None)
    if not callable(stats):
        return {}
    try:
        result = stats()
    except Exception as exc:
        return {"error": str(exc)}
    return result if isinstance(result, dict) else {}


def main() -> int:
    config = load_config()
    backend = build_backend(config)
    emit({"type": "worker_started", "pid": os.getpid(), "backend": getattr(backend, "name", "unknown"), "stats": backend_stats(backend)})
    # Binary stdin to match supervisor's binary-mode Popen. Explicit
    # readline loop avoids `for line in sys.stdin:` which routes through
    # TextIOWrapper and raced on back-to-back streaming requests.
    stdin_buffer = sys.stdin.buffer
    while True:
        raw_bytes = stdin_buffer.readline()
        if not raw_bytes:
            break  # EOF — supervisor closed the pipe
        line = raw_bytes.decode("utf-8").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            emit({"type": "error", "error": "invalid_json"})
            continue

        command = payload.get("command")
        if command == "ping":
            emit({"type": "pong", "pid": os.getpid(), "backend": getattr(backend, "name", "unknown")})
        elif command == "generate":
            emit(handle_generate(backend, payload))
        elif command == "generate_stream":
            handle_generate_stream(backend, payload)
        elif command == "session_generate":
            emit(handle_session_generate(backend, payload))
        elif command == "session_teardown":
            emit(handle_session_teardown(backend, payload))
        elif command == "speech_generate":
            emit(handle_speech_generate(backend, payload))
        elif command == "shutdown":
            emit({"type": "shutdown_ack", "pid": os.getpid()})
            return 0
        else:
            emit({"type": "error", "error": f"unknown_command:{command}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
