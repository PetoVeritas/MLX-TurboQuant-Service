"""Worker process for the MLX Turbo Gemma service."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from worker.backends import BackendResult, BackendStreamChunk, build_backend
from worker.config import load_config


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def handle_generate(backend: Any, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("request_id", "unknown")
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens")
    tools = payload.get("tools")

    try:
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
    }


def handle_generate_stream(backend: Any, payload: dict[str, Any]) -> None:
    request_id = payload.get("request_id", "unknown")
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens")
    tools = payload.get("tools")

    try:
        for event in backend.stream_generate(messages, max_tokens, tools=tools):
            if isinstance(event, BackendStreamChunk):
                emit({"type": "completion_chunk", "request_id": request_id, "content": event.text})
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
                    }
                )
                return
    except RuntimeError as exc:
        emit({"type": "error", "request_id": request_id, "error": str(exc)})
        return

    emit({"type": "error", "request_id": request_id, "error": "stream_generate_returned_no_result"})


def main() -> int:
    config = load_config()
    backend = build_backend(config)
    emit({"type": "worker_started", "pid": os.getpid(), "backend": getattr(backend, "name", "unknown")})
    for raw in sys.stdin:
        line = raw.strip()
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
        elif command == "shutdown":
            emit({"type": "shutdown_ack", "pid": os.getpid()})
            return 0
        else:
            emit({"type": "error", "error": f"unknown_command:{command}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
