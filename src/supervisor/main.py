"""Supervisor entrypoint and minimal Phase 1 HTTP shell."""

from __future__ import annotations

import ipaddress
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from shared.config import load_config
from supervisor.worker_manager import CompletionChunk, CompletionResult, FAILURE_BACKEND, FAILURE_COOLDOWN, FAILURE_CRASH, FAILURE_GOVERNOR, FAILURE_PROTOCOL, FAILURE_STARTUP, FAILURE_TIMEOUT, WorkerManager


LOG = logging.getLogger("mlx_turbo_gemma.supervisor")
ALLOWED_CHAT_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "store",
    "tools",
    "tool_choice",
    "reasoning_effort",
}
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


class App:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.worker = WorkerManager(config)
        self.version = "0.1.0"


def configure_logging(config: dict[str, Any]) -> None:
    level_name = str(config.get("logging", {}).get("level", "info")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def split_failure(error: str) -> tuple[str | None, str]:
    if ":" not in error:
        return None, error
    prefix, remainder = error.split(":", 1)
    if prefix in {FAILURE_BACKEND, FAILURE_PROTOCOL, FAILURE_TIMEOUT, FAILURE_STARTUP, FAILURE_CRASH, FAILURE_GOVERNOR}:
        return prefix, remainder
    return None, error


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MLXTurboGemma/0.1"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_chat_stream(
            self,
            *,
            completion_id: str,
            model: str,
            stream_events: Any,
            include_usage: bool = False,
        ) -> CompletionResult:
            created = int(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

            final_result: CompletionResult | None = None
            streamed_content = False
            for event in stream_events:
                if isinstance(event, CompletionChunk):
                    if not event.content:
                        continue
                    streamed_content = True
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": event.content}, "finish_reason": None}],
                    }
                else:
                    final_result = event
                    # If the worker synthesized content into the final result
                    # (e.g. the hallucinated-tool-call break-glass message) and
                    # nothing was streamed yet, emit it as a content delta now
                    # so the client actually sees the text.
                    if event.content and not streamed_content and not event.tool_calls:
                        catch_up = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": event.content}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(catch_up)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        streamed_content = True
                    if event.tool_calls:
                        indexed_calls = [
                            {**call, "index": idx}
                            for idx, call in enumerate(event.tool_calls)
                        ]
                        tool_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"tool_calls": indexed_calls}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(tool_chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    chunk: dict[str, Any] = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": event.finish_reason}],
                    }
                    if include_usage:
                        chunk["usage"] = event.usage
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            if final_result is None:
                raise RuntimeError(f"{FAILURE_PROTOCOL}:Streaming request completed without final result")
            return final_result

        def _error(self, status: int, error_type: str, message: str, retryable: bool = False) -> None:
            self._send_json(status, {"error": {"type": error_type, "message": message, "retryable": retryable}})

        def _admin_allowed(self) -> bool:
            if not bool(app.config.get("server", {}).get("adminLocalOnly", True)):
                return True
            client_host = str(self.client_address[0]).strip()
            if client_host == "localhost":
                return True
            if client_host.startswith("::ffff:"):
                client_host = client_host.removeprefix("::ffff:")
            try:
                return ipaddress.ip_address(client_host).is_loopback
            except ValueError:
                return False

        def _require_admin_access(self) -> bool:
            if self._admin_allowed():
                return True
            self._error(403, "forbidden", "Admin endpoints are restricted to local requests")
            return False

        def _read_json_body(self) -> dict[str, Any] | None:
            length_header = self.headers.get("Content-Length")
            if not length_header:
                self._error(400, "bad_request", "Missing Content-Length")
                return None
            try:
                length = int(length_header)
            except ValueError:
                self._error(400, "bad_request", "Invalid Content-Length")
                return None
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self._error(400, "bad_request", "Malformed JSON body")
                return None
            if not isinstance(data, dict):
                self._error(400, "bad_request", "JSON body must be an object")
                return None
            return data

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "service": "mlx-turbo-gemma-service", "version": app.version})
                return
            if self.path == "/ready":
                payload = app.worker.ready_payload()
                status = 200 if payload["ok"] or payload.get("cold_load_acceptable") else 503
                self._send_json(status, payload)
                return
            if self.path == "/v1/models":
                self._send_json(200, app.worker.models_payload())
                return
            if self.path == "/admin/stats":
                if not self._require_admin_access():
                    return
                self._send_json(200, app.worker.stats_payload())
                return
            self._error(404, "bad_request", f"Unknown path: {self.path}")

        def do_POST(self) -> None:
            if self.path == "/admin/worker/unload":
                if not self._require_admin_access():
                    return
                try:
                    self._send_json(200, app.worker.unload_worker())
                except RuntimeError as exc:
                    self._error(503, "worker_failed", str(exc), retryable=True)
                return
            if self.path == "/admin/worker/restart":
                if not self._require_admin_access():
                    return
                try:
                    self._send_json(200, app.worker.restart_worker())
                except RuntimeError as exc:
                    self._error(503, "worker_failed", str(exc), retryable=True)
                return
            if self.path != "/v1/chat/completions":
                self._error(404, "bad_request", f"Unknown path: {self.path}")
                return
            payload = self._read_json_body()
            if payload is None:
                return
            error = validate_chat_request(payload, app)
            if error is not None:
                status, error_type, message = error
                self._error(status, error_type, message, retryable=status in {409, 503, 504})
                return
            accepted, reason = app.worker.begin_request()
            if not accepted:
                if reason == "worker_busy":
                    self._error(409, "worker_busy", "Worker is busy and queue depth is zero", retryable=True)
                elif reason == FAILURE_COOLDOWN:
                    stats = app.worker.stats_payload()
                    remaining = stats["worker"].get("cooldown_remaining_s", 0)
                    self._error(503, "worker_failed", f"Worker is cooling down after repeated failures ({remaining}s remaining)", retryable=True)
                else:
                    self._error(503, "worker_failed", "Service is not ready to accept requests", retryable=True)
                return
            request_started = time.perf_counter()
            completion_id = f"chatcmpl_local_{uuid.uuid4().hex[:12]}"
            try:
                max_tokens = payload.get("max_tokens")
                if max_tokens is None:
                    max_tokens = payload.get("max_completion_tokens")
                normalized_messages = normalize_messages(payload["messages"])
                tools = payload.get("tools") if isinstance(payload.get("tools"), list) else None
                if payload.get("stream") is True:
                    stream_options = payload.get("stream_options")
                    include_usage = bool(
                        isinstance(stream_options, dict)
                        and stream_options.get("include_usage") is True
                    )
                    final_result = self._send_chat_stream(
                        completion_id=completion_id,
                        model=payload["model"],
                        stream_events=app.worker.generate_completion_stream(normalized_messages, max_tokens, tools=tools),
                        include_usage=include_usage,
                    )
                    app.worker.complete_request(success=True)
                    LOG.info(
                        "http_completion_success %s",
                        json.dumps(
                            {
                                "completion_id": completion_id,
                                "model": payload["model"],
                                "worker_pid": app.worker.stats_payload()["worker"]["pid"],
                                "usage": final_result.usage,
                                "metrics": final_result.metrics,
                                "elapsed_ms": int((time.perf_counter() - request_started) * 1000),
                                "stream": True,
                            },
                            sort_keys=True,
                        ),
                    )
                else:
                    result = app.worker.generate_completion(normalized_messages, max_tokens, tools=tools)
                    app.worker.complete_request(success=True)
                    response_payload = {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None if result.tool_calls else result.content,
                                    **({"tool_calls": result.tool_calls} if result.tool_calls else {}),
                                },
                                "finish_reason": result.finish_reason,
                            }
                        ],
                        "usage": result.usage,
                        "metrics": result.metrics,
                    }
                    LOG.info(
                        "http_completion_success %s",
                        json.dumps(
                            {
                                "completion_id": completion_id,
                                "model": payload["model"],
                                "worker_pid": app.worker.stats_payload()["worker"]["pid"],
                                "usage": result.usage,
                                "metrics": result.metrics,
                                "elapsed_ms": int((time.perf_counter() - request_started) * 1000),
                                "stream": False,
                            },
                            sort_keys=True,
                        ),
                    )
                    self._send_json(200, response_payload)
            except TimeoutError as exc:
                LOG.warning("http_completion_timeout %s", json.dumps({"model": payload["model"], "error": str(exc)}, sort_keys=True))
                app.worker.complete_request(success=False, error=str(exc), failure_kind=FAILURE_TIMEOUT)
                self._error(504, "timeout", "Worker request timed out", retryable=True)
            except RuntimeError as exc:
                raw_error = str(exc)
                failure_kind, clean_error = split_failure(raw_error)
                LOG.warning(
                    "http_completion_failed %s",
                    json.dumps({"model": payload["model"], "error": clean_error, "failure_kind": failure_kind}, sort_keys=True),
                )
                app.worker.complete_request(success=False, error=clean_error, failure_kind=failure_kind)
                if failure_kind == FAILURE_PROTOCOL:
                    self._error(503, "worker_failed", clean_error, retryable=True)
                elif failure_kind == FAILURE_BACKEND:
                    self._error(503, "worker_failed", clean_error, retryable=True)
                else:
                    self._error(503, "worker_failed", raw_error, retryable=True)
            except Exception as exc:  # pragma: no cover
                LOG.exception("Unexpected supervisor error during completion")
                app.worker.complete_request(success=False, error=str(exc))
                self._error(500, "internal_error", "Unexpected internal error", retryable=False)

    return Handler


def extract_message_text(content: Any, *, allow_non_text: bool = False) -> str | None:
    if content is None:
        return "" if allow_non_text else None
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                return None
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item_type in {"thinking", "reasoning"}:
                continue
            elif allow_non_text and item_type in {"toolCall", "tool_call", "function_call", "tool_use"}:
                continue
            else:
                return None
        return "\n".join(part for part in parts if part)
    return None


def _role_allows_null_content(role: str) -> bool:
    # Assistant messages can have null content when they only carry tool_calls.
    # Tool-role messages can legitimately have null/empty content when a tool
    # returned nothing — upstream mlx_lm.server treats None as "" for tools.
    return role in {"assistant", "tool"}


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        allow_non_text = _role_allows_null_content(role)
        text = extract_message_text(message.get("content"), allow_non_text=allow_non_text)
        normalized_message: dict[str, Any] = {"role": role, "content": text or ""}
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            normalized_message["tool_calls"] = message["tool_calls"]
        if role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id:
                normalized_message["tool_call_id"] = tool_call_id
            tool_name = message.get("name") or message.get("toolName")
            if isinstance(tool_name, str) and tool_name:
                normalized_message["name"] = tool_name
        normalized.append(normalized_message)
    return normalized


def validate_chat_request(payload: dict[str, Any], app: App) -> tuple[int, str, str] | None:
    unknown_fields = set(payload) - ALLOWED_CHAT_FIELDS
    if unknown_fields:
        return 400, "unsupported_request", f"Unsupported top-level fields: {', '.join(sorted(unknown_fields))}"
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return 400, "bad_request", "Field 'model' must be a non-empty string"
    if model != app.worker.model_id:
        return 404, "model_not_found", f"Unknown model id: {model}"
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return 400, "bad_request", "Field 'messages' must be a non-empty array"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return 400, "bad_request", f"Message at index {index} must be an object"
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            return 400, "unsupported_request", f"Unsupported role at index {index}: {role!r}"
        if extract_message_text(message.get("content"), allow_non_text=_role_allows_null_content(role)) is None:
            return 400, "unsupported_request", f"Message content at index {index} must be text or text parts"
    if "stream" in payload:
        stream = payload["stream"]
        if not isinstance(stream, bool):
            return 400, "bad_request", "Field 'stream' must be a boolean when provided"
    if "stream_options" in payload:
        stream_options = payload["stream_options"]
        if stream_options is not None and not isinstance(stream_options, dict):
            return 400, "bad_request", "Field 'stream_options' must be an object when provided"
        if isinstance(stream_options, dict):
            include_usage = stream_options.get("include_usage")
            if include_usage is not None and not isinstance(include_usage, bool):
                return 400, "bad_request", "Field 'stream_options.include_usage' must be a boolean when provided"
    if "max_tokens" in payload and (not isinstance(payload["max_tokens"], int) or payload["max_tokens"] <= 0):
        return 400, "bad_request", "Field 'max_tokens' must be a positive integer"
    if "max_completion_tokens" in payload and (
        not isinstance(payload["max_completion_tokens"], int) or payload["max_completion_tokens"] <= 0
    ):
        return 400, "bad_request", "Field 'max_completion_tokens' must be a positive integer"
    if not app.worker.can_accept_requests():
        reason = app.worker.rejection_reason()
        if reason == "worker_busy":
            return 409, "worker_busy", "Worker is busy and queue depth is zero"
        if reason == FAILURE_COOLDOWN:
            stats = app.worker.stats_payload()
            remaining = stats["worker"].get("cooldown_remaining_s", 0)
            return 503, "worker_failed", f"Worker is cooling down after repeated failures ({remaining}s remaining)"
        return 503, "worker_failed", "Service is not ready to accept requests"
    return None


def main() -> int:
    config = load_config()
    configure_logging(config)
    app = App(config)
    host = str(config["server"]["host"])
    port = int(config["server"]["port"])
    server = ThreadingHTTPServer((host, port), make_handler(app))
    LOG.info("Starting supervisor shell on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down supervisor shell")
    finally:
        app.worker.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
