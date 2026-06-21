"""Supervisor entrypoint and minimal Phase 1 HTTP shell."""

from __future__ import annotations

import ipaddress
import json
import logging
import time
import uuid
import base64
import binascii
import wave
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from shared.backend_adapters import backend_supported_modalities
from shared.config import load_config
from shared.parts import MessagePartError, configured_modalities, extract_message_parts, part_modalities, parts_to_dicts, text_from_parts, validate_part_counts
from supervisor.session_store import SessionStore
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
        self.sessions = SessionStore(
            config,
            on_count_change=self.worker.set_live_session_count,
            on_expire=self.worker.teardown_session,
        )
        self.version = "0.1.0"

    def shutdown(self) -> None:
        self.sessions.shutdown()
        self.worker.shutdown()


def configure_logging(config: dict[str, Any]) -> None:
    level_name = str(config.get("logging", {}).get("level", "info")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def split_failure(error: str) -> tuple[str | None, str]:
    if ":" not in error:
        return None, error
    prefix, remainder = error.split(":", 1)
    if prefix in {FAILURE_BACKEND, FAILURE_PROTOCOL, FAILURE_TIMEOUT, FAILURE_STARTUP, FAILURE_CRASH, FAILURE_GOVERNOR, "unsupported_modality"}:
        return prefix, remainder
    return None, error


def _session_error_status(error_type: str) -> int:
    if error_type == "session_not_found":
        return 404
    if error_type == "session_expired":
        return 410
    if error_type == "session_lost":
        return 409
    if error_type == "max_context_tokens_exceeded":
        return 409
    if error_type in {"max_turns_exceeded", "unsupported_part_type"}:
        return 415 if error_type == "unsupported_part_type" else 409
    return 400


def _wav_duration_seconds(payload: bytes) -> float | None:
    try:
        with wave.open(BytesIO(payload), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, EOFError):
        return None


def _session_modality_error(config: dict[str, Any] | None, modality: str) -> tuple[int, str, str] | None:
    if not config:
        return None
    strict = bool(config.get("modalities", {}).get("strictCapabilityCheck", True))
    if not strict:
        return None
    configured = configured_modalities(config)
    if modality not in configured:
        return 422, "unsupported_modality", f"Modality is disabled: {modality}"
    if modality not in (configured & backend_supported_modalities(config)):
        return 422, "unsupported_modality", f"Backend does not support requested modality: {modality}"
    return None


def normalize_session_parts(payload: dict[str, Any], policy: dict[str, Any], config: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]] | None, tuple[int, str, str] | None]:
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        return None, (400, "bad_request", "Field 'parts' must be a non-empty array")

    normalized: list[dict[str, Any]] = []
    audio_seconds_limit = int(policy["audio_seconds_per_turn"])
    audio_seconds_total = 0.0
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            return None, (400, "bad_request", f"Part at index {index} must be an object")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                return None, (400, "bad_request", f"Text part at index {index} must include string 'text'")
            normalized.append({"type": "text", "text": text})
            continue
        if part_type in {"image", "video", "video_frames"}:
            return None, (415, "unsupported_part_type", f"SI Drone v1 does not support part type: {part_type}")
        if part_type != "audio":
            return None, (400, "bad_request", f"Unsupported part type at index {index}: {part_type!r}")
        modality_error = _session_modality_error(config, "audio")
        if modality_error is not None:
            return None, modality_error

        audio = part.get("audio")
        if not isinstance(audio, dict):
            return None, (400, "bad_request", f"Audio part at index {index} must include an audio object")
        audio_format = str(audio.get("format", "")).strip().lower()
        if audio_format not in {"wav", "x-wav"}:
            return None, (415, "unsupported_part_type", "SI Drone v1 audio input only supports wav")
        data = audio.get("data")
        if not isinstance(data, str) or not data:
            return None, (400, "bad_request", f"Audio part at index {index} must include base64 data")
        try:
            raw_audio = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return None, (400, "bad_request", f"Audio part at index {index} has malformed base64 data")
        duration = _wav_duration_seconds(raw_audio)
        if duration is not None:
            audio_seconds_total += duration
            if audio_seconds_total > audio_seconds_limit:
                return None, (400, "bad_request", "Audio input exceeds session policy audio_seconds_per_turn")
        normalized.append(
            {
                "type": "audio",
                "data_url": f"data:audio/wav;base64,{data}",
                "mime_type": "audio/wav",
                "byte_length": len(raw_audio),
            }
        )
    return normalized, None


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
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/health":
                self._send_json(200, {"ok": True, "service": "mlx-turbo-gemma-service", "version": app.version})
                return
            if path == "/ready":
                payload = app.worker.ready_payload()
                status = 200 if payload["ok"] or payload.get("cold_load_acceptable") else 503
                self._send_json(status, payload)
                return
            if path == "/v1/models":
                self._send_json(200, app.worker.models_payload())
                return
            if path == "/admin/stats":
                if not self._require_admin_access():
                    return
                self._send_json(200, app.worker.stats_payload())
                return
            if path == "/v1/si-drones":
                if not self._require_admin_access():
                    return
                self._send_json(200, {"object": "list", "data": [record.to_public_dict() for record in app.sessions.list()]})
                return
            if path.startswith("/v1/si-drones/") and path.count("/") == 3:
                session_id = path.rsplit("/", 1)[-1]
                record = app.sessions.get(session_id)
                if record is None:
                    self._error(404, "session_not_found", f"Unknown SI Drone session: {session_id}")
                    return
                self._send_json(200, record.to_public_dict())
                return
            self._error(404, "bad_request", f"Unknown path: {self.path}")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/admin/worker/unload":
                if not self._require_admin_access():
                    return
                try:
                    self._send_json(200, app.worker.unload_worker())
                except RuntimeError as exc:
                    self._error(503, "worker_failed", str(exc), retryable=True)
                return
            if path == "/admin/worker/restart":
                if not self._require_admin_access():
                    return
                try:
                    self._send_json(200, app.worker.restart_worker())
                except RuntimeError as exc:
                    self._error(503, "worker_failed", str(exc), retryable=True)
                return
            if path == "/v1/si-drones":
                record = app.sessions.create()
                self._send_json(
                    201,
                    {
                        "session_id": record.session_id,
                        "policy": record.policy,
                        "state": record.state,
                        "created_at": record.created_at,
                        "expires_at": record.expires_at,
                    },
                )
                return
            if path.startswith("/v1/si-drones/") and path.endswith("/turns"):
                session_id = path.removeprefix("/v1/si-drones/").removesuffix("/turns").strip("/")
                if not session_id:
                    self._error(404, "session_not_found", "Missing SI Drone session id")
                    return
                payload = self._read_json_body()
                if payload is None:
                    return
                record = app.sessions.get(session_id)
                if record is None:
                    self._error(404, "session_not_found", f"SI Drone session is not available: {session_id}")
                    return
                if record.turn_count >= int(record.policy["max_turns"]):
                    self._error(409, "max_turns_exceeded", "max_turns_exceeded")
                    return
                current_worker_pid = app.worker.worker_pid()
                if record.turn_count > 0 and record.worker_binding is not None and current_worker_pid != record.worker_binding:
                    app.sessions.delete(session_id)
                    self._error(409, "session_lost", "SI Drone worker session was lost", retryable=False)
                    return
                normalized_parts, part_error = normalize_session_parts(payload, record.policy, config=app.config)
                if part_error is not None:
                    status, error_type, message = part_error
                    self._error(status, error_type, message)
                    return
                max_tokens = payload.get("max_tokens")
                if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
                    self._error(400, "bad_request", "Field 'max_tokens' must be a positive integer when provided")
                    return
                accepted, reason = app.worker.begin_request()
                if not accepted:
                    self._error(409 if reason in {"worker_busy", "queue_full"} else 503, reason or "worker_not_ready", "Worker is not ready for SI Drone turn", retryable=True)
                    return
                try:
                    record = app.sessions.begin_turn(session_id)
                except KeyError as exc:
                    error_type = str(exc).strip("'") or "session_not_found"
                    app.worker.complete_request_rejected(error_type)
                    self._error(_session_error_status(error_type), error_type, f"SI Drone session is not available: {session_id}")
                    return
                except RuntimeError as exc:
                    error_type = str(exc)
                    app.worker.complete_request_rejected(error_type)
                    self._error(_session_error_status(error_type), error_type, error_type)
                    return
                started = time.perf_counter()
                try:
                    result = app.worker.generate_session_turn(
                        session_id,
                        normalized_parts or [],
                        max_tokens=max_tokens,
                        policy=record.policy,
                        turn_index=record.turn_count,
                    )
                    app.sessions.bind_worker(session_id, app.worker.worker_pid())
                    app.worker.complete_request(success=True)
                    self._send_json(
                        200,
                        {
                            "session_id": session_id,
                            "turn_index": record.turn_count,
                            "completion": {
                                "text": result.content,
                                "finish_reason": result.finish_reason,
                            },
                            "metrics": result.metrics,
                            "usage": result.usage,
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        },
                    )
                except TimeoutError:
                    app.worker.complete_request(success=False, error="session_generate_timeout", failure_kind=FAILURE_TIMEOUT)
                    self._error(504, "timeout", "Worker SI Drone turn timed out", retryable=True)
                except RuntimeError as exc:
                    raw_error = str(exc)
                    failure_kind, clean_error = split_failure(raw_error)
                    app.worker.complete_request(success=False, error=clean_error, failure_kind=failure_kind)
                    if clean_error == "session_lost" or raw_error.endswith(":session_lost"):
                        app.sessions.delete(session_id)
                        self._error(409, "session_lost", "SI Drone worker session was lost", retryable=False)
                    elif clean_error == "max_context_tokens_exceeded" or raw_error.endswith(":max_context_tokens_exceeded"):
                        app.sessions.delete(session_id)
                        self._error(409, "max_context_tokens_exceeded", "SI Drone session exceeded max_context_tokens", retryable=False)
                    elif clean_error.startswith("unsupported_part_type") or raw_error.startswith("unsupported_part_type"):
                        self._error(415, "unsupported_part_type", clean_error, retryable=False)
                    elif clean_error.startswith("unsupported_backend") or raw_error.startswith("unsupported_backend"):
                        self._error(503, "unsupported_backend", clean_error, retryable=False)
                    else:
                        self._error(503, "worker_failed", clean_error or raw_error, retryable=True)
                except Exception as exc:  # pragma: no cover
                    LOG.exception("Unexpected supervisor error during SI Drone turn")
                    app.worker.complete_request(success=False, error=str(exc))
                    self._error(500, "internal_error", "Unexpected internal error", retryable=False)
                return
            if path != "/v1/chat/completions":
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
                elif reason == "queue_full":
                    stats = app.worker.stats_payload()
                    worker = stats.get("worker", {})
                    depth = worker.get("queue_depth", 0)
                    max_depth = worker.get("queue_max_depth", 0)
                    self._error(409, "queue_full", f"Worker queue is full ({depth}/{max_depth})", retryable=True)
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
                normalized_messages = normalize_messages(payload["messages"], config=app.config)
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
                if failure_kind == "unsupported_modality":
                    app.worker.complete_request_rejected(clean_error)
                    self._error(422, "unsupported_modality", clean_error, retryable=False)
                elif failure_kind == FAILURE_PROTOCOL:
                    app.worker.complete_request(success=False, error=clean_error, failure_kind=failure_kind)
                    self._error(503, "worker_failed", clean_error, retryable=True)
                elif failure_kind == FAILURE_BACKEND:
                    app.worker.complete_request(success=False, error=clean_error, failure_kind=failure_kind)
                    self._error(503, "worker_failed", clean_error, retryable=True)
                else:
                    app.worker.complete_request(success=False, error=clean_error, failure_kind=failure_kind)
                    self._error(503, "worker_failed", raw_error, retryable=True)
            except Exception as exc:  # pragma: no cover
                LOG.exception("Unexpected supervisor error during completion")
                app.worker.complete_request(success=False, error=str(exc))
                self._error(500, "internal_error", "Unexpected internal error", retryable=False)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/v1/si-drones/") and path.count("/") == 3:
                session_id = path.rsplit("/", 1)[-1]
                record = app.sessions.delete(session_id)
                app.worker.teardown_session(session_id)
                if record is None:
                    self._error(404, "session_not_found", f"Unknown SI Drone session: {session_id}")
                    return
                self._send_json(200, {"ok": True, "session_id": session_id, "state": "deleted"})
                return
            self._error(404, "bad_request", f"Unknown path: {self.path}")

    return Handler


def extract_message_text(content: Any, *, allow_non_text: bool = False, config: dict[str, Any] | None = None) -> str | None:
    parts = extract_message_parts(
        content,
        config=config,
        allow_empty=allow_non_text,
        allow_tool_content=allow_non_text,
    )
    if parts is None:
        return None
    return text_from_parts(parts)


def _role_allows_null_content(role: str) -> bool:
    # Assistant messages can have null content when they only carry tool_calls.
    # Tool-role messages can legitimately have null/empty content when a tool
    # returned nothing — upstream mlx_lm.server treats None as "" for tools.
    return role in {"assistant", "tool"}


def normalize_messages(messages: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        allow_non_text = _role_allows_null_content(role)
        parts = extract_message_parts(
            message.get("content"),
            config=config,
            allow_empty=allow_non_text,
            allow_tool_content=allow_non_text,
        )
        text = text_from_parts(parts or [])
        normalized_message: dict[str, Any] = {"role": role, "content": text or ""}
        if parts and any(part.type != "text" for part in parts):
            normalized_message["parts"] = parts_to_dicts(parts)
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
    config = getattr(app, "config", getattr(app.worker, "_config", {}))
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
    part_counts: dict[str, int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return 400, "bad_request", f"Message at index {index} must be an object"
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            return 400, "unsupported_request", f"Unsupported role at index {index}: {role!r}"
        try:
            parts = extract_message_parts(
                message.get("content"),
                config=config,
                allow_empty=_role_allows_null_content(role),
                allow_tool_content=_role_allows_null_content(role),
            )
        except MessagePartError as exc:
            status = 422 if exc.error_type == "unsupported_modality" else 400
            return status, exc.error_type, exc.message
        if parts is None:
            return 400, "unsupported_request", f"Message content at index {index} must contain supported typed parts"
        for modality in part_modalities(parts):
            if modality != "text":
                part_counts[modality] = part_counts.get(modality, 0) + sum(1 for part in parts if part.type == modality)
    try:
        validate_part_counts(part_counts, config)
    except MessagePartError as exc:
        return 400, exc.error_type, exc.message
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
        if reason == "queue_full":
            stats = app.worker.stats_payload()
            worker = stats.get("worker", {})
            depth = worker.get("queue_depth", 0)
            max_depth = worker.get("queue_max_depth", 0)
            return 409, "queue_full", f"Worker queue is full ({depth}/{max_depth})"
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
        app.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
