#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated E2B mlx-vlm service smokes.")
    parser.add_argument("--model", required=True, help="Local E2B TurboQuant model directory")
    parser.add_argument("--venv", required=True, help="Patched mlx-vlm virtualenv")
    parser.add_argument("--image", required=True, help="Image fixture path")
    parser.add_argument("--audio", required=True, help="Audio fixture path")
    parser.add_argument("--port", type=int, default=4029, help="Local isolated port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args()


def data_url(path: Path, mime_type: str) -> str:
    return f"data:{mime_type};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def request_json(
    method: str,
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        data = json.loads(raw.decode("utf-8")) if raw else {}
        return response.status, data
    finally:
        conn.close()


def wait_for_health(host: str, port: int, deadline_s: float = 15.0) -> dict[str, Any]:
    deadline = time.time() + deadline_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            status, payload = request_json("GET", host, port, "/health", timeout=1.0)
            if status == 200:
                return payload
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"service did not become healthy: {last_error}")


def chat_payload(model_id: str, content: Any, *, max_tokens: int = 32) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }


def chat_completion(host: str, port: int, payload: dict[str, Any], *, timeout: float = 180.0) -> tuple[int, dict[str, Any]]:
    return request_json("POST", host, port, "/v1/chat/completions", payload, timeout=timeout)


def choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def assert_not_degenerate(name: str, text: str, prompt: str) -> None:
    normalized = text.strip()
    if not normalized:
        raise AssertionError(f"{name} smoke returned empty content")
    if normalized == prompt.strip():
        raise AssertionError(f"{name} smoke echoed the prompt verbatim: {text!r}")
    tokens = normalized.split()
    run_length = 1
    previous: str | None = None
    for token in tokens:
        run_length = run_length + 1 if token == previous else 1
        if run_length > 8:
            raise AssertionError(f"{name} smoke repeated token {token!r} more than 8 times: {text!r}")
        previous = token


def assert_status(name: str, status: int, expected: int, payload: dict[str, Any]) -> None:
    if status != expected:
        raise AssertionError(f"{name}: expected HTTP {expected}, got {status}: {payload}")


def load_parity(model_path: Path) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text())
    text = config["text_config"]
    quant = config.get("quantization") or {}
    parity = {
        "max_position_embeddings": text.get("max_position_embeddings"),
        "sliding_window": text.get("sliding_window"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_kv_shared_layers": text.get("num_kv_shared_layers"),
        "first_kv_shared_layer_idx": text.get("num_hidden_layers") - text.get("num_kv_shared_layers"),
        "hidden_size_per_layer_input": text.get("hidden_size_per_layer_input"),
        "attention_k_eq_v": text.get("attention_k_eq_v"),
        "quantization": quant,
    }
    expected = {
        "max_position_embeddings": 131072,
        "sliding_window": 512,
        "num_hidden_layers": 35,
        "num_kv_shared_layers": 20,
        "first_kv_shared_layer_idx": 15,
        "hidden_size_per_layer_input": 256,
        "attention_k_eq_v": False,
    }
    for key, value in expected.items():
        if parity.get(key) != value:
            raise AssertionError(f"parity mismatch for {key}: {parity.get(key)!r} != {value!r}")
    if quant.get("bits") != 4 or quant.get("group_size") != 64 or quant.get("mode") != "affine":
        raise AssertionError(f"quantization mismatch: {quant!r}")
    return parity


def run_cancel_probe(host: str, port: int, model_id: str) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Write a long numbered list of 500 short items."}],
        "max_tokens": 1024,
    }
    started = time.time()
    try:
        chat_completion(host, port, payload, timeout=0.25)
        outcome = "completed_before_client_timeout"
    except (TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
        outcome = f"client_cancelled:{type(exc).__name__}"

    # Service should recover to accepting requests after the in-flight request
    # notices the closed client or completes server-side.
    deadline = time.time() + 45
    last_ready: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            status, ready = request_json("GET", host, port, "/ready", timeout=2.0)
            last_ready = ready
            worker = ready.get("worker", {}) if isinstance(ready, dict) else {}
            if status == 200 and ready.get("accepting_requests") and worker.get("queue_depth", 0) == 0:
                return {
                    "outcome": outcome,
                    "recovered": True,
                    "elapsed_s": round(time.time() - started, 3),
                    "ready": ready,
                }
        except Exception:
            pass
        time.sleep(0.5)
    return {
        "outcome": outcome,
        "recovered": False,
        "elapsed_s": round(time.time() - started, 3),
        "ready": last_ready,
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    venv_path = Path(args.venv).expanduser()
    image_path = Path(args.image).expanduser()
    audio_path = Path(args.audio).expanduser()
    worker_python = venv_path / "bin" / "python"

    if not model_path.exists():
        raise SystemExit(f"missing model path: {model_path}")
    if not worker_python.exists():
        raise SystemExit(f"missing venv python: {worker_python}")
    if not image_path.exists():
        raise SystemExit(f"missing image fixture: {image_path}")
    if not audio_path.exists():
        raise SystemExit(f"missing audio fixture: {audio_path}")

    model_id = "e2b-turboquant-smoke"
    parity = load_parity(model_path)
    report: dict[str, Any] = {
        "model_path": str(model_path),
        "worker_python": str(worker_python),
        "host": args.host,
        "port": args.port,
        "parity": parity,
        "checks": {},
    }

    with tempfile.TemporaryDirectory(prefix="mlx-e2b-smoke-") as tmp:
        tmp_dir = Path(tmp)
        config_path = tmp_dir / "local-smoke.json"
        config_path.write_text(
            json.dumps(
                {
                    "server": {"host": args.host, "port": args.port, "adminLocalOnly": True},
                    "model": {"id": model_id, "path": str(model_path), "maxOutputTokens": 128},
                    "worker": {
                        "backend": "mlx_vlm_turboquant",
                        "pythonExecutable": str(worker_python),
                        "stubMode": False,
                        "lazyLoad": True,
                        "startupTimeoutMs": 120000,
                        "requestTimeoutMs": 180000,
                        "probeTimeoutMs": 5000,
                        "queue": {"maxDepth": 1},
                        "idleUnload": {"enabled": False},
                        "recycle": {"maxConsecutiveErrors": 3, "cooldownMs": 15000},
                    },
                    "governor": {"enabled": False},
                    "modalities": {
                        "text": {"enabled": True},
                        "image": {
                            "enabled": True,
                            "maxInputs": 4,
                            "maxBytesMb": 20,
                            "allowedMimeTypes": ["image/png", "image/jpeg", "image/webp"],
                            "transport": ["data_url"],
                        },
                        "audio": {
                            "enabled": True,
                            "maxInputs": 2,
                            "maxBytesMb": 50,
                            "allowedMimeTypes": ["audio/wav", "audio/mpeg", "audio/mp4", "audio/x-m4a"],
                            "transport": ["data_url"],
                            "transcode": {"enabled": False},
                        },
                        "video": {
                            "enabled": False,
                            "maxInputs": 1,
                            "maxBytesMb": 200,
                            "allowedMimeTypes": ["video/mp4"],
                            "transport": ["data_url"],
                        },
                        "document": {"enabled": False},
                        "strictCapabilityCheck": True,
                    },
                    "logging": {"level": "info"},
                }
            )
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR / "src")
        env["MLX_GEMMA_CONFIG"] = str(config_path)
        supervisor = subprocess.Popen(
            [sys.executable, "-m", "supervisor.main"],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            report["checks"]["health"] = wait_for_health(args.host, args.port)

            ready_status, ready = request_json("GET", args.host, args.port, "/ready")
            assert_status("ready", ready_status, 200, ready)
            report["checks"]["ready_initial"] = ready

            models_status, models = request_json("GET", args.host, args.port, "/v1/models")
            assert_status("models", models_status, 200, models)
            report["checks"]["models"] = models

            text_prompt = "Reply with exactly: text smoke ok"
            text_status, text_response = chat_completion(
                args.host,
                args.port,
                chat_payload(model_id, text_prompt, max_tokens=12),
            )
            assert_status("text", text_status, 200, text_response)
            text = choice_text(text_response)
            assert_not_degenerate("text", text, text_prompt)
            if "text smoke ok" not in text.lower():
                raise AssertionError(f"text smoke unexpected content: {text!r}")
            report["checks"]["text"] = {
                "status": text_status,
                "content": text,
                "usage": text_response.get("usage"),
                "metrics": text_response.get("metrics"),
            }

            image_status, image_response = chat_completion(
                args.host,
                args.port,
                chat_payload(
                    model_id,
                    [
                        {"type": "text", "text": "What color is the box in this image? Answer in one word."},
                        {"type": "image_url", "image_url": {"url": data_url(image_path, "image/png")}},
                    ],
                    max_tokens=24,
                ),
            )
            assert_status("image", image_status, 200, image_response)
            image_text = choice_text(image_response)
            assert_not_degenerate("image", image_text, "What color is the box in this image? Answer in one word.")
            if "red" not in image_text.lower():
                raise AssertionError(f"image smoke unexpected content: {image_text!r}")
            report["checks"]["image"] = {
                "status": image_status,
                "content": image_text,
                "usage": image_response.get("usage"),
                "metrics": image_response.get("metrics"),
            }

            audio_status, audio_response = chat_completion(
                args.host,
                args.port,
                chat_payload(
                    model_id,
                    [
                        {"type": "text", "text": "Transcribe the spoken word in this audio. Answer with one word."},
                        {"type": "input_audio", "data_url": data_url(audio_path, "audio/wav")},
                    ],
                    max_tokens=24,
                ),
            )
            assert_status("audio", audio_status, 200, audio_response)
            audio_text = choice_text(audio_response)
            assert_not_degenerate("audio", audio_text, "Transcribe the spoken word in this audio. Answer with one word.")
            if "banana" not in audio_text.lower():
                raise AssertionError(f"audio smoke unexpected content: {audio_text!r}")
            report["checks"]["audio"] = {
                "status": audio_status,
                "content": audio_text,
                "usage": audio_response.get("usage"),
                "metrics": audio_response.get("metrics"),
            }

            unsupported_status, unsupported_response = chat_completion(
                args.host,
                args.port,
                chat_payload(
                    model_id,
                    [{"type": "input_video", "data_url": "data:video/mp4;base64,AA=="}],
                    max_tokens=8,
                ),
            )
            assert_status("unsupported_modality", unsupported_status, 422, unsupported_response)
            report["checks"]["unsupported_modality"] = {"status": unsupported_status, "response": unsupported_response}

            malformed_status, malformed_response = chat_completion(
                args.host,
                args.port,
                chat_payload(
                    model_id,
                    [{"type": "input_image", "data_url": "not-a-data-url"}],
                    max_tokens=8,
                ),
            )
            assert_status("malformed_data_url", malformed_status, 400, malformed_response)
            report["checks"]["malformed_data_url"] = {"status": malformed_status, "response": malformed_response}

            cancel = run_cancel_probe(args.host, args.port, model_id)
            if not cancel["recovered"]:
                raise AssertionError(f"cancel/timeout probe did not recover: {cancel}")
            report["checks"]["timeout_cancellation"] = cancel

            stats_status, stats = request_json("GET", args.host, args.port, "/admin/stats")
            assert_status("stats", stats_status, 200, stats)
            report["checks"]["stats_final"] = stats
            report["memory"] = {
                "worker_pid": stats.get("worker", {}).get("pid"),
                "governor_rss_estimate_loaded_gb": stats.get("governor", {}).get("rss_estimate_loaded_gb"),
                "text_peak_memory_gb": report["checks"]["text"]["metrics"].get("peak_memory_gb"),
                "image_peak_memory_gb": report["checks"]["image"]["metrics"].get("peak_memory_gb"),
                "audio_peak_memory_gb": report["checks"]["audio"]["metrics"].get("peak_memory_gb"),
            }
        finally:
            supervisor.send_signal(signal.SIGINT)
            try:
                supervisor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait(timeout=10)
            stdout, stderr = supervisor.communicate(timeout=1)
            report["supervisor_returncode"] = supervisor.returncode
            report["supervisor_stdout_tail"] = stdout[-4000:]
            report["supervisor_stderr_tail"] = stderr[-4000:]

    if args.output:
        Path(args.output).expanduser().write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
