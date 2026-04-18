"""Configuration loading for the MLX Turbo Gemma service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.json"
LOCAL_CONFIG_PATH = ROOT_DIR / "config" / "local.json"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)

    host = os.getenv("MLX_GEMMA_HOST")
    port = os.getenv("MLX_GEMMA_PORT")
    model_path = os.getenv("MLX_GEMMA_MODEL_PATH")
    model_id = os.getenv("MLX_GEMMA_MODEL_ID")
    log_level = os.getenv("MLX_GEMMA_LOG_LEVEL")
    stub_mode = os.getenv("MLX_GEMMA_STUB_MODE")
    worker_python = os.getenv("MLX_GEMMA_WORKER_PYTHON")

    if host:
        result.setdefault("server", {})["host"] = host
    if port:
        result.setdefault("server", {})["port"] = int(port)
    if model_path:
        result.setdefault("model", {})["path"] = model_path
    if model_id:
        result.setdefault("model", {})["id"] = model_id
    if log_level:
        result.setdefault("logging", {})["level"] = log_level
    if stub_mode is not None:
        normalized = stub_mode.strip().lower()
        result.setdefault("worker", {})["stubMode"] = normalized in {"1", "true", "yes", "on"}
    if worker_python:
        result.setdefault("worker", {})["pythonExecutable"] = worker_python

    return result


def load_config() -> dict[str, Any]:
    config = load_json_file(DEFAULT_CONFIG_PATH)

    if LOCAL_CONFIG_PATH.exists():
        config = deep_merge(config, load_json_file(LOCAL_CONFIG_PATH))

    config = apply_env_overrides(config)
    return config
