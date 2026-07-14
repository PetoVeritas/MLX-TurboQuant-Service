"""Backend adapter descriptors shared by supervisor and worker code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_MODALITIES = frozenset({"text"})
TURBOQUANT_RUNTIME_MODALITIES = frozenset({"text", "image", "audio"})
DIFFUSION_GEMMA_RUNTIME_MODALITIES = frozenset({"text", "image"})
QWEN3_TTS_RUNTIME_MODALITIES = frozenset({"text"})


@dataclass(frozen=True)
class BackendAdapterDescriptor:
    backend_id: str
    display_name: str
    supported_modalities: frozenset[str]


BACKEND_ADAPTERS: dict[str, BackendAdapterDescriptor] = {
    "stub": BackendAdapterDescriptor("stub", "Stub backend", TEXT_MODALITIES),
    "mlx_vlm_turboquant": BackendAdapterDescriptor(
        "mlx_vlm_turboquant",
        "MLX-VLM TurboQuant backend",
        TURBOQUANT_RUNTIME_MODALITIES,
    ),
    "mlx_vlm_diffusion_gemma": BackendAdapterDescriptor(
        "mlx_vlm_diffusion_gemma",
        "MLX-VLM DiffusionGemma backend",
        DIFFUSION_GEMMA_RUNTIME_MODALITIES,
    ),
    "mlx_audio_qwen3_tts": BackendAdapterDescriptor(
        "mlx_audio_qwen3_tts",
        "MLX-Audio Qwen3-TTS backend",
        QWEN3_TTS_RUNTIME_MODALITIES,
    ),
}


DEFAULT_BACKEND_ID = "mlx_vlm_turboquant"


def configured_backend_id(config: dict[str, Any]) -> str:
    worker_cfg = config.get("worker", {})
    if bool(worker_cfg.get("stubMode", False)):
        return "stub"
    backend_id = worker_cfg.get("backend") or worker_cfg.get("backendAdapter") or DEFAULT_BACKEND_ID
    return str(backend_id).strip() or DEFAULT_BACKEND_ID


def backend_descriptor(config: dict[str, Any]) -> BackendAdapterDescriptor:
    backend_id = configured_backend_id(config)
    try:
        return BACKEND_ADAPTERS[backend_id]
    except KeyError as exc:
        raise ValueError(f"unknown_backend_adapter:{backend_id}") from exc


def _read_model_config(config: dict[str, Any]) -> dict[str, Any] | None:
    raw_model_path = str(config.get("model", {}).get("path", "")).strip()
    if not raw_model_path:
        return None
    model_path = Path(raw_model_path).expanduser()
    config_path = model_path / "config.json" if model_path.is_dir() else model_path
    if not config_path.exists() or config_path.name != "config.json":
        return None
    try:
        return json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _modalities_from_model_config(model_config: dict[str, Any]) -> set[str]:
    supported = {"text"}
    if isinstance(model_config.get("vision_config"), dict):
        supported.add("image")
    if isinstance(model_config.get("audio_config"), dict):
        supported.add("audio")
    return supported


def turboquant_supported_modalities(config: dict[str, Any]) -> set[str]:
    """Return modalities supported by the configured TurboQuant artifact/runtime.

    The runtime can handle text/image/audio, but a particular Gemma 4 artifact
    may omit a tower. When the local config is readable, use the artifact's
    `config.json`; otherwise expose the runtime's known upper bound so status
    endpoints stay useful before a model path is filled in.
    """

    artifact_config = _read_model_config(config)
    if artifact_config is None:
        return set(TURBOQUANT_RUNTIME_MODALITIES)
    return _modalities_from_model_config(artifact_config) & set(TURBOQUANT_RUNTIME_MODALITIES)


def backend_supported_modalities(config: dict[str, Any]) -> set[str]:
    descriptor = backend_descriptor(config)
    if descriptor.backend_id == "mlx_vlm_turboquant":
        return turboquant_supported_modalities(config)
    return set(descriptor.supported_modalities)
