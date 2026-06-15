"""Typed chat message parts and modality validation."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass
from typing import Any


ACTIVE_MODALITIES = ("text", "image", "video", "audio")
RESERVED_MODALITIES = ("document",)
ALL_MODALITIES = ACTIVE_MODALITIES + RESERVED_MODALITIES
NON_TEXT_MODALITIES = tuple(modality for modality in ALL_MODALITIES if modality != "text")


class MessagePartError(ValueError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass(frozen=True)
class TextPart:
    type: str
    text: str


@dataclass(frozen=True)
class MediaPart:
    type: str
    data_url: str
    mime_type: str
    byte_length: int


MessagePart = TextPart | MediaPart


def part_to_dict(part: MessagePart) -> dict[str, Any]:
    return asdict(part)


def parts_to_dicts(parts: list[MessagePart]) -> list[dict[str, Any]]:
    return [part_to_dict(part) for part in parts]


def part_modalities(parts: list[MessagePart] | list[dict[str, Any]]) -> set[str]:
    modalities: set[str] = set()
    for part in parts:
        if isinstance(part, dict):
            part_type = str(part.get("type", ""))
        else:
            part_type = part.type
        if part_type in ALL_MODALITIES:
            modalities.add(part_type)
    return modalities


def configured_modalities(config: dict[str, Any]) -> set[str]:
    modalities_cfg = config.get("modalities", {})
    enabled: set[str] = set()
    for modality in ALL_MODALITIES:
        cfg = modalities_cfg.get(modality, {})
        if isinstance(cfg, dict) and bool(cfg.get("enabled", modality == "text")):
            enabled.add(modality)
    enabled.add("text")
    return enabled


def backend_supported_modalities(_config: dict[str, Any]) -> set[str]:
    # Phase 1 wires the contract and status surface only. Current backends
    # still consume text prompts; real multimodal support lands behind this
    # boundary in later phases.
    return {"text"}


def modalities_status(config: dict[str, Any]) -> dict[str, Any]:
    configured = configured_modalities(config)
    supported = backend_supported_modalities(config)
    effective = configured & supported
    return {
        "configured": sorted(configured),
        "backend_supported": sorted(supported),
        "effective": sorted(effective),
        "vision": {
            "configured": bool({"image", "video"} & configured),
            "effective": bool({"image", "video"} & effective),
        },
        "strict_capability_check": bool(config.get("modalities", {}).get("strictCapabilityCheck", True)),
    }


def _media_config(config: dict[str, Any], modality: str) -> dict[str, Any]:
    value = config.get("modalities", {}).get(modality, {})
    return value if isinstance(value, dict) else {}


def max_inputs(config: dict[str, Any], modality: str) -> int:
    cfg = _media_config(config, modality)
    try:
        value = int(cfg.get("maxInputs", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    return max(0, value)


def _max_bytes(config: dict[str, Any], modality: str) -> int:
    cfg = _media_config(config, modality)
    try:
        max_mb = float(cfg.get("maxBytesMb", 0) or 0)
    except (TypeError, ValueError):
        max_mb = 0
    return int(max_mb * 1024 * 1024) if max_mb > 0 else 0


def _allowed_mime_types(config: dict[str, Any], modality: str) -> set[str]:
    values = _media_config(config, modality).get("allowedMimeTypes", [])
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str) and value}


def _parse_data_url(value: str, *, modality: str, config: dict[str, Any]) -> tuple[str, int]:
    if not value.startswith("data:") or "," not in value:
        raise MessagePartError("bad_request", f"{modality} parts must use data_url transport")

    header, encoded = value[5:].split(",", 1)
    header_parts = header.split(";")
    mime_type = header_parts[0].strip().lower()
    if not mime_type:
        raise MessagePartError("bad_request", f"{modality} data_url is missing a MIME type")
    if "base64" not in {part.lower() for part in header_parts[1:]}:
        raise MessagePartError("bad_request", f"{modality} data_url must be base64 encoded")

    allowed = _allowed_mime_types(config, modality)
    if allowed and mime_type not in allowed:
        raise MessagePartError("unsupported_modality", f"Unsupported {modality} MIME type: {mime_type}")

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MessagePartError("bad_request", f"Malformed {modality} data_url payload") from exc

    max_bytes = _max_bytes(config, modality)
    if max_bytes and len(decoded) > max_bytes:
        raise MessagePartError("bad_request", f"{modality} payload exceeds configured byte limit")
    return mime_type, len(decoded)


def _extract_data_url(item: dict[str, Any], modality: str) -> str | None:
    for key in ("data_url", "url", f"{modality}_url"):
        value = item.get(key)
        if isinstance(value, str):
            return value

    nested = item.get(f"input_{modality}") or item.get(modality)
    if isinstance(nested, dict):
        for key in ("data_url", "url"):
            value = nested.get(key)
            if isinstance(value, str):
                return value
    if isinstance(nested, str):
        return nested

    if modality == "image" and isinstance(item.get("image_url"), dict):
        value = item["image_url"].get("url")
        if isinstance(value, str):
            return value
    return None


def _normalize_part_type(item_type: str) -> str | None:
    if item_type in {"text", "input_text", "output_text"}:
        return "text"
    if item_type in {"image", "image_url", "input_image"}:
        return "image"
    if item_type in {"video", "video_url", "input_video"}:
        return "video"
    if item_type in {"audio", "audio_url", "input_audio"}:
        return "audio"
    if item_type in {"document", "file", "input_file"}:
        return "document"
    return None


def extract_message_parts(
    content: Any,
    *,
    config: dict[str, Any] | None = None,
    allow_empty: bool = False,
    allow_tool_content: bool = False,
) -> list[MessagePart] | None:
    config = config or {}
    if content is None:
        return [] if allow_empty else None
    if isinstance(content, str):
        return [TextPart(type="text", text=content)]
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return None

    parts: list[MessagePart] = []
    for item in content:
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if not isinstance(item_type, str):
            return None
        normalized_type = _normalize_part_type(item_type)
        if normalized_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                return None
            parts.append(TextPart(type="text", text=text))
            continue
        if item_type in {"thinking", "reasoning"}:
            continue
        if allow_tool_content and item_type in {"toolCall", "tool_call", "function_call", "tool_use"}:
            continue
        if normalized_type in NON_TEXT_MODALITIES:
            if normalized_type not in configured_modalities(config):
                raise MessagePartError("unsupported_modality", f"Modality is disabled: {normalized_type}")
            data_url = _extract_data_url(item, normalized_type)
            if not data_url:
                raise MessagePartError("bad_request", f"{normalized_type} part is missing a data_url")
            mime_type, byte_length = _parse_data_url(data_url, modality=normalized_type, config=config)
            parts.append(
                MediaPart(
                    type=normalized_type,
                    data_url=data_url,
                    mime_type=mime_type,
                    byte_length=byte_length,
                )
            )
            continue
        return None
    return parts


def text_from_parts(parts: list[MessagePart]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextPart) and part.text)


def validate_part_counts(counts: dict[str, int], config: dict[str, Any]) -> None:
    for modality in NON_TEXT_MODALITIES:
        limit = max_inputs(config, modality)
        count = counts.get(modality, 0)
        if limit and count > limit:
            raise MessagePartError("bad_request", f"{modality} input count exceeds configured limit")


def unsupported_backend_modalities(messages: list[dict[str, Any]], supported: set[str]) -> set[str]:
    requested: set[str] = set()
    for message in messages:
        parts = message.get("parts")
        if isinstance(parts, list):
            requested |= part_modalities(parts)
        elif "content" in message:
            requested.add("text")
    return requested - supported
