"""Shared helpers for overlays, config, and JSON validation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

# Always load project .env and override stale Windows/user env vars.
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE, override=True)

CINEMATIC_SCHEMA_KEYS = (
    "camera_movement",
    "emotion",
    "shot_style",
    "reasoning",
    "confidence",
)


def patch_gradio_client() -> None:
    """Fix gradio-client crash when JSON schema uses boolean additionalProperties."""
    import gradio_client.utils as client_utils

    original_convert = client_utils._json_schema_to_python_type
    original_get_type = client_utils.get_type

    def safe_get_type(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return "Any"
        return original_get_type(schema)

    def safe_convert(schema: Any, defs: Any) -> str:
        if isinstance(schema, bool):
            return "Any"
        return original_convert(schema, defs)

    client_utils.get_type = safe_get_type
    client_utils._json_schema_to_python_type = safe_convert


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key, default)
    return value.strip() if value else default


def is_valid_gemini_api_key_format(key: Optional[str] = None) -> bool:
    """Google AI Studio keys start with AIza (not AQ. or other prefixes)."""
    value = (key or get_env("GEMINI_API_KEY") or "").strip()
    return bool(value) and value.startswith("AIza") and len(value) >= 35


def gemini_key_hint() -> str:
    key = get_env("GEMINI_API_KEY") or ""
    if not key.strip():
        return (
            "GEMINI_API_KEY is empty in .env. "
            "Create a key at https://aistudio.google.com/apikey (starts with AIza)."
        )
    if key.strip().startswith("AQ."):
        return (
            "Your key starts with AQ. — that is NOT a Google AI Studio API key. "
            "Open https://aistudio.google.com/apikey → Create API key → copy the AIza… key into .env. "
            "Also remove GEMINI_API_KEY from Windows Environment Variables if set there."
        )
    if not key.strip().startswith("AIza"):
        return (
            "GEMINI_API_KEY must start with AIza (Google AI Studio). "
            "Get one at https://aistudio.google.com/apikey — not from Cloud Console OAuth."
        )
    return ""


def gemini_key_debug_label() -> str:
    """Safe one-line summary for terminal (never prints full key)."""
    key = (get_env("GEMINI_API_KEY") or "").strip()
    if not key:
        return "GEMINI_API_KEY: (empty) — add AIza key to .env"
    prefix = key[:4] + "…" if len(key) > 4 else "(short)"
    if key.startswith("AIza"):
        return f"GEMINI_API_KEY: {prefix} ({len(key)} chars) — format OK"
    if key.startswith("AQ."):
        return f"GEMINI_API_KEY: {prefix} — INVALID (use AIza from AI Studio, not AQ.)"
    return f"GEMINI_API_KEY: {prefix} — INVALID (must start with AIza)"


def has_llm_credentials() -> bool:
    if get_env("OPENAI_API_KEY"):
        return True
    return is_valid_gemini_api_key_format()


def has_stt_credentials() -> bool:
    backend = get_env("STT_BACKEND", "").lower()
    if backend == "openai":
        return bool(get_env("OPENAI_API_KEY"))
    if backend == "gemini":
        return is_valid_gemini_api_key_format()
    if backend == "faster-whisper":
        return True
    if get_env("OPENAI_API_KEY"):
        return True
    if is_valid_gemini_api_key_format():
        return True
    return True  # local faster-whisper fallback when no cloud keys


def draw_overlay(
    frame: np.ndarray,
    gesture_label: str,
    action_label: str,
    confidence: float,
    fps: float = 0.0,
    extra_line: str = "",
) -> np.ndarray:
    """Draw gesture/action/confidence/FPS labels on the webcam frame (in-place)."""
    h, w = frame.shape[:2]

    panel_h = 116 if extra_line else 96
    cv2.rectangle(frame, (0, 0), (w, panel_h), (18, 18, 24), -1)
    cv2.rectangle(frame, (0, 0), (w, panel_h), (60, 65, 85), 1)

    lines = [
        f"Gesture: {gesture_label}",
        f"Action: {action_label}",
        f"Conf: {confidence:.0%}" if confidence > 0 else "Conf: --",
        f"FPS: {fps:.0f}" if fps > 0 else "FPS: --",
    ]
    if extra_line:
        lines.append(extra_line[:56])

    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (12, 22 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 240), 1)

    badge = action_label if confidence > 0 else "Awaiting Gesture"
    badge_color = (72, 180, 120) if confidence > 0 else (70, 70, 90)
    cv2.rectangle(frame, (w - 220, h - 36), (w - 8, h - 8), badge_color, -1)
    cv2.putText(frame, badge[:24], (w - 212, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 20), 1)

    return frame


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty content.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object.")
    return parsed


def validate_cinematic_output(data: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in CINEMATIC_SCHEMA_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    confidence = data["confidence"]
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("'confidence' must be a number.") from exc

    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("'confidence' must be between 0.0 and 1.0.")

    return {
        "camera_movement": str(data["camera_movement"]).strip(),
        "emotion": str(data["emotion"]).strip(),
        "shot_style": str(data["shot_style"]).strip(),
        "reasoning": str(data["reasoning"]).strip(),
        "confidence": round(confidence_value, 2),
    }


def format_cinematic_output(data: dict[str, Any]) -> str:
    return (
        f"Camera Movement: {data['camera_movement']}\n"
        f"Emotion: {data['emotion']}\n"
        f"Shot Style: {data['shot_style']}\n"
        f"Confidence: {data['confidence']:.0%}\n\n"
        f"Reasoning:\n{data['reasoning']}"
    )


def format_cinematic_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2)


def format_error(message: str) -> str:
    if "API_KEY_INVALID" in message or "API key not valid" in message:
        hint = gemini_key_hint()
        if hint:
            return f"⚠ Gemini API key rejected. {hint}"
    return f"⚠ {message}"


def system_status() -> str:
    gemini_hint = gemini_key_hint()
    if get_env("LLM_PROVIDER", "").lower() == "gemini" or get_env("GEMINI_API_KEY"):
        if gemini_hint:
            llm = f"[!!] LLM: {gemini_hint}"
        else:
            llm = "[OK] LLM Ready (Gemini)"
    elif get_env("OPENAI_API_KEY"):
        llm = "[OK] LLM Ready (OpenAI)"
    elif has_llm_credentials():
        llm = "[OK] LLM Ready"
    else:
        llm = "[!!] LLM Key Missing"

    stt_backend = get_env("STT_BACKEND", "").lower()
    if stt_backend == "gemini" and gemini_hint:
        stt = f"[!!] STT: {gemini_hint}"
    elif has_stt_credentials():
        stt = "[OK] STT Ready"
    else:
        stt = "[!!] STT Unavailable"
    return f"{llm}  |  {stt}  |  [OK] Gesture Engine Ready"
