"""Shared helpers for overlays, config, and JSON validation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

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


def has_llm_credentials() -> bool:
    return bool(get_env("OPENAI_API_KEY") or get_env("GEMINI_API_KEY"))


def has_stt_credentials() -> bool:
    return bool(get_env("OPENAI_API_KEY")) or get_env("STT_BACKEND", "faster-whisper") == "faster-whisper"


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
    return f"⚠ {message}"


def system_status() -> str:
    llm = "[OK] LLM Ready" if has_llm_credentials() else "[!!] LLM Key Missing"
    stt = "[OK] STT Ready" if has_stt_credentials() else "[!!] STT Unavailable"
    return f"{llm}  |  {stt}  |  [OK] Gesture Engine Ready"
