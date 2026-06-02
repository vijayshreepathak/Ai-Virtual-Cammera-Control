"""LLM-powered structured cinematic output generation."""

from __future__ import annotations

import json
from typing import Any, Optional

from utils import (
    extract_json_object,
    format_cinematic_output,
    format_error,
    get_env,
    validate_cinematic_output,
)

SYSTEM_PROMPT = """You are a cinematic camera director assistant.
Convert the user's spoken instruction into a concise camera direction.

Return ONLY valid JSON with exactly these keys:
{
  "camera_movement": "",
  "emotion": "",
  "shot_style": "",
  "reasoning": "",
  "confidence": 0.0
}

Rules:
- camera_movement: one clear movement (e.g., dolly in, pan left, crane up)
- emotion: mood/tone (e.g., tense, hopeful, melancholic)
- shot_style: framing/style (e.g., close-up, wide establishing, over-the-shoulder)
- reasoning: 1-2 sentences explaining the choice
- confidence: float from 0.0 to 1.0
- Do not include markdown, code fences, or extra keys.
- If the instruction is vague, infer a reasonable cinematic interpretation and lower confidence.
"""


class CinematicGenerator:
    """Generate validated cinematic JSON from transcript text."""

    def __init__(self) -> None:
        self.provider = self._resolve_provider()

    def _resolve_provider(self) -> str:
        if get_env("LLM_PROVIDER"):
            return get_env("LLM_PROVIDER", "openai").lower()
        if get_env("GEMINI_API_KEY"):
            return "gemini"
        if get_env("OPENAI_API_KEY"):
            return "openai"
        return "none"

    def _call_openai(self, transcript: str) -> str:
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=get_env("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response.")
        return content

    def _call_gemini(self, transcript: str) -> str:
        api_key = get_env("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_env("GEMINI_MODEL", "gemini-2.0-flash"))
        prompt = f"{SYSTEM_PROMPT}\n\nUser instruction:\n{transcript}"
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    def generate(self, transcript: str, gesture_context: str = "") -> dict[str, Any]:
        if not transcript.strip():
            raise ValueError("Transcript is empty. Record voice input first.")

        if self.provider == "none":
            raise RuntimeError(
                "No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in your environment."
            )

        user_content = transcript
        if gesture_context.strip():
            user_content = (
                f"Recent gesture context: {gesture_context}\n\n"
                f"Spoken instruction: {transcript}"
            )

        if self.provider == "gemini":
            raw = self._call_gemini(user_content)
        else:
            raw = self._call_openai(user_content)

        parsed = extract_json_object(raw)
        return validate_cinematic_output(parsed)


def generate_cinematic_safe(
    transcript: str, gesture_context: str = ""
) -> tuple[str, str, Optional[str]]:
    """Return (formatted output, json string, optional error)."""
    try:
        generator = CinematicGenerator()
        result = generator.generate(transcript, gesture_context)
        from utils import format_cinematic_json

        return format_cinematic_output(result), format_cinematic_json(result), None
    except Exception as exc:
        fallback = {
            "camera_movement": "Unavailable",
            "emotion": "Unknown",
            "shot_style": "Unknown",
            "reasoning": str(exc),
            "confidence": 0.0,
        }
        from utils import format_cinematic_json

        return (
            format_cinematic_output(fallback),
            format_cinematic_json(fallback),
            format_error(str(exc)),
        )
