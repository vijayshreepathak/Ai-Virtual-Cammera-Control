"""Speech-to-text pipeline using OpenAI Whisper API or faster-whisper."""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import numpy as np

from utils import format_error, get_env


class SpeechToText:
    """Transcribe microphone audio to text."""

    def __init__(self) -> None:
        configured = get_env("STT_BACKEND", "").lower()
        if configured:
            self.backend = configured
        elif get_env("OPENAI_API_KEY"):
            self.backend = "openai"
        else:
            self.backend = "faster-whisper"
        self._model = None

    def preload(self) -> str:
        if self.backend == "openai":
            if not get_env("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI Whisper.")
            return "openai-whisper (cloud)"
        self._load_faster_whisper()
        return f"faster-whisper ({get_env('WHISPER_MODEL_SIZE', 'tiny')})"

    def _load_faster_whisper(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            model_size = get_env("WHISPER_MODEL_SIZE", "tiny")
            print(f"Loading Whisper model '{model_size}' (one-time, may take a minute)…")
            self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("Whisper model ready.")
        return self._model

    def _transcribe_faster_whisper(self, audio_path: str) -> str:
        model = self._load_faster_whisper()
        segments, _ = model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
            language="en",
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    def _transcribe_openai(self, audio_path: str) -> str:
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI Whisper transcription.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=get_env("OPENAI_WHISPER_MODEL", "whisper-1"),
                file=audio_file,
                language="en",
            )
        return response.text.strip()

    def _to_mono_int16(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        return audio_data

    def _save_audio(self, audio: Tuple[int, np.ndarray]) -> str:
        sample_rate, audio_data = audio
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("No audio captured. Select Microphone (not Stereo Mix) and record again.")

        audio_data = self._to_mono_int16(audio_data)

        peak = np.max(np.abs(audio_data))
        if peak < 500:
            raise ValueError(
                "Audio level too low. Select your Microphone input, speak closer, and record 3+ seconds."
            )

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        import scipy.io.wavfile as wavfile

        wavfile.write(temp_path, sample_rate, audio_data)
        return temp_path

    def transcribe(self, audio: Optional[Tuple[int, np.ndarray]]) -> str:
        if audio is None:
            raise ValueError("No audio input received. Click Record and speak first.")

        temp_path = self._save_audio(audio)
        try:
            if self.backend == "openai":
                text = self._transcribe_openai(temp_path)
            else:
                text = self._transcribe_faster_whisper(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not text:
            raise ValueError("Transcription was empty. Speak clearly into your microphone and try again.")

        return text


_stt_singleton: Optional[SpeechToText] = None


def get_stt() -> SpeechToText:
    global _stt_singleton
    if _stt_singleton is None:
        _stt_singleton = SpeechToText()
    return _stt_singleton


def preload_stt() -> str:
    return get_stt().preload()


def transcribe_audio_safe(audio: Optional[Tuple[int, np.ndarray]]) -> tuple[str, Optional[str]]:
    try:
        return get_stt().transcribe(audio), None
    except Exception as exc:
        return "", format_error(str(exc))
