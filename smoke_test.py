"""Quick smoke test for all project dependencies and modules."""

import sys
import traceback

import numpy as np

errors = []


def check(name: str, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as exc:
        errors.append((name, exc))
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()


def test_imports():
    import cv2
    import gradio as gr
    import mediapipe as mp
    import numpy
    import openai
    import scipy
    from dotenv import load_dotenv
    from faster_whisper import WhisperModel

    assert cv2.__version__
    assert gr.__version__
    assert mp.__version__


def test_gesture():
    from gesture_detector import GestureDetector

    detector = GestureDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.process_frame(frame)
    assert result.gesture_label
    assert result.action_label
    assert result.frame.shape == frame.shape
    detector.close()


def test_utils():
    from utils import validate_cinematic_output

    data = validate_cinematic_output(
        {
            "camera_movement": "dolly in",
            "emotion": "tense",
            "shot_style": "close-up",
            "reasoning": "test",
            "confidence": 0.8,
        }
    )
    assert data["confidence"] == 0.8


def test_webcam():
    from webcam import find_working_camera, is_black_frame, read_frame

    cap, info = find_working_camera()
    assert cap is not None, f"No camera: {info}"
    ok, frame = read_frame(cap)
    cap.release()
    assert ok and frame is not None
    assert not is_black_frame(frame), "Camera returned black frames"


def test_ui():
    from app import build_ui, process_webcam_tick

    build_ui()
    outputs = process_webcam_tick()
    assert len(outputs) == 7
    frame = outputs[0]
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_stt_init():
    from voice_to_text import SpeechToText

    stt = SpeechToText()
    assert stt.backend in ("faster-whisper", "openai", "gemini")


def test_llm_init():
    from llm_cinematic import CinematicGenerator

    gen = CinematicGenerator()
    assert gen.provider in ("openai", "gemini", "none")


if __name__ == "__main__":
    print("Running dependency smoke tests...\n")
    check("imports", test_imports)
    check("utils", test_utils)
    check("gesture_detector", test_gesture)
    check("webcam", test_webcam)
    check("voice_to_text init", test_stt_init)
    check("llm_cinematic init", test_llm_init)
    check("gradio UI build", test_ui)

    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s)")
        sys.exit(1)
    print("All smoke tests passed.")
