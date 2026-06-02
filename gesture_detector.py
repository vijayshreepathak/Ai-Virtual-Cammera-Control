"""Webcam gesture detection using MediaPipe Hand Landmarker."""

from __future__ import annotations

import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from camera_simulator import VirtualCamera
from utils import draw_overlay

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WRIST_INDEX = 0
INDEX_TIP = 8
DETECT_WIDTH = 320

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class GestureType(str, Enum):
    NONE = "none"
    PAN_RIGHT = "pan_right"
    PAN_LEFT = "pan_left"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    TILT_UP = "tilt_up"


GESTURE_ACTIONS: dict[GestureType, str] = {
    GestureType.NONE: "Idle",
    GestureType.PAN_RIGHT: "Camera Pan Right",
    GestureType.PAN_LEFT: "Camera Pan Left",
    GestureType.ZOOM_IN: "Zoom In",
    GestureType.ZOOM_OUT: "Zoom Out",
    GestureType.TILT_UP: "Camera Tilt Up",
}


@dataclass
class GestureResult:
    gesture_label: str
    action_label: str
    confidence_score: float
    frame: np.ndarray
    fps: float = 0.0
    camera_state: str = ""


@dataclass
class ActionLogEntry:
    timestamp: str
    gesture: str
    action: str
    confidence: float


@dataclass
class GestureState:
    action_log: Deque[ActionLogEntry] = field(default_factory=lambda: deque(maxlen=12))
    last_gesture_label: str = "No Gesture"
    last_action_label: str = "Idle"
    last_confidence: float = 0.0
    fps: float = 0.0
    frame_count: int = 0


def ensure_hand_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Downloading hand_landmarker.task (one-time setup)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def draw_hand_landmarks(frame: np.ndarray, landmarks, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w * scale_x), int(lm.y * h * scale_y)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 200, 255), -1)


def _movement_stats(values: Deque[float]) -> Tuple[float, float]:
    """Return (signed delta, consistency 0-1) for gesture axis."""
    if len(values) < 6:
        return 0.0, 0.0

    seq = list(values)
    delta = seq[-1] - seq[0]
    if abs(delta) < 1e-8:
        return 0.0, 0.0

    sign = 1.0 if delta > 0 else -1.0
    steps = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    consistent = sum(1 for step in steps if step * sign > 0) / len(steps)
    return delta, consistent


class GestureDetector:
    """High-accuracy gesture detection + virtual camera application."""

    def __init__(
        self,
        history_size: int = 14,
        pan_threshold: float = 0.09,
        zoom_threshold: float = 0.018,
        tilt_threshold: float = 0.07,
        cooldown_seconds: float = 1.4,
        detect_every_n: int = 2,
        min_consistency: float = 0.72,
        min_confidence: float = 0.52,
    ) -> None:
        self.history_size = history_size
        self.pan_threshold = pan_threshold
        self.zoom_threshold = zoom_threshold
        self.tilt_threshold = tilt_threshold
        self.cooldown_seconds = cooldown_seconds
        self.detect_every_n = detect_every_n
        self.min_consistency = min_consistency
        self.min_confidence = min_confidence

        model_path = ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0
        self._tick = 0

        self._wrist_x_history: Deque[float] = deque(maxlen=history_size)
        self._wrist_y_history: Deque[float] = deque(maxlen=history_size)
        self._hand_size_history: Deque[float] = deque(maxlen=history_size)

        self._last_gesture: GestureType = GestureType.NONE
        self._last_confidence: float = 0.0
        self._last_trigger_time: float = 0.0
        self._display_gesture_until: float = 0.0
        self._last_landmarks = None
        self._last_scale = (1.0, 1.0)

        self.state = GestureState()
        self.camera = VirtualCamera()
        self._fps_times: Deque[float] = deque(maxlen=30)

    def set_sensitivity(self, level: float) -> None:
        level = max(0.5, min(1.5, level))
        self.pan_threshold = 0.09 * level
        self.zoom_threshold = 0.018 * level
        self.tilt_threshold = 0.07 * level
        self.min_confidence = 0.52 + (level - 1.0) * 0.08

    def reset_camera_view(self) -> None:
        self.camera.reset()

    def _hand_size(self, landmarks) -> float:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _detect_from_history(self) -> Tuple[GestureType, float]:
        x_delta, x_cons = _movement_stats(self._wrist_x_history)
        y_delta, y_cons = _movement_stats(self._wrist_y_history)
        size_delta, size_cons = _movement_stats(self._hand_size_history)

        abs_x, abs_y, abs_size = abs(x_delta), abs(y_delta), abs(size_delta)
        mean_size = sum(self._hand_size_history) / len(self._hand_size_history)
        size_ratio = abs_size / max(mean_size, 0.01)

        candidates: list[Tuple[GestureType, float]] = []

        # Pan: horizontal wrist motion must dominate clearly
        if (
            abs_x > self.pan_threshold
            and x_cons >= self.min_consistency
            and abs_x > abs_size * 2.2
            and abs_x > abs_y * 1.8
        ):
            score = min(1.0, (abs_x / self.pan_threshold) * x_cons * 0.45)
            candidates.append((GestureType.PAN_RIGHT if x_delta > 0 else GestureType.PAN_LEFT, score))

        # Zoom: hand area change must dominate (move hand toward/away from camera)
        if (
            size_ratio > self.zoom_threshold
            and size_cons >= self.min_consistency
            and abs_size > abs_x * 2.2
            and abs_size > abs_y * 1.8
        ):
            score = min(1.0, (size_ratio / self.zoom_threshold) * size_cons * 0.4)
            candidates.append((GestureType.ZOOM_IN if size_delta > 0 else GestureType.ZOOM_OUT, score))

        # Tilt: vertical wrist motion
        if (
            abs_y > self.tilt_threshold
            and y_cons >= self.min_consistency
            and y_delta < 0
            and abs_y > abs_x * 1.8
            and abs_y > abs_size * 1.8
        ):
            score = min(1.0, (abs_y / self.tilt_threshold) * y_cons * 0.4)
            candidates.append((GestureType.TILT_UP, score))

        if not candidates:
            return GestureType.NONE, 0.0

        return max(candidates, key=lambda item: item[1])

    def _log_action(self, gesture: GestureType, action: str, confidence: float) -> None:
        entry = ActionLogEntry(
            timestamp=time.strftime("%H:%M:%S"),
            gesture=gesture.value.replace("_", " ").title(),
            action=action,
            confidence=confidence,
        )
        self.state.action_log.appendleft(entry)

    def process_frame(self, frame: np.ndarray, *, run_detection: bool = True) -> GestureResult:
        if frame is None:
            raise ValueError("Webcam frame is empty.")

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        gesture = GestureType.NONE
        confidence = 0.0
        now = time.time()

        self._tick += 1
        should_detect = run_detection and (self._tick % self.detect_every_n == 0)

        if should_detect:
            scale = DETECT_WIDTH / w
            small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self._frame_timestamp_ms += 50
            result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
            inv_scale_x = w / DETECT_WIDTH
            inv_scale_y = h / small.shape[0]
            self._last_scale = (inv_scale_x, inv_scale_y)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                self._last_landmarks = hand
                self._wrist_x_history.append(hand[WRIST_INDEX].x)
                self._wrist_y_history.append(hand[WRIST_INDEX].y)
                self._hand_size_history.append(self._hand_size(hand))

                detected, raw_confidence = self._detect_from_history()

                if detected != GestureType.NONE and raw_confidence >= self.min_confidence:
                    if now - self._last_trigger_time >= self.cooldown_seconds:
                        gesture = detected
                        confidence = raw_confidence
                        self._last_gesture = gesture
                        self._last_confidence = confidence
                        self._last_trigger_time = now
                        self._display_gesture_until = now + 1.2
                        self._log_action(gesture, GESTURE_ACTIONS[gesture], confidence)
                        self.camera.apply_gesture(gesture.value)
                    elif now < self._display_gesture_until:
                        gesture = self._last_gesture
                        confidence = self._last_confidence
            else:
                self._last_landmarks = None
                self._wrist_x_history.clear()
                self._wrist_y_history.clear()
                self._hand_size_history.clear()

        elif now < self._display_gesture_until and self._last_gesture != GestureType.NONE:
            gesture = self._last_gesture
            confidence = self._last_confidence

        # Apply virtual camera transform (visible zoom/pan on preview)
        frame = self.camera.apply_to_frame(frame)

        if self._last_landmarks is not None:
            draw_hand_landmarks(frame, self._last_landmarks, *self._last_scale)

        action = GESTURE_ACTIONS.get(gesture, "Idle")
        gesture_label = gesture.value.replace("_", " ").title() if gesture != GestureType.NONE else "No Gesture"

        elapsed = time.perf_counter() - t0
        self._fps_times.append(elapsed)
        avg_frame_time = sum(self._fps_times) / len(self._fps_times)
        fps = min(60.0, 1.0 / avg_frame_time) if avg_frame_time > 0 else 0.0

        cam_label = self.camera.state.label()
        annotated = draw_overlay(
            frame,
            gesture_label=gesture_label,
            action_label=action,
            confidence=confidence,
            fps=fps,
            extra_line=cam_label,
        )

        self.state.last_gesture_label = gesture_label
        self.state.last_action_label = action
        self.state.last_confidence = confidence
        self.state.fps = fps
        self.state.frame_count += 1

        return GestureResult(
            gesture_label=gesture_label,
            action_label=action,
            confidence_score=round(confidence, 2),
            frame=annotated,
            fps=round(fps, 1),
            camera_state=cam_label,
        )

    def format_action_log(self) -> str:
        if not self.state.action_log:
            return "No gestures triggered yet."
        lines = ["Time     | Gesture      | Action                | Conf"]
        lines.append("-" * 58)
        for e in self.state.action_log:
            lines.append(
                f"{e.timestamp} | {e.gesture:<12} | {e.action:<21} | {e.confidence:.0%}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        self._landmarker.close()
