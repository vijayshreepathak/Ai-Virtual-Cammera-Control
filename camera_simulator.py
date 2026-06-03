"""Virtual camera: applies visible zoom/pan/tilt to the live preview."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraViewState:
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def label(self) -> str:
        return f"Zoom {self.zoom:.2f}x | Pan X {self.pan_x:+.0%} | Pan Y {self.pan_y:+.0%}"


class VirtualCamera:
    """Smooth virtual camera rig driven by detected gestures."""

    ZOOM_MIN = 0.72
    ZOOM_MAX = 2.4
    PAN_MAX = 0.38
    ZOOM_STEP = 0.18
    PAN_STEP = 0.10
    TILT_STEP = 0.08

    def __init__(self) -> None:
        self.state = CameraViewState()
        self._target_zoom = 1.0
        self._target_pan_x = 0.0
        self._target_pan_y = 0.0
        self._active_gesture: str = "none"

    def reset(self) -> None:
        self._target_zoom = 1.0
        self._target_pan_x = 0.0
        self._target_pan_y = 0.0
        self._active_gesture = "none"

    def apply_gesture(self, gesture: str) -> None:
        """Move virtual camera when a gesture is confirmed."""
        self._active_gesture = gesture
        if gesture == "zoom_in":
            self._target_zoom = min(self.ZOOM_MAX, self._target_zoom + self.ZOOM_STEP)
        elif gesture == "zoom_out":
            self._target_zoom = max(self.ZOOM_MIN, self._target_zoom - self.ZOOM_STEP)
        elif gesture == "pan_right":
            self._target_pan_x = min(self.PAN_MAX, self._target_pan_x + self.PAN_STEP)
        elif gesture == "pan_left":
            self._target_pan_x = max(-self.PAN_MAX, self._target_pan_x - self.PAN_STEP)
        elif gesture == "tilt_up":
            self._target_pan_y = max(-self.PAN_MAX, self._target_pan_y - self.TILT_STEP)

    def smooth_update(self, alpha: float = 0.28) -> None:
        self.state.zoom += (self._target_zoom - self.state.zoom) * alpha
        self.state.pan_x += (self._target_pan_x - self.state.pan_x) * alpha
        self.state.pan_y += (self._target_pan_y - self.state.pan_y) * alpha

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom (center crop) and pan (shift) — visible on the preview."""
        self.smooth_update()
        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return frame

        zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self.state.zoom))
        crop_w = max(32, int(w / zoom))
        crop_h = max(32, int(h / zoom))

        cx = w // 2 + int(self.state.pan_x * w)
        cy = h // 2 + int(self.state.pan_y * h)

        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame

        output = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        self._draw_camera_ui(output)
        return output

    def _draw_camera_ui(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        bar_w = int(120 * (self.state.zoom - self.ZOOM_MIN) / (self.ZOOM_MAX - self.ZOOM_MIN))
        bar_w = max(4, min(120, bar_w))
        cv2.rectangle(frame, (12, h - 28), (132, h - 12), (40, 40, 55), -1)
        cv2.rectangle(frame, (12, h - 28), (12 + bar_w, h - 12), (72, 180, 120), -1)
        cv2.putText(
            frame,
            f"Z {self.state.zoom:.1f}x",
            (14, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (230, 230, 240),
            1,
        )

        if abs(self.state.pan_x) > 0.02:
            direction = ">>>" if self.state.pan_x > 0 else "<<<"
            cv2.putText(frame, direction, (w // 2 - 30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)

        if abs(self.state.pan_y) > 0.02:
            cv2.putText(frame, "^", (w // 2 - 8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
