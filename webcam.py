"""Webcam capture helpers with auto-detection for Windows."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

MIN_FRAME_MEAN = 8.0
WARMUP_READS = 10
FAST_REJECT_READS = 3


def _try_open(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    for i in range(WARMUP_READS):
        ok, frame = cap.read()
        if i >= FAST_REJECT_READS and ok and frame is not None and float(frame.mean()) < MIN_FRAME_MEAN:
            cap.release()
            return None

    ok, frame = cap.read()
    if not ok or frame is None or float(frame.mean()) < MIN_FRAME_MEAN:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def find_working_camera(max_index: int = 4) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Try camera indices/backends and return the first that produces real frames."""
    attempts = []
    for index in range(max_index):
        for backend_name, backend in (("DSHOW", cv2.CAP_DSHOW), ("DEFAULT", cv2.CAP_ANY)):
            cap = _try_open(index, backend)
            if cap is not None:
                msg = f"Camera index {index} ({backend_name})"
                return cap, msg
            attempts.append(f"{backend_name}:{index}")

    return None, f"No working camera found. Tried: {', '.join(attempts)}"


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Read the freshest frame, draining the buffer."""
    frame = None
    ok = False
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            break
    return ok, frame


def is_black_frame(frame: np.ndarray) -> bool:
    return float(frame.mean()) < MIN_FRAME_MEAN
