"""Threaded webcam capture that only ever keeps the newest frame.

The capture thread reads frames as fast as the camera delivers them, which also
drains any driver-side buffer. Consumers always get the most recent frame;
stale frames are overwritten, never queued.
"""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    bgr: np.ndarray        # Mirrored (if enabled) BGR frame, for the preview.
    rgb: np.ndarray        # Same frame in RGB, for MediaPipe.
    frame_id: int          # Monotonically increasing counter.
    timestamp: float       # time.monotonic() when the frame was grabbed.


def _backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW          # Opens much faster than MSMF and honors buffer size.
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY


class CameraStream:
    def __init__(self, index: int, width: int, height: int, fps: int,
                 mirror: bool = True, use_mjpg: bool = True) -> None:
        self._mirror = mirror
        self._cap = cv2.VideoCapture(index, _backend())
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(index)  # Fall back to whatever backend works.
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}. "
                               "Check the camera permission and --camera index.")

        if use_mjpg:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Ignored by some backends; the thread copes.

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._frame_id = 0
        self._running = False
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.frames_captured = 0  # Read by the FPS logger.

    def start(self) -> "CameraStream":
        self._running = True
        self._thread.start()
        return self

    def _run(self) -> None:
        failures = 0
        backoff = threading.Event()  # Used only to pause briefly after read errors.
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures in (1, 30) or failures % 300 == 0:
                    print(f"[camera] frame grab failed ({failures}x)")
                backoff.wait(0.01)
                continue
            failures = 0
            ts = time.monotonic()
            if self._mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._cond:
                self._frame_id += 1
                self._latest = Frame(frame, rgb, self._frame_id, ts)
                self.frames_captured += 1
                self._cond.notify_all()

    def wait_for_frame(self, after_id: int, timeout: float = 0.5) -> Frame | None:
        """Block until a frame newer than ``after_id`` exists; return the newest one."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._latest is not None and self._latest.frame_id > after_id,
                timeout=timeout,
            ):
                return None
            return self._latest

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
