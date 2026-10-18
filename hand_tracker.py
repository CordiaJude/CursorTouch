"""MediaPipe Hand Landmarker wrapper (Tasks API, LIVE_STREAM mode).

Frames are submitted with ``detect_async`` and results arrive on MediaPipe's
own thread via a callback. MediaPipe drops frames internally when it is busy,
so submitting every camera frame never builds up a backlog.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Landmark indices used across the project.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# Bone list used for drawing the skeleton.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class HandData:
    """One detected hand.

    norm: (x, y) per landmark, normalized to the (mirrored) frame, 0..1.
    px:   (x, y, z) per landmark in pixels. MediaPipe's z uses roughly the same
          scale as x, so it is multiplied by the frame width.
    """
    norm: list[tuple[float, float]]
    px: list[tuple[float, float, float]]
    width: int
    height: int


ResultHandler = Callable[[HandData | None, float], None]


def ensure_model(path: str, url: str) -> str:
    """Download the hand landmarker model on first run."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[tracker] downloading model to {path} ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print("[tracker] model downloaded")
    return path


class HandTracker:
    def __init__(self, model_path: str, on_result: ResultHandler, *,
                 min_detection: float, min_presence: float, min_tracking: float,
                 use_gpu: bool = False) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._on_result = on_result
        self._last_ts_ms = -1
        self.results_received = 0       # Read by the FPS logger.
        self.latency_ms_total = 0.0     # Sum of capture->result latencies since last read.
        self.latency_samples = 0
        self.last_result_time = time.monotonic()  # For the stall watchdog.

        delegate = (mp_tasks.BaseOptions.Delegate.GPU if use_gpu
                    else mp_tasks.BaseOptions.Delegate.CPU)
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path, delegate=delegate),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
            result_callback=self._callback,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_size = (0, 0)

    def submit(self, rgb: np.ndarray, capture_time: float) -> None:
        """Queue a frame. ``capture_time`` is its time.monotonic() grab time."""
        ts_ms = int(capture_time * 1000)
        if ts_ms <= self._last_ts_ms:     # MediaPipe requires strictly increasing timestamps.
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        self._frame_size = (rgb.shape[1], rgb.shape[0])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(image, ts_ms)

    def _callback(self, result, output_image, timestamp_ms: int) -> None:
        # Runs on MediaPipe's thread. Keep it fast and never let it raise.
        self.results_received += 1
        self.last_result_time = time.monotonic()
        self.latency_ms_total += time.monotonic() * 1000.0 - timestamp_ms
        self.latency_samples += 1

        hand: HandData | None = None
        if result.hand_landmarks:
            w, h = output_image.width, output_image.height
            lms = result.hand_landmarks[0]
            norm = [(lm.x, lm.y) for lm in lms]
            px = [(lm.x * w, lm.y * h, lm.z * w) for lm in lms]
            hand = HandData(norm, px, w, h)
        try:
            self._on_result(hand, timestamp_ms / 1000.0)
        except Exception as exc:  # noqa: BLE001 - must not kill MediaPipe's thread
            print(f"[tracker] result handler error: {exc!r}")

    def close(self) -> None:
        self._landmarker.close()
