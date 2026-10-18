"""Lightweight preview drawing (a few dozen OpenCV primitives per frame)."""

from __future__ import annotations

import cv2
import numpy as np

from gestures import FINGER_TIPS, Snapshot
from hand_tracker import HAND_CONNECTIONS, THUMB_TIP

GREEN = (0, 220, 0)
RED = (0, 0, 230)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
ZONE = (255, 180, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(frame: np.ndarray, snap: Snapshot, zone: tuple[float, float, float, float],
         fps: float, latency_ms: float) -> None:
    h, w = frame.shape[:2]

    x0, y0, x1, y1 = zone
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), ZONE, 1)

    hand = snap.hand
    if hand is not None:
        # Scale in case the result came from a frame of a different size.
        sx, sy = w / hand.width, h / hand.height
        pts = [(int(p[0] * sx), int(p[1] * sy)) for p in hand.px]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], WHITE, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, YELLOW, -1)
        # Thumb-to-finger lines: green while touching, red otherwise.
        for name, tip in FINGER_TIPS.items():
            color = GREEN if snap.touching.get(name) else RED
            cv2.line(frame, pts[THUMB_TIP], pts[tip], color, 2, cv2.LINE_AA)

    label_color = YELLOW if snap.paused else GREEN if snap.button_down else WHITE
    cv2.putText(frame, snap.label, (10, 34), FONT, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, snap.label, (10, 34), FONT, 1.0, label_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:4.1f}  latency {latency_ms:3.0f} ms", (10, h - 12),
                FONT, 0.55, WHITE, 1, cv2.LINE_AA)
