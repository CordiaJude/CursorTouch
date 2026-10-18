"""Maps the tracked hand to a smoothed screen position.

Pipeline per frame:
  1. Pick the source point: the index fingertip (landmark 8) normally, or a
     blend of tip and knuckle (landmark 5) while pinching.
  2. Map it from the active zone (a centered rectangle of the camera frame) to
     the screen rectangle.
  3. Add an offset that keeps the cursor continuous when the source point
     switches (tip <-> anchor), so switching never makes the cursor jump.
  4. One Euro filter, then a small dead zone.

Pinch handling ("clicks land where you were pointing"):
  * begin_pinch() rewinds the cursor to where it was PINCH_REWIND_MS ago, i.e.
    before the fingertip started curling toward the thumb, and freezes it there
    for PINCH_FREEZE_MS.
  * While pinched, the cursor follows the stable anchor point instead of the
    moving fingertip, so drags are still accurate.
"""

from __future__ import annotations

import math
from collections import deque

from filters import OneEuroFilter2D
from hand_tracker import INDEX_MCP, INDEX_TIP, HandData
from screen import Rect


class CursorController:
    def __init__(self, screen: Rect, cfg) -> None:
        self.screen = screen
        self.cfg = cfg
        hw, hh = cfg.ACTIVE_ZONE_WIDTH / 2.0, cfg.ACTIVE_ZONE_HEIGHT / 2.0
        cx = min(max(cfg.ACTIVE_ZONE_CENTER_X, hw), 1.0 - hw)
        cy = min(max(cfg.ACTIVE_ZONE_CENTER_Y, hh), 1.0 - hh)
        # Active zone in normalized frame coordinates: (x0, y0, x1, y1).
        self.zone = (cx - hw, cy - hh, cx + hw, cy + hh)

        self._filter = OneEuroFilter2D(cfg.FILTER_MIN_CUTOFF, cfg.FILTER_BETA,
                                       cfg.FILTER_D_CUTOFF)
        self._pinched = False
        self._offset = (0.0, 0.0)          # Screen-space offset added to the mapped source.
        self._offset_decaying = False      # True after a pinch ends: offset fades to zero.
        self._freeze_until = -1.0
        self._last_t: float | None = None
        self._last_target: tuple[float, float] | None = None
        self._emitted: tuple[int, int] | None = None
        self._jump_pending = False
        self._history: deque[tuple[float, float, float]] = deque(maxlen=64)

    # -- public API ----------------------------------------------------------

    def reset(self) -> None:
        """Call when the hand is lost so the next frame starts fresh."""
        self._filter.reset()
        self._pinched = False
        self._offset = (0.0, 0.0)
        self._offset_decaying = False
        self._freeze_until = -1.0
        self._jump_pending = False
        self._last_t = None
        self._last_target = None
        self._history.clear()

    def begin_pinch(self, hand: HandData, t: float) -> None:
        """Switch to the stable anchor point, rewind and freeze the cursor."""
        if self._pinched:
            return
        self._pinched = True
        self._offset_decaying = False

        rewind_s = self.cfg.PINCH_REWIND_MS / 1000.0
        target = self._filter.value or self._last_target
        if rewind_s > 0 and self._history:
            # Newest history entry at or before (t - rewind). If the hand only
            # just appeared, fall back to the oldest one we have.
            cutoff = t - rewind_s
            past = self._history[0]
            for entry in reversed(self._history):
                if entry[0] <= cutoff:
                    past = entry
                    break
            target = (past[1], past[2])
        if target is None:
            return

        self._rebase(hand, target)
        self._filter.reset(target, t)
        self._freeze_until = t + self.cfg.PINCH_FREEZE_MS / 1000.0
        self._emitted = (int(round(target[0])), int(round(target[1])))
        self._jump_pending = True  # Next update() sends the rewound position once.

    def end_pinch(self, hand: HandData | None, t: float) -> None:
        """Go back to following the fingertip, fading out the offset."""
        if not self._pinched:
            return
        self._pinched = False
        self._freeze_until = -1.0
        if hand is not None and self._filter.value is not None:
            self._rebase(hand, self._filter.value)
            self._offset_decaying = True

    @property
    def frozen_position(self) -> tuple[int, int] | None:
        return self._emitted

    def update(self, hand: HandData, t: float) -> tuple[int, int] | None:
        """Return the new cursor position, or None if the cursor should not move."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        if self._offset_decaying and dt > 0:
            k = math.exp(-dt / max(1e-3, self.cfg.ANCHOR_RELEASE_DECAY_MS / 1000.0))
            self._offset = (self._offset[0] * k, self._offset[1] * k)
            if abs(self._offset[0]) < 0.5 and abs(self._offset[1]) < 0.5:
                self._offset = (0.0, 0.0)
                self._offset_decaying = False

        sx, sy = self._map(self._source(hand))
        target = self._clamp(sx + self._offset[0], sy + self._offset[1])
        self._last_target = target

        if t < self._freeze_until:
            if self._jump_pending:
                self._jump_pending = False
                return self._emitted
            return None
        if self._freeze_until > 0:
            # Freeze just ended: continue from the frozen spot, no jump.
            self._freeze_until = -1.0
            frozen = self._filter.value
            if frozen is not None:
                self._rebase(hand, frozen)
                target = frozen

        fx, fy = self._filter(target, t)
        self._history.append((t, fx, fy))

        pos = (int(round(fx)), int(round(fy)))
        if self._emitted is not None:
            if math.hypot(fx - self._emitted[0], fy - self._emitted[1]) < self.cfg.DEAD_ZONE_PX:
                return None
        self._emitted = pos
        return pos

    # -- helpers -------------------------------------------------------------

    def _source(self, hand: HandData) -> tuple[float, float]:
        tip = hand.norm[INDEX_TIP]
        if not self._pinched:
            return tip
        knuckle = hand.norm[INDEX_MCP]
        b = self.cfg.PINCH_ANCHOR_BLEND
        return (tip[0] + (knuckle[0] - tip[0]) * b, tip[1] + (knuckle[1] - tip[1]) * b)

    def _map(self, p: tuple[float, float]) -> tuple[float, float]:
        x0, y0, x1, y1 = self.zone
        u = min(max((p[0] - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((p[1] - y0) / (y1 - y0), 0.0), 1.0)
        s = self.screen
        return (s.left + u * (s.width - 1), s.top + v * (s.height - 1))

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        s = self.screen
        return (min(max(x, s.left), s.left + s.width - 1),
                min(max(y, s.top), s.top + s.height - 1))

    def _rebase(self, hand: HandData, desired: tuple[float, float]) -> None:
        """Choose the offset so the current source maps exactly to ``desired``."""
        sx, sy = self._map(self._source(hand))
        self._offset = (desired[0] - sx, desired[1] - sy)
