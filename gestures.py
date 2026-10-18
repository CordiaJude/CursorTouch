"""Gesture detection: touch hysteresis + a single state machine.

Touches
-------
A "touch" is the thumb tip (4) close to a fingertip (index 8, middle 12,
ring 16). Distances are divided by the hand size (wrist 0 -> middle knuckle 9)
so they are independent of how far the hand is from the camera. Each finger
has two thresholds: a touch starts below TOUCH_ON and ends above TOUCH_OFF.

State machine (only one gesture can be active at a time)
--------------------------------------------------------

    IDLE ──any touch──▶ PINCH_PENDING ──index+ring──▶ GRAB_DRAG (button held)
      ▲                    │  (waits CONFIRM_WINDOW_MS so a grab is never
      │                    │   mistaken for a click or paste)
      │                    ├─index───▶ CLICK_PENDING ──released < TAP_MAX_MS──▶ click
      │                    │               ├─held > TAP_MAX_MS or moved──▶ HIGHLIGHT_DRAG
      │                    │               └─ring joins──▶ GRAB_DRAG
      │                    ├─middle──▶ copy  (once, with cooldown)
      │                    └─ring────▶ paste (once, with cooldown)
      │
      └── WAIT_RELEASE ◀── every gesture ends here and waits until no finger
                            is touching, so leftover contact (e.g. the ring
                            finger after a grab) can never trigger a new gesture.

The cursor keeps moving in every state except during the short freeze at the
start of a pinch (see cursor.py), which is what makes clicks land where you
were pointing.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from actions import InputController
from cursor import CursorController
from hand_tracker import (INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP,
                          MIDDLE_TIP, PINKY_PIP, PINKY_TIP, RING_MCP, RING_PIP, RING_TIP,
                          THUMB_TIP, WRIST, HandData)

FINGER_TIPS = {"index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP}
FINGER_MCPS = {"index": INDEX_MCP, "middle": MIDDLE_MCP, "ring": RING_MCP}


class State(Enum):
    IDLE = "MOVE"
    PINCH_PENDING = "PINCH..."
    CLICK_PENDING = "PINCH"
    HIGHLIGHT_DRAG = "SELECTING"
    GRAB_DRAG = "GRAB"
    WAIT_RELEASE = "RELEASE"
    PAUSED = "PAUSED"


class Controls:
    """Flags shared between threads (hotkey listener, main loop, engine)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.paused = False

    def toggle_pause(self) -> bool:
        with self._lock:
            self.paused = not self.paused
            return self.paused


@dataclass(frozen=True)
class Snapshot:
    """Immutable view of the engine for the preview window (read on another thread)."""
    hand: HandData | None = None
    distances: dict[str, float] = field(default_factory=dict)
    touching: dict[str, bool] = field(default_factory=dict)
    label: str = "NO HAND"
    paused: bool = False
    button_down: bool = False


def _d3(a: tuple[float, float, float], b: tuple[float, float, float], use_z: bool) -> float:
    dz = (a[2] - b[2]) if use_z else 0.0
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + dz * dz)


class GestureEngine:
    def __init__(self, cfg, cursor: CursorController, inputs: InputController,
                 controls: Controls, debug: bool = False) -> None:
        self.cfg = cfg
        self.cursor = cursor
        self.input = inputs
        self.controls = controls
        self.debug = debug

        self.state = State.IDLE
        self.touching = {f: False for f in FINGER_TIPS}
        self.snapshot = Snapshot()     # Replaced atomically every frame.

        self._last_t: float | None = None
        self._hand_lost_since: float | None = None
        self._pinch_start_t = 0.0
        self._pinch_finger = "index"
        self._knuckle_start = (0.0, 0.0)
        self._grab_open_since: float | None = None
        self._last_key_t = {"copy": -1e9, "paste": -1e9}
        self._flash_label = ""
        self._flash_until = 0.0
        self._palm_since: float | None = None
        self._palm_pos = (0.0, 0.0)
        self._palm_latched = False
        self._frames_with_hand = 0
        self._reach: dict[str, float] = {}
        self._extension: dict[str, float] = {}

    # ------------------------------------------------------------------ frame

    def update(self, hand: HandData | None, t: float) -> None:
        """Process one tracking result. Called on MediaPipe's callback thread."""
        try:
            self._update(hand, t)
        except Exception:
            self.input.release_all()   # Never leave the button held after an error.
            self.state = State.WAIT_RELEASE
            raise

    def _update(self, hand: HandData | None, t: float) -> None:
        if self._last_t is not None and t - self._last_t > self.cfg.STALL_RELEASE_MS / 1000.0:
            self._abort(None, t)       # Long gap: the watchdog may have released the button.
        self._last_t = t

        if hand is None:
            self._on_no_hand(t)
            return
        self._hand_lost_since = None
        self._frames_with_hand += 1

        hand_size = max(_d3(hand.px[WRIST], hand.px[MIDDLE_MCP], self.cfg.USE_3D_DISTANCE), 1e-6)
        dists = self._distances(hand, hand_size)
        use_z = self.cfg.USE_3D_DISTANCE
        reach = {f: _d3(hand.px[WRIST], hand.px[tip], use_z) / hand_size
                 for f, tip in FINGER_TIPS.items()}
        extension = {f: _d3(hand.px[FINGER_MCPS[f]], hand.px[tip], use_z) / hand_size
                     for f, tip in FINGER_TIPS.items()}
        self._apply_hysteresis(dists, reach, extension)

        if self.cfg.PALM_PAUSE_ENABLED:
            self._check_palm_toggle(hand, hand_size, t)

        if self.controls.paused:
            if self.state is not State.PAUSED:
                self._abort(hand, t)
                self.cursor.reset()
                self.state = State.PAUSED
            self._publish(hand, dists, t)
            return
        if self.state is State.PAUSED:
            self.state = State.WAIT_RELEASE  # Resume only once all fingers are open.

        post_move: list[Callable[[], None]] = []
        self._step(hand, hand_size, t, post_move)

        pos = self.cursor.update(hand, t)
        if pos is not None:
            self.input.move(pos)
        for action in post_move:       # Clicks/presses happen after the cursor is in place.
            action()

        self._publish(hand, dists, t)

    # ---------------------------------------------------------- state machine

    def _step(self, hand: HandData, hand_size: float, t: float,
              post: list[Callable[[], None]]) -> None:
        cfg = self.cfg
        idx, mid, ring = (self.touching["index"], self.touching["middle"],
                          self.touching["ring"])
        any_touch = idx or mid or ring

        if self.state is State.IDLE:
            if idx and ring:
                self._start_pinch(hand, "index", t)
                self._start_grab(t, post)
            elif any_touch:
                # Remember which finger touched first; index wins ties because
                # it drives click/select, the most common gesture.
                finger = "index" if idx else "ring" if ring else "middle"
                self._start_pinch(hand, finger, t)
                self.state = State.PINCH_PENDING

        elif self.state is State.PINCH_PENDING:
            # Grab has priority over everything: check it first.
            if idx and ring:
                self._start_grab(t, post)
            elif not any_touch:
                # Released inside the confirmation window: a very quick tap of
                # the finger that started it (it can't have been a grab).
                self._fire_tap(self._pinch_finger, t, post)
            elif t - self._pinch_start_t >= cfg.CONFIRM_WINDOW_MS / 1000.0:
                if idx:
                    self.state = State.CLICK_PENDING
                elif ring:
                    self._fire_tap("ring", t, post)
                else:
                    self._fire_tap("middle", t, post)

        elif self.state is State.CLICK_PENDING:
            held = t - self._pinch_start_t
            moved = self._knuckle_moved(hand, hand_size)
            if idx and ring:
                self._start_grab(t, post)
            elif not idx:
                if held <= cfg.TAP_MAX_MS / 1000.0 and moved <= cfg.DRAG_START_MOVE:
                    self._fire_tap("index", t, post)
                else:
                    self.state = State.WAIT_RELEASE
            elif held > cfg.TAP_MAX_MS / 1000.0 or moved > cfg.DRAG_START_MOVE:
                # Pinch held or dragged: press and keep the button down to select.
                post.append(self.input.press_left)
                self.state = State.HIGHLIGHT_DRAG

        elif self.state is State.HIGHLIGHT_DRAG:
            if not idx:
                self.input.release_left()
                self.state = State.WAIT_RELEASE

        elif self.state is State.GRAB_DRAG:
            if not idx and not ring:
                self._end_grab()
            elif not (idx and ring):
                # One finger let go; give the (flicker-prone) ring finger a
                # short grace period before treating the grab as opened.
                if self._grab_open_since is None:
                    self._grab_open_since = t
                elif t - self._grab_open_since > cfg.GRAB_RELEASE_GRACE_MS / 1000.0:
                    self._end_grab()
            else:
                self._grab_open_since = None

        if self.state is State.WAIT_RELEASE and not any_touch:
            self.cursor.end_pinch(hand, t)
            self.state = State.IDLE

    def _start_pinch(self, hand: HandData, finger: str, t: float) -> None:
        self._pinch_start_t = t
        self._pinch_finger = finger
        self._knuckle_start = hand.px[INDEX_MCP][:2]
        self.cursor.begin_pinch(hand, t)   # Rewind + freeze so the click lands on target.

    def _start_grab(self, t: float, post: list[Callable[[], None]]) -> None:
        self._grab_open_since = None
        post.append(self.input.press_left)
        self.state = State.GRAB_DRAG

    def _end_grab(self) -> None:
        self.input.release_left()
        self._grab_open_since = None
        self.state = State.WAIT_RELEASE

    def _fire_tap(self, finger: str, t: float, post: list[Callable[[], None]]) -> None:
        """One-shot actions. Always ends in WAIT_RELEASE so a tap fires only once."""
        if finger == "index":
            post.append(self.input.click)
            self._flash("CLICK", t)
        else:
            action = "copy" if finger == "middle" else "paste"
            if t - self._last_key_t[action] >= self.cfg.KEY_COOLDOWN_MS / 1000.0:
                self._last_key_t[action] = t
                post.append(self.input.copy if action == "copy" else self.input.paste)
                self._flash(action.upper(), t)
        self.state = State.WAIT_RELEASE

    def _abort(self, hand: HandData | None, t: float) -> None:
        """Cancel any gesture and make sure the mouse button is up."""
        self.input.release_all()
        active = self.state not in (State.IDLE, State.PAUSED)
        self.cursor.end_pinch(hand, t)
        self._grab_open_since = None
        if active:
            self.state = State.WAIT_RELEASE
        if hand is None:
            self.touching = {f: False for f in FINGER_TIPS}

    def _on_no_hand(self, t: float) -> None:
        self._frames_with_hand = 0
        if self._hand_lost_since is None:
            self._hand_lost_since = t
        if t - self._hand_lost_since >= self.cfg.HAND_LOST_RELEASE_MS / 1000.0:
            self._abort(None, t)
            self.cursor.reset()
            self._palm_since = None
        self._publish(None, {}, t)

    # --------------------------------------------------------------- helpers

    def _distances(self, hand: HandData, hand_size: float) -> dict[str, float]:
        use_z = self.cfg.USE_3D_DISTANCE
        thumb = hand.px[THUMB_TIP]
        d = {name: _d3(thumb, hand.px[tip], use_z) / hand_size
             for name, tip in FINGER_TIPS.items()}
        d["pinky"] = _d3(thumb, hand.px[PINKY_TIP], use_z) / hand_size  # Debug only.
        return d

    def _apply_hysteresis(self, dists: dict[str, float], reach: dict[str, float],
                          extension: dict[str, float]) -> None:
        settled = self._frames_with_hand > self.cfg.HAND_SETTLE_FRAMES
        for f in FINGER_TIPS:
            if self.touching[f]:
                self.touching[f] = dists[f] < self.cfg.TOUCH_OFF[f]
            else:
                # Start only below the lower threshold AND with the finger reaching
                # out from the palm and not folded up: a thumb resting on curled
                # fingers (pointing pose, fist) is not a pinch.
                self.touching[f] = (settled and dists[f] < self.cfg.TOUCH_ON[f]
                                    and reach[f] >= self.cfg.MIN_TOUCH_REACH
                                    and extension[f] >= self.cfg.MIN_TOUCH_EXTENSION)
        self._reach, self._extension = reach, extension

    def _knuckle_moved(self, hand: HandData, hand_size: float) -> float:
        x, y = hand.px[INDEX_MCP][:2]
        return math.hypot(x - self._knuckle_start[0], y - self._knuckle_start[1]) / hand_size

    def _check_palm_toggle(self, hand: HandData, hand_size: float, t: float) -> None:
        """Open palm (all fingers extended and spread) held still -> toggle pause."""
        if self.state not in (State.IDLE, State.PAUSED) or not self._is_open_palm(hand, hand_size):
            self._palm_since = None
            self._palm_latched = False
            return
        if self._palm_latched:
            return                      # Already toggled; wait for the palm to close.
        pos = hand.px[MIDDLE_MCP][:2]
        if (self._palm_since is None or
                math.hypot(pos[0] - self._palm_pos[0], pos[1] - self._palm_pos[1])
                / hand_size > 0.2):
            self._palm_since, self._palm_pos = t, pos   # Moved: restart the timer.
            return
        if t - self._palm_since >= self.cfg.PALM_PAUSE_HOLD_MS / 1000.0:
            paused = self.controls.toggle_pause()
            print(f"[gesture] open palm -> {'PAUSED' if paused else 'RESUMED'}")
            self._palm_latched = True
            self._palm_since = None

    def _is_open_palm(self, hand: HandData, hand_size: float) -> bool:
        if any(self.touching.values()):
            return False
        p, use_z = hand.px, self.cfg.USE_3D_DISTANCE
        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
            if _d3(p[WRIST], p[tip], use_z) < 1.15 * _d3(p[WRIST], p[pip], use_z):
                return False            # Finger curled.
        return _d3(p[THUMB_TIP], p[INDEX_MCP], use_z) / hand_size > 0.5  # Thumb spread out.

    def _flash(self, label: str, t: float) -> None:
        self._flash_label = label
        self._flash_until = t + self.cfg.FLASH_LABEL_MS / 1000.0

    def _publish(self, hand: HandData | None, dists: dict[str, float], t: float) -> None:
        paused = self.controls.paused
        if paused:
            label = "PAUSED"
        elif hand is None:
            label = "NO HAND"
        elif t < self._flash_until:
            label = self._flash_label
        else:
            label = self.state.value
        self.snapshot = Snapshot(hand, dists, dict(self.touching), label, paused,
                                 self.input.left_down)
        if self.debug and dists:
            touch = "".join(f[0].upper() if self.touching[f] else "-" for f in FINGER_TIPS)
            print(f"[debug] index {dists['index']:.3f}  middle {dists['middle']:.3f}  "
                  f"ring {dists['ring']:.3f}  pinky {dists['pinky']:.3f}  "
                  f"touch {touch}  reach/ext "
                  + " ".join(f"{f[0].upper()}{self._reach.get(f, 0):.2f}/"
                             f"{self._extension.get(f, 0):.2f}" for f in FINGER_TIPS)
                  + f"  state {label}")
