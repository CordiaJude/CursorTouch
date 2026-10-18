"""Gesture state machine tests with synthetic hands (no camera needed).

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from filters import OneEuroFilter2D  # noqa: E402
from cursor import CursorController  # noqa: E402
from gestures import Controls, GestureEngine, State  # noqa: E402
from hand_tracker import HandData  # noqa: E402
from screen import Rect  # noqa: E402

W, H = 640, 480


class FakeInput:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.left_down = False

    def move(self, pos):
        self.events.append(("move", pos))

    def click(self):
        self.events.append("click")

    def press_left(self):
        if not self.left_down:
            self.left_down = True
            self.events.append("press")

    def release_left(self):
        if self.left_down:
            self.left_down = False
            self.events.append("release")

    release_all = release_left

    def copy(self):
        self.events.append("copy")

    def paste(self):
        self.events.append("paste")

    def actions(self):
        return [e for e in self.events if isinstance(e, str)]


def make_hand(touch: set[str] = frozenset(), shift: float = 0.0, curl: float = 0.0) -> HandData:
    """Hand size (wrist->middle knuckle) is 100 px. Touching fingertips sit on the thumb."""
    px = [(320.0 + shift, 400.0, 0.0)] * 21
    px = list(px)
    px[9] = (320.0 + shift, 300.0, 0.0)            # middle knuckle -> hand size 100
    px[5] = (300.0 + shift, 305.0, 0.0)            # index knuckle
    px[13] = (340.0 + shift, 305.0, 0.0)           # ring knuckle
    thumb = (250.0 + shift, 240.0, 0.0)
    px[4] = thumb
    open_tips = {8: (300.0 + shift, 200.0), 12: (320.0 + shift, 195.0),
                 16: (340.0 + shift, 205.0), 20: (360.0 + shift, 220.0)}
    names = {8: "index", 12: "middle", 16: "ring", 20: "pinky"}
    for tip, xy in open_tips.items():
        px[tip] = (thumb[0] + 5, thumb[1], 0.0) if names[tip] in touch else (*xy, 0.0)
    if curl and "index" not in touch:              # index tip part-way toward the thumb
        ox, oy = open_tips[8]
        px[8] = (ox + (thumb[0] + 5 - ox) * curl, oy + (thumb[1] - oy) * curl, 0.0)
    norm = [(p[0] / W, p[1] / H) for p in px]
    return HandData(norm, px, W, H)


class GestureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inp = FakeInput()
        self.controls = Controls()
        cursor = CursorController(Rect(0, 0, 1920, 1080, True), config)
        self.eng = GestureEngine(config, cursor, self.inp, self.controls)
        self.t = 0.0

    def run_frames(self, n: int, touch=frozenset(), shift=0.0, dt=1 / 30, curl=0.0):
        for _ in range(n):
            self.t += dt
            self.eng.update(make_hand(set(touch), shift, curl), self.t)

    def test_quick_tap_is_one_click(self):
        self.run_frames(5)
        self.run_frames(4, {"index"})          # ~133 ms pinch
        self.run_frames(5)
        self.assertEqual(self.inp.actions(), ["click"])
        self.assertIs(self.eng.state, State.IDLE)

    def test_long_pinch_selects_and_releases(self):
        self.run_frames(5)
        self.run_frames(15, {"index"})         # 500 ms > TAP_MAX_MS
        self.assertIs(self.eng.state, State.HIGHLIGHT_DRAG)
        self.assertTrue(self.inp.left_down)
        self.run_frames(3)
        self.assertEqual(self.inp.actions(), ["press", "release"])

    def test_pinch_and_move_starts_drag_early(self):
        self.run_frames(5)
        self.run_frames(3, {"index"})          # past the confirm window
        for i in range(3):
            self.run_frames(1, {"index"}, shift=10.0 * (i + 1))
        self.assertIs(self.eng.state, State.HIGHLIGHT_DRAG)

    def test_copy_and_paste_fire_once(self):
        self.run_frames(5)
        self.run_frames(10, {"middle"})
        self.run_frames(5)
        self.run_frames(10, {"ring"})
        self.run_frames(5)
        self.assertEqual(self.inp.actions(), ["copy", "paste"])

    def test_copy_cooldown(self):
        self.run_frames(5)
        self.run_frames(3, {"middle"})
        self.run_frames(2)
        self.run_frames(3, {"middle"})         # second tap within 400 ms
        self.run_frames(5)
        self.assertEqual(self.inp.actions(), ["copy"])

    def test_grab_beats_ring_and_index(self):
        # Ring touches a frame before index: must become a grab, never a paste.
        self.run_frames(5)
        self.run_frames(1, {"ring"})
        self.run_frames(10, {"index", "ring"})
        self.assertIs(self.eng.state, State.GRAB_DRAG)
        # Index lets go a little early, ring follows: still one press/release, no paste.
        self.run_frames(1, {"ring"})
        self.run_frames(5)
        self.assertEqual(self.inp.actions(), ["press", "release"])
        self.assertIs(self.eng.state, State.IDLE)

    def test_hand_lost_releases_button(self):
        self.run_frames(5)
        self.run_frames(15, {"index"})
        self.assertTrue(self.inp.left_down)
        self.t += 1 / 30
        self.eng.update(None, self.t)
        self.assertFalse(self.inp.left_down)

    def test_pause_releases_and_blocks(self):
        self.run_frames(5)
        self.run_frames(15, {"index"})
        self.controls.toggle_pause()
        self.run_frames(2, {"index"})
        self.assertFalse(self.inp.left_down)
        self.assertIs(self.eng.state, State.PAUSED)
        self.run_frames(10, {"middle"})
        self.assertNotIn("copy", self.inp.actions())

    def test_click_lands_where_pointing(self):
        self.run_frames(10)
        before = [e for e in self.inp.events if isinstance(e, tuple)][-1][1]
        self.inp.events.clear()
        for curl in (0.25, 0.5, 0.75):         # fingertip curls toward the thumb...
            self.run_frames(1, curl=curl)
        self.run_frames(4, {"index"})          # ...then touches
        moves = [e[1] for e in self.inp.events if isinstance(e, tuple)]
        self.assertTrue(moves)                 # the curl did drag the cursor away...
        last = moves[-1]                       # ...but the click position is restored
        self.assertLess(abs(last[0] - before[0]) + abs(last[1] - before[1]), 4)
        self.assertEqual(self.inp.actions(), [])


    def test_thumb_on_curled_middle_is_not_copy(self):
        # Pointing pose: thumb tucked over the middle finger curled into the palm.
        hand = make_hand()
        px = list(hand.px)
        px[4] = (290.0, 340.0, 0.0)
        px[12] = (295.0, 345.0, 0.0)           # 0.07 hand units from the thumb, but reach 0.6
        curled = HandData([(p[0] / W, p[1] / H) for p in px], px, W, H)
        for _ in range(20):
            self.t += 1 / 30
            self.eng.update(curled, self.t)
        self.assertEqual(self.inp.actions(), [])
        self.assertFalse(self.eng.touching["middle"])


class FilterTests(unittest.TestCase):
    def test_still_input_is_steady_and_moves_follow(self):
        f = OneEuroFilter2D(1.0, 0.01, 1.0)
        t = 0.0
        out = (0.0, 0.0)
        for i in range(60):                    # +-3 px jitter around 100
            t += 1 / 30
            out = f((100.0 + (3 if i % 2 else -3), 100.0), t)
        self.assertLess(abs(out[0] - 100.0), 1.5)
        for _ in range(10):                    # fast jump to 1000
            t += 1 / 30
            out = f((1000.0, 100.0), t)
        self.assertGreater(out[0], 950.0)


if __name__ == "__main__":
    unittest.main()
