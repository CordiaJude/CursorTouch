"""Mouse and keyboard output via pynput.

All methods are thread-safe: the gesture engine calls them from MediaPipe's
callback thread, while the pause hotkey and shutdown code call
``release_all()`` from other threads. The held-button state is tracked here
so a button can always be released, and releasing twice is harmless.
"""

from __future__ import annotations

import threading

from screen import IS_MAC


class InputController:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._lock = threading.RLock()
        self._left_down = False
        if dry_run:
            self._mouse = self._kb = None
            return
        from pynput import keyboard, mouse
        self._Button = mouse.Button
        self._mouse = mouse.Controller()
        self._kb = keyboard.Controller()
        # Cmd on macOS, Ctrl everywhere else.
        self._modifier = keyboard.Key.cmd if IS_MAC else keyboard.Key.ctrl

    @property
    def left_down(self) -> bool:
        return self._left_down

    def move(self, pos: tuple[int, int]) -> None:
        if self._mouse is not None:
            self._mouse.position = pos

    def click(self) -> None:
        with self._lock:
            if self._left_down:          # Never click on top of a held drag.
                return
            if self._mouse is not None:
                self._mouse.click(self._Button.left, 1)

    def press_left(self) -> None:
        with self._lock:
            if self._left_down:
                return
            self._left_down = True
            if self._mouse is not None:
                self._mouse.press(self._Button.left)

    def release_left(self) -> None:
        with self._lock:
            if not self._left_down:
                return
            self._left_down = False
            if self._mouse is not None:
                self._mouse.release(self._Button.left)

    def release_all(self) -> None:
        """Safety net: make sure no mouse button is left held down."""
        try:
            self.release_left()
        except Exception as exc:  # noqa: BLE001 - called from cleanup paths
            print(f"[input] release failed: {exc!r}")

    def _shortcut(self, char: str) -> None:
        if self._kb is None:
            return
        with self._kb.pressed(self._modifier):
            self._kb.tap(char)

    def copy(self) -> None:
        self._shortcut("c")

    def paste(self) -> None:
        self._shortcut("v")
