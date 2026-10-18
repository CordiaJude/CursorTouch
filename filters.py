"""One Euro Filter (Casiez, Roussel & Vogel, CHI 2012).

An adaptive low-pass filter: the cutoff frequency rises with speed, so the
output is heavily smoothed when the hand is still (no jitter) and lightly
smoothed when it moves fast (no lag).

    cutoff = min_cutoff + beta * |speed|

This 2D version uses the speed of the point as a whole (vector magnitude), so
diagonal movement is treated the same as horizontal or vertical movement.
Plain floats are used instead of numpy: for two values they are much faster.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: tuple[float, float] | None = None
        self._dx: tuple[float, float] = (0.0, 0.0)
        self._t: float | None = None

    @property
    def value(self) -> tuple[float, float] | None:
        return self._x

    def reset(self, x: tuple[float, float] | None = None, t: float | None = None) -> None:
        """Forget history. If ``x`` is given, restart the filter at that value."""
        self._x = x
        self._dx = (0.0, 0.0)
        self._t = t if x is not None else None

    def __call__(self, x: tuple[float, float], t: float) -> tuple[float, float]:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0.0:
            dt = 1e-3
        self._t = t

        # Smoothed velocity estimate.
        a_d = _alpha(self.d_cutoff, dt)
        vx = (x[0] - self._x[0]) / dt
        vy = (x[1] - self._x[1]) / dt
        dx = (a_d * vx + (1.0 - a_d) * self._dx[0],
              a_d * vy + (1.0 - a_d) * self._dx[1])
        self._dx = dx

        # Speed-dependent cutoff, then a plain exponential low-pass.
        cutoff = self.min_cutoff + self.beta * math.hypot(dx[0], dx[1])
        a = _alpha(cutoff, dt)
        self._x = (a * x[0] + (1.0 - a) * self._x[0],
                   a * x[1] + (1.0 - a) * self._x[1])
        return self._x
