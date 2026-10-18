"""Monitor geometry and high-DPI handling for Windows, macOS and Linux (X11).

Coordinates returned here are in the same space pynput uses to move the mouse:
  * Windows: physical pixels (we make the process per-monitor DPI aware first).
  * macOS:   logical points (Quartz). Retina scaling is handled by the OS.
  * Linux:   X11 root-window pixels.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MAC = SYSTEM == "Darwin"


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int
    primary: bool = False

    def __str__(self) -> str:
        tag = " (primary)" if self.primary else ""
        return f"{self.width}x{self.height} at ({self.left}, {self.top}){tag}"


def enable_dpi_awareness() -> None:
    """Windows only: stop the OS from scaling our coordinates on high-DPI screens.

    Must run before any window is created or the screen size is queried.
    """
    if not IS_WINDOWS:
        return
    import ctypes
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE_V2 (Windows 10 1703+)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _monitors_windows() -> list[Rect]:
    import ctypes
    from ctypes import wintypes

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    user32 = ctypes.windll.user32
    monitors: list[Rect] = []
    proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                   ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def callback(hmon, _hdc, _rect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcMonitor
            monitors.append(Rect(r.left, r.top, r.right - r.left, r.bottom - r.top,
                                 bool(info.dwFlags & 1)))
        return 1

    user32.EnumDisplayMonitors(None, None, proc_type(callback), 0)
    return monitors


def _monitors_mac() -> list[Rect]:
    import Quartz  # Installed with pynput on macOS (pyobjc-framework-Quartz).

    main_id = Quartz.CGMainDisplayID()
    err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    if err:
        ids = [main_id]
    monitors = []
    for did in ids[:count] if not err else ids:
        b = Quartz.CGDisplayBounds(did)
        monitors.append(Rect(int(b.origin.x), int(b.origin.y),
                             int(b.size.width), int(b.size.height), did == main_id))
    return monitors


def _monitors_x11() -> list[Rect]:
    from Xlib import display  # Installed with pynput on Linux (python-xlib).

    d = display.Display()
    try:
        root = d.screen().root
        try:
            mons = root.xrandr_get_monitors().monitors
            rects = [Rect(m.x, m.y, m.width_in_pixels, m.height_in_pixels, bool(m.primary))
                     for m in mons]
            if rects:
                if not any(r.primary for r in rects):
                    rects[0] = Rect(rects[0].left, rects[0].top, rects[0].width,
                                    rects[0].height, True)
                return rects
        except Exception:  # noqa: BLE001 - RandR missing or too old
            pass
        s = d.screen()
        return [Rect(0, 0, s.width_in_pixels, s.height_in_pixels, True)]
    finally:
        d.close()


def _monitors_tk() -> list[Rect]:
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    try:
        return [Rect(0, 0, root.winfo_screenwidth(), root.winfo_screenheight(), True)]
    finally:
        root.destroy()


def list_monitors() -> list[Rect]:
    getters = ([_monitors_windows] if IS_WINDOWS else
               [_monitors_mac] if IS_MAC else [_monitors_x11])
    getters.append(_monitors_tk)
    for getter in getters:
        try:
            mons = getter()
            if mons:
                # Primary first, then left-to-right, so indices are predictable.
                return sorted(mons, key=lambda r: (not r.primary, r.left, r.top))
        except Exception as exc:  # noqa: BLE001
            print(f"[screen] {getter.__name__} failed: {exc!r}")
    print("[screen] could not detect monitors, assuming 1920x1080")
    return [Rect(0, 0, 1920, 1080, True)]


def target_rect(selection: str | int = "all") -> Rect:
    """Screen area the active zone is mapped onto."""
    mons = list_monitors()
    if isinstance(selection, int) or (isinstance(selection, str) and selection.isdigit()):
        idx = int(selection)
        if 0 <= idx < len(mons):
            return mons[idx]
        print(f"[screen] monitor {idx} not found, using primary")
        selection = "primary"
    if selection == "primary":
        return next((m for m in mons if m.primary), mons[0])
    left = min(m.left for m in mons)
    top = min(m.top for m in mons)
    right = max(m.left + m.width for m in mons)
    bottom = max(m.top + m.height for m in mons)
    return Rect(left, top, right - left, bottom - top, True)
