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


@dataclass(frozen=True)
class CameraInfo:
    index: int             # The OpenCV index for this camera.
    name: str
    builtin: bool          # The computer's own camera (e.g. "MacBook Air Camera").
    phone: bool            # An iPhone/iPad used as a Continuity Camera.


def list_cameras_mac() -> list[CameraInfo] | None:
    """Name every camera in the same order OpenCV's AVFoundation backend indexes them.

    Returns None if the names can't be read (pyobjc-framework-AVFoundation missing).
    """
    try:
        import AVFoundation as av
    except ImportError:
        return None
    # OpenCV builds its index list as: video devices, then muxed devices.
    devices = list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo) or [])
    devices += list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed) or [])
    builtin_type = getattr(av, "AVCaptureDeviceTypeBuiltInWideAngleCamera", None)
    continuity_type = getattr(av, "AVCaptureDeviceTypeContinuityCamera", None)
    cams = []
    for i, dev in enumerate(devices):
        name = str(dev.localizedName())
        dtype = dev.deviceType() if hasattr(dev, "deviceType") else None
        model = str(dev.modelID() or "") if hasattr(dev, "modelID") else ""
        phone = ((continuity_type is not None and dtype == continuity_type)
                 or any(k in (name + " " + model) for k in ("iPhone", "iPad")))
        builtin = not phone and (
            (builtin_type is not None and dtype == builtin_type)
            or any(k in name for k in ("FaceTime", "MacBook", "Built-in", "iMac")))
        cams.append(CameraInfo(i, name, builtin, phone))
    return cams


def resolve_camera_index(requested: int, prefer_builtin: bool, allow_phone: bool) -> int:
    """Pick the camera to open. On macOS: the built-in camera, never an iPhone.

    Raises RuntimeError rather than silently opening a phone camera.
    """
    if platform.system() != "Darwin":
        return requested
    cams = list_cameras_mac()
    if cams is None:
        print("[camera] can't read camera names (run `pip install -r requirements.txt`); "
              f"using index {requested}")
        return requested
    for c in cams:
        tag = " [built-in]" if c.builtin else " [iPhone - blocked]" if c.phone and not allow_phone else ""
        print(f"[camera] found {c.index}: {c.name}{tag}")
    if prefer_builtin:
        builtin = next((c for c in cams if c.builtin), None)
        if builtin is not None:
            return builtin.index
        print("[camera] no built-in camera found")
    chosen = next((c for c in cams if c.index == requested), None)
    if chosen is not None and chosen.phone and not allow_phone:
        others = [c for c in cams if not c.phone]
        if others:
            print(f"[camera] camera {requested} is an iPhone; using {others[0].name} instead")
            return others[0].index
        raise RuntimeError("The only camera available is an iPhone, and phone cameras are "
                           "disabled (ALLOW_PHONE_CAMERA in config.py).")
    return requested


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

        # MJPG helps USB webcams on Windows/Linux reach 30+ FPS, but on macOS
        # (AVFoundation) forcing it can produce black frames, so skip it there.
        if use_mjpg and platform.system() != "Darwin":
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
        dark_since: float | None = None
        warned_dark = False
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
            # Warn once if the camera only delivers black frames (wrong camera,
            # an iPhone Continuity Camera that isn't active, lens covered...).
            if not warned_dark and self.frames_captured % 15 == 0:
                if float(frame[::16, ::16].mean()) < 8.0:
                    dark_since = dark_since or ts
                    if ts - dark_since > 3.0:
                        warned_dark = True
                        print("[camera] frames are completely black. Wrong camera? "
                              "Run `python main.py --list-cameras` and pick one with --camera N.")
                else:
                    dark_since = None
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

    @staticmethod
    def probe(max_index: int = 5) -> None:
        """Print which camera indices open and whether they deliver real images."""
        backoff = threading.Event()
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, _backend())
            if not cap.isOpened():
                cap.release()
                print(f"  camera {idx}: not available")
                continue
            brightness, size = None, ""
            deadline = time.monotonic() + 3.0  # Cameras often send dark frames while warming up.
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None:
                    size = f"{frame.shape[1]}x{frame.shape[0]}"
                    brightness = float(frame.mean())
                    if brightness >= 8.0:
                        break
                else:
                    backoff.wait(0.05)
            cap.release()
            if brightness is None:
                print(f"  camera {idx}: opens but gives no frames")
            else:
                verdict = "OK" if brightness >= 8.0 else "BLACK (probably not this one)"
                print(f"  camera {idx}: {size}, brightness {brightness:.0f} -> {verdict}")

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
