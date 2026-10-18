"""CursorTouch: control the mouse with one hand through your webcam.

Threads:
  camera     - grabs frames, keeps only the newest one (camera.py)
  mediapipe  - runs hand tracking and calls GestureEngine.update(), which moves
               the cursor immediately when a result is ready (lowest latency)
  main       - submits frames to MediaPipe, draws the optional preview, logs FPS
  hotkey     - pynput keyboard listener for the pause key
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import config
import screen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-gesture mouse controller")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preview", dest="preview", action="store_true", default=None,
                   help="show the camera preview window (default from config.py)")
    g.add_argument("--no-preview", dest="preview", action="store_false",
                   help="run without a preview window (lowest overhead)")
    p.add_argument("--debug", action="store_true",
                   help="print normalized thumb-to-finger distances every frame")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="camera index")
    p.add_argument("--monitor", default=str(config.MONITOR),
                   help='"all", "primary" or a monitor index')
    p.add_argument("--list-monitors", action="store_true", help="print monitors and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="track and recognise gestures without moving the mouse or typing")
    return p.parse_args()


def start_hotkey_listener(controls, inputs):
    """Global pause/resume key. Returns the listener, or None if unavailable."""
    try:
        from pynput import keyboard
    except Exception as exc:  # noqa: BLE001
        print(f"[hotkey] unavailable: {exc!r}")
        return None
    name = config.PAUSE_HOTKEY.lower()
    hotkey = getattr(keyboard.Key, name, None) or keyboard.KeyCode.from_char(name)

    def on_press(key):
        if key == hotkey:
            paused = controls.toggle_pause()
            if paused:
                inputs.release_all()       # Instantly, without waiting for the next frame.
            print(f"[hotkey] {'PAUSED' if paused else 'RESUMED'}")

    listener = keyboard.Listener(on_press=on_press, name="hotkey")
    listener.daemon = True
    listener.start()
    return listener


def main() -> int:
    args = parse_args()
    screen.enable_dpi_awareness()   # Before any window or screen query (Windows).

    if args.list_monitors:
        for i, m in enumerate(screen.list_monitors()):
            print(f"  {i}: {m}")
        return 0

    import cv2

    import preview
    from actions import InputController
    from camera import CameraStream
    from cursor import CursorController
    from gestures import Controls, GestureEngine
    from hand_tracker import HandTracker, ensure_model

    show_preview = config.SHOW_PREVIEW if args.preview is None else args.preview
    target = screen.target_rect(args.monitor)
    print(f"[screen] mapping to {target}  (os={screen.SYSTEM})")

    inputs = InputController(dry_run=args.dry_run)
    controls = Controls()
    cursor = CursorController(target, config)
    engine = GestureEngine(config, cursor, inputs, controls, debug=args.debug)

    model = ensure_model(config.MODEL_PATH, config.MODEL_URL)
    tracker = HandTracker(model, engine.update,
                          min_detection=config.MIN_DETECTION_CONFIDENCE,
                          min_presence=config.MIN_PRESENCE_CONFIDENCE,
                          min_tracking=config.MIN_TRACKING_CONFIDENCE,
                          use_gpu=config.USE_GPU_DELEGATE)
    camera = None
    listener = None

    # SIGTERM -> same clean shutdown path as Ctrl+C.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        camera = CameraStream(args.camera, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                              config.CAMERA_FPS, config.MIRROR, config.CAMERA_USE_MJPG).start()
        print(f"[camera] {camera.width}x{camera.height} @ index {args.camera}")
        listener = start_hotkey_listener(controls, inputs)
        print(f"[ready] {config.PAUSE_HOTKEY.upper()} = pause/resume, "
              f"{'Q/Esc in preview or ' if show_preview else ''}Ctrl+C = quit")
        if show_preview:
            cv2.namedWindow(config.PREVIEW_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        last_id = 0
        fps = latency = 0.0
        last_log = time.monotonic()
        last_results = tracker.results_received
        last_frames = camera.frames_captured
        stall_s = config.STALL_RELEASE_MS / 1000.0

        while True:
            # Event-driven: wakes as soon as the camera thread has a new frame.
            frame = camera.wait_for_frame(last_id, timeout=0.25)
            if frame is not None:
                last_id = frame.frame_id
                tracker.submit(frame.rgb, frame.timestamp)

            now = time.monotonic()
            # Watchdog: tracking stalled (camera gone?) while the button is held.
            if inputs.left_down and now - tracker.last_result_time > stall_s:
                print("[safety] tracking stalled, releasing mouse button")
                inputs.release_all()

            if now - last_log >= 1.0:
                span = now - last_log
                fps = (tracker.results_received - last_results) / span
                cam_fps = (camera.frames_captured - last_frames) / span
                if tracker.latency_samples:
                    latency = tracker.latency_ms_total / tracker.latency_samples
                tracker.latency_ms_total, tracker.latency_samples = 0.0, 0
                last_results, last_frames = tracker.results_received, camera.frames_captured
                last_log = now
                print(f"[fps] tracking {fps:5.1f} | camera {cam_fps:5.1f} | "
                      f"capture->cursor {latency:5.1f} ms | {engine.snapshot.label}")

            if show_preview:
                if frame is not None:
                    preview.draw(frame.bgr, engine.snapshot, cursor.zone, fps, latency)
                    cv2.imshow(config.PREVIEW_WINDOW_NAME, frame.bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(config.PREVIEW_WINDOW_NAME,
                                         cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:          # e.g. camera could not be opened
        print(f"[error] {exc}")
        return 1
    finally:
        inputs.release_all()
        if listener is not None:
            listener.stop()
        if camera is not None:
            camera.stop()
        tracker.close()
        inputs.release_all()  # Again, in case a late callback pressed it during shutdown.
        if show_preview:
            cv2.destroyAllWindows()
        print("[exit] stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
