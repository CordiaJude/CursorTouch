# CursorTouch

Control your mouse with one hand through your webcam. CursorTouch uses MediaPipe's 21-point hand tracking, a One Euro filter for smoothing, and a gesture state machine. It aims for low latency: on a laptop CPU it runs at the camera's full frame rate (30 FPS), and hand tracking takes about 10–15 ms per frame.

| Gesture | Action |
|---|---|
| Point with your **index fingertip** | Move the cursor |
| Quick **index + thumb** tap (< 250 ms) | Left click |
| **Index + thumb** pinch, hold or move | Hold the left button and drag to select text |
| **Middle + thumb** tap | Copy (Cmd+C on macOS, Ctrl+C elsewhere) |
| **Ring + thumb** tap | Paste (Cmd+V on macOS, Ctrl+V elsewhere) |
| **Index + ring + thumb** together | Grab: hold the left button to drag windows or files |
| **F8** | Pause or resume hand control |
| Open palm held for 1 s *(optional, off by default)* | Pause or resume |
| **Q** / **Esc** in the preview, or **Ctrl+C** in the terminal | Quit (always releases the mouse button) |

---

## Setup

You need Python 3.10 or newer and a webcam. If pip can't find MediaPipe for your Python version, use Python 3.11 or 3.12.

```bash
git clone <this repo> CursorTouch
cd CursorTouch
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

On the first run, CursorTouch downloads the hand model (`models/hand_landmarker.task`, about 8 MB). If you're offline, download it from this link and save it under that name:
<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>

> MediaPipe installs `opencv-contrib-python`. **Don't also install `opencv-python`**: both packages ship `cv2` and they overwrite each other. If `import cv2` breaks, run `pip uninstall -y opencv-python opencv-contrib-python` and then `pip install -r requirements.txt`.

### macOS permissions (required)

macOS grants these permissions to the app that runs Python: Terminal, iTerm, VS Code, PyCharm, and so on. Grant them to that app, then **quit and reopen it**.

1. **Camera**: go to System Settings → Privacy & Security → Camera and enable your terminal app. macOS also asks automatically the first time you run CursorTouch.
2. **Accessibility**: go to System Settings → Privacy & Security → Accessibility, click **+**, and add your terminal app. Without this permission the cursor won't move and clicks won't register, and macOS shows no error.
3. **Input Monitoring**, for the F8 pause key: go to System Settings → Privacy & Security → Input Monitoring and add your terminal app.

On a Mac keyboard, F8 may control media playback. Press **fn+F8**, or turn on "Use F1, F2, etc. keys as standard function keys". You can also change `PAUSE_HOTKEY` in `config.py`.

CursorTouch always uses the Mac's **built-in camera** and never opens an iPhone or iPad (Continuity Camera), whatever order macOS lists the cameras in. The startup log shows which cameras were found and which one is used. To use an external webcam instead, pass `--camera N` (see `python main.py --list-cameras`). Phone cameras stay blocked unless you set `ALLOW_PHONE_CAMERA = True` in `config.py`.

Retina displays work without any setup. Cursor coordinates use macOS points, so the active zone always covers the whole screen.

### Windows

No extra permissions are needed. CursorTouch makes itself *per-monitor DPI aware*, so it moves the cursor correctly at 125%, 150% and 200% display scaling. The cursor can't control windows running as Administrator unless you also run CursorTouch from an Administrator terminal.

### Linux

- CursorTouch needs an **X11 session**. On Wayland, pynput can only control XWayland windows, so log in with "Ubuntu on Xorg" or a similar session.
- Your user needs access to `/dev/video0`. Usually that means being in the `video` group: `sudo usermod -aG video $USER`, then log out and back in.
- On a headless or minimal system, MediaPipe needs `libegl1` and `libgles2`.

---

## Running

```bash
python main.py                 # with the preview window (default)
python main.py --no-preview    # no window, lowest overhead
python main.py --debug         # print live finger distances, for tuning
python main.py --dry-run       # recognize gestures without moving the mouse or typing
python main.py --camera 1      # use a different webcam
python main.py --list-cameras  # test each camera; use the "OK" one with --camera N
python main.py --list-monitors # list monitors, then choose one with --monitor 1
python main.py --monitor primary
```

The console prints a status line once per second:

```
[fps] tracking  30.0 | camera  30.0 | capture->cursor  12.4 ms | MOVE
```

`capture->cursor` is the time from grabbing a frame to moving the cursor. The camera adds its own exposure and USB delay on top, usually 15–30 ms, which software can't measure.

### Testing each gesture

Sit about 50–80 cm from the camera, raise your hand with your palm facing the camera, and keep the preview window open. The orange rectangle is the **active zone**: moving your fingertip across it moves the cursor across the whole screen.

Tip: start with `python main.py --dry-run`. The preview shows each gesture label, but nothing gets clicked or typed. Once the gestures look right, run it for real.

1. **Move**: point with your index finger and trace the active zone. The label reads `MOVE`. The cursor should stay steady while your hand is still and keep up when you move quickly.
2. **Click**: hover over a desktop icon or a link, then quickly tap your index fingertip to your thumb and let go. The thumb–index line turns green, the label flashes `CLICK`, and one click happens where you were pointing.
3. **Select text**: open a text document and point at the start of a sentence. Pinch index and thumb and keep holding: after 250 ms, or as soon as you move, the label changes to `SELECTING` and the button is held down. Move along the sentence, then open the pinch to release.
4. **Copy**: with text selected, tap your middle fingertip to your thumb. The label flashes `COPY`. Tapping again within 0.4 s does nothing, by design.
5. **Paste**: click into a text field and tap your ring fingertip to your thumb. The label flashes `PASTE`.
6. **Grab**: point at a window's title bar, then touch both your index and ring fingertips to your thumb, like a claw. The label reads `GRAB`. Move your hand to drag the window, then open your fingers to drop it.
7. **Pause**: press **F8**. The label reads `PAUSED`, the cursor stops following your hand, and any held button is released. Press F8 again to resume.
8. **Safety check**: start a selection, then pull your hand out of the frame. The button is released immediately.

---

## Tuning

Every setting is in [`config.py`](config.py), and each one has a comment explaining it. The settings you're most likely to change:

| Problem | Change |
|---|---|
| Can't reach the screen edges comfortably | Lower `ACTIVE_ZONE_WIDTH` / `ACTIVE_ZONE_HEIGHT`, or move `ACTIVE_ZONE_CENTER_*` |
| Cursor jitters while your hand is still | Lower `FILTER_MIN_CUTOFF` (e.g. 0.6) or raise `DEAD_ZONE_PX` |
| Cursor lags during fast moves | Raise `FILTER_BETA` (e.g. 0.02) |
| Pinches aren't detected | Run `--debug`, pinch, note the distance, and set `TOUCH_ON` slightly above it |
| Gestures fire on their own | Lower `TOUCH_ON`. Keep `TOUCH_OFF` about 0.1 higher |
| Clicks turn into selections | Raise `TAP_MAX_MS` or `DRAG_START_MOVE` |
| Clicks land slightly off target | Raise `PINCH_REWIND_MS` or `PINCH_FREEZE_MS` |
| Grab sometimes starts a paste | Raise `CONFIRM_WINDOW_MS` (e.g. 100) |
| Drags drop when tracking glitches | Set `HAND_LOST_RELEASE_MS` to about 100 |

`--debug` prints one line per frame:

```
[debug] index 0.153  middle 0.912  ring 1.044  pinky 1.210  touch I--  reach/ext I1.21/0.58 M1.80/0.90 R1.70/0.84  state PINCH
```

The first four numbers are thumb-to-fingertip distances divided by hand size. `touch` shows which fingers count as touching: `I`, `M`, `R`. `reach/ext` shows how far each fingertip is from the wrist and from its own knuckle. A touch can only start when both values are above `MIN_TOUCH_REACH` and `MIN_TOUCH_EXTENSION`. This stops your thumb resting on curled fingers, as in a normal pointing pose or a fist, from counting as a pinch.

---

## How it works

```
camera thread ──latest frame only──▶ main thread ──detect_async──▶ MediaPipe thread
 (640x480, buffer 1,                 (submit frame, preview,        (hand landmarks
  mirror + RGB convert)               FPS log, watchdog)             ─▶ GestureEngine.update
                                                                      ─▶ cursor move / click)
```

| File | Purpose |
|---|---|
| `main.py` | Entry point, CLI flags, main loop, F8 listener, clean shutdown |
| `config.py` | Every setting |
| `camera.py` | Threaded capture that keeps only the newest frame |
| `hand_tracker.py` | MediaPipe Hand Landmarker in LIVE_STREAM mode, plus model download |
| `gestures.py` | Touch detection with hysteresis, and the gesture state machine |
| `cursor.py` | Active-zone mapping, pinch stabilization, dead zone |
| `filters.py` | One Euro filter (own implementation) |
| `actions.py` | Thread-safe pynput mouse and keyboard output |
| `screen.py` | Monitor geometry and DPI handling per OS |
| `preview.py` | Preview drawing |

These choices keep latency low:

- The cursor moves **inside MediaPipe's result callback**, so the result never has to be handed back to the main loop first.
- The camera thread drains the driver buffer and keeps only the newest frame. MediaPipe's LIVE_STREAM mode skips frames while it's busy, so a backlog never builds up.
- The main loop waits on a condition variable for each new frame. It never sleeps, and drawing is limited to about 60 OpenCV primitives per frame.

The **state machine** in `gestures.py` works like this:

- `IDLE` becomes `PINCH_PENDING` when any touch starts. It waits 70 ms so a grab is never mistaken for a click or paste. Then it moves to one of:
  - `CLICK_PENDING`, which becomes a click on a quick release, or `HIGHLIGHT_DRAG` if held or moved.
  - `GRAB_DRAG`, if the index and ring fingers both touch. A grab can also start from `CLICK_PENDING`.
  - Copy or paste.
- Every gesture ends in `WAIT_RELEASE`. Nothing new can start until all fingers are open, so leftover contact never fires a second action.

**Clicks land where you pointed**, because three things happen when a pinch starts:

1. The cursor snaps back to where it was 100 ms earlier, before your fingertip started to curl.
2. It freezes there for 130 ms.
3. While you pinch, it follows a blend of the fingertip and the index knuckle (landmark 5), which barely moves when you pinch.

When the cursor switches between the fingertip and this blended point, it's offset to stay continuous, so it never jumps.

**Safety**: CursorTouch releases the mouse button when the hand leaves the frame, when tracking stops for 500 ms, when you pause, when you quit, on Ctrl+C or SIGTERM, and on errors.

## Tests

The gesture state machine and the filter have unit tests that use synthetic hands, so they don't need a camera:

```bash
python -m unittest discover -s tests -v
```
