"""
CursorTouch configuration.

Every tunable lives here so you never have to touch the main code.
Run `python main.py --debug` to print live normalized finger distances while
you tune the touch thresholds below.

Units used throughout:
  * "hand units"  -> a distance divided by the hand size (wrist -> middle-finger
                     knuckle, landmarks 0 -> 9). 1.0 == one hand size. This makes
                     gestures work the same near or far from the camera.
  * "normalized"  -> fraction of the camera frame (0.0 .. 1.0).
  * "px"          -> screen pixels (logical points on macOS).
  * "ms"          -> milliseconds.
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0            # Which webcam to open (0 = default). Override with --camera.
PREFER_BUILTIN_CAMERA = True  # macOS: always use the computer's own camera, whatever its index.
                              # Set False to use CAMERA_INDEX / --camera instead.
ALLOW_PHONE_CAMERA = False    # macOS: never open an iPhone/iPad (Continuity Camera), even if
                              # asked for by index.
CAMERA_WIDTH = 640          # Capture resolution. 640x480 keeps MediaPipe fast.
CAMERA_HEIGHT = 480
CAMERA_FPS = 30             # Requested FPS. Some webcams can do 60 at 640x480; try it.
CAMERA_USE_MJPG = True      # Ask for MJPG; many USB webcams only reach 30+ FPS in MJPG.
MIRROR = True               # Flip the image horizontally so moving right moves the cursor right.

# ---------------------------------------------------------------------------
# Hand tracking (MediaPipe Hand Landmarker)
# ---------------------------------------------------------------------------
MODEL_PATH = "models/hand_landmarker.task"   # Downloaded automatically on first run.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MIN_DETECTION_CONFIDENCE = 0.6   # Palm detector confidence needed to start tracking a hand.
MIN_PRESENCE_CONFIDENCE = 0.5    # Below this the hand is considered lost and re-detected.
MIN_TRACKING_CONFIDENCE = 0.5    # Landmark tracking confidence; lower = stickier, may drift.
USE_GPU_DELEGATE = False         # Try MediaPipe's GPU delegate (Linux/macOS only). CPU is
                                 # usually fast enough and more reliable.

# ---------------------------------------------------------------------------
# Screen / monitors
# ---------------------------------------------------------------------------
# "all"     -> map the active zone onto the bounding box of every monitor.
# "primary" -> only the primary monitor.
# 0, 1, ... -> a specific monitor by index (run with --list-monitors to see them).
MONITOR = "all"

# ---------------------------------------------------------------------------
# Cursor mapping
# ---------------------------------------------------------------------------
# A smaller "active zone" in the middle of the camera frame is stretched over
# the whole screen, so you can reach the screen edges without your hand
# leaving the camera view.
ACTIVE_ZONE_WIDTH = 0.65    # Fraction of the frame width used (0.3 .. 1.0).
ACTIVE_ZONE_HEIGHT = 0.65   # Fraction of the frame height used.
ACTIVE_ZONE_CENTER_X = 0.5  # Center of the zone (normalized). Shift if you sit off-center.
ACTIVE_ZONE_CENTER_Y = 0.45 # Slightly above center: a raised hand is less tiring.

# ---------------------------------------------------------------------------
# Smoothing (One Euro Filter) - applied in screen pixels
# ---------------------------------------------------------------------------
# min_cutoff: lower = steadier when still, but more lag on slow moves.
# beta:       higher = less lag on fast moves (but more jitter during them).
# Tuning recipe: set BETA = 0, lower MIN_CUTOFF until a still hand is steady,
# then raise BETA until fast flicks no longer lag.
FILTER_MIN_CUTOFF = 1.0     # Hz
FILTER_BETA = 0.01          # 1/px
FILTER_D_CUTOFF = 1.0       # Hz, cutoff used for the speed estimate. Rarely needs changing.

DEAD_ZONE_PX = 1.5          # Ignore filtered moves smaller than this (kills micro-jitter).

# ---------------------------------------------------------------------------
# Touch detection (thumb tip vs each fingertip), in hand units
# ---------------------------------------------------------------------------
# Hysteresis: a touch STARTS when the distance drops below TOUCH_ON and only
# ENDS when it rises above TOUCH_OFF. The gap stops gestures flickering.
TOUCH_ON = {
    "index": 0.22,
    "middle": 0.22,
    "ring": 0.25,
}
TOUCH_OFF = {
    "index": 0.32,
    "middle": 0.32,
    "ring": 0.36,
}
MIN_TOUCH_REACH = 0.9       # A touch can only START if that fingertip is at least this far from
                            # the wrist (hand units). Fingers curled into the palm reach ~0.7;
                            # a deliberate pinch reaches ~1.0+. This stops the thumb resting on
                            # curled fingers (the normal pointing pose) from triggering copy/paste.
MIN_TOUCH_EXTENSION = 0.45  # ...and at least this far from its own knuckle (hand units). A fist
                            # curls the fingertips to ~0.3 from their knuckles; an index-thumb
                            # pinch keeps the index tip ~0.5-0.7 away.
HAND_SETTLE_FRAMES = 3      # Ignore new touches for this many frames after a hand appears; the
                            # first detection frame is noisier than tracked frames.
USE_3D_DISTANCE = True      # Include MediaPipe's depth (z) in distances. Helps when the thumb
                            # is in front of/behind a finger from the camera's point of view.

# ---------------------------------------------------------------------------
# Gesture timing
# ---------------------------------------------------------------------------
CONFIRM_WINDOW_MS = 70      # After a pinch starts, wait this long to see if it becomes a GRAB
                            # (index + ring) before committing to click/select/copy/paste.
TAP_MAX_MS = 250            # Index pinch released faster than this = click. Held longer = select.
DRAG_START_MOVE = 0.12      # Hand movement (hand units, measured at the index knuckle) while
                            # pinched that turns a tap into a select-drag even before TAP_MAX_MS.
KEY_COOLDOWN_MS = 400       # Minimum time between two copies (or two pastes).
GRAB_RELEASE_GRACE_MS = 120 # During a GRAB, if only one of index/ring lets go, wait this long
                            # before releasing (the ring finger tends to flicker). Both open
                            # releases immediately.

# ---------------------------------------------------------------------------
# Click accuracy (stop the cursor jumping as the index finger curls in)
# ---------------------------------------------------------------------------
PINCH_REWIND_MS = 100       # On pinch start, snap the cursor back to where it was this long
                            # ago (before the fingertip started curling). 0 disables.
PINCH_FREEZE_MS = 130       # Then hold the cursor completely still for this long.
PINCH_ANCHOR_BLEND = 0.7    # While pinched, drive the cursor from a blend of the index tip (0.0)
                            # and the index knuckle, landmark 5 (1.0). The knuckle barely moves
                            # when you pinch, so drags stay accurate.
ANCHOR_RELEASE_DECAY_MS = 150  # After a pinch ends, blend smoothly back to the fingertip over
                               # roughly this time constant instead of jumping.

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
PAUSE_HOTKEY = "f8"         # Global key that pauses/resumes hand control (any pynput Key name).
HAND_LOST_RELEASE_MS = 0    # Release a held mouse button this long after the hand disappears.
                            # 0 = immediately (safest). Raise to ~100 if drags drop on brief
                            # tracking glitches.
STALL_RELEASE_MS = 500      # Release held buttons if tracking produces no results this long
                            # (e.g. camera unplugged).

# Optional: hold an open palm (all five fingers spread) still for a moment to
# toggle pause. Off by default because an open hand is also a natural pointing pose.
PALM_PAUSE_ENABLED = False
PALM_PAUSE_HOLD_MS = 1000

# ---------------------------------------------------------------------------
# Preview window
# ---------------------------------------------------------------------------
SHOW_PREVIEW = True         # Show the camera preview. Override with --no-preview / --preview.
PREVIEW_WINDOW_NAME = "CursorTouch"
FLASH_LABEL_MS = 350        # How long one-shot labels (CLICK / COPY / PASTE) stay on screen.
