"""Shared constants: which structures exist, how they are stored, and UI tuning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    name: str      # label shown in the UI
    channel: int   # channel index in the (H, W, Phase, Slice, 3) mask array
    color: str     # overlay colour (used by the GUI and the viewers)
    key: str       # keyboard shortcut that selects it


# Order == channel order in the mask files: 0 = RV, 1 = myocardium/epicardium, 2 = LV/endocardium.
STRUCTURES = (
    Structure("RV Epicardium", 0, "#ff4040", "1"),
    Structure("Epicardium", 1, "#40e040", "2"),
    Structure("Endocardium", 2, "#4090ff", "3"),
)
N_CHANNELS = len(STRUCTURES)
EPICARDIUM = 1
ENDOCARDIUM = 2

# File layout
IMAGE_SUFFIX = "_image.npy"
MASK_SUFFIX = "_mask.npy"
OUTPUT_DIRNAME = "seg_outputs"          # created next to the image files
CLEANED_DIRNAME = "seg_outputs_cleaned"  # default output of process_masks.py

# View
ZOOM_STEP = 0.85          # span multiplier per zoom-in step
MIN_VIEW_SPAN_PX = 10     # can't zoom in further than this many image pixels
SCROLL_ZOOM_INTERVAL = 0.03  # seconds; throttles very chatty trackpads

# Brush
BRUSH_DEFAULT_RADIUS = 6
BRUSH_MIN_RADIUS = 1
BRUSH_MAX_RADIUS = 30

# Editing
UNDO_LIMIT = 50           # undo steps kept per frame
VERTEX_HIT_PX = 8         # how close (screen px) a click must be to grab a vertex
EDGE_HIT_PX = 15          # how close a shift-click must be to an edge to insert a vertex
SNAP_PX = 6               # endocardium vertices snap to the epicardium edge within this distance

SHORTCUTS = (
    ("Tools", (
        ("X", "Polygon tool"),
        ("C", "Lasso tool"),
        ("Z", "Brush tool"),
    )),
    ("Structures", (
        ("1 / 2 / 3", "Select RV Epicardium / Epicardium / Endocardium"),
        ("V then 1 / 2 / 3", "Toggle visibility of that structure"),
    )),
    ("Region mode (polygon & lasso)", (
        ("N", "New: replace the structure with the region"),
        ("Q", "Add: union with the region"),
        ("R", "Erase: remove the region"),
    )),
    ("Brush", (
        ("T / F", "Paint / Erase"),
        ("- / =", "Smaller / larger brush"),
    )),
    ("Polygon", (
        ("Click", "Add a vertex (click the first vertex to close)"),
        ("Enter or right-click", "Close and apply the polygon"),
        ("Drag a vertex", "Move it"),
        ("Shift+Click", "Insert a vertex on the nearest edge"),
        ("Ctrl/Cmd+Drag", "Move the whole polygon"),
        ("Right-click a vertex", "Delete it"),
        ("Backspace", "Remove the last vertex"),
        ("Esc", "Discard the polygon"),
    )),
    ("Navigation", (
        ("W / S  or  Up / Down", "Next / previous slice"),
        ("D / A  or  Right / Left", "Next / previous frame (phase)"),
        ("Page Up / Page Down", "Previous / next slice"),
        ("Scroll wheel", "Zoom at cursor"),
        ("Middle-drag or Alt/Option+drag", "Pan"),
        ("Ctrl/Cmd + R", "Reset view"),
        ("Ctrl/Cmd + = / -", "Zoom in / out"),
    )),
    ("File & editing", (
        ("Ctrl/Cmd + O", "Open folder"),
        ("Ctrl/Cmd + S", "Save patient"),
        ("Ctrl/Cmd + Z / Y", "Undo / redo (per frame)"),
        ("F1 or ?", "Show this help"),
    )),
)
