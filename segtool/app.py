"""Interactive cardiac MRI segmentation GUI (Tkinter + matplotlib).

Run with ``python seg.py [data_folder]``. Press F1 inside the app for all shortcuts.

Design notes
------------
* The whole patient is held as two arrays: ``image`` (H, W, Phase, Slice) and
  ``mask`` (H, W, Phase, Slice, 3). Edits write straight into ``mask``; saving is a
  single ``np.save``. No resizing happens, so saved masks always match the image.
* All matplotlib artists are created once and updated with ``set_data`` - nothing
  is cleared and redrawn on mouse moves.
* Rotation only affects the view: artists are drawn through a rotation transform and
  mouse positions are mapped back to image pixels, so masks are never resampled.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D

from . import geometry
from .config import (
    BRUSH_DEFAULT_RADIUS, BRUSH_MAX_RADIUS, BRUSH_MIN_RADIUS, EDGE_HIT_PX, ENDOCARDIUM, EPICARDIUM,
    MIN_VIEW_SPAN_PX, OUTPUT_DIRNAME, SCROLL_ZOOM_INTERVAL, SHORTCUTS, SNAP_PX, STRUCTURES,
    UNDO_LIMIT, VERTEX_HIT_PX, ZOOM_STEP,
)
from .data import PatientFiles, apply_window, discover_patients, load_image, load_mask_for, save_mask, window_params

SETTINGS_FILE = Path.home() / ".segtool.json"
IS_MAC = sys.platform == "darwin"
MOD = "Command" if IS_MAC else "Control"
# Tk event.state bits for Ctrl, Cmd/Alt (Mod1) and, on macOS, Option (Mod2).
MODIFIER_MASK = 0x4 | 0x8 | (0x10 if IS_MAC else 0)
STRUCTURE_BY_KEY = {s.key: s for s in STRUCTURES}
BG = "#1e1e1e"


@dataclass
class Snapshot:
    """One undo step for one frame."""
    mask: np.ndarray
    polygons: dict


@dataclass
class FrameHistory:
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)


@dataclass
class Drag:
    """What the current mouse drag is doing (only one at a time)."""
    kind: str                      # pan | brush | lasso | vertex | polygon
    start_px: tuple = (0, 0)
    start_xlim: tuple = (0, 1)
    start_ylim: tuple = (0, 1)
    last: tuple | None = None      # last image-space point
    vertex: int | None = None
    points: list = field(default_factory=list)


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def store_settings(**values) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps({**load_settings(), **values}, indent=2))
    except OSError:
        pass


class SegmentationApp:
    def __init__(self, root: tk.Tk):
        self.root = root

        # --- patient state ---
        self.patients: dict[str, PatientFiles] = {}
        self.patient_key: str | None = None
        self.image: np.ndarray | None = None   # (H, W, P, S)
        self.mask: np.ndarray | None = None    # (H, W, P, S, 3) uint8
        self.lo, self.hi = 0.0, 1.0
        self.slice = self.phase = 0
        self.dirty = False
        self.history: dict[tuple[int, int], FrameHistory] = {}
        # in-progress polygon per structure channel, as [(x, y), ...] in image pixels
        self.polygons: dict[int, list] = {s.channel: [] for s in STRUCTURES}

        # --- interaction state ---
        self.drag: Drag | None = None
        self.cursor: tuple | None = None
        self.pending_visibility = False
        self._last_scroll = 0.0

        # --- UI variables ---
        self.tool = tk.StringVar(value="polygon")
        self.active = tk.IntVar(value=EPICARDIUM)
        self.region_mode = tk.StringVar(value="New")
        self.brush_mode = tk.StringVar(value="Paint")
        self.brush_radius = tk.IntVar(value=BRUSH_DEFAULT_RADIUS)
        self.visible = {s.channel: tk.BooleanVar(value=True) for s in STRUCTURES}
        self.show_mask = tk.BooleanVar(value=True)
        self.show_polygons = tk.BooleanVar(value=True)
        self.show_neighbors = tk.BooleanVar(value=False)
        self.view_mode = tk.StringVar(value="Normal")
        self.opacity = tk.DoubleVar(value=0.35)
        self.brightness = tk.DoubleVar(value=0.0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.rotation = tk.IntVar(value=0)
        self.slice_var = tk.IntVar(value=1)
        self.phase_var = tk.IntVar(value=1)
        self.status = tk.StringVar(value="Open a data folder to begin.")
        self.cursor_info = tk.StringVar()
        self.position_info = tk.StringVar()

        self._view_tf = Affine2D()  # image pixels -> rotated view coordinates

        self._build_ui()
        self._build_artists()
        self._bind_keys()
        for var in (self.tool, self.active, self.region_mode, self.brush_mode, self.brush_radius):
            var.trace_add("write", lambda *_: self._on_tool_state_changed())
        self._update_title()
        self._update_tool_status()

    # ================================================================ UI layout

    def _build_ui(self):
        self.root.title("Segmentation Tool")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1400, sw - 60)}x{min(900, sh - 100)}")
        self.root.minsize(1000, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # Toolbar
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        mod = "Cmd" if IS_MAC else "Ctrl"
        for text, cmd in ((f"Open Folder… ({mod}+O)", self.open_folder_dialog),
                          (f"Save ({mod}+S)", self.save),
                          ("Undo", self.undo), ("Redo", self.redo),
                          ("Reset View", self.reset_view),
                          ("Shortcuts (F1)", self.show_help)):
            ttk.Button(bar, text=text, command=cmd, takefocus=False).pack(side=tk.LEFT, padx=2)
        self.patient_label = ttk.Label(bar, text="", font=("TkDefaultFont", 12, "bold"))
        self.patient_label.pack(side=tk.RIGHT, padx=8)

        # Status bar
        status = ttk.Frame(self.root, padding=(6, 2), relief=tk.SUNKEN)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.cursor_info, width=34, anchor="e").pack(side=tk.RIGHT)
        self.tool_status = ttk.Label(status, anchor="e")
        self.tool_status.pack(side=tk.RIGHT, padx=12)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: patient list
        left = ttk.Frame(body, padding=4, width=170)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        ttk.Label(left, text="Patients  (✓ = saved edits)").pack(anchor="w")
        tree_box = ttk.Frame(left)
        tree_box.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_box, selectmode="browse", show="tree")
        sb = ttk.Scrollbar(tree_box, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_patient_selected)

        # Right: tool panel (scrollable)
        self._build_tool_panel(body)

        # Neighbour previews (packed on demand) + main canvas
        center = ttk.Frame(body)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.neighbor_frame = tk.Frame(center, width=210, bg=BG)
        self.neighbor_frame.pack_propagate(False)
        self.previews = {}
        for key in ("above", "below"):
            label = tk.Label(self.neighbor_frame, bg=BG, fg="#aaaaaa")
            label.pack(fill=tk.X)
            fig = Figure(figsize=(2, 2), facecolor=BG)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=self.neighbor_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            img = ax.imshow(np.zeros((2, 2)), cmap="gray", vmin=0, vmax=1)
            ov = ax.imshow(np.zeros((2, 2, 4)))
            self.previews[key] = dict(label=label, ax=ax, canvas=canvas, img=img, ov=ov)

        main = ttk.Frame(center)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_frame = main
        self.fig = Figure(facecolor="black")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        widget.bind("<Enter>", lambda e: widget.focus_set())

        nav = ttk.Frame(main, padding=(4, 2))
        nav.pack(side=tk.BOTTOM, fill=tk.X)
        self.slice_scale = self._nav_slider(nav, "Slice", self.slice_var, 0)
        self.phase_scale = self._nav_slider(nav, "Frame", self.phase_var, 1)
        ttk.Label(nav, textvariable=self.position_info, width=26).grid(row=0, column=3, rowspan=2, padx=6)
        nav.columnconfigure(1, weight=1)

        for name, handler in (("button_press_event", self._on_press),
                              ("button_release_event", self._on_release),
                              ("motion_notify_event", self._on_motion),
                              ("scroll_event", self._on_scroll),
                              ("figure_leave_event", self._on_leave)):
            self.canvas.mpl_connect(name, handler)

    def _nav_slider(self, parent, text, var, row):
        ttk.Label(parent, text=text, width=6).grid(row=row, column=0, sticky="w")
        scale = tk.Scale(parent, from_=1, to=1, orient=tk.HORIZONTAL, variable=var, showvalue=True,
                         takefocus=0, command=lambda _v: self._on_nav_slider())
        scale.grid(row=row, column=1, sticky="ew")
        return scale

    def _build_tool_panel(self, parent):
        outer = ttk.Frame(parent, width=290)
        outer.pack(side=tk.RIGHT, fill=tk.Y)
        outer.pack_propagate(False)
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        panel = ttk.Frame(canvas, padding=6)
        window = canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))

        def on_wheel(event):
            w = event.widget.winfo_containing(event.x_root, event.y_root)
            while w is not None and w is not outer:
                w = w.master
            if w is None:
                return
            up = event.delta > 0 if event.num not in (4, 5) else event.num == 4
            canvas.yview_scroll(-1 if up else 1, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, on_wheel, add="+")

        def section(title):
            f = ttk.LabelFrame(panel, text=title, padding=6)
            f.pack(fill=tk.X, pady=4)
            return f

        def radio_row(parent_, var, options):
            row = ttk.Frame(parent_)
            row.pack(fill=tk.X, pady=1)
            for label, value in options:
                ttk.Radiobutton(row, text=label, variable=var, value=value, takefocus=False).pack(side=tk.LEFT)
            return row

        def slider(parent_, label, var, lo, hi, res, command=None):
            tk.Scale(parent_, label=label, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, takefocus=0, command=command).pack(fill=tk.X)

        # Structures
        f = section("Structure")
        for s in STRUCTURES:
            row = ttk.Frame(f)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, bg=s.color, width=2).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Radiobutton(row, text=f"{s.key}  {s.name}", variable=self.active, value=s.channel,
                            takefocus=False).pack(side=tk.LEFT)
            ttk.Checkbutton(row, text="visible", variable=self.visible[s.channel], takefocus=False,
                            command=self._refresh_all).pack(side=tk.RIGHT)
        btns = ttk.Frame(f)
        btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btns, text="Clear structure", command=self.clear_active, takefocus=False).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btns, text="Clear frame", command=self.clear_frame, takefocus=False).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        # Tools
        f = section("Tool")
        radio_row(f, self.tool, (("Polygon (X)", "polygon"), ("Lasso (C)", "lasso"), ("Brush (Z)", "brush")))
        ttk.Label(f, text="Polygon / lasso mode:").pack(anchor="w", pady=(6, 0))
        radio_row(f, self.region_mode, (("New (N)", "New"), ("Add (Q)", "Add"), ("Erase (R)", "Erase")))
        ttk.Label(f, text="Brush mode:").pack(anchor="w", pady=(6, 0))
        radio_row(f, self.brush_mode, (("Paint (T)", "Paint"), ("Erase (F)", "Erase")))
        slider(f, "Brush radius (- / =)", self.brush_radius, BRUSH_MIN_RADIUS, BRUSH_MAX_RADIUS, 1)

        # Display
        f = section("Display")
        for text, var, cmd in (("Show mask", self.show_mask, self._refresh_masks),
                               ("Show polygon outlines", self.show_polygons, self._refresh_polygons),
                               ("Show neighbour slices", self.show_neighbors, self._toggle_neighbors)):
            ttk.Checkbutton(f, text=text, variable=var, command=cmd, takefocus=False).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="View:").pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=self.view_mode, state="readonly", width=16, takefocus=False,
                             values=("Normal", "Isolate active", "Mask only"))
        combo.pack(side=tk.LEFT, padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: (self._refresh_all(), self.canvas.get_tk_widget().focus_set()))
        slider(f, "Mask opacity", self.opacity, 0, 1, 0.05, lambda _v: self._refresh_masks())
        slider(f, "Brightness", self.brightness, -1, 1, 0.05, lambda _v: self._refresh_image())
        slider(f, "Contrast", self.contrast, 0.1, 3, 0.1, lambda _v: self._refresh_image())
        slider(f, "Rotation (view only, °)", self.rotation, -180, 180, 1, lambda _v: self._apply_rotation())
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Zoom in", command=lambda: self.zoom(ZOOM_STEP), takefocus=False).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Zoom out", command=lambda: self.zoom(1 / ZOOM_STEP), takefocus=False).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Reset display", command=self.reset_display, takefocus=False).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

    def _build_artists(self):
        tf = self._view_tf + self.ax.transData
        self.img_artist = self.ax.imshow(np.zeros((2, 2)), cmap="gray", vmin=0, vmax=1,
                                         interpolation="bicubic", transform=tf, visible=False)
        self.mask_artist = self.ax.imshow(np.zeros((2, 2, 4)), interpolation="nearest",
                                          transform=tf, visible=False)
        self.poly_lines, self.poly_points = {}, {}
        for s in STRUCTURES:
            self.poly_lines[s.channel] = self.ax.add_line(Line2D([], [], color=s.color, lw=1.2, transform=tf))
            self.poly_points[s.channel] = self.ax.add_line(Line2D(
                [], [], color=s.color, ls="none", marker="o", ms=3.5, transform=tf))
        self.highlight = self.ax.add_line(Line2D([], [], ls="none", marker="o", ms=8, mfc="none",
                                                 mec="white", transform=tf))
        self.lasso_line = self.ax.add_line(Line2D([], [], color="yellow", lw=1.2, ls="--", transform=tf))
        self.brush_circle = Circle((0, 0), 1, fill=False, color="cyan", ls="--", lw=1, transform=tf,
                                   visible=False)
        self.ax.add_patch(self.brush_circle)
        self.ax.set_autoscale_on(False)
        self.placeholder = self.ax.text(0.5, 0.5, "Open a data folder to begin", color="#888888",
                                        ha="center", va="center", fontsize=14, transform=self.ax.transAxes)

    # ================================================================ keyboard

    def _bind_keys(self):
        r = self.root
        for mod in {"Control", MOD}:
            r.bind(f"<{mod}-o>", lambda e: self.open_folder_dialog())
            r.bind(f"<{mod}-s>", lambda e: self.save())
            r.bind(f"<{mod}-z>", lambda e: self.undo())
            r.bind(f"<{mod}-y>", lambda e: self.redo())
            r.bind(f"<{mod}-Shift-Z>", lambda e: self.redo())
            r.bind(f"<{mod}-r>", lambda e: self.reset_view())
            for k in ("plus", "equal"):
                r.bind(f"<{mod}-{k}>", lambda e: self.zoom(ZOOM_STEP))
            r.bind(f"<{mod}-minus>", lambda e: self.zoom(1 / ZOOM_STEP))
        r.bind("<F1>", lambda e: self.show_help())
        r.bind("<KeyPress>", self._on_key)

    def _on_key(self, event):
        # Ctrl/Cmd/Alt combos are handled by explicit bindings above.
        if event.state & MODIFIER_MASK:
            return
        # Leave arrow keys to list / slider widgets that have keyboard focus.
        if isinstance(event.widget, (ttk.Treeview, tk.Scale, ttk.Combobox)) and event.keysym in (
                "Up", "Down", "Left", "Right"):
            return
        key = event.keysym.lower() if len(event.keysym) == 1 else event.keysym

        if self.pending_visibility:
            self.pending_visibility = False
            if key in STRUCTURE_BY_KEY:
                var = self.visible[STRUCTURE_BY_KEY[key].channel]
                var.set(not var.get())
                self._refresh_masks()
                self._update_tool_status()
                return
            self._update_tool_status()

        actions = {
            "x": lambda: self.tool.set("polygon"),
            "c": lambda: self.tool.set("lasso"),
            "z": lambda: self.tool.set("brush"),
            "n": lambda: self.region_mode.set("New"),
            "q": lambda: self.region_mode.set("Add"),
            "r": lambda: self.region_mode.set("Erase"),
            "t": lambda: self.brush_mode.set("Paint"),
            "f": lambda: self.brush_mode.set("Erase"),
            "minus": lambda: self._step_brush(-1),
            "equal": lambda: self._step_brush(+1),
            "plus": lambda: self._step_brush(+1),
            "v": self._start_visibility_toggle,
            "Return": self.apply_polygon,
            "KP_Enter": self.apply_polygon,
            "BackSpace": self._remove_last_vertex,
            "Escape": self._discard_polygon,
            "Left": lambda: self.goto(phase=self.phase - 1), "a": lambda: self.goto(phase=self.phase - 1),
            "Right": lambda: self.goto(phase=self.phase + 1), "d": lambda: self.goto(phase=self.phase + 1),
            "Up": lambda: self.goto(slice_=self.slice + 1), "w": lambda: self.goto(slice_=self.slice + 1),
            "Down": lambda: self.goto(slice_=self.slice - 1), "s": lambda: self.goto(slice_=self.slice - 1),
            "Prior": lambda: self.goto(slice_=self.slice - 1),
            "Next": lambda: self.goto(slice_=self.slice + 1),
            "question": self.show_help,
        }
        for s in STRUCTURES:
            actions[s.key] = lambda s=s: self.active.set(s.channel)
        action = actions.get(key)
        if action:
            action()
            return "break"

    def _start_visibility_toggle(self):
        self.pending_visibility = True
        self.tool_status.configure(text="Toggle visibility: press 1 / 2 / 3")

    def _step_brush(self, delta):
        self.brush_radius.set(int(np.clip(self.brush_radius.get() + delta, BRUSH_MIN_RADIUS, BRUSH_MAX_RADIUS)))

    def _on_tool_state_changed(self):
        if self.tool.get() != "brush":
            self.brush_circle.set_visible(False)
        else:
            self.brush_circle.set_radius(self.brush_radius.get())
        self._refresh_masks(draw=False)
        self._refresh_polygons()

    def _update_tool_status(self):
        s = STRUCTURES[self.active.get()]
        tool = self.tool.get()
        detail = (f"{self.brush_mode.get()}, r={self.brush_radius.get()}" if tool == "brush"
                  else self.region_mode.get())
        hidden = [x.key for x in STRUCTURES if not self.visible[x.channel].get()]
        extra = f"   hidden: {', '.join(hidden)}" if hidden else ""
        self.tool_status.configure(text=f"{tool.capitalize()} ({detail})  ·  {s.name}{extra}", foreground=s.color)

    # ================================================================ loading / saving

    def open_folder_dialog(self):
        folder = filedialog.askdirectory(title="Select the folder with *_image.npy files",
                                         initialdir=load_settings().get("last_folder"))
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder):
        if not self._confirm_discard():
            return
        folder = Path(folder)
        try:
            patients = discover_patients(folder)
        except OSError as exc:
            messagebox.showerror("Could not read folder", str(exc))
            return
        if not patients:
            messagebox.showerror("No data", f"No '*_image.npy' files found in\n{folder}")
            return
        store_settings(last_folder=str(folder))
        self._unload()
        self.patients = patients
        self.tree.delete(*self.tree.get_children())
        for key, pf in patients.items():
            self.tree.insert("", tk.END, iid=key, text=self._tree_text(key, pf))
        self.status.set(f"Found {len(patients)} patients in {folder}")
        first = next(iter(patients))
        self.tree.selection_set(first)  # triggers loading

    @staticmethod
    def _tree_text(key, pf: PatientFiles):
        return f"{'✓' if pf.has_saved_mask else '   '} {key}"

    def _unload(self):
        self.patient_key = self.image = self.mask = None
        self.dirty = False
        self.history.clear()
        for pts in self.polygons.values():
            pts.clear()
        for art in (self.img_artist, self.mask_artist):
            art.set_visible(False)
        self.placeholder.set_visible(True)
        self.position_info.set("")
        self._update_title()
        self.canvas.draw_idle()

    def _on_patient_selected(self, _event=None):
        # Re-selecting the current patient (e.g. after a cancelled switch) is a no-op.
        sel = self.tree.selection()
        if not sel or sel[0] == self.patient_key:
            return
        if not self._confirm_discard():
            self._select_tree(self.patient_key)
            return
        self.load_patient(sel[0])

    def _select_tree(self, key):
        if key and self.tree.exists(key):
            self.tree.selection_set(key)
            self.tree.see(key)
        else:
            self.tree.selection_remove(*self.tree.selection())

    def load_patient(self, key):
        pf = self.patients[key]
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            image = load_image(pf.image_path)
            mask, source = load_mask_for(pf, image.shape)
            lo, hi = window_params(image)
        except Exception as exc:
            messagebox.showerror("Could not load patient", f"{key}:\n{exc}")
            self._select_tree(self.patient_key)
            return
        finally:
            self.root.config(cursor="")

        self.patient_key, self.image, self.mask = key, image, mask
        self.lo, self.hi = lo, hi
        self.dirty = False
        self.history.clear()
        for pts in self.polygons.values():
            pts.clear()
        _, _, n_phases, n_slices = image.shape
        self.slice_scale.configure(to=max(n_slices, 1))
        self.phase_scale.configure(to=max(n_phases, 1))
        self.slice = self.phase = 0
        self.placeholder.set_visible(False)
        self.img_artist.set_visible(True)
        self.mask_artist.set_visible(True)
        self._set_rotation_center()
        source_text = {"saved": f"your saved edits ({OUTPUT_DIRNAME}/)",
                       "original": "the original mask",
                       "empty": "no mask found - starting empty"}[source]
        self.status.set(f"Loaded {key}: {n_slices} slices × {n_phases} frames, mask from {source_text}")
        self._update_title()
        self._show_frame()
        self.reset_view()
        self.canvas.get_tk_widget().focus_set()

    def save(self):
        if self.mask is None:
            return False
        self._commit_polygons()
        pf = self.patients[self.patient_key]
        try:
            path = save_mask(pf, self.mask)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save {self.patient_key}:\n{exc}")
            return False
        self.dirty = False
        self.tree.item(self.patient_key, text=self._tree_text(self.patient_key, pf))
        self.status.set(f"Saved {self.patient_key} → {path}")
        self._update_title()
        self._refresh_all()
        return True

    def _confirm_discard(self):
        """Ask to save unsaved edits. Returns False if the user cancelled."""
        if self.mask is None or not (self.dirty or self._has_open_polygon()):
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes", f"Save your changes to {self.patient_key} first?")
        if answer is None:
            return False
        return self.save() if answer else True

    def quit(self):
        if self._confirm_discard():
            self.root.destroy()

    def _update_title(self):
        name = f" — {self.patient_key}{' *' if self.dirty else ''}" if self.patient_key else ""
        self.root.title(f"Segmentation Tool{name}")
        self.patient_label.configure(text=f"{self.patient_key}{'  (unsaved)' if self.dirty else ''}"
                                     if self.patient_key else "")

    def _mark_dirty(self):
        if not self.dirty:
            self.dirty = True
            self._update_title()

    # ================================================================ navigation

    @property
    def frame_mask(self) -> np.ndarray:
        """(H, W, 3) view into the mask of the current slice/frame (edits write through)."""
        return self.mask[:, :, self.phase, self.slice, :]

    def goto(self, slice_=None, phase=None):
        if self.image is None:
            return
        _, _, n_phases, n_slices = self.image.shape
        s = self.slice if slice_ is None else int(np.clip(slice_, 0, n_slices - 1))
        p = self.phase if phase is None else int(np.clip(phase, 0, n_phases - 1))
        if (s, p) == (self.slice, self.phase):
            return
        self._commit_polygons()  # don't lose a half-finished polygon (undoable when you come back)
        self.slice, self.phase = s, p
        self._show_frame()

    def _on_nav_slider(self):
        self.goto(slice_=self.slice_var.get() - 1, phase=self.phase_var.get() - 1)

    def _show_frame(self):
        _, _, n_phases, n_slices = self.image.shape
        self.slice_var.set(self.slice + 1)
        self.phase_var.set(self.phase + 1)
        self.position_info.set(f"Slice {self.slice + 1}/{n_slices}   Frame {self.phase + 1}/{n_phases}")
        self._refresh_all()

    # ================================================================ rendering

    def _refresh_all(self):
        self._refresh_image(draw=False)
        self._refresh_masks(draw=False)
        self._refresh_polygons(draw=False)
        self._refresh_neighbors()
        self.canvas.draw_idle()

    def _refresh_image(self, draw=True):
        if self.image is None:
            return
        if self.view_mode.get() == "Mask only":
            img = np.zeros(self.image.shape[:2], dtype=np.float32)
        else:
            img = apply_window(self.image[:, :, self.phase, self.slice], self.lo, self.hi,
                               self.brightness.get(), self.contrast.get())
        self.img_artist.set_data(img)
        h, w = img.shape
        self.img_artist.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        if draw:
            self._refresh_neighbors()
            self.canvas.draw_idle()

    def _visible_channels(self):
        isolate = self.view_mode.get() == "Isolate active"
        return [s.channel for s in STRUCTURES
                if self.visible[s.channel].get() and (not isolate or s.channel == self.active.get())]

    def _overlay(self, mask3):
        rgba = np.zeros((*mask3.shape[:2], 4), dtype=np.float32)
        if self.show_mask.get():
            alpha = self.opacity.get()
            for ch in self._visible_channels():
                r, g, b, _ = to_rgba(STRUCTURES[ch].color)
                rgba[mask3[..., ch] > 0] = (r, g, b, alpha)
        return rgba

    def _refresh_masks(self, draw=True):
        if self.mask is None:
            return
        rgba = self._overlay(self.frame_mask)
        self.mask_artist.set_data(rgba)
        h, w = rgba.shape[:2]
        self.mask_artist.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        self._update_tool_status()
        if draw:
            self._refresh_neighbors()
            self.canvas.draw_idle()

    def _refresh_polygons(self, draw=True):
        show = self.show_polygons.get()
        visible = set(self._visible_channels())
        active = self.active.get()
        for ch, pts in self.polygons.items():
            on = show and ch in visible and bool(pts)
            xy = np.asarray(pts, dtype=float).reshape(-1, 2)
            self.poly_lines[ch].set_data(xy[:, 0], xy[:, 1])
            self.poly_lines[ch].set_visible(on)
            self.poly_points[ch].set_data(xy[:, 0], xy[:, 1])
            self.poly_points[ch].set_visible(on and ch == active and self.tool.get() == "polygon")
        if draw:
            self.canvas.draw_idle()

    def _toggle_neighbors(self):
        if self.show_neighbors.get():
            self.neighbor_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.main_frame)
            self._refresh_neighbors()
        else:
            self.neighbor_frame.pack_forget()

    def _refresh_neighbors(self):
        if not self.show_neighbors.get() or self.image is None:
            return
        n_slices = self.image.shape[3]
        for key, s in (("above", self.slice - 1), ("below", self.slice + 1)):
            pv = self.previews[key]
            arrow = "▲" if key == "above" else "▼"
            if 0 <= s < n_slices:
                pv["label"].configure(text=f"{arrow} Slice {s + 1}")
                pv["img"].set_data(apply_window(self.image[:, :, self.phase, s], self.lo, self.hi,
                                                self.brightness.get(), self.contrast.get()))
                pv["ov"].set_data(self._overlay(self.mask[:, :, self.phase, s, :]))
                h, w = self.image.shape[:2]
                for art in (pv["img"], pv["ov"]):
                    art.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
                    art.set_visible(True)
            else:
                pv["label"].configure(text=f"{arrow} (none)")
                pv["img"].set_visible(False)
                pv["ov"].set_visible(False)
            pv["canvas"].draw_idle()

    # ================================================================ view: zoom / pan / rotate

    def _full_limits(self):
        h, w = self.image.shape[:2]
        return (-0.5, w - 0.5), (h - 0.5, -0.5)

    def reset_view(self):
        if self.image is None:
            return
        xlim, ylim = self._full_limits()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.canvas.draw_idle()

    def reset_display(self):
        self.brightness.set(0.0)
        self.contrast.set(1.0)
        self.rotation.set(0)
        self.opacity.set(0.35)
        self._apply_rotation()
        self.reset_view()
        self._refresh_all()

    def _set_limits(self, xlim, ylim):
        """Set axis limits, clamped so the view never leaves the image or zooms in absurdly far."""
        (fx0, fx1), (fy1, fy0) = self._full_limits()   # y axis is inverted
        def clamp(a, b, lo, hi):
            span = min(max(b - a, MIN_VIEW_SPAN_PX), hi - lo)
            c = np.clip((a + b) / 2, lo + span / 2, hi - span / 2)
            return c - span / 2, c + span / 2
        x0, x1 = clamp(*sorted(xlim), fx0, fx1)
        y0, y1 = clamp(*sorted(ylim), fy0, fy1)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)

    def zoom(self, factor, center=None):
        if self.image is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = center if center else ((x0 + x1) / 2, (y0 + y1) / 2)
        self._set_limits((cx + (x0 - cx) * factor, cx + (x1 - cx) * factor),
                         (cy + (y0 - cy) * factor, cy + (y1 - cy) * factor))
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if self.image is None or event.inaxes is not self.ax or event.step == 0:
            return
        now = time.monotonic()
        if now - self._last_scroll < SCROLL_ZOOM_INTERVAL:
            return
        self._last_scroll = now
        self.zoom(ZOOM_STEP if event.step > 0 else 1 / ZOOM_STEP, (event.xdata, event.ydata))

    def _set_rotation_center(self):
        h, w = self.image.shape[:2]
        self._rot_center = ((w - 1) / 2, (h - 1) / 2)
        self._apply_rotation()

    def _apply_rotation(self):
        if self.image is None:
            return
        cx, cy = self._rot_center
        # The y axis points down on screen, so negate to make positive angles counter-clockwise.
        self._view_tf.clear().rotate_deg_around(cx, cy, -self.rotation.get())
        self.canvas.draw_idle()

    # ================================================================ coordinates

    def _to_image(self, event):
        """Mouse event -> (x, y) in image pixel coordinates (undoing the view rotation)."""
        x, y = self._view_tf.inverted().transform((event.xdata, event.ydata))
        return float(x), float(y)

    def _to_screen(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        return (self._view_tf + self.ax.transData).transform(pts) if len(pts) else pts

    def _in_image(self, x, y):
        h, w = self.image.shape[:2]
        return -0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5

    # ================================================================ undo / redo

    def _frame_history(self) -> FrameHistory:
        return self.history.setdefault((self.slice, self.phase), FrameHistory())

    def _snapshot(self) -> Snapshot:
        return Snapshot(self.frame_mask.copy(), copy.deepcopy(self.polygons))

    def _push_undo(self):
        hist = self._frame_history()
        hist.undo.append(self._snapshot())
        del hist.undo[:-UNDO_LIMIT]
        hist.redo.clear()

    def _restore(self, snap: Snapshot):
        self.frame_mask[...] = snap.mask
        self.polygons = snap.polygons
        self._mark_dirty()
        self._refresh_all()

    def undo(self):
        if self.mask is None or self.drag:
            return
        hist = self._frame_history()
        if not hist.undo:
            self.status.set("Nothing to undo on this frame.")
            return
        hist.redo.append(self._snapshot())
        self._restore(hist.undo.pop())

    def redo(self):
        if self.mask is None or self.drag:
            return
        hist = self._frame_history()
        if not hist.redo:
            self.status.set("Nothing to redo on this frame.")
            return
        hist.undo.append(self._snapshot())
        self._restore(hist.redo.pop())

    # ================================================================ editing operations

    def _active_channel_view(self) -> np.ndarray:
        return self.frame_mask[..., self.active.get()]

    def _has_open_polygon(self):
        return any(len(p) >= 3 for p in self.polygons.values())

    def apply_polygon(self, channel=None, push_undo=True):
        ch = self.active.get() if channel is None else channel
        pts = self.polygons[ch]
        if self.mask is None or len(pts) < 3:
            return
        if push_undo:
            self._push_undo()
        region = geometry.rasterize_polygon(pts, self.image.shape[:2])
        geometry.apply_region(self.frame_mask[..., ch], region, self.region_mode.get())
        pts.clear()
        self._mark_dirty()
        self._refresh_all()

    def _commit_polygons(self):
        """Apply every in-progress polygon with >= 3 points (before navigating/saving)."""
        if self.mask is None:
            return
        open_channels = [ch for ch, p in self.polygons.items() if len(p) >= 3]
        if open_channels:
            self._push_undo()
            for ch in open_channels:
                self.apply_polygon(ch, push_undo=False)
        for pts in self.polygons.values():
            pts.clear()

    def _remove_last_vertex(self):
        pts = self.polygons[self.active.get()]
        if pts:
            self._push_undo()
            pts.pop()
            self._refresh_polygons()

    def _discard_polygon(self):
        pts = self.polygons[self.active.get()]
        if pts:
            self._push_undo()
            pts.clear()
            self._refresh_polygons()

    def clear_active(self):
        if self.mask is None:
            return
        self._push_undo()
        self._active_channel_view()[...] = 0
        self.polygons[self.active.get()].clear()
        self._mark_dirty()
        self._refresh_all()

    def clear_frame(self):
        if self.mask is None:
            return
        self._push_undo()
        self.frame_mask[...] = 0
        for pts in self.polygons.values():
            pts.clear()
        self._mark_dirty()
        self._refresh_all()

    def _snap(self, x, y):
        """Endocardium vertices snap onto the epicardium edge when close to it."""
        if self.active.get() != ENDOCARDIUM:
            return x, y
        edge = geometry.boundary_pixels(self.frame_mask[..., EPICARDIUM])
        if len(edge) == 0:
            return x, y
        sx, sy = self._to_screen([(x, y)])[0]
        i = geometry.nearest_point(self._to_screen(edge), (sx, sy), SNAP_PX)
        return (float(edge[i, 0]), float(edge[i, 1])) if i is not None else (x, y)

    # ================================================================ mouse

    def _on_press(self, event):
        if self.image is None or event.inaxes is not self.ax or event.xdata is None or self.drag:
            return
        mods = set(event.modifiers)
        if event.button == 2 or (event.button == 1 and "alt" in mods):
            self.drag = Drag("pan", start_px=(event.x, event.y),
                             start_xlim=self.ax.get_xlim(), start_ylim=self.ax.get_ylim())
            return

        x, y = self._to_image(event)
        tool = self.tool.get()
        ch = self.active.get()
        pts = self.polygons[ch]

        if event.button == 3:
            if tool == "polygon":
                i = geometry.nearest_point(self._to_screen(pts), (event.x, event.y), VERTEX_HIT_PX)
                if i is not None:
                    self._push_undo()
                    pts.pop(i)
                    self._refresh_polygons()
                else:
                    self.apply_polygon()
            return
        if event.button != 1:
            return

        if tool == "brush":
            self._push_undo()
            self.drag = Drag("brush", last=(x, y))
            self._paint(None, (x, y))
        elif tool == "lasso":
            self.drag = Drag("lasso", points=[(x, y)])
        elif tool == "polygon":
            self._polygon_click(event, x, y, mods, pts)

    def _polygon_click(self, event, x, y, mods, pts):
        screen = self._to_screen(pts)
        if mods & {"ctrl", "cmd"} and pts:
            self._push_undo()
            self.drag = Drag("polygon", last=(x, y))
            return
        if "shift" in mods and len(pts) >= 2:
            i = geometry.nearest_edge_insert_index(screen, (event.x, event.y), EDGE_HIT_PX,
                                                   closed=len(pts) >= 3)
            if i is not None:
                self._push_undo()
                pts.insert(i, self._snap(x, y))
                self._refresh_polygons()
            return
        i = geometry.nearest_point(screen, (event.x, event.y), VERTEX_HIT_PX)
        if i == 0 and len(pts) >= 3:
            self.apply_polygon()   # clicked the first vertex: close the polygon
            return
        if i is not None:
            self._push_undo()
            self.drag = Drag("vertex", vertex=i)
            self.highlight.set_data([pts[i][0]], [pts[i][1]])
            self.highlight.set_visible(True)
            self.canvas.draw_idle()
            return
        if not self._in_image(x, y):
            return
        self._push_undo()
        pts.append(self._snap(x, y))
        self._refresh_polygons()

    def _on_motion(self, event):
        if self.image is None:
            return
        d = self.drag
        if d and d.kind == "pan":
            self._pan(event)
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x, y = self._to_image(event)
        self._update_cursor_info(x, y)

        if self.tool.get() == "brush":
            self.brush_circle.set_center((x, y))
            self.brush_circle.set_radius(self.brush_radius.get())
            self.brush_circle.set_visible(True)
        if d is None:
            if self.tool.get() == "brush":
                self.canvas.draw_idle()
            return

        if d.kind == "brush":
            self._paint(d.last, (x, y))
            d.last = (x, y)
        elif d.kind == "lasso":
            d.points.append((x, y))
            xy = np.asarray(d.points)
            self.lasso_line.set_data(xy[:, 0], xy[:, 1])
            self.lasso_line.set_visible(True)
            self.canvas.draw_idle()
        elif d.kind == "vertex":
            pts = self.polygons[self.active.get()]
            if d.vertex < len(pts):
                pts[d.vertex] = self._snap(x, y)
                self.highlight.set_data([pts[d.vertex][0]], [pts[d.vertex][1]])
                self._refresh_polygons()
        elif d.kind == "polygon":
            dx, dy = x - d.last[0], y - d.last[1]
            ch = self.active.get()
            self.polygons[ch] = [(px + dx, py + dy) for px, py in self.polygons[ch]]
            d.last = (x, y)
            self._refresh_polygons()

    def _on_release(self, event):
        d, self.drag = self.drag, None
        if d is None:
            return
        if d.kind == "brush":
            self._refresh_neighbors()
        elif d.kind == "lasso":
            self.lasso_line.set_visible(False)
            if len(d.points) >= 3:
                self._push_undo()
                region = geometry.rasterize_polygon(d.points, self.image.shape[:2])
                geometry.apply_region(self._active_channel_view(), region, self.region_mode.get())
                self._mark_dirty()
                self._refresh_masks()
            else:
                self.canvas.draw_idle()
        elif d.kind == "vertex":
            self.highlight.set_visible(False)
            self.canvas.draw_idle()

    def _on_leave(self, _event):
        self.cursor_info.set("")
        if self.brush_circle.get_visible():
            self.brush_circle.set_visible(False)
            self.canvas.draw_idle()

    def _pan(self, event):
        d = self.drag
        bbox = self.ax.get_window_extent()
        (x0, x1), (y0, y1) = d.start_xlim, d.start_ylim
        dx = (event.x - d.start_px[0]) / bbox.width * (x1 - x0)
        dy = (event.y - d.start_px[1]) / bbox.height * (y1 - y0)
        self._set_limits((x0 - dx, x1 - dx), (y0 - dy, y1 - dy))
        self.canvas.draw_idle()

    def _paint(self, p0, p1):
        value = 1 if self.brush_mode.get() == "Paint" else 0
        geometry.paint_stroke(self._active_channel_view(), p0, p1, self.brush_radius.get(), value)
        self._mark_dirty()
        self._refresh_masks(draw=False)
        self.canvas.draw_idle()

    def _update_cursor_info(self, x, y):
        if not self._in_image(x, y):
            self.cursor_info.set("")
            return
        col, row = int(round(x)), int(round(y))
        val = float(self.image[row, col, self.phase, self.slice])
        labels = [s.name for s in STRUCTURES if self.frame_mask[row, col, s.channel]]
        self.cursor_info.set(f"x={col} y={row}  value={val:.1f}  {'+'.join(labels)}")

    # ================================================================ help

    def show_help(self):
        win = tk.Toplevel(self.root)
        win.title("Keyboard & mouse shortcuts")
        win.transient(self.root)
        text = tk.Text(win, width=72, height=40, wrap="word", padx=12, pady=10, relief=tk.FLAT)
        text.tag_configure("h", font=("TkDefaultFont", 12, "bold"), spacing1=8)
        text.tag_configure("k", font=("TkFixedFont", 11))
        for title, rows in SHORTCUTS:
            text.insert(tk.END, title + "\n", "h")
            for keys, desc in rows:
                text.insert(tk.END, f"  {keys:<32}", "k")
                text.insert(tk.END, desc + "\n")
        text.insert(tk.END, "\nNotes\n", "h")
        text.insert(tk.END,
                    "  • Masks are saved to <data folder>/seg_outputs/<patient>_mask.npy and are loaded "
                    "from there next time (the original mask is never overwritten).\n"
                    "  • Rotation, zoom, brightness and contrast only change the view.\n"
                    "  • Undo history is kept per frame. Moving to another frame applies any "
                    "unfinished polygon with 3+ points.\n"
                    "  • Use process_masks.py afterwards to clean masks and turn a filled "
                    "epicardium into a myocardium ring.\n")
        text.configure(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)
        win.bind("<Escape>", lambda e: win.destroy())
        win.focus_set()


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Cardiac MRI segmentation GUI.")
    parser.add_argument("folder", nargs="?", help="data folder with *_image.npy files "
                        "(default: the last folder you opened)")
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = SegmentationApp(root)
    folder = args.folder or load_settings().get("last_folder")
    if folder and Path(folder).is_dir():
        root.after(100, lambda: app.open_folder(folder))
    elif args.folder:
        messagebox.showerror("Folder not found", args.folder)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
