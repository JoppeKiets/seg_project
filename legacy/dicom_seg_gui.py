# =============================================================================
# File: mri_segmentation_gui.py
# =============================================================================
"""
MRI DICOM Segmentation – Polygon & Lasso GUI
Author:      Burak Demirel
Affiliation: Philips
Date:        2025-09-23
Version:     1.4  (6 contours + view-only zoom + view-only brightness/contrast
                   + store raw + store norm params + re-normalize at save)

Zoom behavior
-------------
Zoom only changes the VIEW (matplotlib axis limits). All annotations are still
created in ORIGINAL IMAGE COORDINATES, and masks are ALWAYS saved at the original
(H, W) resolution. No cropped/zoomed mask saving happens.

Brightness/Contrast behavior
----------------------------
Brightness/contrast only change the VIEW. Masks are drawn/saved in original
image coordinates. Saved NPZ image is normalized using the stored per-slice
normalization parameters (1st–99th percentile), re-applied at save time.

Rotation behavior (UPDATED)
---------------------------
Rotate Left/Right rotates the ACTUAL in-memory image by 5° per click.
To prevent cumulative blurring, the displayed/annotated image is ALWAYS
recomputed from the original unrotated image using the total rotation angle.
Masks are rotated only by the incremental delta (nearest-neighbor).
On save, the NPZ filename is tagged with:
  - _rot_plus_X or _rot_neg_X   (X is abs(degrees))
and rot_deg is stored in the NPZ for reproducibility.
"""

## how to read back

#import numpy as np
#import matplotlib.pyplot as plt

# Path to your saved NPZ
#npz_path = r"path/to/seg_outputs/<SOPInstanceUID>.npz"

# Load data
#data = np.load(npz_path)
#image = data["image"]          # (H, W), float32 in [0, 1]
#masks = data["masks"]          # (H, W, 6), uint8
#rot_deg = int(data.get("rot_deg", 0))

#print("Image shape:", image.shape)
#print("Masks shape:", masks.shape)
#print("Rotation:", rot_deg)

# Display
#plt.figure(figsize=(15, 15))
#plt.imshow(image, cmap="gray", vmin=0, vmax=1)

# Overlay all contours
#for i in range(masks.shape[-1]):
#    if np.any(masks[..., i]):
#        plt.imshow(
#            np.ma.masked_where(masks[..., i] == 0, masks[..., i]),
#            alpha=0.35
#        )

#plt.axis("off")
#plt.title("Image with all contours")
#plt.show()


import os
import sys
import numpy as np
import pydicom
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import LassoSelector
from PIL import Image  # <-- ADDED

# =============================
# Configuration
# =============================
N_CONTOURS = 6
SAVE_EXCEL_LOG = False
# Zoom factors
ZOOM_IN_FACTOR = 0.80   # smaller span => zoom in
ZOOM_OUT_FACTOR = 1.25  # larger span => zoom out
MIN_VIEW_SPAN_PX = 10   # prevent zooming into nothing

# Rotation step (degrees per click)
ROT_STEP_DEG = 5  # <-- UPDATED


# =============================
# Utility functions
# =============================
def compute_norm_params(arr_float32):
    """
    Compute normalization parameters exactly like before:
    1st–99th percentile window.
    Returns (lo, hi) as float32.
    """
    lo, hi = np.percentile(arr_float32, [1, 99]).astype(np.float32)
    return float(lo), float(hi)


def apply_norm(arr_float32, lo, hi):
    """
    Apply normalization (lo/hi) to [0,1] as before.
    """
    arr = arr_float32.astype(np.float32, copy=False)

    if hi > lo:
        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32, copy=False)

    # Fallback: original behavior when hi == lo
    out = arr - float(arr.min())
    mx = float(out.max())
    if mx > 0:
        out = out / mx
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def load_dicom_image(fp):
    """
    Load a DICOM image and return:
      (raw_rescaled_float32, meta_dict)

    NOTE:
    - We now return RAW (after RescaleSlope/Intercept), not normalized.
    - We compute norm params here and store them in meta.
    """
    ds = pydicom.dcmread(fp, force=True)

    if not any(tag in ds for tag in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")):
        raise RuntimeError("No pixel data in this DICOM; skipping.")

    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as dec_err:
        raise RuntimeError(f"Pixel data not decodable; skipping. Details: {dec_err}")

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter  # raw (rescaled)

    norm_lo, norm_hi = compute_norm_params(arr)

    meta = {
        "SOPInstanceUID": str(getattr(ds, "SOPInstanceUID", "")),
        "InstanceNumber": int(getattr(ds, "InstanceNumber", 0) or 0),
        "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "")),
        "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
        "filepath": fp,
        "PatientID": str(getattr(ds, "PatientID", "")),
        # NOTE: store normalization parameters so we can re-apply at save time
        "norm_lo": float(norm_lo),
        "norm_hi": float(norm_hi),
    }
    return arr.astype(np.float32, copy=False), meta


def list_dicom_files(dirpath):
    """Return a sorted list of *all* files under dirpath (we'll filter later while loading)."""
    files = []
    for root, _, fnames in os.walk(dirpath):
        for f in fnames:
            files.append(os.path.join(root, f))
    return sorted(files)


# =============================
# GUI Application
# =============================
class SegmentationGUI:
    def __init__(self, master):
        self.master = master
        master.title("MRI DICOM Segmentation")
        master.geometry("1200x800")

        # State
        self.current_dir = ""
        self.recent_dirs = []
        self.files = []
        self.index = -1

        # NOTE: we now store raw and norm params separately
        self.image_raw = None     # (H, W) float32 (CURRENT / ROTATED for annotation)
        self.image_raw_base = None  # (H, W) float32 (ORIGINAL, unrotated)  <-- ADDED
        self.norm_lo = None
        self.norm_hi = None

        self.meta = None
        self.masks = None         # (H, W, N_CONTOURS) uint8

        # Polygon drawing state
        self.poly_vertices = []  # list of (x, y)
        self.preview_line = None
        self.vertex_scat = None
        self.vertex_color = "yellow"
        self.drag_idx = None

        # Lasso state
        self.lasso = None
        self.lasso_color = "yellow"

        # Output/logging
        self.output_dir = None
        self.log_path = None

        # Processed tracking
        self.processed_files = set()
        self.processed_uids = set()

        # View/zoom state (axis limits in data coords)
        self._full_xlim = None
        self._full_ylim = None

        # Brightness/contrast (VIEW ONLY)
        self.brightness_var = tk.DoubleVar(value=0.0)  # additive offset in [-0.5, 0.5]
        self.contrast_var = tk.DoubleVar(value=1.0)    # gain in [0.2, 3.0]

        # Rotation state (degrees)
        self.rot_deg = 0  # int  <-- UPDATED

        self._build_ui()

    # --------------------------- UI ---------------------------
    def _build_ui(self):
        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="DICOM Folder:").pack(side=tk.LEFT)
        self.dir_combo = ttk.Combobox(top, values=self.recent_dirs, width=80)
        self.dir_combo.pack(side=tk.LEFT, padx=6)
        self.dir_combo.bind("<<ComboboxSelected>>", lambda e: self.on_select_dir(self.dir_combo.get()))

        ttk.Button(top, text="Browse…", command=self.browse_dir).pack(side=tk.LEFT)
        ttk.Button(top, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=6)

        # Canvas / Matplotlib figure
        mid = ttk.Frame(self.master)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right control panel
        right = ttk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8)

        self.info_var = tk.StringVar(value="No folder selected")
        ttk.Label(right, textvariable=self.info_var, wraplength=260, justify=tk.LEFT).pack(anchor="w", pady=(0, 10))

        # --- Zoom controls ---
        ttk.Label(right, text="Zoom").pack(anchor="w")
        zrow = ttk.Frame(right)
        zrow.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(zrow, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(zrow, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(right, text="Reset View", command=self.reset_view).pack(fill=tk.X)
        ttk.Label(right, text="Tip: mouse wheel zooms at cursor").pack(anchor="w", pady=(2, 8))

        # --- Rotation controls ---
        ttk.Label(right, text="Rotate").pack(anchor="w")
        rrow = ttk.Frame(right)
        rrow.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(rrow, text="⟲ Rotate Left", command=self.rotate_left).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(rrow, text="⟳ Rotate Right", command=self.rotate_right).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # --- Brightness/Contrast controls (view only) ---
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Display (view-only)").pack(anchor="w")

        ttk.Label(right, text="Brightness").pack(anchor="w")
        bright = ttk.Scale(
            right,
            from_=-0.5, to=0.5,
            variable=self.brightness_var,
            command=lambda _v: self._refresh_display()
        )
        bright.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(right, text="Contrast").pack(anchor="w")
        contr = ttk.Scale(
            right,
            from_=0.2, to=3.0,
            variable=self.contrast_var,
            command=lambda _v: self._refresh_display()
        )
        contr.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(
            right,
            text="Reset Bright/Contrast",
            command=lambda: (self.brightness_var.set(0.0),
                             self.contrast_var.set(1.0),
                             self._refresh_display())
        ).pack(fill=tk.X, pady=(0, 6))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Contour controls
        ttk.Label(right, text="Contours").pack(anchor="w")
        self.curr_contour_var = tk.IntVar(value=1)
        for i in range(1, N_CONTOURS + 1):
            ttk.Radiobutton(
                right, text=f"Active: Contour {i}", variable=self.curr_contour_var, value=i
            ).pack(anchor="w")

        # Clear buttons (C1..C6) in 2 columns
        btns = ttk.Frame(right)
        btns.pack(fill=tk.X)

        for r in range(0, N_CONTOURS, 2):
            row = ttk.Frame(btns)
            row.pack(fill=tk.X)
            ttk.Button(row, text=f"Clear C{r+1}", command=lambda k=r: self.clear_contour(k)).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=1
            )
            if r + 1 < N_CONTOURS:
                ttk.Button(row, text=f"Clear C{r+2}", command=lambda k=r + 1: self.clear_contour(k)).pack(
                    side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=1
                )

        # Skip buttons (C2..C6) in 2 columns
        skipf = ttk.Frame(right)
        skipf.pack(fill=tk.X, pady=(6, 0))

        for r in range(1, N_CONTOURS, 2):
            row = ttk.Frame(skipf)
            row.pack(fill=tk.X)
            ttk.Button(row, text=f"Skip C{r+1}", command=lambda k=r: self.skip_contour(k)).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=1
            )
            if r + 1 < N_CONTOURS:
                ttk.Button(row, text=f"Skip C{r+2}", command=lambda k=r + 1: self.skip_contour(k)).pack(
                    side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=1
                )

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Button(right, text="Finalize & Save", command=self.finalize_current).pack(fill=tk.X)
        ttk.Button(right, text="◀ Previous Image", command=self.prev_image).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(right, text="Next Image ▶", command=self.next_image).pack(fill=tk.X, pady=(0, 0))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(right, text="Polygon drawing:").pack(anchor="w")
        ttk.Label(
            right,
            text=(
                "• Left-click to add points; left-click a point to drag\n"
                "• Enter / Right-click to close & apply\n"
                "• Backspace undo last point\n"
                "• Esc cancel"
            ),
        ).pack(anchor="w")

        ttk.Label(right, text="Lasso (freehand) drawing:").pack(anchor="w")
        ttk.Label(
            right,
            text=(f"• Drag to draw; release to apply\n• Auto-advances C1→…→C{N_CONTOURS}"),
        ).pack(anchor="w")

        # Bindings
        self.canvas.mpl_connect("button_press_event", self.on_click_poly)
        self.canvas.mpl_connect("motion_notify_event", self.on_move_poly)
        self.canvas.mpl_connect("key_press_event", self.on_key_poly)
        self.canvas.mpl_connect("button_release_event", self.on_release_poly)

        # Mouse wheel zoom
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)

        # Initialize lasso
        self._init_lasso()

    # ======================= Image (normalized) + Bright/Contrast (view only) =======================
    def _get_norm_image(self):
        """
        Returns the normalized image in [0,1] using stored norm_lo/norm_hi.
        Does NOT modify stored raw.
        """
        if self.image_raw is None or self.norm_lo is None or self.norm_hi is None:
            return None
        return apply_norm(self.image_raw, self.norm_lo, self.norm_hi)

    def _get_display_image(self):
        """
        Returns a view-only transformed image for display:
        normalized -> brightness/contrast.
        """
        img = self._get_norm_image()
        if img is None:
            return None

        b = float(self.brightness_var.get())
        c = float(self.contrast_var.get())

        disp = (img - 0.5) * c + 0.5 + b
        disp = np.clip(disp, 0.0, 1.0)
        return disp.astype(np.float32, copy=False)

    # ======================= Rotation helpers (5° per click, no cumulative blur) =======================
    def _rotation_deg_and_tag(self):
        """
        Returns (deg, tag_str) for current rotation.
        Uses:
          +deg => "_rot_plus_X"
          -deg => "_rot_neg_X"
          0    => ""
        """
        deg = int(self.rot_deg)
        if deg == 0:
            return 0, ""
        if deg > 0:
            return deg, f"_rot_plus_{abs(deg)}"
        return deg, f"_rot_neg_{abs(deg)}"

    def _rotate_image_from_base(self):
        """
        Recompute image_raw from the original unrotated base using total rot_deg.
        Uses high-quality resampling to minimize blur (single resample from base).
        """
        if self.image_raw_base is None:
            return
        deg = float(self.rot_deg)
        if deg == 0:
            self.image_raw = self.image_raw_base.astype(np.float32, copy=False)
            return

        # PIL float32 via mode 'F'
        pil = Image.fromarray(self.image_raw_base.astype(np.float32), mode="F")
        pil_r = pil.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=0.0)
        self.image_raw = np.asarray(pil_r, dtype=np.float32)

    def _rotate_mask_delta(self, mask2d_uint8, delta_deg):
        """
        Rotate a single uint8 mask by delta_deg using NEAREST (keeps crisp edges).
        """
        if mask2d_uint8 is None:
            return None
        if int(delta_deg) == 0:
            return mask2d_uint8
        pil = Image.fromarray(mask2d_uint8.astype(np.uint8), mode="L")
        pil_r = pil.rotate(float(delta_deg), resample=Image.NEAREST, expand=False, fillcolor=0)
        return np.asarray(pil_r, dtype=np.uint8)

    def _apply_rotation_delta(self, delta_deg):
        """
        Apply incremental rotation:
          - Update total rot_deg
          - Recompute image_raw from base (prevents cumulative blur)
          - Rotate masks by delta only (nearest)
        """
        if self.image_raw_base is None:
            return

        old_deg = int(self.rot_deg)
        new_deg = int(old_deg + int(delta_deg))
        self.rot_deg = new_deg

        # Recompute image from base at total rotation (no cumulative blur)
        self._rotate_image_from_base()

        # Rotate masks by delta only (so existing annotations follow)
        if self.masks is not None:
            d = int(delta_deg)
            if d != 0:
                h, w, c = self.masks.shape
                new_masks = np.zeros_like(self.masks, dtype=np.uint8)
                for k in range(c):
                    if np.any(self.masks[..., k]):
                        new_masks[..., k] = self._rotate_mask_delta(self.masks[..., k], d)
                self.masks = new_masks

        # Clear any in-progress polygon vertices (they won't match after rotation)
        self.poly_vertices = []
        self._clear_temp_poly_artists()

        # Update view extents and redraw
        self._compute_full_view()
        self._refresh_display()
        self.reset_view()

        # Recreate lasso
        self._init_lasso()

    def rotate_left(self):
        """Rotate +ROT_STEP_DEG (counter-clockwise)."""
        self._apply_rotation_delta(+ROT_STEP_DEG)

    def rotate_right(self):
        """Rotate -ROT_STEP_DEG (clockwise)."""
        self._apply_rotation_delta(-ROT_STEP_DEG)

    # ======================= Zoom helpers =======================
    def _compute_full_view(self):
        """Set full-view xlim/ylim for the current image based on imshow default extent."""
        if self.image_raw is None:
            self._full_xlim = None
            self._full_ylim = None
            return
        h, w = self.image_raw.shape
        # For imshow(origin='upper'), default extent is x=[-0.5, w-0.5], y=[h-0.5, -0.5]
        self._full_xlim = (-0.5, w - 0.5)
        self._full_ylim = (h - 0.5, -0.5)

    def reset_view(self):
        """Reset to full view."""
        if self.image_raw is None:
            return
        if self._full_xlim is None or self._full_ylim is None:
            self._compute_full_view()
        self.ax.set_xlim(*self._full_xlim)
        self.ax.set_ylim(*self._full_ylim)
        self.canvas.draw_idle()

    def _clamp_view(self, xlim, ylim):
        """Clamp view limits to the image full extent and prevent too-small spans."""
        if self._full_xlim is None or self._full_ylim is None:
            self._compute_full_view()
        fx0, fx1 = self._full_xlim
        fy0, fy1 = self._full_ylim  # note: fy0 > fy1 because origin='upper'

        x0, x1 = xlim
        y0, y1 = ylim

        # Ensure span not too small
        if abs(x1 - x0) < MIN_VIEW_SPAN_PX:
            cx = 0.5 * (x0 + x1)
            x0, x1 = cx - MIN_VIEW_SPAN_PX / 2, cx + MIN_VIEW_SPAN_PX / 2
        if abs(y1 - y0) < MIN_VIEW_SPAN_PX:
            cy = 0.5 * (y0 + y1)
            y0, y1 = cy - MIN_VIEW_SPAN_PX / 2, cy + MIN_VIEW_SPAN_PX / 2

        # Clamp to full extent (x)
        span_x = x1 - x0
        if x0 < fx0:
            x0 = fx0
            x1 = fx0 + span_x
        if x1 > fx1:
            x1 = fx1
            x0 = fx1 - span_x

        # Clamp to full extent (y) where fy0>fy1
        span_y = y1 - y0
        y_min = min(fy0, fy1)
        y_max = max(fy0, fy1)
        if y0 < y_min:
            y0 = y_min
            y1 = y_min + span_y
        if y1 > y_max:
            y1 = y_max
            y0 = y_max - span_y

        # Prevent zooming out beyond full view
        if (x1 - x0) >= (fx1 - fx0):
            x0, x1 = fx0, fx1
        if (y1 - y0) >= (y_max - y_min):
            y0, y1 = (fy0, fy1)

        return (x0, x1), (y0, y1)

    def _zoom(self, factor, center=None):
        """Zoom view by factor around center=(cx,cy) in data coords."""
        if self.image_raw is None:
            return
        if self._full_xlim is None or self._full_ylim is None:
            self._compute_full_view()

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        if center is None:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
        else:
            cx, cy = center

        span_x = (x1 - x0) * factor
        span_y = (y1 - y0) * factor

        nx0 = cx - span_x / 2
        nx1 = cx + span_x / 2
        ny0 = cy - span_y / 2
        ny1 = cy + span_y / 2

        (nx0, nx1), (ny0, ny1) = self._clamp_view((nx0, nx1), (ny0, ny1))
        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)
        self.canvas.draw_idle()

    def zoom_in(self):
        self._zoom(ZOOM_IN_FACTOR)

    def zoom_out(self):
        self._zoom(ZOOM_OUT_FACTOR)

    def on_scroll_zoom(self, event):
        """Mouse wheel zoom at cursor location."""
        if self.image_raw is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            center = None
        else:
            center = (float(event.xdata), float(event.ydata))

        if event.button == "up":
            self._zoom(ZOOM_IN_FACTOR, center=center)
        elif event.button == "down":
            self._zoom(ZOOM_OUT_FACTOR, center=center)

    # ======================= Lasso setup & callback =======================
    def _init_lasso(self):
        if self.lasso is not None:
            try:
                self.lasso.disconnect_events()
            except Exception:
                pass
            self.lasso = None

        self.lasso = LassoSelector(
            self.ax,
            self._on_lasso_select,
            useblit=True,
            props=dict(color=self.lasso_color, linewidth=1.5, alpha=0.9),
        )

    def _on_lasso_select(self, verts):
        if self.image_raw is None or self.masks is None or not verts or len(verts) < 3:
            return

        poly = np.asarray(verts, dtype=np.float32)
        h, w = self.image_raw.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        pts = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        try:
            path = MplPath(poly, closed=True)
            inside = path.contains_points(pts)
            mask = inside.reshape(h, w)
        except Exception:
            mask = np.zeros((h, w), dtype=bool)

        idx = self.curr_contour_var.get() - 1
        if idx < 0 or idx >= self.masks.shape[-1]:
            return

        self.masks[..., idx] = mask.astype(np.uint8)
        self._refresh_display()

        if idx < N_CONTOURS - 1:
            self.curr_contour_var.set(idx + 2)

    # --------------------------- Contour utilities ---------------------------
    def clear_contour(self, idx):
        if self.masks is None:
            return
        if 0 <= idx < self.masks.shape[-1]:
            self.masks[..., idx] = 0
            self._refresh_display()

    def skip_contour(self, idx):
        if self.masks is None:
            return
        if 0 <= idx < self.masks.shape[-1]:
            self.masks[..., idx] = 0
            active_idx = self.curr_contour_var.get() - 1
            if active_idx == idx and idx < N_CONTOURS - 1:
                self.curr_contour_var.set(idx + 2)
            self._refresh_display()

    # --------------------------- Polygon helpers ---------------------------
    def _nearest_vertex(self, event, tol_px=8):
        if not self.poly_vertices or event.x is None or event.y is None:
            return None, None

        trans = self.ax.transData
        ex, ey = event.x, event.y

        best_idx, best_d2 = None, float("inf")
        for i, (x, y) in enumerate(self.poly_vertices):
            px, py = trans.transform((x, y))
            d2 = (px - ex) ** 2 + (py - ey) ** 2
            if d2 < best_d2:
                best_idx, best_d2 = i, d2

        if best_idx is None:
            return None, None
        if best_d2 <= tol_px**2:
            return best_idx, best_d2**0.5
        return None, None

    def _clear_temp_poly_artists(self):
        if self.preview_line is not None:
            try:
                self.preview_line.remove()
            except Exception:
                pass
            self.preview_line = None
        if self.vertex_scat is not None:
            try:
                self.vertex_scat.remove()
            except Exception:
                pass
            self.vertex_scat = None

    def _update_poly_artists(self):
        if self.preview_line is not None:
            try:
                self.preview_line.remove()
            except Exception:
                pass
            self.preview_line = None

        if not self.poly_vertices:
            if self.vertex_scat is not None:
                try:
                    self.vertex_scat.remove()
                except Exception:
                    pass
                self.vertex_scat = None
            return

        xs, ys = zip(*self.poly_vertices)
        if self.vertex_scat is None:
            self.vertex_scat = self.ax.scatter(xs, ys, s=18, c=self.vertex_color)
        else:
            self.vertex_scat.set_offsets(np.c_[xs, ys])

    def on_click_poly(self, event):
        if self.image_raw is None or event.inaxes != self.ax:
            return

        if event.button == 3:
            self.close_and_apply_polygon()
            return

        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            idx, _ = self._nearest_vertex(event, tol_px=8)
            if idx is not None:
                self.drag_idx = idx
                return

            x, y = float(event.xdata), float(event.ydata)

            if len(self.poly_vertices) >= 3:
                trans = self.ax.transData
                fx, fy = self.poly_vertices[0]
                fpx, fpy = trans.transform((fx, fy))
                d2 = (fpx - event.x) ** 2 + (fpy - event.y) ** 2
                if d2 <= (8.0**2):
                    self.close_and_apply_polygon()
                    return

            self.poly_vertices.append((x, y))
            self._update_poly_artists()
            self.canvas.draw_idle()

    def on_move_poly(self, event):
        if self.image_raw is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self.drag_idx is not None and 0 <= self.drag_idx < len(self.poly_vertices):
            self.poly_vertices[self.drag_idx] = (float(event.xdata), float(event.ydata))
            self._update_poly_artists()
            self.canvas.draw_idle()
            return

        if self.poly_vertices:
            self._update_poly_artists()
            self.canvas.draw_idle()

    def on_release_poly(self, event):
        if self.drag_idx is not None:
            self.drag_idx = None
            self.canvas.draw_idle()

    def on_key_poly(self, event):
        if self.image_raw is None:
            return
        if event.key in ("enter", "return"):
            self.close_and_apply_polygon()
        elif event.key == "backspace":
            if self.poly_vertices:
                self.poly_vertices.pop()
                self._update_poly_artists()
                self.canvas.draw_idle()
        elif event.key == "escape":
            self.poly_vertices = []
            self._clear_temp_poly_artists()
            self.canvas.draw_idle()

    def close_and_apply_polygon(self):
        if len(self.poly_vertices) < 3 or self.image_raw is None or self.masks is None:
            return

        poly = np.asarray(self.poly_vertices, dtype=np.float32)
        h, w = self.image_raw.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        pts = np.vstack((grid_x.ravel(), grid_y.ravel())).T
        try:
            path = MplPath(poly, closed=True)
            inside = path.contains_points(pts)
            mask = inside.reshape(h, w)
        except Exception:
            mask = np.zeros((h, w), dtype=bool)

        idx = self.curr_contour_var.get() - 1
        if idx < 0 or idx >= self.masks.shape[-1]:
            return

        self.masks[..., idx] = mask.astype(np.uint8)

        self._clear_temp_poly_artists()
        self.poly_vertices = []
        self._refresh_display()

        if idx < N_CONTOURS - 1:
            self.curr_contour_var.set(idx + 2)

    # --------------------------- Directory handling ---------------------------
    def browse_dir(self):
        d = filedialog.askdirectory(title="Select DICOM folder")
        if d:
            self.dir_combo.set(d)
            self.on_select_dir(d)

    def on_select_dir(self, d):
        if d and os.path.isdir(d):
            self.current_dir = d
            if d not in self.recent_dirs:
                self.recent_dirs = [d] + self.recent_dirs[:9]
                self.dir_combo["values"] = self.recent_dirs
            self.info_var.set(f"Selected folder:\n{d}")

    def load_folder(self):
        if not self.current_dir:
            messagebox.showwarning("Folder", "Please select a folder first.")
            return

        self.files = list_dicom_files(self.current_dir)
        if not self.files:
            messagebox.showinfo("Folder", "No DICOM-like files found in this folder.")
            return

        self.output_dir = os.path.join(self.current_dir, "seg_outputs_myo")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_path = os.path.join(self.output_dir, "seg_log.xlsx")

        self.processed_files = set()
        self.processed_uids = set()
        try:
            if os.path.exists(self.log_path):
                df = pd.read_excel(self.log_path)
                if "status" in df.columns:
                    df = df[df["status"].astype(str).str.lower() == "done"]
                if "file" in df.columns:
                    self.processed_files = set(df["file"].dropna().astype(str).tolist())
                if "SOPInstanceUID" in df.columns:
                    self.processed_uids = set(df["SOPInstanceUID"].dropna().astype(str).tolist())
        except Exception as e:
            print(f"Warning: could not read existing log for skip list: {e}")

        self.index = -1
        self.next_image()

    # --------------------------- Image navigation ---------------------------
    def prev_image(self):
        if not self.files:
            messagebox.showinfo("Done", "No folder loaded.")
            return

        while True:
            self.index -= 1
            if self.index < 0:
                messagebox.showinfo("Start", "You are at the first available image.")
                self.index = 0
                return

            fp = self.files[self.index]
            try:
                img_raw, meta = load_dicom_image(fp)
            except Exception as e:
                print(f"Skipping previous (not decodable): {fp} | {e}")
                continue

            rel = os.path.relpath(meta.get("filepath", fp), self.current_dir)
            sop = str(meta.get("SOPInstanceUID", "") or "")
            if (rel in self.processed_files) or (sop and sop in self.processed_uids):
                print(f"Skipping previous (already processed): '{rel}' uid='{sop}'")
                continue

            break

        self._load_image_state(img_raw, meta)

    def next_image(self):
        if not self.files:
            messagebox.showinfo("Done", "No folder loaded.")
            return

        while True:
            self.index += 1
            if self.index >= len(self.files):
                self.show_no_more()
                return

            fp = self.files[self.index]
            try:
                img_raw, meta = load_dicom_image(fp)
            except Exception as e:
                print(f"Skipping file (not DICOM image / not decodable): {fp} | {e}")
                continue

            rel = os.path.relpath(meta.get("filepath", fp), self.current_dir)
            sop = str(meta.get("SOPInstanceUID", "") or "")
            if (rel in self.processed_files) or (sop and sop in self.processed_uids):
                print(f"Skipping already processed: file='{rel}' uid='{sop}'")
                continue

            break

        self._load_image_state(img_raw, meta)

    def _load_image_state(self, img_raw, meta):
        """Common logic for loading a new image and resetting state."""
        self.image_raw_base = img_raw.astype(np.float32, copy=True)  # <-- store base (unrotated)
        self.image_raw = self.image_raw_base.astype(np.float32, copy=False)
        self.meta = meta

        # NOTE: store norm params from meta
        self.norm_lo = float(meta.get("norm_lo"))
        self.norm_hi = float(meta.get("norm_hi"))

        self.masks = np.zeros((img_raw.shape[0], img_raw.shape[1], N_CONTOURS), dtype=np.uint8)

        # Reset display controls on each new image
        self.brightness_var.set(0.0)
        self.contrast_var.set(1.0)

        # Reset rotation state for each new image
        self.rot_deg = 0

        # Reset drawing state
        self.poly_vertices = []
        self._clear_temp_poly_artists()

        # Set full view extents and reset view
        self._compute_full_view()
        self._refresh_display()
        self.reset_view()

        self.info_var.set(
            f"Folder: {self.current_dir}\n"
            f"Image {self.index+1}/{len(self.files)}\n"
            f"File: {os.path.relpath(self.files[self.index], self.current_dir)}\n"
            f"InstanceNumber: {meta.get('InstanceNumber', 0)}"
        )

        # Recreate lasso
        self._init_lasso()

    def show_no_more(self):
        messagebox.showinfo("Done", "No more decodable DICOM images in this folder.")

    # --------------------------- Rendering ---------------------------
    def _refresh_display(self):
        """
        IMPORTANT:
        - Preserve current xlim/ylim so zoom doesn't reset when masks update.
        - Brightness/contrast are view-only.
        """
        prev_xlim = self.ax.get_xlim()
        prev_ylim = self.ax.get_ylim()

        self.ax.clear()
        self.ax.set_axis_off()

        if self.image_raw is None:
            self.canvas.draw_idle()
            return

        disp = self._get_display_image()
        if disp is None:
            self.canvas.draw_idle()
            return

        self.ax.imshow(
            disp,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="bicubic",
        )

        for i in range(N_CONTOURS):
            if self.masks is not None and np.any(self.masks[..., i]):
                self.ax.imshow(
                    np.ma.masked_where(self.masks[..., i] == 0, self.masks[..., i]),
                    alpha=0.35,
                )

        if self._full_xlim is None or self._full_ylim is None:
            self._compute_full_view()

        if prev_xlim == (0.0, 1.0) and prev_ylim == (0.0, 1.0):
            self.ax.set_xlim(*self._full_xlim)
            self.ax.set_ylim(*self._full_ylim)
        else:
            (nx0, nx1), (ny0, ny1) = self._clamp_view(prev_xlim, prev_ylim)
            self.ax.set_xlim(nx0, nx1)
            self.ax.set_ylim(ny0, ny1)

        self.canvas.draw_idle()

    # --------------------------- Saving / Finalize ---------------------------
    def finalize_current(self):
        if self.image_raw is None or self.meta is None or self.masks is None:
            return

        base_name = self.meta.get("SOPInstanceUID") or os.path.basename(
            self.meta.get("filepath", f"img_{self.index:04d}")
        )
        base_name = str(base_name).replace("/", "_").replace("\\", "_")

        rot_deg, rot_tag = self._rotation_deg_and_tag()
        out_npz = os.path.join(self.output_dir, f"{base_name}{rot_tag}.npz")

        # NOTE: re-apply normalization right before saving (on the ROTATED image_raw)
        try:
            image_norm = apply_norm(self.image_raw, self.norm_lo, self.norm_hi).astype(np.float32, copy=False)
        except Exception as e:
            messagebox.showerror("Save", f"Failed to normalize image for saving: {e}")
            return

        try:
            np.savez_compressed(
                out_npz,
                masks=self.masks.astype(np.uint8),
                image=image_norm,              # normalized [0,1]
                norm_lo=float(self.norm_lo),   # store params for reproducibility
                norm_hi=float(self.norm_hi),
                rot_deg=int(rot_deg),          # rotation at save time
            )
        except Exception as e:
            messagebox.showerror("Save", f"Failed to save NPZ: {e}")
            return

        row = {
            "folder": self.current_dir,
            "file": os.path.relpath(self.meta.get("filepath", ""), self.current_dir),
            "SOPInstanceUID": self.meta.get("SOPInstanceUID", ""),
            "InstanceNumber": self.meta.get("InstanceNumber", 0),
            "status": "Done",
            "output_npz": os.path.relpath(out_npz, self.current_dir),
        }

        if SAVE_EXCEL_LOG:
            try:
                if os.path.exists(self.log_path):
                    df = pd.read_excel(self.log_path)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])

                with pd.ExcelWriter(self.log_path, engine="openpyxl", mode="w") as writer:
                    df.to_excel(writer, index=False, sheet_name="log")
            except Exception as e:
                messagebox.showwarning("Excel", f"Saved masks, but failed to update Excel log: {e}")

        messagebox.showinfo("Saved", f"Contours saved and logged for image {self.index+1}.")

        # Update in-memory skip sets
        try:
            rel = os.path.relpath(self.meta.get("filepath", ""), self.current_dir)
            if rel:
                self.processed_files.add(rel)
            sop = str(self.meta.get("SOPInstanceUID", "") or "")
            if sop:
                self.processed_uids.add(sop)
        except Exception:
            pass

        self.next_image()


def main():
    root = tk.Tk()
    try:
        from ctypes import windll

        root.call("tk", "scaling", 1.25)
        if sys.platform.startswith("win"):
            windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = SegmentationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
