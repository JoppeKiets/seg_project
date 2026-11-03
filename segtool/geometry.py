"""Pure mask/polygon helpers (no GUI state), so they can be unit tested."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_erosion


def rasterize_polygon(points, shape: tuple[int, int]) -> np.ndarray:
    """Fill a polygon given as ``[(x, y), ...]`` in pixel coordinates into a uint8 mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=float)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x0 > x1 or y0 > y1:
        return mask
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = MplPath(pts).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    mask[y0:y1 + 1, x0:x1 + 1] = inside.reshape(yy.shape)
    return mask


def apply_region(channel: np.ndarray, region: np.ndarray, mode: str) -> None:
    """Combine ``region`` into ``channel`` in place. ``mode`` is New / Add / Erase."""
    if mode == "New":
        channel[...] = region
    elif mode == "Add":
        channel |= region
    elif mode == "Erase":
        channel[region > 0] = 0
    else:
        raise ValueError(f"unknown mode {mode!r}")


def paint_stroke(channel: np.ndarray, p0, p1, radius: float, value: int) -> None:
    """Paint a round brush of ``radius`` along the segment p0 -> p1 (in place).

    Stamps are spaced at most half a radius apart, so fast mouse drags leave no gaps.
    """
    h, w = channel.shape
    p0 = np.asarray(p0 if p0 is not None else p1, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    r = max(float(radius), 0.5)
    x0 = int(max(np.floor(min(p0[0], p1[0]) - r), 0))
    x1 = int(min(np.ceil(max(p0[0], p1[0]) + r), w - 1))
    y0 = int(max(np.floor(min(p0[1], p1[1]) - r), 0))
    y1 = int(min(np.ceil(max(p0[1], p1[1]) + r), h - 1))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    # Distance from every pixel in the window to the segment.
    seg = p1 - p0
    denom = float(seg @ seg)
    if denom == 0:
        t = np.zeros_like(xx, dtype=float)
    else:
        t = np.clip(((xx - p0[0]) * seg[0] + (yy - p0[1]) * seg[1]) / denom, 0, 1)
    dist = np.hypot(xx - (p0[0] + t * seg[0]), yy - (p0[1] + t * seg[1]))
    channel[y0:y1 + 1, x0:x1 + 1][dist <= r] = value


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """``(N, 2)`` array of ``(x, y)`` coordinates of the pixels on the edge of ``mask``."""
    m = mask.astype(bool)
    if not m.any():
        return np.empty((0, 2))
    edge = m & ~binary_erosion(m)
    ys, xs = np.nonzero(edge)
    return np.column_stack([xs, ys]).astype(float)


def nearest_point(points_px: np.ndarray, target_px, max_dist: float) -> int | None:
    """Index of the point closest to ``target_px`` if within ``max_dist``, else None."""
    if len(points_px) == 0:
        return None
    d = np.hypot(points_px[:, 0] - target_px[0], points_px[:, 1] - target_px[1])
    i = int(np.argmin(d))
    return i if d[i] <= max_dist else None


def nearest_edge_insert_index(points_px: np.ndarray, target_px, max_dist: float,
                              closed: bool = True) -> int | None:
    """Index at which to insert a vertex so it lands on the edge nearest ``target_px``."""
    n = len(points_px)
    if n < 2:
        return None
    p = np.asarray(target_px, dtype=float)
    best_idx, best_d = None, np.inf
    for i in range(n if closed else n - 1):
        a, b = points_px[i], points_px[(i + 1) % n]
        seg = b - a
        denom = float(seg @ seg)
        if denom == 0:
            continue
        t = np.clip(float((p - a) @ seg) / denom, 0, 1)
        d = float(np.hypot(*(p - (a + t * seg))))
        if d < best_d:
            best_idx, best_d = i + 1, d
    return best_idx if best_d <= max_dist else None
