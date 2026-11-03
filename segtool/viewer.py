"""Quick read-only viewers for a patient's image + mask.

    python view_masks.py segmentation_ivana/seg_outputs/S_001_mask.npy        # interactive
    python view_masks.py segmentation_ivana/S_001_image.npy --grid            # all slices, 2 phases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.widgets import CheckButtons, Slider

from .config import IMAGE_SUFFIX, MASK_SUFFIX, OUTPUT_DIRNAME, STRUCTURES
from .data import apply_window, find_image_for_mask, window_params


def resolve_paths(path: Path, image: Path | None) -> tuple[Path, Path]:
    """Accept either an image or a mask path and work out the other one."""
    if path.name.endswith(IMAGE_SUFFIX):
        image_path = path
        pid = path.name[: -len(IMAGE_SUFFIX)]
        candidates = [path.parent / OUTPUT_DIRNAME / (pid + MASK_SUFFIX), path.with_name(pid + MASK_SUFFIX)]
        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is None:
            raise FileNotFoundError(f"no mask found for {path.name} (looked in: {', '.join(map(str, candidates))})")
    else:
        mask_path = path
        image_path = image or find_image_for_mask(path)
        if image_path is None:
            raise FileNotFoundError(f"could not find the image for {path.name}; pass it with --image")
    return image_path, mask_path


def overlay_rgba(mask_2d: np.ndarray, visible=None, alpha: float = 0.5) -> np.ndarray:
    """(H, W, 3) mask -> (H, W, 4) colour overlay in the standard structure colours."""
    out = np.zeros((*mask_2d.shape[:2], 4), dtype=np.float32)
    for s in STRUCTURES:
        if visible is None or visible[s.channel]:
            r, g, b, _ = to_rgba(s.color)
            out[mask_2d[..., s.channel] > 0] = (r, g, b, alpha)
    return out


def show_interactive(image: np.ndarray, mask: np.ndarray, title: str) -> None:
    lo, hi = window_params(image)
    _, _, n_phases, n_slices = image.shape
    state = {"slice": 0, "phase": 0, "visible": [True] * len(STRUCTURES)}

    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_axes([0.05, 0.15, 0.7, 0.8])
    ax.set_axis_off()
    im = ax.imshow(np.zeros(image.shape[:2]), cmap="gray", vmin=0, vmax=1)
    ov = ax.imshow(np.zeros((*image.shape[:2], 4)))

    def redraw(_=None):
        s, p = state["slice"], state["phase"]
        im.set_data(apply_window(image[:, :, p, s], lo, hi))
        ov.set_data(overlay_rgba(mask[:, :, p, s], state["visible"], 0.6))
        ax.set_title(f"{title}   slice {s + 1}/{n_slices}   phase {p + 1}/{n_phases}")
        fig.canvas.draw_idle()

    check = CheckButtons(fig.add_axes([0.78, 0.6, 0.2, 0.2]), [s.name for s in STRUCTURES],
                         state["visible"], label_props={"color": [s.color for s in STRUCTURES]})

    def on_check(label):
        idx = [s.name for s in STRUCTURES].index(label)
        state["visible"][idx] = not state["visible"][idx]
        redraw()

    check.on_clicked(on_check)
    sliders = []
    for i, (key, n) in enumerate((("slice", n_slices), ("phase", n_phases))):
        sl = Slider(fig.add_axes([0.15, 0.07 - i * 0.04, 0.6, 0.025]), key.capitalize(),
                    1, max(n, 1), valinit=1, valstep=1)
        sl.on_changed(lambda v, key=key: (state.__setitem__(key, int(v) - 1), redraw()))
        sliders.append(sl)  # keep references alive
    redraw()
    plt.show()


def show_grid(image: np.ndarray, mask: np.ndarray, title: str, phases=None) -> None:
    lo, hi = window_params(image)
    _, _, n_phases, n_slices = image.shape
    phases = phases or [0, n_phases // 2]
    for p in phases:
        fig, axes = plt.subplots(2, n_slices, figsize=(2.2 * n_slices, 4.6), squeeze=False)
        fig.canvas.manager.set_window_title(f"{title} - phase {p + 1}")
        for s in range(n_slices):
            img = apply_window(image[:, :, p, s], lo, hi)
            axes[0, s].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, s].set_title(f"slice {s + 1}", fontsize=8)
            axes[1, s].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[1, s].imshow(overlay_rgba(mask[:, :, p, s]))
            for a in axes[:, s]:
                a.set_axis_off()
        fig.suptitle(f"{title} - phase {p + 1}/{n_phases}")
        fig.tight_layout()
    plt.show()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="View a patient's MRI with its segmentation mask.")
    parser.add_argument("path", type=Path, help=f"a *{MASK_SUFFIX} or *{IMAGE_SUFFIX} file")
    parser.add_argument("--image", type=Path, help="image file, if it can't be found automatically")
    parser.add_argument("--grid", action="store_true", help="static grid of all slices instead of sliders")
    parser.add_argument("--phases", type=int, nargs="+", help="(with --grid) 1-based phases to show")
    args = parser.parse_args(argv)

    try:
        image_path, mask_path = resolve_paths(args.path, args.image)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    image, mask = np.load(image_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")
    if image.shape != mask.shape[:4]:
        parser.error(f"image {image.shape} and mask {mask.shape} do not match")
    print(f"image: {image_path}\nmask:  {mask_path}\nshape: {image.shape} (H, W, Phase, Slice)")

    title = image_path.name[: -len(IMAGE_SUFFIX)]
    if args.grid:
        show_grid(image, mask, title, [p - 1 for p in args.phases] if args.phases else None)
    else:
        show_interactive(image, mask, title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
