"""Clean up saved masks.

For every 2-D slice/phase and every channel:
  1. keep only the largest connected component (drops stray islands)
  2. morphological closing (fills small gaps)
  3. morphological opening (removes frayed edges / thin spurs)
then resolve overlaps between channels with priority LV (endocardium) > RV > myocardium.
Because the LV is removed from the myocardium channel, a filled epicardium drawing
turns into a proper myocardium ring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label

from .config import CLEANED_DIRNAME, ENDOCARDIUM, EPICARDIUM, MASK_SUFFIX, OUTPUT_DIRNAME

RV = 0
KERNEL = np.ones((3, 3), dtype=bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())[1:]
    return labeled == (np.argmax(sizes) + 1)


def clean_2d(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    m = largest_component(m)
    m = binary_closing(m, structure=KERNEL)
    m = binary_opening(m, structure=KERNEL)
    return m


def resolve_overlaps(masks: np.ndarray) -> np.ndarray:
    """``masks[..., 3]`` -> same, with LV > RV > myocardium priority where they overlap."""
    lv = masks[..., ENDOCARDIUM] > 0
    rv = (masks[..., RV] > 0) & ~lv
    myo = (masks[..., EPICARDIUM] > 0) & ~lv & ~rv
    out = np.zeros(masks.shape, dtype=np.uint8)
    out[..., RV], out[..., EPICARDIUM], out[..., ENDOCARDIUM] = rv, myo, lv
    return out


def clean_mask(masks: np.ndarray) -> np.ndarray:
    """Clean a ``(H, W, 3)`` or ``(H, W, Phase, Slice, 3)`` mask array."""
    if masks.shape[-1] != 3 or masks.ndim not in (3, 5):
        raise ValueError(f"expected (H, W, 3) or (H, W, Phase, Slice, 3), got {masks.shape}")
    if masks.ndim == 3:
        masks = masks[:, :, None, None, :]
        squeeze = True
    else:
        squeeze = False
    out = np.zeros(masks.shape, dtype=np.uint8)
    _, _, n_phases, n_slices, n_ch = masks.shape
    for p in range(n_phases):
        for s in range(n_slices):
            for c in range(n_ch):
                out[:, :, p, s, c] = clean_2d(masks[:, :, p, s, c])
    out = resolve_overlaps(out)
    return out[:, :, 0, 0, :] if squeeze else out


def process_folder(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    files = sorted(input_dir.glob("*" + MASK_SUFFIX))
    if not files:
        print(f"No *{MASK_SUFFIX} files found in {input_dir}")
        return 0, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cleaning {len(files)} mask(s): {input_dir} -> {output_dir}\n")
    ok = failed = 0
    for f in files:
        try:
            original = np.load(f)
            cleaned = clean_mask(original)
            np.save(output_dir / f.name, cleaned)
            changed = int((original.astype(np.uint8) != cleaned).sum())
            print(f"  ok   {f.name:24s} {changed:9d} voxels changed")
            ok += 1
        except Exception as exc:  # keep going; report at the end
            print(f"  FAIL {f.name:24s} {exc}")
            failed += 1
    print(f"\nDone: {ok} cleaned, {failed} failed.")
    return ok, failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean segmentation masks (largest component, closing, opening, overlap resolution).")
    parser.add_argument("input_dir", type=Path,
                        help=f"folder containing *{MASK_SUFFIX} files, e.g. segmentation_ivana/{OUTPUT_DIRNAME}")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help=f"where to write cleaned masks (default: <input_dir>/../{CLEANED_DIRNAME})")
    parser.add_argument("--in-place", action="store_true", help="overwrite the input files instead")
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"not a folder: {args.input_dir}")
    if args.in_place and args.output_dir:
        parser.error("use either --in-place or --output-dir, not both")
    output_dir = args.input_dir if args.in_place else (
        args.output_dir or args.input_dir.resolve().parent / CLEANED_DIRNAME)
    _, failed = process_folder(args.input_dir, output_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
