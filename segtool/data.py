"""Finding, loading and saving patient volumes.

Files on disk (per patient ``S_001``):
    S_001_image.npy                (H, W, Phase, Slice) float
    S_001_mask.npy                 (H, W, Phase, Slice, 3) uint8, the original/reference mask
    seg_outputs/S_001_mask.npy     same layout, written by the GUI (takes precedence when present)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from .config import IMAGE_SUFFIX, MASK_SUFFIX, N_CHANNELS, OUTPUT_DIRNAME


@dataclass(frozen=True)
class PatientFiles:
    patient_id: str
    image_path: Path

    @property
    def original_mask_path(self) -> Path:
        return self.image_path.with_name(self.patient_id + MASK_SUFFIX)

    @property
    def saved_mask_path(self) -> Path:
        return self.image_path.parent / OUTPUT_DIRNAME / (self.patient_id + MASK_SUFFIX)

    @property
    def has_saved_mask(self) -> bool:
        return self.saved_mask_path.exists()


def discover_patients(folder: str | os.PathLike) -> dict[str, PatientFiles]:
    """Find every ``*_image.npy`` below ``folder`` (output folders are skipped)."""
    patients: dict[str, PatientFiles] = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith((OUTPUT_DIRNAME, ".")))
        for name in sorted(files):
            if name.endswith(IMAGE_SUFFIX):
                pid = name[: -len(IMAGE_SUFFIX)]
                # Same ID in two sub-folders: keep both, distinguished by folder name.
                key = pid if pid not in patients else f"{pid} ({os.path.basename(root)})"
                patients[key] = PatientFiles(pid, Path(root) / name)
    return dict(sorted(patients.items()))


def resize_nearest(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize the first two axes of ``arr`` to ``hw`` with nearest-neighbour sampling."""
    if arr.shape[:2] == tuple(hw):
        return arr
    factors = (hw[0] / arr.shape[0], hw[1] / arr.shape[1]) + (1,) * (arr.ndim - 2)
    return zoom(arr, factors, order=0)


def load_image(path: str | os.PathLike) -> np.ndarray:
    image = np.load(path)
    if image.ndim != 4:
        raise ValueError(f"{Path(path).name}: expected (H, W, Phase, Slice), got shape {image.shape}")
    return image


def load_mask_for(patient: PatientFiles, image_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    """Load the best available mask for ``patient``.

    Returns ``(mask, source)`` where source is ``"saved"``, ``"original"`` or ``"empty"``.
    Masks saved at a different in-plane size (older versions resized to 256x256) are
    resampled to the image size.
    """
    for path, source in ((patient.saved_mask_path, "saved"), (patient.original_mask_path, "original")):
        if path.exists():
            mask = np.load(path)
            expected = (*image_shape[2:4], N_CHANNELS)
            if mask.ndim != 5 or (mask.shape[2], mask.shape[3], mask.shape[4]) != expected:
                raise ValueError(
                    f"{path.name}: expected mask shape (H, W, {expected[0]}, {expected[1]}, {expected[2]}) "
                    f"to match the image, got {mask.shape}"
                )
            mask = resize_nearest((mask > 0).astype(np.uint8), image_shape[:2])
            return np.ascontiguousarray(mask), source
    return np.zeros((*image_shape, N_CHANNELS), dtype=np.uint8), "empty"


def save_mask(patient: PatientFiles, mask: np.ndarray) -> Path:
    """Write ``mask`` to ``seg_outputs/<id>_mask.npy`` atomically and return the path."""
    out = patient.saved_mask_path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, mask.astype(np.uint8, copy=False))
    os.replace(tmp, out)
    return out


def window_params(volume: np.ndarray) -> tuple[float, float]:
    """1st/99th percentile intensity window, estimated on a subsample of the whole volume."""
    sample = np.asarray(volume[::2, ::2], dtype=np.float32)
    lo, hi = np.percentile(sample, [1, 99])
    if not hi > lo:
        lo, hi = sample.min(), sample.max()
    if not hi > lo:  # constant image
        hi = lo + 1.0
    return float(lo), float(hi)


def apply_window(img: np.ndarray, lo: float, hi: float,
                 brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """Map raw intensities to [0, 1] for display."""
    out = (np.asarray(img, dtype=np.float32) - lo) / (hi - lo)
    out = (np.clip(out, 0, 1) - 0.5) * contrast + 0.5 + brightness
    return np.clip(out, 0, 1, out=out)


def find_image_for_mask(mask_path: str | os.PathLike) -> Path | None:
    """``.../seg_outputs/S_001_mask.npy`` -> ``.../S_001_image.npy`` (also checks the same folder)."""
    mask_path = Path(mask_path)
    pid = mask_path.name[: -len(MASK_SUFFIX)] if mask_path.name.endswith(MASK_SUFFIX) else mask_path.stem
    for folder in (mask_path.parent.parent, mask_path.parent):
        candidate = folder / (pid + IMAGE_SUFFIX)
        if candidate.exists():
            return candidate
    return None
