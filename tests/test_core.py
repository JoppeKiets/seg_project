"""Unit tests for the non-GUI parts.  Run:  python -m pytest"""

import numpy as np
import pytest

from segtool import geometry, postprocess
from segtool.data import (PatientFiles, apply_window, discover_patients, find_image_for_mask,
                          load_mask_for, save_mask, window_params)


# ---------------------------------------------------------------- geometry

def test_rasterize_square():
    m = geometry.rasterize_polygon([(2, 2), (6, 2), (6, 6), (2, 6)], (10, 10))
    assert m.dtype == np.uint8
    assert m[4, 4] == 1 and m[0, 0] == 0 and m[8, 8] == 0
    assert 16 <= m.sum() <= 25


def test_rasterize_degenerate_and_offscreen():
    assert geometry.rasterize_polygon([(1, 1), (2, 2)], (5, 5)).sum() == 0
    assert geometry.rasterize_polygon([(50, 50), (60, 50), (60, 60)], (5, 5)).sum() == 0


def test_rasterize_clipped_to_image():
    m = geometry.rasterize_polygon([(-5, -5), (20, -5), (20, 20), (-5, 20)], (8, 8))
    assert m.sum() == 64


@pytest.mark.parametrize("mode,expected", [("New", [0, 1]), ("Add", [1, 1]), ("Erase", [1, 0])])
def test_apply_region(mode, expected):
    ch = np.array([1, 1], dtype=np.uint8) if mode == "Erase" else np.array([1, 0], dtype=np.uint8)
    region = np.array([0, 1], dtype=np.uint8)
    geometry.apply_region(ch, region, mode)
    assert ch.tolist() == expected


def test_apply_region_writes_through_views():
    vol = np.zeros((4, 4, 2, 3), dtype=np.uint8)
    view = vol[:, :, 1, 2]
    geometry.apply_region(view, np.ones((4, 4), dtype=np.uint8), "Add")
    assert vol[:, :, 1, 2].all() and not vol[:, :, 0].any()


def test_paint_stroke_has_no_gaps():
    ch = np.zeros((20, 40), dtype=np.uint8)
    geometry.paint_stroke(ch, (2, 10), (37, 10), radius=1.5, value=1)
    assert ch[10, 2:38].all()
    geometry.paint_stroke(ch, None, (20, 10), radius=3, value=0)
    assert ch[10, 18:23].sum() == 0 and ch[10, 5] == 1


def test_paint_stroke_outside_image_is_noop():
    ch = np.zeros((5, 5), dtype=np.uint8)
    geometry.paint_stroke(ch, (50, 50), (60, 60), radius=2, value=1)
    assert ch.sum() == 0


def test_nearest_point_and_edge():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    assert geometry.nearest_point(pts, (9, 1), 3) == 1
    assert geometry.nearest_point(pts, (5, 5), 3) is None
    assert geometry.nearest_edge_insert_index(pts, (5, 1), 3) == 1      # between 0 and 1
    assert geometry.nearest_edge_insert_index(pts, (1, 5), 3) == 4      # closing edge 3 -> 0
    assert geometry.nearest_edge_insert_index(pts, (1, 5), 3, closed=False) is None


def test_boundary_pixels():
    m = np.zeros((7, 7), dtype=np.uint8)
    m[1:6, 1:6] = 1
    edge = geometry.boundary_pixels(m)
    assert len(edge) == 16 and [3, 3] not in edge.tolist()
    assert len(geometry.boundary_pixels(np.zeros((3, 3)))) == 0


# ---------------------------------------------------------------- data

def make_patient(tmp_path, pid="S_001", shape=(16, 16, 3, 2), with_mask=True):
    img = np.random.default_rng(0).random(shape).astype(np.float32) * 100
    np.save(tmp_path / f"{pid}_image.npy", img)
    if with_mask:
        mask = np.zeros((*shape, 3), dtype=np.uint8)
        mask[4:8, 4:8, 1, 0, 2] = 1
        np.save(tmp_path / f"{pid}_mask.npy", mask)
    return img


def test_discover_and_mask_priority(tmp_path):
    img = make_patient(tmp_path)
    make_patient(tmp_path, "S_002", with_mask=False)
    patients = discover_patients(tmp_path)
    assert list(patients) == ["S_001", "S_002"]

    pf = patients["S_001"]
    mask, source = load_mask_for(pf, img.shape)
    assert source == "original" and mask[5, 5, 1, 0, 2] == 1

    mask[0, 0, 0, 0, 0] = 1
    out = save_mask(pf, mask)
    assert out == tmp_path / "seg_outputs" / "S_001_mask.npy" and pf.has_saved_mask
    assert not list((tmp_path / "seg_outputs").glob("*.tmp*"))
    again, source = load_mask_for(pf, img.shape)
    assert source == "saved" and np.array_equal(again, mask)
    # the saved output folder is not mistaken for a patient folder
    assert list(discover_patients(tmp_path)) == ["S_001", "S_002"]

    empty, source = load_mask_for(patients["S_002"], img.shape)
    assert source == "empty" and empty.shape == (*img.shape, 3) and not empty.any()


def test_old_256_masks_are_resampled(tmp_path):
    img = make_patient(tmp_path, shape=(32, 32, 2, 2), with_mask=False)
    big = np.zeros((64, 64, 2, 2, 3), dtype=np.uint8)
    big[16:48, 16:48, 0, 1, 0] = 1
    np.save(tmp_path / "S_001_mask.npy", big)
    mask, _ = load_mask_for(PatientFiles("S_001", tmp_path / "S_001_image.npy"), img.shape)
    assert mask.shape == (32, 32, 2, 2, 3) and mask[16, 16, 0, 1, 0] == 1 and mask[2, 2, 0, 1, 0] == 0


def test_mismatched_mask_is_rejected(tmp_path):
    img = make_patient(tmp_path, with_mask=False)
    np.save(tmp_path / "S_001_mask.npy", np.zeros((16, 16, 5, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected mask shape"):
        load_mask_for(PatientFiles("S_001", tmp_path / "S_001_image.npy"), img.shape)


def test_find_image_for_mask(tmp_path):
    make_patient(tmp_path)
    (tmp_path / "seg_outputs").mkdir()
    assert find_image_for_mask(tmp_path / "seg_outputs" / "S_001_mask.npy") == tmp_path / "S_001_image.npy"
    assert find_image_for_mask(tmp_path / "S_001_mask.npy") == tmp_path / "S_001_image.npy"


def test_window():
    vol = np.linspace(0, 100, 4 * 4 * 2 * 2, dtype=np.float32).reshape(4, 4, 2, 2)
    lo, hi = window_params(vol)
    assert 0 <= lo < hi <= 100
    out = apply_window(vol[..., 0, 0], lo, hi)
    assert out.min() >= 0 and out.max() <= 1
    assert window_params(np.ones((4, 4, 1, 1))) == (1.0, 2.0)  # constant image doesn't divide by 0


# ---------------------------------------------------------------- postprocess

def test_clean_mask_removes_islands_and_resolves_overlap():
    m = np.zeros((40, 40, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:40, :40]
    disk = lambda r: (np.hypot(xx - 20, yy - 20) <= r).astype(np.uint8)
    m[..., 1] = disk(12)          # filled epicardium
    m[..., 2] = disk(7)           # LV
    m[2, 2, 1] = 1                # stray island
    out = postprocess.clean_mask(m)
    assert out[2, 2, 1] == 0                      # island removed
    assert out[20, 20, 2] == 1 and out[20, 20, 1] == 0   # LV wins; myocardium becomes a ring
    assert out[20, 30, 1] == 1
    assert not (out[..., 1] & out[..., 2]).any()


def test_clean_mask_5d_shape():
    m = np.zeros((10, 10, 2, 3, 3), dtype=np.uint8)
    assert postprocess.clean_mask(m).shape == m.shape
    with pytest.raises(ValueError):
        postprocess.clean_mask(np.zeros((10, 10, 2)))


def test_process_folder_cli(tmp_path):
    src = tmp_path / "seg_outputs"
    src.mkdir()
    m = np.zeros((10, 10, 1, 1, 3), dtype=np.uint8)
    m[2:8, 2:8, 0, 0, 0] = 1
    np.save(src / "S_001_mask.npy", m)
    assert postprocess.main([str(src)]) == 0
    assert (tmp_path / "seg_outputs_cleaned" / "S_001_mask.npy").exists()
    assert np.array_equal(np.load(src / "S_001_mask.npy"), m)   # input untouched by default
