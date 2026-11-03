# Cardiac MRI Segmentation Tool

A desktop tool for drawing and correcting RV, epicardium and endocardium segmentations on
cine cardiac MRI stacks. It also includes scripts to clean the masks and to view them.

## Setup

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib (+ pytest for the tests)
```

Tkinter comes with most Python installs. On Linux you may need `sudo apt install python3-tk`.

## Data layout

```
data_folder/
  S_001_image.npy        (H, W, Phase, Slice)       float image
  S_001_mask.npy         (H, W, Phase, Slice, 3)    original mask, channels = RV, Epicardium/Myo, Endocardium/LV
  seg_outputs/
    S_001_mask.npy       your edited mask (same layout), written by the GUI
```

The original `*_mask.npy` files are never overwritten. When `seg_outputs/<id>_mask.npy` exists,
the GUI loads it instead of the original.

## 1. Segmentation GUI

```bash
python seg.py                       # reopens the last folder you used
python seg.py path/to/data_folder
```

* Pick a patient on the left. A ✓ means the patient already has saved edits.
* Pick a structure (`1` `2` `3`) and a tool: polygon (`X`), lasso (`C`) or brush (`Z`).
* Move through slices with `W`/`S` (or ↑/↓) and through frames with `A`/`D` (or ←/→), or use the sliders under the image.
* Save with `Ctrl/Cmd+S`. The app asks before you switch patients or quit with unsaved changes.
* Press **F1** to see every shortcut.

Zoom (scroll wheel), pan (middle-drag or Alt/Option+drag), rotation, brightness and contrast
only change what you see. Masks always stay in the image's own pixel grid.

## 2. Clean up masks

```bash
python process_masks.py data_folder/seg_outputs            # writes data_folder/seg_outputs_cleaned/
python process_masks.py data_folder/seg_outputs -o out_dir
python process_masks.py data_folder/seg_outputs --in-place
```

Each structure in each slice and frame goes through these steps in order:

1. Keep only the largest connected region.
2. Apply a 3×3 morphological closing, then a 3×3 opening.
3. Resolve overlaps with the priority LV > RV > myocardium.

Step 3 turns a filled epicardium into a myocardium ring.

## 3. View masks

```bash
python view_masks.py data_folder/seg_outputs/S_001_mask.npy        # sliders + structure toggles
python view_masks.py data_folder/S_001_image.npy --grid            # every slice, two phases
python view_masks.py data_folder/S_001_image.npy --grid --phases 1 10
```

## Tests

```bash
python -m pytest
```

## Code layout

| Path | Contents |
|---|---|
| `segtool/config.py` | Structure names, colours, channels, shortcuts and tuning constants |
| `segtool/data.py` | Finding patients, loading and saving masks, intensity windowing |
| `segtool/geometry.py` | Polygon filling, brush strokes and hit-testing (no GUI code) |
| `segtool/app.py` | The Tkinter GUI |
| `segtool/postprocess.py` | Mask cleaning (`process_masks.py`) |
| `segtool/viewer.py` | Viewers (`view_masks.py`) |
| `legacy/dicom_seg_gui.py` | Older DICOM-based tool with 6 contours that saves NPZ files. It is kept unchanged and needs `pydicom` and `pandas`. |

Patient data (`*.npy`, `segmentation_ivana/`) is listed in `.gitignore`. Don't commit it.
