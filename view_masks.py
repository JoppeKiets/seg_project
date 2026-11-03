"""View an image + mask:  python view_masks.py path/to/S_001_mask.npy [--grid]"""
import sys

from segtool.viewer import main

if __name__ == "__main__":
    sys.exit(main())
