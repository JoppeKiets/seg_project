"""Clean saved masks:  python process_masks.py segmentation_ivana/seg_outputs [-o OUT | --in-place]"""
import sys

from segtool.postprocess import main

if __name__ == "__main__":
    sys.exit(main())
