#!/usr/bin/env python3
"""
Visualize an instance-mask PNG by mapping each integer ID to a bright, distinct color.

Usage:
    python visualize_mask.py cam00_mask.png
    python visualize_mask.py cam00_mask.png -o out.png     # choose output name
    python visualize_mask.py "renders/*_mask*.png"          # glob many at once
"""

import sys
import os
import glob
import argparse
import colorsys
import numpy as np
from PIL import Image


def id_to_colors(max_id):
    """Distinct bright RGB per id; id 0 -> black (background)."""
    colors = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for i in range(1, max_id + 1):
        h = (i * 0.61803398875) % 1.0        # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def visualize(path, out=None):
    m = np.array(Image.open(path))
    if m.ndim == 3:                          # collapse RGB(A) -> single channel
        m = m[..., 0]
    m = np.rint(m).astype(int)

    ids = np.unique(m)
    print(f"{os.path.basename(path)}: shape={m.shape} ids={ids.tolist()}")

    lut = id_to_colors(int(m.max()) if m.max() > 0 else 0)
    rgb = lut[m]                             # fancy-index each pixel to its color

    if out is None:
        base, _ = os.path.splitext(path)
        out = base + "_vis.png"
    Image.fromarray(rgb, "RGB").save(out)
    print(f"  -> {out}  ({len(ids) - (1 if 0 in ids else 0)} instances)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="mask png(s) or glob pattern")
    ap.add_argument("-o", "--out", default=None, help="output file (single input only)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    if not files:
        sys.exit("No files matched.")
    if args.out and len(files) > 1:
        sys.exit("-o only works with a single input file.")

    for f in files:
        visualize(f, args.out)


if __name__ == "__main__":
    main()
