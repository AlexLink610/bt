"""
filter_conf.py -- Confidence-based alternative to the radius outlier filter.

Marks the apple pixels with the lowest VGGT confidence as invalid (NaN):
the threshold is the given quantile of the confidence over all apple pixels
of all views, so --drop 0.25 removes the least confident 25 %. Other pixels
are left untouched, exactly like filter_pointmap.py.

Usage:
    python filter_conf.py --pointmap <..._pointmap_head.npy>
        --conf <..._pointconf_raw.npy> --filenames <..._filenames.txt>
        --masks <masks dir> --drop 0.25 --out <..._head_conf25.npy>
"""

import os
import argparse
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--drop", type=float, required=True,
                    help="fraction of apple pixels to remove (least confident)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pm = np.load(args.pointmap)
    conf = np.load(args.conf)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    apple = np.zeros((N, H, W), bool)
    for i, fn in enumerate(fnames):
        path = os.path.join(args.masks, f"mask_{os.path.splitext(fn)[0]}.png")
        if os.path.exists(path):
            m = np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))
            apple[i] = m > 0
    valid = apple & ~np.isnan(pm).any(-1)
    thr = np.quantile(conf[valid], args.drop)
    remove = valid & (conf < thr)

    out = pm.copy()
    out[remove] = np.nan
    np.save(args.out, out)
    print(f"Total apple points: {valid.sum():,} -> {(valid & ~remove).sum():,}  "
          f"(removed {remove.sum():,}, {remove.sum() / valid.sum():.1%}; "
          f"confidence threshold {thr:.3f})")


if __name__ == "__main__":
    main()
