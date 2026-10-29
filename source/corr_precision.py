"""
corr_precision.py -- How reliable are pointmap correspondences, per threshold?

For every pair of views (i < j) and every valid apple pixel of view i, the
nearest apple point of view j is found (as in the association). With the
Blender ground truth (map_<stem>.json: local mask ID -> global apple ID) each
correspondence is either correct (same apple) or wrong (different apple).
For a list of thresholds tau it reports

  precision  share of accepted correspondences (distance < tau) that land on
             the same apple        -> low precision = merge errors
  recall     share of apple pixels whose apple is also visible in view j that
             get a correct correspondence   -> low recall = split errors
             (the denominator counts all mask pixels, including points removed
             by an outlier filter, so that recall is comparable across filters)

This isolates the quality of the reconstruction from the rest of the pipeline;
run it once per pointmap (point head / depth+camera) to compare them.

Usage:
    python corr_precision.py --pointmap <..._pointmap_head.npy> \
        --filenames <..._filenames.txt> --masks <run>/masks_gt \
        [--taus 0.005 0.01 0.015 0.02 0.03] [--out table.csv]
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def gid_image(masks_dir, stem, W, H):
    mpath = os.path.join(masks_dir, f"mask_{stem}.png")
    jpath = os.path.join(masks_dir, f"map_{stem}.json")
    if not (os.path.exists(mpath) and os.path.exists(jpath)):
        return None
    mask = np.array(Image.open(mpath).convert("L").resize((W, H), Image.NEAREST))
    with open(jpath) as f:
        mapping = {int(k): int(v) for k, v in json.load(f).items()}
    lut = np.zeros(256, np.int64)
    for local, g in mapping.items():
        lut[local] = g
    return lut[mask].reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.04])
    ap.add_argument("--out", default=None)
    ap.add_argument("--views", default=None,
                    help="txt file with image names: evaluate only pairs among these "
                         "views (e.g. the views shared by several runs)")
    args = ap.parse_args()

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    only = None
    if args.views:
        with open(args.views) as f:
            only = {os.path.basename(l.strip()) for l in f if l.strip()}

    views = []
    for i, fn in enumerate(fnames):
        if only is not None and os.path.basename(fn) not in only:
            views.append(None)
            continue
        g = gid_image(args.masks, os.path.splitext(fn)[0], W, H)
        p = pm[i].reshape(-1, 3)
        if g is None:
            views.append(None)
            continue
        valid = (g > 0) & ~np.isnan(p).any(axis=1)
        # third entry: apple IDs of ALL mask pixels, including those removed
        # by a filter, so that recall stays comparable across filters
        views.append((p[valid], g[valid], g[g > 0]))

    taus = sorted(args.taus)
    n_corr = np.zeros(len(taus))
    n_same = np.zeros(len(taus))
    n_visible = 0
    for i in range(N):
        if views[i] is None or len(views[i][1]) == 0:
            continue
        pi, gi, gi_all = views[i]
        for j in range(i + 1, N):
            if views[j] is None or len(views[j][1]) == 0:
                continue
            pj, gj, gj_all = views[j]
            d, nn = cKDTree(pj).query(pi, workers=-1)
            same = gi == gj[nn]
            # denominator: all apple pixels of view i (filtered ones too)
            # whose apple appears in the masks of view j
            n_visible += int(np.isin(gi_all, np.unique(gj_all)).sum())
            for k, t in enumerate(taus):
                m = d < t
                n_corr[k] += m.sum()
                n_same[k] += (m & same).sum()

    print(f"{'tau':>8} {'precision':>10} {'recall':>8} {'corr/pixel':>11}")
    rows = []
    for k, t in enumerate(taus):
        prec = n_same[k] / max(n_corr[k], 1)
        rec = n_same[k] / max(n_visible, 1)
        rows.append((t, prec, rec))
        print(f"{t:8.4f} {prec:10.3f} {rec:8.3f} {n_corr[k] / max(n_visible, 1):11.3f}")
    if args.out:
        with open(args.out, "w") as f:
            f.write("tau,precision,recall\n")
            for t, p, r in rows:
                f.write(f"{t},{p:.4f},{r:.4f}\n")


if __name__ == "__main__":
    main()
