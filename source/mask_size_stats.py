"""
mask_size_stats.py -- Apple size and spacing from ordinary instance masks
(no ground truth needed, e.g. SAM3 masks of a real scene).

For every mask (one apple in one view) it takes the 3D points of the mask and
measures their extent along the main direction (2nd-98th percentile span).
For an unoccluded apple this is about its diameter; occlusion only makes it
smaller, so an upper quantile is used as the size estimate. Within each view
it also measures the distance between the point centroids of different masks
(a lower bound for the spacing of neighbouring apples).

Prints start values for the pipeline thresholds in the scene units of the
given pointmap (heuristics, relative to the estimated apple radius r).

Usage:
    python mask_size_stats.py --pointmap <..._pointmap_head.npy> \
        --filenames <..._filenames.txt> --masks <mask dir>
"""

import os
import argparse
import numpy as np
from PIL import Image

MIN_POINTS = 30


def load_mask(path, W, H):
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def extents(p):
    """Span (2nd-98th percentile) along the first and second principal axis.
    Depth smearing at mask edges stretches the points along the viewing ray
    and takes over the first axis; the second axis then still spans the
    silhouette, i.e. about the apple diameter."""
    c = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    spans = []
    for k in range(2):
        proj = c @ vt[k]
        spans.append(float(np.percentile(proj, 98) - np.percentile(proj, 2)))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    args = ap.parse_args()

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    ext1, ext2, nn = [], [], []
    n_masks = 0
    for i, fn in enumerate(fnames):
        path = os.path.join(args.masks, f"mask_{os.path.splitext(fn)[0]}.png")
        if not os.path.exists(path):
            print(f"  missing {path}")
            continue
        mask = load_mask(path, W, H)
        P = pm[i]
        cents = []
        for m in np.unique(mask):
            if m == 0:
                continue
            pts = P[mask == m]
            pts = pts[~np.isnan(pts).any(axis=1)]
            if len(pts) < MIN_POINTS:
                continue
            n_masks += 1
            e1, e2 = extents(pts)
            ext1.append(e1)
            ext2.append(e2)
            cents.append(pts.mean(axis=0))
        if len(cents) >= 2:
            C = np.array(cents)
            D = np.linalg.norm(C[:, None] - C[None], axis=2)
            np.fill_diagonal(D, np.inf)
            nn.extend(D.min(axis=1).tolist())

    e1, ext = np.array(ext1), np.array(ext2)
    nn = np.array(nn)
    r = float(np.percentile(ext, 75)) / 2
    print(f"masks used: {n_masks} in {len(fnames)} views")
    print(f"extent axis 1 median {np.median(e1):.4f}  p75 {np.percentile(e1, 75):.4f}"
          f"  p90 {np.percentile(e1, 90):.4f}   (includes depth smearing)")
    print(f"extent axis 2 median {np.median(ext):.4f}  p75 {np.percentile(ext, 75):.4f}"
          f"  p90 {np.percentile(ext, 90):.4f}   (~ silhouette diameter)")
    if len(nn):
        print(f"centroid NN   p5 {np.percentile(nn, 5):.4f}  median {np.median(nn):.4f}")
    print(f"estimated apple radius r = p75 axis-2 extent / 2 = {r:.4f}\n")
    print("start values (heuristic, relative to r):")
    print(f"  --corr_3d_thresh   {0.5 * r:.4f}   (0.5 r)")
    print(f"  --sphere_thresh    {0.25 * r:.4f}   (0.25 r)")
    print(f"  --min_radius       {0.5 * r:.4f}   (0.5 r)")
    print(f"  --max_radius_multi {1.5 * r:.4f}   (1.5 r)")
    print(f"  --max_radius_single {2.0 * r:.4f}  (2 r)")
    print(f"  --max_extent_multi {2.6 * r:.4f}   (2.6 r)")
    print(f"  --max_extent_single {3.0 * r:.4f}  (3 r)")
    cl = min(r, 0.8 * np.percentile(nn, 5)) if len(nn) else r
    print(f"  --cluster_dist     {cl:.4f}   (min(r, 0.8 x centroid NN p5))")
    print(f"  filter --radius    {0.35 * r:.4f}   (0.35 r)")


if __name__ == "__main__":
    main()
