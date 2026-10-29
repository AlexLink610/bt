"""
gt_apple_stats.py -- Ground-truth statistics of a VGGT pointmap, per apple.

Uses the Blender ground-truth masks (mask_<stem>.png + map_<stem>.json with
global apple IDs from decode_gt_masks.py) to collect, for every apple, its 3D
points from all views in which it is visible. From these it measures

  * apple radius     robust sphere fit to all points of an apple
  * center spacing   distance from each apple center to the nearest other one
  * consistency      distance of each view's points to the apple's sphere
                     (how well the views agree -> reconstruction error)

and prints start values for the pipeline thresholds. All values are in the
scene units of the given pointmap, so run it once per pointmap variant, e.g.
point head vs. depth+camera, to compare their accuracy.

Usage:
    python gt_apple_stats.py --pointmap  <run>/vggt/..._pointmap_head.npy \
                             --filenames <run>/vggt/..._filenames.txt \
                             --masks     <run>/masks_gt  [--out stats.json]
"""

import os
import json
import argparse
import numpy as np
from PIL import Image

MIN_VIEW_POINTS = 20     # points of one apple in one view to use that view
MIN_APPLE_POINTS = 100   # points of one apple in total to fit a sphere


def load_mask(path, W, H):
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def fit_sphere(p):
    """Algebraic least-squares sphere: |p|^2 = 2 p.c + (r^2 - |c|^2)."""
    A = np.hstack([2.0 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 0.0)))


def robust_sphere(p, iters=3):
    """Least-squares fit, then refit on points within 3 robust sigmas."""
    c, r = fit_sphere(p)
    for _ in range(iters):
        res = np.abs(np.linalg.norm(p - c, axis=1) - r)
        sigma = 1.4826 * np.median(res) + 1e-12
        keep = res < 3.0 * sigma
        if keep.sum() < 10:
            break
        c, r = fit_sphere(p[keep])
    return c, r


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]
    print(f"Pointmap {pm.shape}, {len(fnames)} views")

    # gid -> {view index: points}
    apples = {}
    for i, fn in enumerate(fnames):
        stem = os.path.splitext(fn)[0]
        mpath = os.path.join(args.masks, f"mask_{stem}.png")
        jpath = os.path.join(args.masks, f"map_{stem}.json")
        if not (os.path.exists(mpath) and os.path.exists(jpath)):
            print(f"  missing mask/map for {stem}, skipped")
            continue
        mask = load_mask(mpath, W, H)
        with open(jpath) as f:
            mapping = {int(k): int(v) for k, v in json.load(f).items()}
        P = pm[i]
        for local, gid in mapping.items():
            pts = P[mask == local]
            pts = pts[~np.isnan(pts).any(axis=1)]
            if len(pts) >= MIN_VIEW_POINTS:
                apples.setdefault(gid, {})[i] = pts

    # per apple: sphere from all views + residuals per view
    radii, centers, n_views, view_res = [], [], [], []
    for gid, views in apples.items():
        allp = np.concatenate(list(views.values()))
        n_views.append(len(views))
        if len(allp) < MIN_APPLE_POINTS or len(views) < 2:
            continue
        c, r = robust_sphere(allp)
        radii.append(r)
        centers.append(c)
        for pts in views.values():
            res = np.abs(np.linalg.norm(pts - c, axis=1) - r)
            view_res.append(float(np.median(res)))

    radii = np.array(radii)
    centers = np.array(centers)
    view_res = np.array(view_res)
    n_views = np.array(n_views)

    nn = np.array([])
    if len(centers) >= 2:
        D = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)

    r_hat = float(np.median(radii)) if len(radii) else float("nan")
    stats = {
        "apples_with_points": len(apples),
        "apples_fitted": int(len(radii)),
        "views_per_apple": {"median": pct(n_views, 50), "min": int(n_views.min()),
                            "max": int(n_views.max())} if len(n_views) else {},
        "radius": {"median": r_hat, "p10": pct(radii, 10), "p90": pct(radii, 90)},
        "center_nn_distance": {"p5": pct(nn, 5), "median": pct(nn, 50)},
        "view_residual": {"median": pct(view_res, 50), "p90": pct(view_res, 90)},
    }

    # start values (heuristics, to be tuned against the GT count)
    res90 = stats["view_residual"]["p90"]
    nn5 = stats["center_nn_distance"]["p5"]
    stats["suggested_start_values"] = {
        "tau_corr (2 x view residual p90, <= 0.5 r)": min(2 * res90, 0.5 * r_hat),
        "sphere_delta (view residual p90)": res90,
        "r_min (0.5 r)": 0.5 * r_hat,
        "r_max (1.5 r)": 1.5 * r_hat,
        "d_cl upper bound (center NN p5)": nn5,
        "d_cl start (min(r, 0.8 NN p5))": min(r_hat, 0.8 * nn5),
    }

    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
