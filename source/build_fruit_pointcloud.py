"""
build_fruit_pointcloud.py -- Fuse per-view pointmaps + instance masks into one
apple-only point cloud (.ply), suitable as input to FruitNeRF's clustering
(clustering/run_clustering.py) or your own cascaded clustering.

Pools every apple-masked, non-NaN 3D point from all views into one cloud.
Optionally colors by source RGB (for inspection) -- FruitNeRF's clustering
reads geometry only, so color is cosmetic.

Usage:
    python build_fruit_pointcloud.py \
        --pointmap  ~/ba/output_vggt/t02_360_122v_pointmap.npy \
        --filenames ~/ba/output_vggt/t02_360_122v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --out       ~/ba/output_vggt/t02_360_122v_fruit.ply

Optional:
    --confmap        confmap .npy (N,H,W); with --conf_thresh, drop low-conf points
    --conf_thresh    keep points with normalized conf >= this (default: none)
    --image_dir      color points by true RGB from source images (else uniform red)
    --voxel          voxel-downsample size in scene units (default: 0 = off)
"""

import os
import argparse
import numpy as np
from PIL import Image


def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def voxel_downsample(pts, cols, voxel):
    """Simple voxel-grid downsample: one point (centroid) per occupied voxel."""
    if voxel <= 0 or len(pts) == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    # unique voxel -> first occurrence (fast; centroid would be nicer but slower)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return pts[idx], cols[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",  required=True)
    parser.add_argument("--filenames", required=True)
    parser.add_argument("--masks",     required=True)
    parser.add_argument("--out",       required=True)
    parser.add_argument("--confmap",   default=None)
    parser.add_argument("--conf_thresh", type=float, default=None)
    parser.add_argument("--image_dir", default=None,
                        help="Color points by true RGB from source images.")
    parser.add_argument("--voxel",     type=float, default=0.0,
                        help="Voxel downsample size in scene units (0 = off).")
    args = parser.parse_args()

    pm = np.load(args.pointmap)              # (N,H,W,3)
    N, H, W, _ = pm.shape
    print(f"Pointmap: {pm.shape}")

    with open(args.filenames) as f:
        filenames = [l.strip() for l in f if l.strip()]

    conf = None
    if args.confmap is not None:
        conf = np.load(args.confmap)         # (N,H,W)
        cmin, cmax = conf.min(), conf.max()

    all_pts, all_cols = [], []
    total_apple_px = 0

    for i, fname in enumerate(filenames):
        mask = load_mask(mask_path_for(args.masks, fname), W, H)
        if mask is None:
            continue
        apple = mask.reshape(-1) > 0
        pts = pm[i].reshape(-1, 3)
        valid = apple & ~np.isnan(pts).any(axis=1)

        if conf is not None and args.conf_thresh is not None:
            cn = (conf[i].reshape(-1) - cmin) / (cmax - cmin + 1e-8)
            valid &= (cn >= args.conf_thresh)

        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        total_apple_px += idx.size
        pts_v = pts[idx]

        # Colors
        if args.image_dir is not None:
            ipath = os.path.join(args.image_dir, fname)
            if not os.path.exists(ipath):
                stem = os.path.splitext(fname)[0]
                cands = [c for c in os.listdir(args.image_dir)
                         if os.path.splitext(c)[0] == stem]
                ipath = os.path.join(args.image_dir, cands[0]) if cands else None
            if ipath and os.path.exists(ipath):
                img = np.array(Image.open(ipath).convert("RGB").resize((W, H), Image.NEAREST))
                cols_v = img.reshape(-1, 3)[idx]
            else:
                cols_v = np.tile([255, 0, 0], (len(pts_v), 1))
        else:
            cols_v = np.tile([255, 0, 0], (len(pts_v), 1))   # uniform red

        all_pts.append(pts_v)
        all_cols.append(cols_v.astype(np.uint8))

    if not all_pts:
        print("No apple points found -- check masks path / format.")
        return

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    print(f"Fused apple points: {len(pts):,} (from {total_apple_px:,} masked pixels)")

    if args.voxel > 0:
        before = len(pts)
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        print(f"Voxel downsample ({args.voxel}): {before:,} -> {len(pts):,}")

    # Report extent (helps set FruitNeRF eps / outlier radius for your scale)
    print(f"Extent: X[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
          f"Y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
          f"Z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(pts.astype(np.float32), cols.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(c))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
