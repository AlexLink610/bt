"""
filter_pointmap.py -- Apply radius/statistical outlier removal to the APPLE points
of a pointmap, writing a new pointmap (.npy) where rejected points are set to NaN.

Unlike filter_pointcloud.py (which outputs a flat .ply), this preserves the
(N,H,W,3) structure, so the filtered pointmap drops straight into
count_spheres.py / associate_masks_graph.py with no other changes --
those scripts already skip NaN points.

Only apple-masked points are considered/filtered; non-apple pixels are passed
through untouched (they're ignored downstream anyway).

Usage:
    python filter_pointmap.py \
        --pointmap  ~/ba/output_vggt/old/t02_360_32v_pointmap.npy \
        --filenames ~/ba/output_vggt/old/t02_360_32v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --out       ~/ba/output_vggt/old/t02_360_32v_pointmap_r12.npy \
        --method radius --nb_points 12 --radius 0.0076

Then count as usual, pointing --pointmap at the filtered file:
    python count_spheres.py --pointmap ..._pointmap_r12.npy --filenames ... --masks ...
"""

import os
import argparse
import numpy as np
from PIL import Image

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d not installed. Run:  pip install open3d")


def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",  required=True)
    parser.add_argument("--filenames", required=True)
    parser.add_argument("--masks",     required=True)
    parser.add_argument("--out",       required=True)
    parser.add_argument("--method", default="radius",
                        choices=["radius", "statistical", "both"])
    parser.add_argument("--nb_points",    type=int,   default=12,
                        help="radius: min neighbours within --radius (default: 12)")
    parser.add_argument("--radius",       type=float, default=0.0076,
                        help="radius: neighbourhood radius in scene units")
    parser.add_argument("--nb_neighbors", type=int,   default=20,
                        help="statistical: k nearest neighbours")
    parser.add_argument("--std_ratio",    type=float, default=2.0,
                        help="statistical: std-dev multiplier")
    parser.add_argument("--save_ply", default=None,
                        help="Optional: also save the kept apple points as a .ply")
    args = parser.parse_args()

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    print(f"Pointmap: {pm.shape}")

    with open(args.filenames) as f:
        filenames = [l.strip() for l in f if l.strip()]

    # --- Gather apple points and remember where each came from ---------------
    pts_list, idx_list = [], []   # idx = flat index into (N*H*W)
    for i, fname in enumerate(filenames):
        mask = load_mask(mask_path_for(args.masks, fname), W, H)
        if mask is None:
            continue
        apple = mask.reshape(-1) > 0
        p = pm[i].reshape(-1, 3)
        valid = apple & ~np.isnan(p).any(axis=1)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        pts_list.append(p[idx])
        idx_list.append(idx + i * H * W)   # global flat index

    if not pts_list:
        raise SystemExit("No apple points found -- check masks path/format.")

    pts = np.concatenate(pts_list, axis=0)
    gidx = np.concatenate(idx_list, axis=0)
    n0 = len(pts)
    print(f"Apple points: {n0:,}")

    # --- Filter --------------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    keep = np.arange(n0)

    if args.method in ("statistical", "both"):
        before = len(keep)
        _, sel = pcd.remove_statistical_outlier(
            nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
        keep = keep[sel]
        pcd = pcd.select_by_index(sel)
        print(f"Statistical (k={args.nb_neighbors}, std={args.std_ratio}): "
              f"{before:,} -> {len(keep):,}  "
              f"(removed {before-len(keep):,}, {100*(before-len(keep))/before:.1f}%)")

    if args.method in ("radius", "both"):
        before = len(keep)
        _, sel = pcd.remove_radius_outlier(
            nb_points=args.nb_points, radius=args.radius)
        keep = keep[sel]
        pcd = pcd.select_by_index(sel)
        print(f"Radius (nb_points={args.nb_points}, radius={args.radius}): "
              f"{before:,} -> {len(keep):,}  "
              f"(removed {before-len(keep):,}, {100*(before-len(keep))/before:.1f}%)")

    n1 = len(keep)
    print(f"\nTotal apple points: {n0:,} -> {n1:,}  "
          f"(removed {n0-n1:,}, {100*(n0-n1)/n0:.1f}%)")

    # --- Write filtered pointmap (rejected apple points -> NaN) --------------
    removed_global = np.setdiff1d(gidx, gidx[keep], assume_unique=False)
    pm_out = pm.copy()
    flat = pm_out.reshape(-1, 3)
    flat[removed_global] = np.nan
    pm_out = flat.reshape(N, H, W, 3)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, pm_out.astype(np.float32))
    print(f"Saved filtered pointmap: {args.out}")

    if args.save_ply:
        kept_pts = pts[keep].astype(np.float32)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(kept_pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        with open(args.save_ply, "wb") as f:
            f.write(header.encode("ascii"))
            for p in kept_pts:
                f.write(p.tobytes())
        print(f"Saved kept-points PLY: {args.save_ply}  ({len(kept_pts):,} points)")


if __name__ == "__main__":
    main()
