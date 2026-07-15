"""
visualize_conf_filter.py  --  Show which apple pixels survive confidence filtering.

For each --conf_thresh value, saves a PLY where:
  GREEN  = apple pixel that PASSES the threshold (used in correspondence)
  RED    = apple pixel that FAILS  the threshold (excluded)

Load all PLYs in MeshLab side by side to see what gets cut at each level.

Usage:
    python3 visualize_conf_filter.py \
        --pointmap  ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --confmap   ~/ba/output_vggt/t02_360_64v_confmap.npy \
        --filenames ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --out_dir   ~/ba/output_vggt/conf_vis \
        --thresholds 0.0 0.1 0.2 0.3 0.4 0.5 0.6
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


def save_ply(points, colors, path):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(points.astype(np.float32), colors.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(c))
    print(f"  Saved {len(points):,} points → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",    required=True)
    parser.add_argument("--confmap",     required=True)
    parser.add_argument("--filenames",   required=True)
    parser.add_argument("--masks",       required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--thresholds",  type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading pointmap...")
    point_map = np.load(args.pointmap)   # (N, H, W, 3)
    N, H, W, _ = point_map.shape
    print(f"  Shape: {point_map.shape}")

    print("Loading confmap...")
    conf_map = np.load(args.confmap)     # (N, H, W)
    pcts = np.percentile(conf_map, [10, 25, 50, 75, 90])
    print(f"  Conf percentiles [10,25,50,75,90]: {pcts.round(3)}")

    print("Loading filenames...")
    with open(args.filenames) as f:
        filenames = [l.strip() for l in f if l.strip()]

    print(f"Loading {N} masks...")
    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        masks.append(load_mask(mpath, W, H))
    loaded = sum(1 for m in masks if m is not None)
    print(f"  Loaded {loaded}/{N} masks")

    # --- collect all apple pixels once across all images ----------------------
    print("\nCollecting apple pixels...")
    all_pts   = []   # (K, 3)
    all_confs = []   # (K,)

    for i in range(N):
        if masks[i] is None:
            continue
        apple_mask = masks[i].reshape(-1) > 0          # (H*W,)
        pts_flat   = point_map[i].reshape(-1, 3)        # (H*W, 3)
        conf_flat  = conf_map[i].reshape(-1)            # (H*W,)

        valid = apple_mask & ~np.isnan(pts_flat).any(axis=1)
        if valid.sum() == 0:
            continue

        all_pts.append(pts_flat[valid])
        all_confs.append(conf_flat[valid])

    all_pts   = np.concatenate(all_pts,   axis=0)   # (K, 3)
    all_confs = np.concatenate(all_confs, axis=0)   # (K,)
    print(f"  Total apple points: {len(all_pts):,}")

    # --- for each threshold, color and save -----------------------------------
    GREEN = np.array([0,   220,  80], dtype=np.uint8)
    RED   = np.array([220,  30,  30], dtype=np.uint8)

    for thresh in args.thresholds:
        keep = all_confs >= thresh
        n_keep = keep.sum()
        n_drop = (~keep).sum()
        pct_kept = 100 * n_keep / max(len(keep), 1)
        print(f"\nthresh={thresh:.2f}:  kept={n_keep:,} ({pct_kept:.1f}%)  dropped={n_drop:,}")

        colors = np.where(keep[:, None], GREEN, RED)   # (K, 3)

        fname = os.path.join(args.out_dir, f"conf_vis_{thresh:.2f}.ply")
        save_ply(all_pts, colors, fname)

    print(f"\nDone. Open all PLYs in MeshLab from {args.out_dir}")


if __name__ == "__main__":
    main()
