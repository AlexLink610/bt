"""
pointmap_to_ply.py -- Dump a pointmap (.npy, shape N,H,W,3) to a PLY point cloud
for visualization in MeshLab. Optionally color by confidence.

Usage:
    python pointmap_to_ply.py \
        --pointmap ~/ba/output_da3/t02_da3_360_32v_pointmap.npy \
        --out      ~/ba/output_da3/t02_da3_360_32v_cloud.ply

    # color by confidence (blue=low, red=high):
    python pointmap_to_ply.py \
        --pointmap ~/ba/output_da3/t02_da3_360_32v_pointmap.npy \
        --confmap  ~/ba/output_da3/t02_da3_360_32v_confmap.npy \
        --out      ~/ba/output_da3/t02_da3_360_32v_cloud.ply

    # subsample to keep file small (every Nth point):
    python pointmap_to_ply.py --pointmap ... --out ... --stride 4
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap", required=True)
    parser.add_argument("--confmap",  default=None,
                        help="Optional confmap .npy (N,H,W) to color points by confidence.")
    parser.add_argument("--out",      required=True)
    parser.add_argument("--stride",   type=int, default=2,
                        help="Keep every Nth point to reduce file size (default: 2).")
    args = parser.parse_args()

    pm = np.load(args.pointmap)            # (N, H, W, 3)
    print(f"Pointmap shape: {pm.shape}")
    N, H, W, _ = pm.shape

    pts = pm.reshape(-1, 3)

    conf = None
    if args.confmap is not None:
        conf = np.load(args.confmap).reshape(-1)
        print(f"Confmap shape: {conf.shape}")

    # Valid (non-NaN) points
    valid = ~np.isnan(pts).any(axis=1)
    pts = pts[valid]
    if conf is not None:
        conf = conf[valid]

    # Subsample
    if args.stride > 1:
        pts = pts[::args.stride]
        if conf is not None:
            conf = conf[::args.stride]

    print(f"Writing {len(pts):,} points...")

    # Colors
    if conf is not None:
        # Normalize conf to 0-1, map to blue(low)->red(high)
        c = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        r = (c * 255).astype(np.uint8)
        b = ((1 - c) * 255).astype(np.uint8)
        g = np.zeros_like(r)
        colors = np.stack([r, g, b], axis=1)
    else:
        # Uniform light gray
        colors = np.full((len(pts), 3), 200, dtype=np.uint8)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(args.out, "wb") as f:
        f.write(header.encode("ascii"))
        for p, col in zip(pts.astype(np.float32), colors):
            f.write(p.tobytes())
            f.write(bytes(col))

    print(f"Saved: {args.out}  ({len(pts):,} points)")


if __name__ == "__main__":
    main()
