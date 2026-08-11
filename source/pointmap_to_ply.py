"""
pointmap_to_ply.py -- Dump a pointmap (.npy, shape N,H,W,3) to a PLY point cloud.

Color modes (pick one):
  --image_dir DIR   color each point with the RGB of its source image pixel (true color)
  --confmap FILE    color by confidence (blue=low, red=high)
  (neither)         uniform gray

Usage (true RGB):
    python pointmap_to_ply.py \
        --pointmap  table_naive_5v_pointmap_r12.npy \
        --filenames table_naive_5v_filenames.txt \
        --image_dir ~/ba/data/table \
        --out       table_r12_apples_rgb.ply --stride 1
"""

import os
import argparse
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",  required=True)
    parser.add_argument("--filenames", default=None,
                        help="Filenames .txt (one per line); required for --image_dir.")
    parser.add_argument("--image_dir", default=None,
                        help="Dir with source images; color points by true RGB.")
    parser.add_argument("--confmap",   default=None,
                        help="Confmap .npy (N,H,W); color by confidence "
                             "(ignored if --image_dir given).")
    parser.add_argument("--conf_thresh", type=float, default=None,
                        help="Drop points with normalized confidence below this "
                             "(requires --confmap).")
    parser.add_argument("--out",       required=True)
    parser.add_argument("--stride",    type=int, default=2,
                        help="Keep every Nth point (default: 2).")
    args = parser.parse_args()

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    print(f"Pointmap shape: {pm.shape}")

    rgb_mode = args.image_dir is not None

    colors_all = None
    if rgb_mode:
        if args.filenames is None:
            raise SystemExit("--image_dir requires --filenames")
        with open(args.filenames) as f:
            fnames = [l.strip() for l in f if l.strip()]
        if len(fnames) != N:
            print(f"WARNING: {len(fnames)} filenames but pointmap N={N}")
        color_stack = np.zeros((N, H, W, 3), dtype=np.uint8)
        for i, fn in enumerate(fnames):
            ipath = os.path.join(args.image_dir, fn)
            if not os.path.exists(ipath):
                stem = os.path.splitext(fn)[0]
                cands = [c for c in os.listdir(args.image_dir)
                         if os.path.splitext(c)[0] == stem]
                if cands:
                    ipath = os.path.join(args.image_dir, cands[0])
                else:
                    print(f"  missing image for {fn}, using gray")
                    color_stack[i] = 128
                    continue
            img = Image.open(ipath).convert("RGB").resize((W, H), Image.NEAREST)
            color_stack[i] = np.array(img)
        colors_all = color_stack.reshape(-1, 3)
        print("Coloring by true RGB from source images.")

    conf_all = None
    if args.confmap is not None:
        conf_all = np.load(args.confmap).reshape(-1)

    pts = pm.reshape(-1, 3)
    valid = ~np.isnan(pts).any(axis=1)
    if args.conf_thresh is not None and conf_all is not None:
        cnorm = (conf_all - conf_all.min()) / (conf_all.max() - conf_all.min() + 1e-8)
        valid &= (cnorm >= args.conf_thresh)
        print(f"Confidence threshold {args.conf_thresh}: {valid.sum():,} kept")

    pts = pts[valid]

    if rgb_mode:
        colors = colors_all[valid]
    elif args.confmap is not None:
        c = conf_all[valid]
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        colors = np.stack([(c * 255).astype(np.uint8),
                           np.zeros(len(c), np.uint8),
                           ((1 - c) * 255).astype(np.uint8)], axis=1)
    else:
        colors = np.full((len(pts), 3), 200, dtype=np.uint8)

    if args.stride > 1:
        pts = pts[::args.stride]
        colors = colors[::args.stride]

    print(f"Writing {len(pts):,} points...")
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
        for p, col in zip(pts.astype(np.float32), colors.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(col))
    print(f"Saved: {args.out}  ({len(pts):,} points)")


if __name__ == "__main__":
    main()
