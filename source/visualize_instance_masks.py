"""
visualize_instance_masks.py -- Vividly color SAM3 instance masks.

Two modes:
  --mode 2d   overlay colored instances on each RGB image -> PNGs
  --mode 3d   color the 3D apple points by instance -> single PLY
              (needs --pointmap + --filenames)

Usage (2D overlays):
    python visualize_instance_masks.py --mode 2d \
        --masks     ~/ba/output_sam/table \
        --image_dir ~/ba/data/table \
        --out_dir   ~/ba/output_vggt/old/table/mask_vis

Usage (3D instance-colored cloud):
    python visualize_instance_masks.py --mode 3d \
        --masks     ~/ba/output_sam/table \
        --pointmap  ~/ba/output_vggt/old/table/table_naive_5v_pointmap.npy \
        --filenames ~/ba/output_vggt/old/table/table_naive_5v_filenames.txt \
        --out       ~/ba/output_vggt/old/table/table_masks_3d.ply
"""

import os
import argparse
import numpy as np
from PIL import Image


def vivid_colors(n):
    """n vivid, maximally distinct RGB colors via golden-ratio hue stepping."""
    import colorsys
    cols = []
    golden = 0.618033988749895
    h = 0.15
    for i in range(n):
        h = (h + golden) % 1.0
        s = 0.85 + 0.15 * (i % 2)      # high saturation
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return cols


def get_instance_ids(mask):
    ids = np.unique(mask)
    return ids[ids != 0].tolist()


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def mode_2d(args):
    os.makedirs(args.out_dir, exist_ok=True)
    imgs = sorted(f for f in os.listdir(args.image_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for fn in imgs:
        mpath = mask_path_for(args.masks, fn)
        if not os.path.exists(mpath):
            continue
        img = np.array(Image.open(os.path.join(args.image_dir, fn)).convert("RGB"))
        mask = np.array(Image.open(mpath).convert("L").resize(
            (img.shape[1], img.shape[0]), Image.NEAREST))
        ids = get_instance_ids(mask)
        cols = vivid_colors(max(1, len(ids)))
        overlay = np.zeros_like(img)   # black background
        for k, iid in enumerate(ids):
            sel = mask == iid
            overlay[sel] = cols[k]      # solid vivid color, no blend
        out = os.path.join(args.out_dir, f"maskvis_{os.path.splitext(fn)[0]}.png")
        Image.fromarray(overlay).save(out)
        print(f"  {out}  ({len(ids)} instances)")
    print(f"Done -> {args.out_dir}")


def mode_3d(args):
    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    all_pts, all_cols = [], []
    for i, fn in enumerate(fnames):
        mpath = mask_path_for(args.masks, fn)
        if not os.path.exists(mpath):
            continue
        mask = np.array(Image.open(mpath).convert("L").resize((W, H), Image.NEAREST))
        ids = get_instance_ids(mask)
        cols = vivid_colors(max(1, len(ids)))
        mask_flat = mask.reshape(-1)
        pts_flat = pm[i].reshape(-1, 3)
        for k, iid in enumerate(ids):
            px = np.where(mask_flat == iid)[0]
            pts = pts_flat[px]
            good = ~np.isnan(pts).any(axis=1)
            pts = pts[good]
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_cols.append(np.tile(cols[k], (len(pts), 1)))

    if not all_pts:
        print("No apple points found.")
        return
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0).astype(np.uint8)

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
        for p, c in zip(pts.astype(np.float32), cols):
            f.write(p.tobytes()); f.write(bytes(c))
    print(f"Saved: {args.out}  ({len(pts):,} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2d", "3d"], required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pointmap", default=None)
    ap.add_argument("--filenames", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "2d":
        if not (args.image_dir and args.out_dir):
            raise SystemExit("2d mode needs --image_dir and --out_dir")
        mode_2d(args)
    else:
        if not (args.pointmap and args.filenames and args.out):
            raise SystemExit("3d mode needs --pointmap, --filenames, --out")
        mode_3d(args)


if __name__ == "__main__":
    main()
