"""
apple_ply.py -- Export only the apple pixels of a VGGT pointmap as PLY.

Uses the ground-truth masks (mask_<stem>.png + map_<stem>.json from
decode_gt_masks.py) and writes, for each color mode:

  <out>/<name>_by_apple.ply  one color per ground-truth apple
                             (a clean reconstruction shows compact spheres,
                             an inconsistent one shows thick or doubled shells)
  <out>/<name>_by_view.ply   color ramp over the view index
  <out>/<name>_by_ring.ply   one color per camera ring (views split into
                             --rings equally sized, consecutive groups)

Run it once per pointmap variant (point head, depth+camera, filtered) with
the same --out to compare them side by side in MeshLab.

Usage:
    python apple_ply.py --pointmap <..._pointmap_head.npy> --name head
        --filenames <..._filenames.txt> --masks <run>/masks_gt --out <dir>
        [--rings 3] [--apples 5 12]
"""

import os
import json
import argparse
import colorsys
import numpy as np
from PIL import Image


def palette(n, seed=0.15):
    cols, h = [], seed
    for i in range(max(1, n)):
        h = (h + 0.618033988749895) % 1.0
        v = 0.95 if i % 2 == 0 else 0.7
        cols.append([int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.85, v)])
    return np.array(cols, np.uint8)


def ramp(t):
    """blue -> green -> red for t in [0, 1]."""
    t = np.clip(t, 0, 1)
    r = np.clip(2 * t - 1, 0, 1)
    g = 1 - np.abs(2 * t - 1)
    b = np.clip(1 - 2 * t, 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def write_ply(path, pts, cols):
    v = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["red"], v["green"], v["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        f.write(v.tobytes())
    print(f"  {path}  ({len(v):,} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="pointmap")
    ap.add_argument("--rings", type=int, default=3)
    ap.add_argument("--apples", type=int, nargs="+", default=None,
                    help="export only these ground-truth apple IDs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    P, G, V = [], [], []
    for i, fn in enumerate(fnames):
        stem = os.path.splitext(os.path.basename(fn))[0]
        mpath = os.path.join(args.masks, f"mask_{stem}.png")
        jpath = os.path.join(args.masks, f"map_{stem}.json")
        if not (os.path.exists(mpath) and os.path.exists(jpath)):
            continue
        mask = np.array(Image.open(mpath).convert("L").resize((W, H), Image.NEAREST))
        with open(jpath) as f:
            mapping = {int(k): int(v) for k, v in json.load(f).items()}
        lut = np.zeros(256, np.int64)
        for local, g in mapping.items():
            lut[local] = g
        gid = lut[mask].reshape(-1)
        pts = pm[i].reshape(-1, 3)
        keep = (gid > 0) & ~np.isnan(pts).any(axis=1)
        if args.apples:
            keep &= np.isin(gid, args.apples)
        P.append(pts[keep])
        G.append(gid[keep])
        V.append(np.full(keep.sum(), i))

    P = np.concatenate(P).astype(np.float32)
    G = np.concatenate(G)
    V = np.concatenate(V)
    print(f"{len(P):,} apple points, {len(np.unique(G))} apples, "
          f"{len(np.unique(V))} views")

    pal = palette(G.max() + 1)
    write_ply(os.path.join(args.out, f"{args.name}_by_apple.ply"), P, pal[G])
    write_ply(os.path.join(args.out, f"{args.name}_by_view.ply"), P,
              ramp(V / max(1, N - 1)))
    ring = (V * args.rings) // N
    ring_cols = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200],
                          [245, 130, 48], [145, 30, 180]], np.uint8)
    write_ply(os.path.join(args.out, f"{args.name}_by_ring.ply"), P,
              ring_cols[ring % len(ring_cols)])


if __name__ == "__main__":
    main()
