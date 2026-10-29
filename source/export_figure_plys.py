"""
export_figure_plys.py -- colored PLYs for the two synthetic-tree figures.

status  groups of masks colored by correctness (fig:results_status)
        green: one apple, not split   orange: one apple, split
        red: several apples           gray: discarded
        -> <out>_initial.ply, <out>_final.ply
apples  apple points colored by GT apple ID (fig:results_apples)
        -> <out>.ply

Only masks that are nodes of the initial graph are drawn in status mode.
--scene adds the non-apple pixels of a full point map in light gray
(pass the same file as --pointmap if it still contains the background).

Usage:
  python export_figure_plys.py status --pointmap PM.npy --filenames FN.txt
      --masks masks_gt --dump run.json --out figs/status [--scene PM.npy]
  python export_figure_plys.py apples --pointmap PM.npy --filenames FN.txt
      --masks masks_gt --out figs/apples_head [--scene PM.npy]
"""

import os
import json
import argparse
import colorsys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

MASK_PATTERN = "mask_{stem}.png"
GREEN = (60, 170, 60)
ORANGE = (245, 150, 20)
RED = (215, 40, 40)
GRAY = (110, 110, 110)       # discarded
SCENE = (215, 215, 215)      # background context
SCENE_STRIDE = 3             # keep every n-th background pixel per axis


def load_masks(masks_dir, fnames, H, W):
    out = []
    for fn in fnames:
        stem = os.path.splitext(fn)[0]
        path = os.path.join(masks_dir, MASK_PATTERN.format(stem=stem))
        m = np.array(Image.open(path)).astype(np.int32)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (H, W):  # nearest neighbor, never mixes IDs
            ys = ((np.arange(H) + 0.5) * m.shape[0] / H).astype(int)
            xs = ((np.arange(W) + 0.5) * m.shape[1] / W).astype(int)
            m = m[ys][:, xs]
        out.append(m)
    return np.stack(out)


def load_gid_map(masks_dir, fnames):
    gid = {}
    for i, fn in enumerate(fnames):
        stem = os.path.splitext(fn)[0]
        path = os.path.join(masks_dir, f"map_{stem}.json")
        if os.path.exists(path):
            with open(path) as f:
                for local, g in json.load(f).items():
                    gid[(i, int(local))] = int(g)
    return gid


def apple_color(a):
    h = (a * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def group_status(groups, gid):
    per_group = [Counter(gid[tuple(n)] for n in g if tuple(n) in gid)
                 for g in groups]
    frag = Counter()
    for c in per_group:
        for a in c:
            frag[a] += 1
    status = []
    for c in per_group:
        if len(c) > 1:
            status.append(RED)
        elif len(c) == 1 and frag[next(iter(c))] > 1:
            status.append(ORANGE)
        elif len(c) == 1:
            status.append(GREEN)
        else:
            status.append(GRAY)
    return status


def points_by_node(pm, masks, color_of):
    pts, cols = [], []
    for i in range(len(masks)):
        for iid in np.unique(masks[i]):
            if iid == 0:
                continue
            c = color_of((i, int(iid)))
            if c is None:
                continue
            p = pm[i][masks[i] == iid]
            pts.append(p)
            cols.append(np.tile(c, (len(p), 1)))
    return np.concatenate(pts), np.concatenate(cols).astype(np.uint8)


def scene_points(path, masks):
    pm = np.load(path)
    grid = np.zeros(masks.shape[1:], bool)
    grid[::SCENE_STRIDE, ::SCENE_STRIDE] = True
    p = pm[(masks == 0) & grid[None]]
    return p, np.tile(SCENE, (len(p), 1)).astype(np.uint8)


def write_ply(path, pts, cols, extra=None):
    if extra is not None:
        pts = np.concatenate([extra[0], pts])
        cols = np.concatenate([extra[1], cols])
    ok = np.isfinite(pts).all(1)
    pts, cols = pts[ok], cols[ok]
    v = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"),
                                  ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts.T
    v["red"], v["green"], v["blue"] = cols.T
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(v)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\n"
                 "property uchar blue\nend_header\n").encode("ascii"))
        f.write(v.tobytes())
    print(f"  saved {path}  ({len(v)} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["status", "apples"])
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--out", required=True, help="output path without .ply")
    args = ap.parse_args()

    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]
    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    assert N == len(fnames), f"{N} point maps vs {len(fnames)} filenames"
    masks = load_masks(args.masks, fnames, H, W)
    gid = load_gid_map(args.masks, fnames)
    extra = scene_points(args.scene, masks) if args.scene else None

    if args.mode == "apples":
        pts, cols = points_by_node(
            pm, masks, lambda n: apple_color(gid[n]) if n in gid else None)
        write_ply(f"{args.out}.ply", pts, cols, extra)
        print(f"  apples: {len(set(gid.values()))}")
        return

    with open(args.dump) as f:
        d = json.load(f)
    clusters = defaultdict(list)
    for comp, lab in zip(d["accepted"], d["labels"]):
        clusters[lab].extend(comp)
    nodes = {tuple(n) for g in d["initial"] for n in g}

    for stage, groups in (("initial", d["initial"]),
                          ("final", list(clusters.values()))):
        status = group_status(groups, gid)
        col = {}
        for g, s in zip(groups, status):
            for n in g:
                col[tuple(n)] = s
        pts, cols = points_by_node(
            pm, masks,
            lambda n: (col.get(n, GRAY) if n in nodes else None))
        k = Counter(status)
        print(f"[{stage}] groups {len(groups)}: green {k[GREEN]}  "
              f"orange {k[ORANGE]}  red {k[RED]}  "
              f"discarded masks {len(nodes - set(col))}")
        write_ply(f"{args.out}_{stage}.ply", pts, cols, extra)


if __name__ == "__main__":
    main()
