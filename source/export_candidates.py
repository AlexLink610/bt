"""
export_candidates.py -- One PLY per component / candidate / cluster.

Reads the --dump JSON of associate_masks_graph_iterative.py and writes, into
--out, the points of

  initial_<q>.ply   initial graph component q (before any split), gray
  cand_<k>.ply      accepted candidate k, one color per candidate
  cluster_<c>.ply   final cluster c, one color per cluster

Numbering is 1-based, as in `eval_gt_assoc.py --details`. Load the files you
need into MeshLab (together with the point head PLY for context).

With --gt <dir with map_<stem>.json> it also writes, colored by correctness:

  initial_status.ply   initial graph components
  clusters_status.ply  final clusters
    green   exactly one apple, and this apple in no other group (correct)
    orange  exactly one apple, but the apple is split over several groups
    red     masks of several apples (merge error)
    gray    masks that are in no group (discarded)

--no_singles skips the per-component / per-candidate / per-cluster files.

Usage:
    python export_candidates.py --dump <run.json> --pointmap <..._rf.npy>
        --filenames <..._filenames.txt> --masks <mask dir> --out <dir>
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from associate_masks_graph_iterative import (  # noqa: E402
    load_filenames, mask_path_for, load_mask)

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
    [250, 190, 212], [0, 128, 128], [170, 110, 40], [128, 0, 0],
], dtype=np.uint8)
GRAY = np.array([150, 150, 150], np.uint8)
GREEN = np.array([60, 180, 75], np.uint8)
ORANGE = np.array([245, 130, 48], np.uint8)
RED = np.array([230, 25, 75], np.uint8)


def write_ply(path, pts, col):
    v = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    col = np.asarray(col, np.uint8)
    if col.ndim == 1:                     # one color for all points
        col = np.tile(col, (len(pts), 1))
    v["red"], v["green"], v["blue"] = col[:, 0], col[:, 1], col[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None,
                    help="directory with map_<stem>.json (ground-truth apple IDs)")
    ap.add_argument("--no_singles", action="store_true")
    args = ap.parse_args()

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    fnames = load_filenames(args.filenames)
    masks = [load_mask(mask_path_for(args.masks, f), W, H) for f in fnames]
    with open(args.dump) as f:
        d = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    def points(comp):
        ps = [pm[im][masks[im] == iid] for im, iid in comp if masks[im] is not None]
        p = np.concatenate(ps) if ps else np.empty((0, 3))
        return p[~np.isnan(p).any(axis=1)].astype(np.float32)

    clusters = {}
    for comp, lab in zip(d["accepted"], d["labels"]):
        clusters.setdefault(lab, []).extend(comp)

    if not args.no_singles:
        for q, comp in enumerate(d["initial"]):
            write_ply(os.path.join(args.out, f"initial_{q + 1}.ply"),
                      points(comp), GRAY)
        for k, comp in enumerate(d["accepted"]):
            write_ply(os.path.join(args.out, f"cand_{k + 1}.ply"), points(comp),
                      PALETTE[k % len(PALETTE)])
        for lab, comp in clusters.items():
            write_ply(os.path.join(args.out, f"cluster_{lab + 1}.ply"),
                      points(comp), PALETTE[lab % len(PALETTE)])

    if args.gt:
        gid = {}
        for i, fn in enumerate(fnames):
            stem = os.path.splitext(os.path.basename(fn))[0]
            jp = os.path.join(args.gt, f"map_{stem}.json")
            if os.path.exists(jp):
                with open(jp) as f:
                    for local, g in json.load(f).items():
                        gid[(i, int(local))] = int(g)
        all_nodes = [(i, int(m)) for i, mk in enumerate(masks) if mk is not None
                     for m in np.unique(mk) if m != 0]

        def status_ply(groups, name):
            sets = [{gid[tuple(n)] for n in g if tuple(n) in gid} for g in groups]
            frag = {}
            for gs in sets:
                for a in gs:
                    frag[a] = frag.get(a, 0) + 1
            P, C, covered = [], [], set()
            counts = {"correct": 0, "split": 0, "merge": 0}
            for g, gs in zip(groups, sets):
                covered |= {tuple(n) for n in g}
                if len(gs) > 1:
                    col, key = RED, "merge"
                elif len(gs) == 1 and frag[next(iter(gs))] == 1:
                    col, key = GREEN, "correct"
                else:
                    col, key = ORANGE, "split"
                counts[key] += 1
                p = points(g)
                P.append(p)
                C.append(np.tile(col, (len(p), 1)))
            dropped = [n for n in all_nodes if n not in covered]
            if dropped:
                p = points(dropped)
                P.append(p)
                C.append(np.tile(GRAY, (len(p), 1)))
            v = np.concatenate(P)
            write_ply(os.path.join(args.out, f"{name}.ply"), v, np.concatenate(C))
            print(f"{name}: {counts['correct']} correct, {counts['split']} split "
                  f"fragments, {counts['merge']} mixed, {len(dropped)} masks discarded")

        status_ply(d["initial"], "initial_status")
        status_ply(list(clusters.values()), "clusters_status")

    print(f"{len(d['initial'])} initial components, {len(d['accepted'])} "
          f"candidates, {len(clusters)} clusters -> {args.out}")


if __name__ == "__main__":
    main()
