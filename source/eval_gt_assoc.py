"""
eval_gt_assoc.py -- Split/merge evaluation of the association against the
Blender ground truth.

Reads the --dump JSON of associate_masks_graph_iterative.py (initial graph
components, accepted candidates, cluster labels) and the GT maps
(map_<stem>.json: local mask ID -> global apple ID) and reports, for three
stages of the pipeline,

  initial   connected components of the first association graph
  accepted  candidates accepted after sphere check / refinement
  final     clusters after clustering (= the count)

  groups        number of groups at this stage
  pure / mixed  groups containing one / several different apples (merge errors)
  apples found  GT apples that occur in at least one group
  fragments     groups per apple (split errors): mean, max, apples with > 1
  F1            groups and GT apples matched one-to-one (Hungarian) on their
                pixel IoU over all views; a match with IoU >= 0.5 is a true
                positive. precision = TP / groups, recall = TP / GT apples.

Usage:
    python eval_gt_assoc.py --dump <run>/assoc/xyz.json \
        --filenames <run>/vggt/..._filenames.txt --masks <run>/masks_gt
"""

import os
import json
import argparse
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

IOU_TP = 0.5


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


def mask_pixels(masks_dir, fnames):
    """(view, local mask id) -> number of pixels (native mask resolution)."""
    pix = {}
    for i, fn in enumerate(fnames):
        stem = os.path.splitext(fn)[0]
        path = os.path.join(masks_dir, f"mask_{stem}.png")
        if not os.path.exists(path):
            continue
        m = np.array(Image.open(path))
        vals, cnt = np.unique(m[m > 0], return_counts=True)
        for v, c in zip(vals, cnt):
            pix[(i, int(v))] = int(c)
    return pix


def f1_iou(groups, gid, pix, n_gt, thr=IOU_TP):
    """One-to-one matching of groups and GT apples on pixel IoU.
    Returns precision, recall, F1, mean IoU of the true positives and the
    list of true positives as (group index, GT apple id, IoU)."""
    apple_pix = defaultdict(int)
    for node, g in gid.items():
        apple_pix[g] += pix.get(node, 0)
    apples = sorted(apple_pix)
    if not groups or not apples:
        return 0.0, 0.0, 0.0, 0.0, []
    col = {a: k for k, a in enumerate(apples)}
    inter = np.zeros((len(groups), len(apples)))
    size = np.zeros(len(groups))
    for k, grp in enumerate(groups):
        for n in grp:
            n = tuple(n)
            p = pix.get(n, 0)
            size[k] += p
            if n in gid:
                inter[k, col[gid[n]]] += p
    union = size[:, None] + np.array([apple_pix[a] for a in apples])[None] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    r, c = linear_sum_assignment(-iou)
    tps = [(int(k), apples[j], float(iou[k, j])) for k, j in zip(r, c)
           if iou[k, j] >= thr]
    tp = len(tps)
    precision = tp / len(groups)
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    miou = float(np.mean([t[2] for t in tps])) if tps else 0.0
    return precision, recall, f1, miou, tps


def evaluate(name, groups, gid, n_gt, pix=None):
    per_group = [Counter(gid[tuple(n)] for n in g if tuple(n) in gid)
                 for g in groups]
    per_group = [c for c in per_group if c]
    mixed = [c for c in per_group if len(c) > 1]
    frag = defaultdict(int)
    for c in per_group:
        for g in c:
            frag[g] += 1
    n_frag = list(frag.values())
    split_excess = sum(v - 1 for v in n_frag)
    merge_loss = sum(len(c) - 1 for c in mixed)
    # apples recovered exactly: in one group only, and that group is pure
    perfect = sum(1 for c in per_group
                  if len(c) == 1 and frag[next(iter(c))] == 1)
    print(f"\n[{name}]")
    print(f"  groups            {len(per_group)}")
    print(f"  pure / mixed      {len(per_group) - len(mixed)} / {len(mixed)}"
          f"   (mixed = merge errors)")
    print(f"  apples found      {len(frag)} / {n_gt}")
    if n_frag:
        print(f"  fragments/apple   mean {sum(n_frag) / len(n_frag):.2f}  "
              f"max {max(n_frag)}  apples split: "
              f"{sum(1 for v in n_frag if v > 1)}")
    print(f"  split excess {split_excess:+d}  merge loss {-merge_loss:+d}  "
          f"missing {-(n_gt - len(frag)):+d}")
    print(f"  perfectly recovered apples  {perfect} / {n_gt}")
    res = {"groups": len(per_group), "mixed": len(mixed), "perfect": perfect,
           "apples_found": len(frag), "split_excess": split_excess,
           "merge_loss": merge_loss, "missing": n_gt - len(frag)}
    if pix is not None:
        p, r, f1, miou, _ = f1_iou(groups, gid, pix, n_gt)
        print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}  "
              f"(IoU >= {IOU_TP}; mean IoU of matches {miou:.3f})")
        res.update({"precision": p, "recall": r, "f1": f1, "miou": miou})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None,
                    help="append a one-line summary to this CSV")
    ap.add_argument("--details", action="store_true",
                    help="list every accepted candidate with its GT apples; "
                         "candidates that are only part of an initial component "
                         "(= produced by a split) are marked")
    ap.add_argument("--tag", default="",
                    help="name of the run in the CSV")
    args = ap.parse_args()

    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]
    gid = load_gid_map(args.masks, fnames)
    # ground truth = apples visible in at least one view OF THIS RUN, so that
    # runs on view subsets are evaluated against their own visible apples
    n_gt = len(set(gid.values()))
    if n_gt == 0:
        with open(os.path.join(args.masks, "gt_summary.json")) as f:
            n_gt = json.load(f)["visible_apples"]
    with open(args.dump) as f:
        d = json.load(f)

    clusters = defaultdict(list)
    for comp, lab in zip(d["accepted"], d["labels"]):
        clusters[lab].extend(comp)

    if args.details:
        initial = [set(map(tuple, c)) for c in d["initial"]]
        print("\naccepted candidates (masks -> GT apple: count):")
        for k, (comp, lab) in enumerate(zip(d["accepted"], d["labels"])):
            cs = set(map(tuple, comp))
            src = next((q for q, ini in enumerate(initial) if cs <= ini), None)
            split = src is not None and cs != initial[src]
            comp_gids = Counter(gid[n] for n in cs if n in gid)
            print(f"  cand {k + 1:2d}  cluster {lab + 1:2d}  {len(cs)} masks  "
                  f"apples {dict(comp_gids)}"
                  + (f"   <- split from initial component {src + 1}" if split else ""))

    pix = mask_pixels(args.masks, fnames)
    res = {
        "initial": evaluate("initial graph", d["initial"], gid, n_gt, pix),
        "accepted": evaluate("accepted candidates", d["accepted"], gid, n_gt, pix),
        "final": evaluate("final clusters", list(clusters.values()), gid, n_gt, pix),
    }
    print(f"\nGT apples: {n_gt}   final count: {len(clusters)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    if args.csv:
        new = not os.path.exists(args.csv)
        i, fi = res["initial"], res["final"]
        with open(args.csv, "a") as f:
            if new:
                f.write("tag,count,gt,init_groups,init_split,init_merge,"
                        "final_split,final_merge,final_missing,final_perfect,"
                        "precision,recall,f1,miou\n")
            f.write(f"{args.tag},{len(clusters)},{n_gt},{i['groups']},"
                    f"{i['split_excess']},{i['merge_loss']},{fi['split_excess']},"
                    f"{fi['merge_loss']},{fi['missing']},{fi['perfect']},"
                    f"{fi['precision']:.3f},{fi['recall']:.3f},{fi['f1']:.3f},"
                    f"{fi['miou']:.3f}\n")


if __name__ == "__main__":
    main()
