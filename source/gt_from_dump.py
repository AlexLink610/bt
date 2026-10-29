"""
gt_from_dump.py -- Use a trusted association as ground truth.

For a scene without rendered ground truth (e.g. the table scene), a run whose
result has been checked visually (every apple exactly one instance) can serve
as reference. This script turns its --dump JSON into the files that
eval_gt_assoc.py reads:

    map_<stem>.json    local mask ID -> apple ID (= final cluster of the run)
    gt_summary.json    {"visible_apples": number of clusters}

Usage:
    python gt_from_dump.py --dump <trusted.json> --filenames <..._filenames.txt>
                           --out <gt dir>
Then evaluate other runs with
    python eval_gt_assoc.py --dump <other.json> --filenames ... --masks <gt dir>
"""

import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]
    with open(args.dump) as f:
        d = json.load(f)

    maps = {i: {} for i in range(len(fnames))}
    for comp, lab in zip(d["accepted"], d["labels"]):
        for im, iid in comp:
            maps[im][int(iid)] = int(lab) + 1

    os.makedirs(args.out, exist_ok=True)
    for i, fn in enumerate(fnames):
        stem = os.path.splitext(os.path.basename(fn))[0]
        with open(os.path.join(args.out, f"map_{stem}.json"), "w") as f:
            json.dump(maps[i], f)
    n = len(set(d["labels"]))
    with open(os.path.join(args.out, "gt_summary.json"), "w") as f:
        json.dump({"visible_apples": n}, f)
    n_masks = sum(len(m) for m in maps.values())
    print(f"reference: {n} apples, {n_masks} masks -> {args.out}")


if __name__ == "__main__":
    main()
