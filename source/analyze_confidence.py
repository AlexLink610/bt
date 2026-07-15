"""
analyze_confidence.py -- Quantify how VGGT confidence varies with view count
and angular coverage, optionally restricted to APPLE pixels only (via masks).

Parses filenames of the form  <prefix>_<arc>_<N>v_confmap.npy.

Two modes:
  - default: stats over ALL pixels in each confmap
  - --masks DIR: stats over APPLE pixels only (mask_<stem>.png per view),
    which answers "how confident is VGGT specifically where the apples are".

Usage:
    python analyze_confidence.py --dir ~/ba/output_vggt/old --prefix t02
    python analyze_confidence.py --dir ~/ba/output_vggt/old --prefix t02 \
        --masks ~/ba/output_sam/tree_02/semantics_sam3

Note: confmaps saved by run_vggt.py are per-scene min-max normalized. Apple-only
stats show where apples fall WITHIN each scene's own confidence distribution,
which is comparable across configs even under per-scene normalization.
"""

import os
import re
import glob
import argparse
import numpy as np
from PIL import Image


PATTERN = re.compile(r"(.+?)_(\d+)_(\d+)v_confmap\.npy$")


def parse_name(path):
    m = PATTERN.search(os.path.basename(path))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_filenames(confmap_path):
    base = confmap_path.replace("_confmap.npy", "_filenames.txt")
    if not os.path.exists(base):
        return None
    with open(base) as f:
        return [l.strip() for l in f if l.strip()]


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def apple_conf_values(conf, filenames, masks_dir):
    """Return the confidence values at apple-masked pixels across all views."""
    N, H, W = conf.shape
    vals = []
    matched = 0
    for i, fname in enumerate(filenames):
        m = load_mask(mask_path_for(masks_dir, fname), W, H)
        if m is None:
            continue
        matched += 1
        apple = m > 0
        cv = conf[i][apple]
        cv = cv[~np.isnan(cv)]
        vals.append(cv)
    if not vals:
        return None, matched
    return np.concatenate(vals), matched


def stats_for(c):
    c = c[~np.isnan(c)]
    if c.size == 0:
        return None
    p = np.percentile(c, [10, 25, 50, 75, 90])
    return {
        "n": c.size,
        "mean": float(c.mean()),
        "median": float(p[2]),
        "p10": float(p[0]),
        "p25": float(p[1]),
        "p75": float(p[3]),
        "p90": float(p[4]),
        "frac_above_0.5": float((c >= 0.5).mean()),
        "frac_above_0.3": float((c >= 0.3).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--masks", default=None,
                        help="If given, restrict stats to apple-masked pixels.")
    parser.add_argument("--metric", default="mean")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_confmap.npy")))
    rows = []
    for f in files:
        parsed = parse_name(f)
        if parsed is None:
            continue
        prefix, arc, views = parsed
        if args.prefix and prefix != args.prefix:
            continue

        conf = np.load(f)
        if conf.ndim == 2:               # single view edge case
            conf = conf[None]

        if args.masks:
            filenames = load_filenames(f)
            if filenames is None:
                print(f"  no filenames for {os.path.basename(f)}, skipping")
                continue
            cvals, matched = apple_conf_values(conf, filenames, args.masks)
            if cvals is None:
                print(f"  no masks matched for {os.path.basename(f)}, skipping")
                continue
        else:
            cvals = conf.reshape(-1)
            matched = conf.shape[0]

        s = stats_for(cvals)
        if s is None:
            continue
        s.update({"prefix": prefix, "arc": arc, "views": views, "matched": matched})
        rows.append(s)

    if not rows:
        print("No matching files.")
        return

    rows.sort(key=lambda r: (r["arc"], r["views"]))

    mode = "APPLE PIXELS ONLY" if args.masks else "ALL PIXELS"
    print(f"\n{'='*95}")
    print(f"Per-config confidence statistics  ({mode})")
    print(f"{'='*95}")
    hdr = f"{'arc':>4} {'views':>6} {'mean':>7} {'median':>7} {'p10':>7} {'p25':>7} " \
          f"{'>0.5':>7} {'>0.3':>7} {'points':>11} {'imgs':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arc']:>4} {r['views']:>6} {r['mean']:>7.3f} {r['median']:>7.3f} "
              f"{r['p10']:>7.3f} {r['p25']:>7.3f} "
              f"{r['frac_above_0.5']:>7.3f} {r['frac_above_0.3']:>7.3f} "
              f"{r['n']:>11,} {r['matched']:>5}")

    metric = args.metric if args.metric in rows[0] else "mean"
    arcs  = sorted(set(r["arc"] for r in rows))
    views = sorted(set(r["views"] for r in rows))
    table = {(r["arc"], r["views"]): r[metric] for r in rows}

    print(f"\n{'='*95}")
    print(f"Pivot: {metric}  ({mode})  (rows=arc, cols=views)")
    print(f"{'='*95}")
    print(f"{'arc/views':>10} " + " ".join(f"{v:>8}" for v in views))
    for a in arcs:
        cells = [f"{table.get((a,v)):>8.3f}" if table.get((a,v)) is not None
                 else f"{'--':>8}" for v in views]
        print(f"{a:>10} " + " ".join(cells))


if __name__ == "__main__":
    main()
