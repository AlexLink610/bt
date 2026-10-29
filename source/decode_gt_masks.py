"""
decode_gt_masks.py -- Turn the apple-ID EXRs from render_gt_masks.py into
pipeline-ready instance masks.

For every <stem>.exr in <run>/masks_exr it writes to <run>/masks_gt:
    mask_<stem>.png     8-bit, background 0, apples 1..K (per view)
    map_<stem>.json     local ID -> global apple ID (same apple = same ID
                        in every view; for split/merge evaluation)
    preview_<stem>.png  random colors, only for looking at the masks
and a summary gt_summary.json with the per-view counts and the number of
apples visible in at least one view (= ground truth for counting).

Requires:  pip install "OpenEXR>=3.3"   (e.g. in the sam3 env)

Usage:
    C:\\Users\\alex\\miniconda3\\envs\\sam3\\python.exe decode_gt_masks.py ^
        C:\\Users\\alex\\BA\\syn\\runs\\S04_env7tree_r3x16_close_big_nobg
"""

import os
import sys
import glob
import json
import numpy as np
import OpenEXR
from PIL import Image

AOV = "apple_id"
MIN_PIXELS = 100      # ignore apple fragments smaller than this (render px)
DECIMALS = 6          # rounding that merges float noise of one instance


def load_aov(path):
    # Blender 5 writes multi-part EXRs (one part per pass), so search all
    # parts; the pass name may be in the part name or in the channel name
    found, names = None, []
    with OpenEXR.File(path, separate_channels=True) as f:
        for part in f.parts:
            pname = part.name() or ""
            for cname, ch in part.channels.items():
                full = f"{pname}.{cname}" if pname else cname
                names.append(full)
                if found is None and AOV in full:
                    found = np.array(ch.pixels, dtype=np.float64)
    if found is None:
        sys.exit(f"No '{AOV}' channel in {path}. Channels: {names}")
    return found


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: decode_gt_masks.py <run_dir>")
    run = sys.argv[1]
    exrs = sorted(glob.glob(os.path.join(run, "masks_exr", "*.exr")))
    if not exrs:
        sys.exit("no EXRs in " + os.path.join(run, "masks_exr"))
    out = os.path.join(run, "masks_gt")
    os.makedirs(out, exist_ok=True)

    rng = np.random.default_rng(0)
    palette = rng.integers(40, 256, size=(256, 3), dtype=np.uint8)
    palette[0] = 0

    global_ids = {}          # rounded AOV value -> global apple id
    per_view = {}
    for exr in exrs:
        stem = os.path.splitext(os.path.basename(exr))[0]
        key = np.round(load_aov(exr), DECIMALS)
        vals, counts = np.unique(key[key > 0.5], return_counts=True)

        mask = np.zeros(key.shape, np.uint8)
        mapping = {}
        for val, cnt in zip(vals, counts):
            if cnt < MIN_PIXELS:
                continue
            local = len(mapping) + 1
            if local > 255:
                print(f"  WARNING {stem}: more than 255 apples, rest skipped")
                break
            gid = global_ids.setdefault(float(val), len(global_ids) + 1)
            mask[key == val] = local
            mapping[local] = gid

        Image.fromarray(mask).save(os.path.join(out, f"mask_{stem}.png"))
        Image.fromarray(palette[mask]).save(
            os.path.join(out, f"preview_{stem}.png"))
        with open(os.path.join(out, f"map_{stem}.json"), "w") as f:
            json.dump(mapping, f)
        per_view[stem] = len(mapping)
        print(f"{stem}: {len(mapping)} apples")

    summary = {"visible_apples": len(global_ids), "per_view": per_view}
    with open(os.path.join(out, "gt_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nApples visible in at least one view: {len(global_ids)}")


if __name__ == "__main__":
    main()
