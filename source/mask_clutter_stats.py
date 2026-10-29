"""
mask_clutter_stats.py -- How cluttered is a scene? Statistics of the instance
masks, independent of the image resolution, for comparing datasets.

Per dataset:
  masks per image             mean / median / max
  mask size                   median area in % of the image, share < 0.05 %
  touching masks              share of masks that touch another mask (1 px),
                              i.e. apples that border each other in the image
  neighbor distance           distance of a mask's centroid to the nearest
                              other centroid, in equivalent mask radii
                              (sqrt(area / pi)); < 2 means the neighbor is
                              closer than one apple diameter in the image
  size at VGGT input          median mask area in pixels of the VGGT input
                              (--input_hw, default 350 x 518) and in 14 x 14
                              patches, i.e. how many tokens an apple covers

Usage:
    python mask_clutter_stats.py --masks <dir> [--filenames <..._filenames.txt>]
        [--name tree_02] [--csv clutter.csv]
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--filenames", default=None,
                    help="only these images (default: all mask_*.png)")
    ap.add_argument("--name", default="")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--input_hw", type=int, nargs=2, default=[350, 518],
                    help="VGGT input resolution H W")
    args = ap.parse_args()

    if args.filenames:
        with open(args.filenames) as f:
            stems = [os.path.splitext(os.path.basename(l.strip()))[0] for l in f if l.strip()]
        paths = [os.path.join(args.masks, f"mask_{s}.png") for s in stems]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted(glob.glob(os.path.join(args.masks, "mask_*.png")))
    if not paths:
        raise SystemExit("no masks found")

    per_img, areas, touch, nn = [], [], [], []
    for n_done, p in enumerate(paths, 1):
        m = np.array(Image.open(p).convert("L"))
        ids = [int(v) for v in np.unique(m) if v != 0]
        per_img.append(len(ids))
        if ids:
            img_area = m.size
            counts = np.bincount(m.ravel(), minlength=256)
            boxes = ndimage.find_objects(m)          # bounding box per label
            cents = np.array(ndimage.center_of_mass(m > 0, labels=m, index=ids))
            rads = np.sqrt(counts[ids] / np.pi)
            for v in ids:
                areas.append(counts[v] / img_area)
                sl = boxes[v - 1]
                # work only inside the mask's bounding box (+1 px border)
                sl = tuple(slice(max(a.start - 1, 0), min(a.stop + 1, dim))
                           for a, dim in zip(sl, m.shape))
                sub = m[sl]
                mk = sub == v
                ring = ndimage.binary_dilation(mk) & ~mk
                touch.append(bool(np.any((sub[ring] != 0) & (sub[ring] != v))))
            if len(ids) > 1:
                D = np.linalg.norm(cents[:, None] - cents[None], axis=2)
                np.fill_diagonal(D, np.inf)
                nn.extend(D.min(1) / rads)
        if n_done % 10 == 0 or n_done == len(paths):
            print(f"  {n_done}/{len(paths)} images", flush=True)

    per_img, areas, nn = np.array(per_img), np.array(areas), np.array(nn)
    res = {"name": args.name, "images": len(paths),
           "masks_mean": per_img.mean(), "masks_median": float(np.median(per_img)),
           "masks_max": int(per_img.max()),
           "area_median_pct": 100 * float(np.median(areas)),
           "small_share": float(np.mean(areas < 0.0005)),
           "touching_share": float(np.mean(touch)),
           "nn_median_radii": float(np.median(nn)) if len(nn) else float("nan"),
           "nn_lt2_share": float(np.mean(nn < 2)) if len(nn) else float("nan")}
    px = float(np.median(areas)) * args.input_hw[0] * args.input_hw[1]
    res["vggt_px_median"] = px
    res["vggt_patches_median"] = px / 14 ** 2
    res["below_one_patch_share"] = float(np.mean(
        areas * args.input_hw[0] * args.input_hw[1] < 14 ** 2))
    print(f"[{args.name}] {res['images']} images")
    print(f"  masks per image     mean {res['masks_mean']:.1f}  median "
          f"{res['masks_median']:.0f}  max {res['masks_max']}")
    print(f"  mask size           median {res['area_median_pct']:.3f} % of the image, "
          f"< 0.05 %: {res['small_share']:.0%}")
    print(f"  touching another    {res['touching_share']:.0%} of the masks")
    print(f"  nearest neighbor    median {res['nn_median_radii']:.2f} mask radii, "
          f"< 2 radii: {res['nn_lt2_share']:.0%}")
    print(f"  at VGGT input       median {px:.0f} px = {res['vggt_patches_median']:.2f} "
          f"patches of 14 x 14; smaller than one patch: {res['below_one_patch_share']:.0%}")
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write(",".join(res) + "\n")
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                             for v in res.values()) + "\n")


if __name__ == "__main__":
    main()
