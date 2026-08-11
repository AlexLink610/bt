"""
dump_walkthrough.py -- Generate all artefacts to walk through the graph
association algorithm on ONE image pair, for teaching/presentation.

Produces:
  1. Labeled 2D masks: RGB with vivid instance colors + node-ID text
     (e.g. "I1_3") at each instance centroid  -> <out_dir>/labeled_<stem>.png
  2. Per-image instance point clouds (instance-colored) -> <out_dir>/cloud_imgA.ply,
     cloud_imgB.ply  (each contains ONLY that image's apple points)
  3. Instance-matching walkthrough printed to console:
     - instance sizes per image
     - pairwise overlap fractions (the cost matrix inputs)
     - which pairs are discarded (< min_overlap_pct)
     - Hungarian optimal assignment + which matches become edges

Usage:
    python dump_walkthrough.py \
        --pointmap  ~/ba/output_vggt/old/table/table_naive_5v_pointmap.npy \
        --filenames ~/ba/output_vggt/old/table/table_naive_5v_filenames.txt \
        --masks     ~/ba/output_sam/table \
        --image_dir ~/ba/data/table \
        --img_a 0 --img_b 1 \
        --corr_thresh 0.020 --min_overlap_pct 5.0 --min_match_overlap 0.01 \
        --out_dir   ~/ba/output_vggt/old/table/walkthrough
"""

import os
import argparse
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def vivid_colors(n):
    cols, golden, h = [], 0.618033988749895, 0.15
    for i in range(n):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85 + 0.15 * (i % 2), 0.95)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return cols


def get_instance_ids(mask):
    ids = np.unique(mask)
    return ids[ids != 0].tolist()


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def load_mask(path, W, H):
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def labeled_mask_image(img_rgb, mask, ids, cols, label_prefix, out_path):
    """Solid vivid instance colors on black background + node-ID text."""
    H, W = mask.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)   # schwarzer Hintergrund
    for k, iid in enumerate(ids):
        overlay[mask == iid] = cols[k]              # solide Farbe, kein Blend
    im = Image.fromarray(overlay)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, W // 40))
    except Exception:
        font = ImageFont.load_default()
    for k, iid in enumerate(ids):
        ys, xs = np.where(mask == iid)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = f"{label_prefix}_{iid}"
        for dx in (-2, 2):
            for dy in (-2, 2):
                draw.text((cx+dx, cy+dy), label, fill=(0,0,0), font=font, anchor="mm")
        draw.text((cx, cy), label, fill=(255,255,255), font=font, anchor="mm")
    im.save(out_path)
    print(f"  saved {out_path}")


def save_instance_cloud(pm_img, mask, ids, cols, out_path):
    """PLY of ONE image's apple points, colored by instance."""
    H, W = mask.shape
    mask_flat = mask.reshape(-1)
    pts_flat = pm_img.reshape(-1, 3)
    all_pts, all_cols = [], []
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
        print(f"  (no points for {out_path})")
        return
    pts = np.concatenate(all_pts).astype(np.float32)
    cols_arr = np.concatenate(all_cols).astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(pts, cols_arr):
            f.write(p.tobytes()); f.write(bytes(c))
    print(f"  saved {out_path}  ({len(pts):,} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--img_a", type=int, default=0)
    ap.add_argument("--img_b", type=int, default=1)
    ap.add_argument("--corr_thresh", type=float, default=0.020)
    ap.add_argument("--min_overlap_pct", type=float, default=5.0)
    ap.add_argument("--min_match_overlap", type=float, default=0.01)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    ia, ib = args.img_a, args.img_b
    fa, fb = fnames[ia], fnames[ib]
    print(f"\nImage A = [{ia}] {fa}")
    print(f"Image B = [{ib}] {fb}\n")

    mask_a = load_mask(mask_path_for(args.masks, fa), W, H)
    mask_b = load_mask(mask_path_for(args.masks, fb), W, H)
    ids_a, ids_b = get_instance_ids(mask_a), get_instance_ids(mask_b)
    cols_a, cols_b = vivid_colors(len(ids_a)), vivid_colors(len(ids_b))

    def load_rgb(fn):
        p = os.path.join(args.image_dir, fn)
        if not os.path.exists(p):
            stem = os.path.splitext(fn)[0]
            c = [x for x in os.listdir(args.image_dir) if os.path.splitext(x)[0] == stem]
            p = os.path.join(args.image_dir, c[0]) if c else None
        return np.array(Image.open(p).convert("RGB").resize((W, H))) if p else np.zeros((H, W, 3), np.uint8)

    rgb_a, rgb_b = load_rgb(fa), load_rgb(fb)

    # ---- 1. Labeled 2D masks ------------------------------------------------
    print("=== 1. Labeled 2D masks ===")
    labeled_mask_image(rgb_a, mask_a, ids_a, cols_a, "I1",
                       os.path.join(args.out_dir, f"labeled_A_{os.path.splitext(fa)[0]}.png"))
    labeled_mask_image(rgb_b, mask_b, ids_b, cols_b, "I2",
                       os.path.join(args.out_dir, f"labeled_B_{os.path.splitext(fb)[0]}.png"))

    # ---- 2. Per-image instance clouds --------------------------------------
    print("\n=== 2. Per-image instance point clouds ===")
    save_instance_cloud(pm[ia], mask_a, ids_a, cols_a,
                        os.path.join(args.out_dir, "cloud_imgA.ply"))
    save_instance_cloud(pm[ib], mask_b, ids_b, cols_b,
                        os.path.join(args.out_dir, "cloud_imgB.ply"))

    # ---- 3. Instance-matching walkthrough ----------------------------------
    print("\n=== 3. Instance matching walkthrough ===")

    # instance sizes
    print("\nInstance sizes (apple pixels):")
    print("  Image A:  " + "  ".join(f"I1_{iid}={int((mask_a==iid).sum())}" for iid in ids_a))
    print("  Image B:  " + "  ".join(f"I2_{iid}={int((mask_b==iid).sum())}" for iid in ids_b))

    # pixel correspondence A -> B (KD-tree over B's apple points)
    pts_b_flat = pm[ib].reshape(-1, 3)
    apple_b = (mask_b.reshape(-1) > 0) & ~np.isnan(pts_b_flat).any(axis=1)
    idx_b = np.where(apple_b)[0]
    tree = cKDTree(pts_b_flat[apple_b])

    pts_a_flat = pm[ia].reshape(-1, 3)
    apple_a = (mask_a.reshape(-1) > 0) & ~np.isnan(pts_a_flat).any(axis=1)
    idx_a = np.where(apple_a)[0]
    dist, nn = tree.query(pts_a_flat[apple_a], workers=-1)
    good = dist < args.corr_thresh
    corr = np.full(H * W, -1, dtype=np.int64)
    corr[idx_a[good]] = idx_b[nn[good]]
    print(f"\nPixel correspondence A->B: {good.sum():,}/{len(idx_a):,} apple pixels "
          f"matched within corr_thresh={args.corr_thresh}")

    # overlap fractions -> cost matrix
    mask_b_flat = mask_b.reshape(-1)
    mask_a_flat = mask_a.reshape(-1)
    print(f"\nOverlap fractions  ω = |a∈X: φ(a)∈Y| / min(|X|,|Y|)   "
          f"(discard if ω < {args.min_overlap_pct}%):\n")

    header = "        " + "  ".join(f"I2_{b:<4}" for b in ids_b)
    print(header)
    C = np.ones((len(ids_a), len(ids_b)), dtype=np.float32)
    overlaps = np.zeros_like(C)
    for ai, a in enumerate(ids_a):
        pix_a = np.where(mask_a_flat == a)[0]
        size_a = len(pix_a)
        corr_a = corr[pix_a]
        valid = corr_a >= 0
        landed = mask_b_flat[corr_a[valid]] if valid.any() else np.array([], int)
        row = []
        for bi, b in enumerate(ids_b):
            size_b = int((mask_b_flat == b).sum())
            ov = int((landed == b).sum())
            frac = ov / max(1, min(size_a, size_b))
            overlaps[ai, bi] = frac
            if frac >= args.min_overlap_pct / 100.0:
                C[ai, bi] = 1.0 - frac
                row.append(f"{frac*100:5.1f}%")
            else:
                row.append(f"  -  ")   # discarded
        print(f"I1_{a:<4}  " + "  ".join(f"{x:>6}" for x in row))

    print("\n('-' = discarded, overlap below threshold)")

    # Hungarian
    row_ind, col_ind = linear_sum_assignment(C)
    print("\nHungarian optimal assignment:")
    edges = []
    for ai, bi in zip(row_ind, col_ind):
        a, b = ids_a[ai], ids_b[bi]
        cost = C[ai, bi]
        ov = overlaps[ai, bi]
        if cost < 1.0 - args.min_match_overlap:
            edges.append((a, b))
            print(f"  I1_{a}  <->  I2_{b}   (overlap {ov*100:.1f}%, cost {cost:.3f})  -> EDGE")
        else:
            print(f"  I1_{a}  ..   I2_{b}   (overlap {ov*100:.1f}%, cost {cost:.3f})  -> no edge")

    print(f"\nEdges created this pair: {len(edges)}")
    for a, b in edges:
        print(f"  ({fa}, inst {a})  --  ({fb}, inst {b})")


if __name__ == "__main__":
    main()
