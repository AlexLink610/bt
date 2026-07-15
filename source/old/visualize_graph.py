"""
visualize_graph.py  --  Render the mask association graph as an image.

Shows:
  - One column per image
  - One node per mask instance (colored circle with ID)
  - Edges between matched instances across images
  - Edge thickness proportional to overlap score

Usage:
    python3 visualize_graph.py \
        --pointmap   ~/ba/output_vggt/old_room_naive_3v_pointmap.npy \
        --filenames  ~/ba/output_vggt/old_room_naive_3v_filenames.txt \
        --masks      ~/ba/output_sam/old_room/semantics_sam3 \
        --corr_thresh       0.018 \
        --min_overlap_pct   5 \
        --min_match_overlap 0.01 \
        --out        ~/ba/output_vggt/old_room_graph.png
"""

import os
import argparse
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


# ── reuse association logic ────────────────────────────────────────────────────

def load_filenames(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")

def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))

def get_instance_ids(mask):
    if mask is None:
        return []
    ids = np.unique(mask)
    return ids[ids != 0].tolist()

def compute_correspondence(pm1, pm2, mask1, mask2, H, W, corr_thresh):
    pts2_flat = pm2.reshape(-1, 3)
    apple2_flat = (mask2.reshape(-1) > 0) if mask2 is not None else np.ones(H*W, bool)
    valid2 = apple2_flat & ~np.isnan(pts2_flat).any(axis=1)
    valid2_idx = np.where(valid2)[0]
    if valid2_idx.size == 0:
        return np.full(H*W, -1, dtype=np.int32)
    tree = cKDTree(pts2_flat[valid2])
    apple1_flat = (mask1.reshape(-1) > 0) if mask1 is not None else np.ones(H*W, bool)
    valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
    apple1_idx = np.where(apple1_flat & valid1_flat)[0]
    if apple1_idx.size == 0:
        return np.full(H*W, -1, dtype=np.int32)
    distances, nn = tree.query(pm1.reshape(-1, 3)[apple1_idx], workers=-1)
    nn_flat = valid2_idx[nn]
    corr = np.full(H*W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = nn_flat[good]
    return corr

def compute_overlap(mask1, mask2, corr, ids1, ids2, min_overlap_pct):
    """Returns list of (i, j, overlap_frac) for accepted matches."""
    matches = []
    if not ids1 or not ids2:
        return matches
    mask1_flat = mask1.reshape(-1)
    mask2_flat = mask2.reshape(-1)
    min_frac = min_overlap_pct / 100.0
    C = np.ones((len(ids1), len(ids2)), dtype=np.float32)
    for ai, a in enumerate(ids1):
        pixels_a = np.where(mask1_flat == a)[0]
        size_a = len(pixels_a)
        if size_a == 0:
            continue
        corr_a = corr[pixels_a]
        valid = corr_a >= 0
        corr_a_valid = corr_a[valid]
        if len(corr_a_valid) == 0:
            continue
        landed = mask2_flat[corr_a_valid]
        for bi, b in enumerate(ids2):
            overlap = int((landed == b).sum())
            size_b = int((mask2_flat == b).sum())
            if min(size_a, size_b) == 0:
                continue
            frac = overlap / min(size_a, size_b)
            if frac >= min_frac:
                C[ai, bi] = 1.0 - frac
    row_ind, col_ind = linear_sum_assignment(C)
    for ai, bi in zip(row_ind, col_ind):
        frac = 1.0 - C[ai, bi]
        if frac > 0:
            matches.append((ai, bi, frac))
    return matches


def make_colors(n):
    colors = []
    golden = 0.618033988749895
    h = 0.0
    for i in range(n):
        h = (h + golden) % 1.0
        v = 1.0 if i % 2 == 0 else 0.75
        hi = int(h * 6)
        f = h * 6 - hi
        p = 0.0
        q = v * (1 - f)
        t = v * f
        hi = hi % 6
        if hi == 0: r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        else:         r, g, b = v, p, q
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",          required=True)
    parser.add_argument("--filenames",         required=True)
    parser.add_argument("--masks",             required=True)
    parser.add_argument("--corr_thresh",       type=float, default=0.018)
    parser.add_argument("--min_overlap_pct",   type=float, default=5.0)
    parser.add_argument("--min_match_overlap", type=float, default=0.01)
    parser.add_argument("--out",               default="graph.png")
    args = parser.parse_args()

    print("Loading data...")
    point_map = np.load(args.pointmap)
    N, H, W, _ = point_map.shape
    filenames = load_filenames(args.filenames)

    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        m = load_mask(mpath, W, H)
        masks.append(m)
        ids = get_instance_ids(m)
        print(f"  {fname}: {len(ids)} instances  {ids}")

    # ── collect all edges ──────────────────────────────────────────────────────
    edges = []  # (img_i, inst_i_idx, img_j, inst_j_idx, frac)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    for i, j in pairs:
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])
        if not ids_i or not ids_j or masks[i] is None or masks[j] is None:
            continue
        corr = compute_correspondence(
            point_map[i], point_map[j], masks[i], masks[j], H, W, args.corr_thresh
        )
        matches = compute_overlap(masks[i], masks[j], corr, ids_i, ids_j, args.min_overlap_pct)
        for ai, bi, frac in matches:
            if frac >= args.min_match_overlap:
                edges.append((i, ai, j, bi, frac))

    print(f"\nEdges found: {len(edges)}")

    # ── layout ─────────────────────────────────────────────────────────────────
    COL_W      = 200          # width per image column
    ROW_H      = 80           # height per node row
    NODE_R     = 22           # node circle radius
    MARGIN     = 60           # outer margin
    HEADER_H   = 50           # space for image name header

    max_instances = max((len(get_instance_ids(m)) for m in masks if m is not None), default=1)
    img_w = MARGIN * 2 + COL_W * N
    img_h = MARGIN + HEADER_H + ROW_H * max_instances + MARGIN

    canvas = Image.new("RGB", (img_w, img_h), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)

    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font_label = ImageFont.load_default()
        font_header = font_label

    # node positions: node_pos[img_idx][inst_list_idx] = (x, y)
    node_pos = {}
    for col, fname in enumerate(filenames):
        ids = get_instance_ids(masks[col])
        node_pos[col] = {}
        cx = MARGIN + col * COL_W + COL_W // 2
        # draw header
        short = os.path.splitext(fname)[0]
        draw.text((cx - 30, MARGIN + 10), short, fill=(180, 180, 200), font=font_header)
        for row, iid_idx in enumerate(range(len(ids))):
            cy = MARGIN + HEADER_H + row * ROW_H + ROW_H // 2
            node_pos[col][iid_idx] = (cx, cy)

    # assign instance colors per image independently
    instance_colors = {}
    for col in range(N):
        ids = get_instance_ids(masks[col])
        colors = make_colors(len(ids))
        for k, iid in enumerate(ids):
            instance_colors[(col, iid)] = colors[k]

    # draw edges first (behind nodes)
    for (i, ai, j, bi, frac) in edges:
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])
        if ai >= len(ids_i) or bi >= len(ids_j):
            continue
        p1 = node_pos[i][ai]
        p2 = node_pos[j][bi]
        alpha = int(80 + 160 * frac)
        thickness = max(1, int(frac * 6))
        # draw thick line as series of slightly offset lines
        for t in range(thickness):
            offset = t - thickness // 2
            draw.line(
                [(p1[0], p1[1] + offset), (p2[0], p2[1] + offset)],
                fill=(100, 200, 255, alpha),
                width=1
            )
        # overlap % label at midpoint
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        draw.text((mx - 12, my - 8), f"{frac*100:.0f}%", fill=(180, 230, 255), font=font_label)

    # draw nodes
    for col in range(N):
        ids = get_instance_ids(masks[col])
        for k, iid in enumerate(ids):
            if k not in node_pos[col]:
                continue
            cx, cy = node_pos[col][k]
            color = instance_colors.get((col, iid), (200, 200, 200))
            # glow
            for r_off in [NODE_R + 6, NODE_R + 3]:
                glow = tuple(int(c * 0.3) for c in color)
                draw.ellipse([cx - r_off, cy - r_off, cx + r_off, cy + r_off], fill=glow)
            # main circle
            draw.ellipse([cx - NODE_R, cy - NODE_R, cx + NODE_R, cy + NODE_R], fill=color)
            # ID label
            label = str(iid)
            draw.text((cx - 6, cy - 8), label, fill=(10, 10, 10), font=font_label)

    canvas.save(args.out)
    print(f"\nGraph saved: {args.out}  ({img_w}x{img_h}px)")
    print(f"  {N} images, {sum(len(get_instance_ids(m)) for m in masks if m is not None)} total nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()
