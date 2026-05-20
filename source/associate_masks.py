"""
associate_masks.py  --  Count apples using pointmap-based mask association
                        (simplified CCGS Section 3.2)

Improvements over v1:
    - TSP ordering: images sorted by greedy nearest-neighbour camera position
      to maximise viewpoint continuity between consecutive pairs
    - Apple-pixel-only correspondence: KD-tree built from apple pixels only,
      giving more reliable overlaps in the regions that matter
    - Bad pair skipping: pairs with < min_overlap_pct good correspondences
      are skipped entirely -- global IDs carried forward unchanged rather
      than declaring everything new

Usage:
    python associate_masks.py \
        --pointmap  ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --filenames ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --transforms /path/to/transforms.json \
        --out       ~/ba/output_vggt/t02_360_64v_assoc_count.txt

Optional:
    --corr_thresh       max distance for a valid pixel correspondence (default: 0.05)
    --min_overlap_pct   min % of apple pixels within corr_thresh to run Hungarian (default: 10)
    --min_match_overlap min overlap fraction to accept a Hungarian match (default: 0.01)
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pointmap(path):
    pm = np.load(path)
    print(f"  Pointmap shape: {pm.shape}  ({pm.nbytes / 1e6:.1f} MB)")
    return pm


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


# ── TSP ordering ──────────────────────────────────────────────────────────────

def tsp_order(fnames, transforms_path):
    """
    Greedy nearest-neighbour TSP ordering by camera position.
    Returns reordered list of filenames.
    """
    import json
    with open(transforms_path) as f:
        data = json.load(f)

    name_to_pos = {}
    for frame in data["frames"]:
        M   = np.array(frame["transform_matrix"])
        pos = np.array([M[0, 3], M[1, 3], M[2, 3]])
        name_to_pos[os.path.basename(frame["file_path"])] = pos

    positions = np.array([name_to_pos[f] for f in fnames])
    N         = len(fnames)
    visited   = [False] * N
    order     = [0]
    visited[0] = True

    for _ in range(N - 1):
        current   = order[-1]
        best_dist = np.inf
        best_idx  = -1
        for j in range(N):
            if not visited[j]:
                d = np.linalg.norm(positions[current] - positions[j])
                if d < best_dist:
                    best_dist = d
                    best_idx  = j
        order.append(best_idx)
        visited[best_idx] = True

    return [fnames[i] for i in order]


# ── Apple-pixel correspondence ────────────────────────────────────────────────

def compute_apple_correspondence(pm1, pm2, mask1, H, W, corr_thresh):
    """
    For apple pixels in image 1, find nearest pixel in image 2 using KD-tree.
    Only apple pixels in image 2 are included in the tree.

    Returns:
        corr_flat : (H*W,) int array — flat index in pm2 for each pixel in pm1
                    -1 if no valid correspondence
        pct_good  : % of apple pixels in image 1 with distance < corr_thresh
    """
    # Apple pixels in image 2 for the tree
    apple2 = (mask1 > 0) if mask1 is not None else np.ones((H, W), dtype=bool)
    pts2_all   = pm2.reshape(-1, 3)
    valid2_all = ~np.isnan(pts2_all).any(axis=1)

    # Use all valid pts2 for the tree (not just apple) so we can find
    # correspondences for any pixel in image 1
    valid2_idx = np.where(valid2_all)[0]
    if valid2_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32), 0.0

    tree = cKDTree(pts2_all[valid2_all])

    # Query apple pixels of image 1
    if mask1 is None:
        apple1_idx = np.where(~np.isnan(pm1.reshape(-1, 3)).any(axis=1))[0]
    else:
        apple1_flat = (mask1.reshape(-1) > 0)
        valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
        apple1_idx  = np.where(apple1_flat & valid1_flat)[0]

    if apple1_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32), 0.0

    distances, nn_in_valid2 = tree.query(pm1.reshape(-1, 3)[apple1_idx],
                                          workers=-1)
    nn_flat = valid2_idx[nn_in_valid2]

    pct_good = float((distances < corr_thresh).mean() * 100)

    corr = np.full(H * W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = nn_flat[good]

    return corr, pct_good


# ── Cost matrix ───────────────────────────────────────────────────────────────

def compute_cost_matrix(mask1, mask2, corr, ids1, ids2):
    if not ids1 or not ids2:
        return np.ones((max(1, len(ids1)), max(1, len(ids2))), dtype=np.float32)

    mask1_flat = mask1.reshape(-1)
    mask2_flat = mask2.reshape(-1)
    C = np.ones((len(ids1), len(ids2)), dtype=np.float32)

    for ai, a in enumerate(ids1):
        pixels_a = np.where(mask1_flat == a)[0]
        size_a   = len(pixels_a)
        if size_a == 0:
            continue
        corr_a       = corr[pixels_a]
        valid        = corr_a >= 0
        corr_a_valid = corr_a[valid]
        if len(corr_a_valid) == 0:
            continue
        landed_ids = mask2_flat[corr_a_valid]
        for bi, b in enumerate(ids2):
            overlap = int((landed_ids == b).sum())
            size_b  = int((mask2_flat == b).sum())
            if min(size_a, size_b) > 0:
                C[ai, bi] = 1.0 - overlap / min(size_a, size_b)

    return C


# ── Hungarian assignment ──────────────────────────────────────────────────────

def assign_global_ids(ids1, ids2, cost_matrix, global_ids_1,
                      next_global_id, min_match_overlap):
    global_ids_2 = {}

    if not ids1 or not ids2:
        for b in ids2:
            global_ids_2[b] = next_global_id
            next_global_id += 1
        return global_ids_2, next_global_id

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_b = set()

    for ai, bi in zip(row_ind, col_ind):
        if cost_matrix[ai, bi] < (1.0 - min_match_overlap):
            a = ids1[ai]
            b = ids2[bi]
            global_ids_2[b] = global_ids_1[a]
            matched_b.add(b)

    for b in ids2:
        if b not in matched_b:
            global_ids_2[b] = next_global_id
            next_global_id += 1

    return global_ids_2, next_global_id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",          required=True)
    parser.add_argument("--filenames",         required=True)
    parser.add_argument("--masks",             required=True)
    parser.add_argument("--transforms",        default=None,
                        help="Path to transforms.json for TSP ordering. "
                             "If omitted, original filename order is used.")
    parser.add_argument("--out",               default=None)
    parser.add_argument("--corr_thresh",       type=float, default=0.05,
                        help="Max distance for valid pixel correspondence (default: 0.05)")
    parser.add_argument("--min_overlap_pct",   type=float, default=10.0,
                        help="Min %% apple pixels within corr_thresh to run Hungarian (default: 10)")
    parser.add_argument("--min_match_overlap", type=float, default=0.01,
                        help="Min overlap fraction to accept a match (default: 0.01)")
    args = parser.parse_args()

    print("=" * 60)
    print("Apple Counter -- Pointmap Association + Hungarian v2")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading inputs...")
    point_map = load_pointmap(args.pointmap)
    filenames = load_filenames(args.filenames)
    N, H, W, _ = point_map.shape

    if len(filenames) != N:
        print(f"ERROR: filenames ({len(filenames)}) != pointmap N ({N})")
        sys.exit(1)

    # ── TSP ordering ──────────────────────────────────────────────────────────
    if args.transforms:
        print("Computing TSP ordering...")
        filenames = tsp_order(filenames, args.transforms)
        print(f"  Reordered {N} images by camera proximity")
    else:
        print("  Using original filename order (no --transforms provided)")

    # Build index map from reordered filenames back to pointmap slices
    orig_filenames = load_filenames(args.filenames)
    orig_idx = {f: i for i, f in enumerate(orig_filenames)}

    # ── Load all masks ────────────────────────────────────────────────────────
    print(f"Loading {N} masks...")
    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        masks.append(load_mask(mpath, W, H))
    loaded = sum(1 for m in masks if m is not None)
    print(f"  Loaded {loaded}/{N} masks")

    # ── Sequential association ────────────────────────────────────────────────
    print(f"\nAssociating instances across {N} images...")
    print(f"  corr_thresh={args.corr_thresh}  "
          f"min_overlap_pct={args.min_overlap_pct}%  "
          f"min_match_overlap={args.min_match_overlap}")

    ids0 = get_instance_ids(masks[0])
    next_global_id = 1
    global_ids = [{}] * N

    if ids0:
        global_ids[0] = {iid: next_global_id + i for i, iid in enumerate(ids0)}
        next_global_id += len(ids0)

    skipped = 0
    matched_total = 0
    new_total = 0

    for i in range(N - 1):
        pm_i  = point_map[orig_idx[filenames[i]]]
        pm_i1 = point_map[orig_idx[filenames[i + 1]]]
        ids_i  = get_instance_ids(masks[i])
        ids_i1 = get_instance_ids(masks[i + 1])

        if not ids_i1:
            global_ids[i + 1] = {}
            print(f"  [{i+1:3d}] {filenames[i+1]}: no instances")
            continue

        if not ids_i or masks[i] is None:
            global_ids[i + 1] = {}
            for iid in ids_i1:
                global_ids[i + 1][iid] = next_global_id
                next_global_id += 1
            print(f"  [{i+1:3d}] {filenames[i+1]}: {len(ids_i1)} new (no prev)")
            new_total += len(ids_i1)
            continue

        # Apple-pixel correspondence
        corr, pct_good = compute_apple_correspondence(
            pm_i, pm_i1, masks[i], H, W, args.corr_thresh
        )

        if pct_good < args.min_overlap_pct:
            # Bad pair — carry forward previous global IDs unchanged
            global_ids[i + 1] = dict(global_ids[i])
            # Add any new instances not in image i
            prev_local = set(global_ids[i].keys())
            for iid in ids_i1:
                if iid not in prev_local:
                    global_ids[i + 1][iid] = next_global_id
                    next_global_id += 1
            skipped += 1
            print(f"  [{i+1:3d}] {filenames[i+1]}: SKIPPED "
                  f"(overlap {pct_good:.1f}% < {args.min_overlap_pct}%)")
            continue

        # Cost matrix + Hungarian
        C = compute_cost_matrix(masks[i], masks[i + 1], corr,
                                ids_i, ids_i1)
        global_ids[i + 1], next_global_id = assign_global_ids(
            ids_i, ids_i1, C, global_ids[i],
            next_global_id, args.min_match_overlap
        )

        matched = sum(1 for iid in ids_i1
                      if global_ids[i + 1].get(iid) in global_ids[i].values())
        new     = len(ids_i1) - matched
        matched_total += matched
        new_total     += new
        print(f"  [{i+1:3d}] {filenames[i+1]}: "
              f"{len(ids_i1)} instances, "
              f"{matched} matched, {new} new  "
              f"(overlap {pct_good:.1f}%)")

    # ── Count ─────────────────────────────────────────────────────────────────
    all_global_ids = set()
    for gids in global_ids:
        all_global_ids.update(gids.values())
    apple_count = len(all_global_ids)

    print(f"\n  Pairs skipped (bad overlap): {skipped}/{N-1}")
    print(f"  Total matched instances:     {matched_total}")
    print(f"  Total new instances:         {new_total}")

    print(f"\n{'='*60}")
    print(f"  APPLE COUNT: {apple_count}")
    print(f"  (ground truth tree_02: 113)")
    print(f"  Error: {apple_count - 113:+d}  ({100*(apple_count-113)/113:+.1f}%)")
    print(f"{'='*60}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"apple_count: {apple_count}\n")
            f.write(f"ground_truth: 113\n")
            f.write(f"error: {apple_count - 113:+d}\n")
            f.write(f"corr_thresh: {args.corr_thresh}\n")
            f.write(f"min_overlap_pct: {args.min_overlap_pct}\n")
            f.write(f"min_match_overlap: {args.min_match_overlap}\n")
            f.write(f"pairs_skipped: {skipped}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
