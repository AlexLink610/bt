"""
Pipeline:
    1. Load _pointmap.npy (N, H, W, 3) and _filenames.txt
    2. For each pair of adjacent images (i, i+1):
         a. Build KD-tree from image i+1's pointmap
         b. For each pixel in image i, find its corresponding pixel in image i+1
            via nearest-neighbor search in 3D space
         c. For each pair of instances (a from image i, b from image i+1),
            compute overlap: how many pixels of a map into b
         d. Build cost matrix and solve with Hungarian algorithm
         e. Assign consistent global IDs to image i+1's instances
    3. Count unique global IDs = apple count

Usage:
    python associate_masks.py \
        --pointmap  ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --filenames ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --out       ~/ba/output_vggt/t02_360_64v_assoc_count.txt

Optional:
    --max_dist    max 3D distance for a pixel correspondence to be valid (default: 0.5)
    --min_overlap min overlap fraction to consider a match (default: 0.1)
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
    """Load instance mask resized to (H, W). Returns None if not found."""
    if not os.path.exists(path):
        return None
    mask = np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))
    return mask


def get_instance_ids(mask):
    """Return sorted list of non-zero instance IDs in mask."""
    if mask is None:
        return []
    ids = np.unique(mask)
    return ids[ids != 0].tolist()


# ── Pixel correspondence via KD-tree ─────────────────────────────────────────

def build_correspondence(pm1, pm2, max_dist=0.5):
    """
    For each valid pixel in pm1, find the nearest pixel in pm2 in 3D space.

    Returns:
        corr: (H*W,) int array — flat index into pm2 for each pixel in pm1.
              -1 if no valid correspondence found (NaN or distance > max_dist)
    """
    H, W = pm1.shape[:2]

    pts1 = pm1.reshape(-1, 3)   # (H*W, 3)
    pts2 = pm2.reshape(-1, 3)   # (H*W, 3)

    # Valid (non-NaN) pixels in each image
    valid1 = ~np.isnan(pts1).any(axis=1)
    valid2 = ~np.isnan(pts2).any(axis=1)

    valid2_idx = np.where(valid2)[0]

    if valid2_idx.size == 0 or valid1.sum() == 0:
        return np.full(H * W, -1, dtype=np.int32)

    # Build KD-tree from image 2's valid points
    tree = cKDTree(pts2[valid2])

    # Query image 1's valid points
    valid1_idx = np.where(valid1)[0]
    distances, nn_in_valid2 = tree.query(pts1[valid1], workers=-1)

    # Map back to flat indices in pm2
    nn_flat = valid2_idx[nn_in_valid2]

    # Build full correspondence array
    corr = np.full(H * W, -1, dtype=np.int32)
    # Only keep correspondences within max_dist
    good = distances <= max_dist
    corr[valid1_idx[good]] = nn_flat[good]

    return corr   # (H*W,) — for pixel flat_idx in image1, gives flat_idx in image2


# ── Cost matrix ───────────────────────────────────────────────────────────────

def compute_cost_matrix(mask1, mask2, corr, ids1, ids2, H, W):
    """
    Build cost matrix C where C[a,b] = 1 - overlap(a,b)/min(|a|,|b|).
    overlap(a,b) = pixels in instance a whose correspondence falls in instance b.
    """
    if not ids1 or not ids2:
        return np.ones((max(1, len(ids1)), max(1, len(ids2))), dtype=np.float32)

    mask1_flat = mask1.reshape(-1)   # (H*W,)
    mask2_flat = mask2.reshape(-1)   # (H*W,)

    C = np.ones((len(ids1), len(ids2)), dtype=np.float32)

    for ai, a in enumerate(ids1):
        pixels_a = np.where(mask1_flat == a)[0]   # flat indices in image1
        size_a   = len(pixels_a)
        if size_a == 0:
            continue

        # Find where these pixels correspond to in image2
        corr_a = corr[pixels_a]                   # flat indices in image2
        valid  = corr_a >= 0
        corr_a_valid = corr_a[valid]

        if len(corr_a_valid) == 0:
            continue

        # Which instance in image2 do they land in?
        landed_ids = mask2_flat[corr_a_valid]

        for bi, b in enumerate(ids2):
            overlap = int((landed_ids == b).sum())
            size_b  = int((mask2_flat == b).sum())
            if min(size_a, size_b) > 0:
                C[ai, bi] = 1.0 - overlap / min(size_a, size_b)

    return C


# ── Hungarian assignment ──────────────────────────────────────────────────────

def assign_global_ids(ids1, ids2, cost_matrix, global_ids_1,
                      next_global_id, min_overlap=0.1):
    """
    Run Hungarian algorithm on cost matrix.
    Instances in image2 that match image1 inherit global IDs.
    Unmatched instances in image2 get new global IDs.

    Returns:
        global_ids_2: dict {local_id -> global_id} for image2
        next_global_id: updated counter
    """
    global_ids_2 = {}

    if not ids1 or not ids2:
        # No match possible — assign new IDs to everything in image2
        for b in ids2:
            global_ids_2[b] = next_global_id
            next_global_id += 1
        return global_ids_2, next_global_id

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_b = set()
    for ai, bi in zip(row_ind, col_ind):
        cost = cost_matrix[ai, bi]
        if cost < (1.0 - min_overlap):   # sufficient overlap
            a = ids1[ai]
            b = ids2[bi]
            global_ids_2[b] = global_ids_1[a]
            matched_b.add(b)

    # Unmatched instances in image2 → new global IDs
    for b in ids2:
        if b not in matched_b:
            global_ids_2[b] = next_global_id
            next_global_id += 1

    return global_ids_2, next_global_id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",    required=True)
    parser.add_argument("--filenames",   required=True)
    parser.add_argument("--masks",       required=True)
    parser.add_argument("--out",         default=None)
    parser.add_argument("--max_dist",    type=float, default=0.5,
                        help="Max 3D distance for valid pixel correspondence (default: 0.5)")
    parser.add_argument("--min_overlap", type=float, default=0.1,
                        help="Min overlap fraction to accept a match (default: 0.1)")
    args = parser.parse_args()

    print("=" * 60)
    print("Apple Counter -- Pointmap Association + Hungarian")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading inputs...")
    point_map = load_pointmap(args.pointmap)
    filenames = load_filenames(args.filenames)
    N, H, W, _ = point_map.shape

    if len(filenames) != N:
        print(f"ERROR: filenames ({len(filenames)}) != pointmap N ({N})")
        sys.exit(1)

    # ── Load all masks upfront ────────────────────────────────────────────────
    print(f"Loading {N} masks...")
    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        masks.append(load_mask(mpath, W, H))

    loaded = sum(1 for m in masks if m is not None)
    print(f"  Loaded {loaded}/{N} masks")

    # ── Sequential association ────────────────────────────────────────────────
    print(f"\nAssociating instances across {N} images...")

    # Assign initial global IDs to image 0
    ids0 = get_instance_ids(masks[0])
    next_global_id = 1
    global_ids = [{}] * N

    if ids0:
        global_ids[0] = {iid: next_global_id + i for i, iid in enumerate(ids0)}
        next_global_id += len(ids0)
    else:
        global_ids[0] = {}

    print(f"  [0] {filenames[0]}: {len(ids0)} instances, "
          f"global IDs {list(global_ids[0].values())[:5]}{'...' if len(ids0)>5 else ''}")

    for i in range(N - 1):
        ids_i   = get_instance_ids(masks[i])
        ids_i1  = get_instance_ids(masks[i + 1])

        if not ids_i1:
            global_ids[i + 1] = {}
            print(f"  [{i+1}] {filenames[i+1]}: no instances")
            continue

        if not ids_i or masks[i] is None:
            # No previous instances to match against — assign new IDs
            global_ids[i + 1] = {}
            for iid in ids_i1:
                global_ids[i + 1][iid] = next_global_id
                next_global_id += 1
            print(f"  [{i+1}] {filenames[i+1]}: {len(ids_i1)} new instances (no prev)")
            continue

        # Pixel correspondence via KD-tree
        corr = build_correspondence(point_map[i], point_map[i + 1],
                                    max_dist=args.max_dist)

        # Cost matrix
        C = compute_cost_matrix(masks[i], masks[i + 1], corr,
                                ids_i, ids_i1, H, W)

        # Hungarian assignment
        global_ids[i + 1], next_global_id = assign_global_ids(
            ids_i, ids_i1, C, global_ids[i],
            next_global_id, args.min_overlap
        )

        matched = sum(1 for iid in ids_i1
                      if global_ids[i + 1].get(iid) in global_ids[i].values())
        print(f"  [{i+1}] {filenames[i+1]}: {len(ids_i1)} instances, "
              f"{matched} matched, {len(ids_i1)-matched} new")

    # ── Count unique global IDs ───────────────────────────────────────────────
    all_global_ids = set()
    for gids in global_ids:
        all_global_ids.update(gids.values())

    apple_count = len(all_global_ids)

    print(f"\n{'='*60}")
    print(f"  APPLE COUNT: {apple_count}")
    print(f"  (ground truth tree_02: 113)")
    print(f"  Error: {apple_count - 113:+d}  ({100*(apple_count-113)/113:+.1f}%)")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"apple_count: {apple_count}\n")
            f.write(f"ground_truth: 113\n")
            f.write(f"error: {apple_count - 113:+d}\n")
            f.write(f"max_dist: {args.max_dist}\n")
            f.write(f"min_overlap: {args.min_overlap}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()