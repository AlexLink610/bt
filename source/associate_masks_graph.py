"""
associate_masks_graph.py  --  Count apples using graph-based mask association.

Instead of sequential association (which suffers from track fragmentation),
this builds a graph where:
  - Nodes  = (image_index, instance_id)  one per SAM3 instance per image
  - Edges  = Hungarian matches between instances across image pairs

Only image pairs within a camera distance threshold are considered,
keeping computation tractable.

Connected components of the graph = unique physical apples.

Usage:
    python associate_masks_graph.py \
        --pointmap   ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --filenames  ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks      ~/ba/output_sam/tree_02/semantics_sam3 \
        --transforms ~/ba/data/FruitNeRF_Real/FruitNeRF_Dataset/tree_02/transforms.json \
        --out        ~/ba/output_vggt/t02_360_64v_graph_count.txt

Optional:
    --cam_dist_thresh    max camera distance to consider a pair (default: 2.0)
    --corr_thresh        max 3D distance for valid pixel correspondence (default: 0.05)
    --min_overlap_pct    min % apple pixels within corr_thresh to run Hungarian (default: 10)
    --min_match_overlap  min overlap fraction to accept a Hungarian match (default: 0.05)
"""

import os
import sys
import json
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


def load_camera_positions(transforms_path, fnames):
    with open(transforms_path) as f:
        data = json.load(f)
    name_to_pos = {}
    for frame in data["frames"]:
        M = np.array(frame["transform_matrix"])
        name_to_pos[os.path.basename(frame["file_path"])] = np.array([M[0,3], M[1,3], M[2,3]])
    return np.array([name_to_pos[f] for f in fnames])


# ── Correspondence + cost matrix ──────────────────────────────────────────────

def compute_correspondence(pm1, pm2, mask1, mask2, H, W, corr_thresh):
    """
    For apple pixels in image 1, find nearest apple pixel in image 2.
    KD-tree built from apple pixels of image 2 only, so correspondences
    are forced to land on actual apple detections rather than background.
    Returns (corr flat array, pct_good).
    """
    pts2_flat = pm2.reshape(-1, 3)

    # Build KD-tree from apple pixels of image 2 only
    if mask2 is not None:
        apple2_flat = (mask2.reshape(-1) > 0)
        valid2      = apple2_flat & ~np.isnan(pts2_flat).any(axis=1)
    else:
        valid2      = ~np.isnan(pts2_flat).any(axis=1)

    valid2_idx = np.where(valid2)[0]

    if valid2_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32), 0.0

    tree = cKDTree(pts2_flat[valid2])

    if mask1 is not None:
        apple1_flat = (mask1.reshape(-1) > 0)
        valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
        apple1_idx  = np.where(apple1_flat & valid1_flat)[0]
    else:
        apple1_idx = np.where(~np.isnan(pm1.reshape(-1, 3)).any(axis=1))[0]

    if apple1_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32), 0.0

    distances, nn = tree.query(pm1.reshape(-1, 3)[apple1_idx], workers=-1)
    nn_flat  = valid2_idx[nn]
    pct_good = float((distances < corr_thresh).mean() * 100)

    corr = np.full(H * W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = nn_flat[good]

    return corr, pct_good


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
        landed = mask2_flat[corr_a_valid]
        for bi, b in enumerate(ids2):
            overlap = int((landed == b).sum())
            size_b  = int((mask2_flat == b).sum())
            if min(size_a, size_b) > 0:
                C[ai, bi] = 1.0 - overlap / min(size_a, size_b)

    return C


# ── Union-Find for connected components ───────────────────────────────────────

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def components(self):
        roots = set(self.find(x) for x in self.parent)
        return len(roots)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",          required=True)
    parser.add_argument("--filenames",         required=True)
    parser.add_argument("--masks",             required=True)
    parser.add_argument("--transforms",        required=True)
    parser.add_argument("--out",               default=None)
    parser.add_argument("--cam_dist_thresh",   type=float, default=2.0,
                        help="Max camera distance to consider a pair (default: 2.0)")
    parser.add_argument("--corr_thresh",       type=float, default=0.05,
                        help="Max 3D distance for valid correspondence (default: 0.05)")
    parser.add_argument("--min_overlap_pct",   type=float, default=10.0,
                        help="Min %% apple pixels within corr_thresh (default: 10)")
    parser.add_argument("--min_match_overlap", type=float, default=0.05,
                        help="Min overlap fraction to accept a match (default: 0.05)")
    args = parser.parse_args()

    print("=" * 60)
    print("Apple Counter -- Graph-based Association")
    print("=" * 60)
    print(f"  cam_dist_thresh={args.cam_dist_thresh}  "
          f"corr_thresh={args.corr_thresh}  "
          f"min_overlap_pct={args.min_overlap_pct}%  "
          f"min_match_overlap={args.min_match_overlap}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading inputs...")
    point_map = load_pointmap(args.pointmap)
    filenames = load_filenames(args.filenames)
    N, H, W, _ = point_map.shape

    positions = load_camera_positions(args.transforms, filenames)

    print(f"Loading {N} masks...")
    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        masks.append(load_mask(mpath, W, H))
    loaded = sum(1 for m in masks if m is not None)
    print(f"  Loaded {loaded}/{N} masks")

    # ── Find candidate pairs ──────────────────────────────────────────────────
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < args.cam_dist_thresh:
                pairs.append((i, j))

    print(f"\nCandidate pairs (camera dist < {args.cam_dist_thresh}): {len(pairs)}")

    # ── Build graph ───────────────────────────────────────────────────────────
    uf = UnionFind()

    # Register all instances as nodes
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            uf.find((i, iid))

    total_nodes = len(uf.parent)
    print(f"Total instance nodes: {total_nodes}")

    # Process each pair
    print(f"\nProcessing {len(pairs)} pairs...")
    edges_added = 0
    pairs_skipped = 0

    for pair_idx, (i, j) in enumerate(pairs):
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])

        if not ids_i or not ids_j:
            continue

        # Correspondence i → j
        corr_ij, pct_ij = compute_correspondence(
            point_map[i], point_map[j], masks[i], masks[j], H, W, args.corr_thresh
        )

        if pct_ij < args.min_overlap_pct:
            pairs_skipped += 1
            continue

        # Cost matrix + Hungarian
        C = compute_cost_matrix(masks[i], masks[j], corr_ij, ids_i, ids_j)
        row_ind, col_ind = linear_sum_assignment(C)

        matched = 0
        for ai, bi in zip(row_ind, col_ind):
            if C[ai, bi] < (1.0 - args.min_match_overlap):
                uf.union((i, ids_i[ai]), (j, ids_j[bi]))
                edges_added += 1
                matched += 1

        if (pair_idx + 1) % 10 == 0 or pair_idx == len(pairs) - 1:
            print(f"  [{pair_idx+1:3d}/{len(pairs)}] "
                  f"pair ({filenames[i]} <-> {filenames[j]}): "
                  f"overlap={pct_ij:.1f}%  matched={matched}")

    # ── Count connected components ────────────────────────────────────────────
    apple_count = uf.components()

    print(f"\n  Pairs processed: {len(pairs) - pairs_skipped}/{len(pairs)}")
    print(f"  Pairs skipped (low overlap): {pairs_skipped}")
    print(f"  Edges added (matches): {edges_added}")
    print(f"  Total nodes: {total_nodes}")

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
            f.write(f"cam_dist_thresh: {args.cam_dist_thresh}\n")
            f.write(f"corr_thresh: {args.corr_thresh}\n")
            f.write(f"min_overlap_pct: {args.min_overlap_pct}\n")
            f.write(f"min_match_overlap: {args.min_match_overlap}\n")
            f.write(f"pairs_processed: {len(pairs) - pairs_skipped}\n")
            f.write(f"edges_added: {edges_added}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
