"""
singleton_diagnosis.py -- Classify graph singletons as type (a) rescuable
(a candidate partner exists at relaxed thresholds) or type (b) unfixable
(no partner exists at any relaxation -- the apple genuinely has no
correspondence to any other view in the data).

This reuses the same correspondence machinery as associate_masks_graph.py
but with thresholds relaxed to their loosest plausible values, and only
checks singleton nodes (no merging, no graph construction).

Usage:
    python singleton_diagnosis.py \
        --pointmap  ~/ba/output_vggt/t02_360_122v_pointmap.npy \
        --filenames ~/ba/output_vggt/t02_360_122v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --corr_thresh 0.020 \
        --min_overlap_pct 5.0

Loose thresholds for the rescue check:
    --loose_corr_thresh      max 3D distance, relaxed (default: 0.5 units)
    --loose_min_overlap_pct  min overlap %, relaxed (default: 0.0, i.e. any overlap)
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from collections import Counter


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
    apple2_flat = (mask2.reshape(-1) > 0) if mask2 is not None else np.ones(H * W, bool)
    valid2 = apple2_flat & ~np.isnan(pts2_flat).any(axis=1)
    valid2_idx = np.where(valid2)[0]
    if valid2_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32)

    tree = cKDTree(pts2_flat[valid2])

    apple1_flat = (mask1.reshape(-1) > 0) if mask1 is not None else np.ones(H * W, bool)
    valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
    apple1_idx = np.where(apple1_flat & valid1_flat)[0]
    if apple1_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32)

    distances, nn = tree.query(pm1.reshape(-1, 3)[apple1_idx], workers=-1)
    nn_flat = valid2_idx[nn]

    corr = np.full(H * W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = nn_flat[good]
    return corr


def compute_overlap(mask1, mask2, corr, instance_id, ids2):
    """For a single instance in image 1, return max overlap fraction
    against any instance in image 2."""
    mask1_flat = mask1.reshape(-1)
    mask2_flat = mask2.reshape(-1)
    pixels_a = np.where(mask1_flat == instance_id)[0]
    size_a = len(pixels_a)
    if size_a == 0 or not ids2:
        return 0.0, None

    corr_a = corr[pixels_a]
    valid = corr_a >= 0
    corr_a_valid = corr_a[valid]
    if len(corr_a_valid) == 0:
        return 0.0, None

    landed = mask2_flat[corr_a_valid]
    best_frac = 0.0
    best_id = None
    for b in ids2:
        overlap = int((landed == b).sum())
        size_b = int((mask2_flat == b).sum())
        if min(size_a, size_b) == 0:
            continue
        frac = overlap / min(size_a, size_b)
        if frac > best_frac:
            best_frac = frac
            best_id = b
    return best_frac, best_id


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

    def component_sizes_and_map(self):
        roots = {x: self.find(x) for x in self.parent}
        root_counts = Counter(roots.values())
        return roots, root_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",  required=True)
    parser.add_argument("--filenames", required=True)
    parser.add_argument("--masks",     required=True)
    parser.add_argument("--corr_thresh",      type=float, default=0.020,
                        help="Original corr_thresh used to find singletons (default: 0.020).")
    parser.add_argument("--min_overlap_pct",  type=float, default=5.0,
                        help="Original min_overlap_pct used to find singletons (default: 5.0).")
    parser.add_argument("--loose_corr_thresh", type=float, default=0.5,
                        help="Relaxed corr_thresh for rescue check (default: 0.5 units).")
    parser.add_argument("--loose_min_overlap_pct", type=float, default=0.0,
                        help="Relaxed min_overlap_pct for rescue check (default: 0.0, any overlap).")
    parser.add_argument("--max_singletons_check", type=int, default=None,
                        help="Optional cap on number of singletons to check (for speed testing).")
    args = parser.parse_args()

    print("=" * 60)
    print("Singleton Type (a)/(b) Diagnosis")
    print("=" * 60)
    print(f"  Original thresholds: corr_thresh={args.corr_thresh}  "
          f"min_overlap_pct={args.min_overlap_pct}%")
    print(f"  Loose thresholds:    corr_thresh={args.loose_corr_thresh}  "
          f"min_overlap_pct={args.loose_min_overlap_pct}%")

    print("\nLoading inputs...")
    point_map = np.load(args.pointmap)
    N, H, W, _ = point_map.shape
    print(f"  Pointmap shape: {point_map.shape}")

    filenames = load_filenames(args.filenames)

    print(f"Loading {N} masks...")
    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        masks.append(load_mask(mpath, W, H))
    loaded = sum(1 for m in masks if m is not None)
    print(f"  Loaded {loaded}/{N} masks")

    # ── Step 1: rebuild the original graph to find singletons ────────────────
    print(f"\nRebuilding original graph (corr_thresh={args.corr_thresh}, "
          f"min_overlap_pct={args.min_overlap_pct}%) to find singletons...")

    uf = UnionFind()
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            uf.find((i, iid))

    total_nodes = len(uf.parent)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    min_overlap_frac = args.min_overlap_pct / 100.0

    for pair_idx, (i, j) in enumerate(pairs):
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])
        if not ids_i or not ids_j:
            continue

        corr_ij = compute_correspondence(
            point_map[i], point_map[j], masks[i], masks[j], H, W, args.corr_thresh
        )

        mask1_flat = masks[i].reshape(-1)
        mask2_flat = masks[j].reshape(-1)
        C = np.ones((len(ids_i), len(ids_j)), dtype=np.float32)
        for ai, a in enumerate(ids_i):
            pixels_a = np.where(mask1_flat == a)[0]
            size_a = len(pixels_a)
            if size_a == 0:
                continue
            corr_a = corr_ij[pixels_a]
            valid = corr_a >= 0
            corr_a_valid = corr_a[valid]
            if len(corr_a_valid) == 0:
                continue
            landed = mask2_flat[corr_a_valid]
            for bi, b in enumerate(ids_j):
                overlap = int((landed == b).sum())
                size_b = int((mask2_flat == b).sum())
                if min(size_a, size_b) == 0:
                    continue
                frac = overlap / min(size_a, size_b)
                C[ai, bi] = 1.0 if frac < min_overlap_frac else 1.0 - frac

        row_ind, col_ind = linear_sum_assignment(C)
        for ai, bi in zip(row_ind, col_ind):
            if C[ai, bi] < (1.0 - 0.01):
                uf.union((i, ids_i[ai]), (j, ids_j[bi]))

        if (pair_idx + 1) % 500 == 0 or pair_idx == len(pairs) - 1:
            print(f"  [{pair_idx+1:4d}/{len(pairs)}]")

    roots, root_counts = uf.component_sizes_and_map()
    singletons = [node for node, root in roots.items() if root_counts[root] == 1]
    print(f"\nTotal nodes: {total_nodes}")
    print(f"Singletons found: {len(singletons)}")

    if args.max_singletons_check is not None:
        singletons = singletons[:args.max_singletons_check]
        print(f"Checking first {len(singletons)} singletons (capped)")

    # ── Step 2: for each singleton, check at LOOSE thresholds ────────────────
    print(f"\nChecking each singleton at loose thresholds "
          f"(corr_thresh={args.loose_corr_thresh}, "
          f"min_overlap_pct={args.loose_min_overlap_pct}%)...")

    type_a = 0  # rescuable: some partner exists at loose thresholds
    type_b = 0  # unfixable: no partner exists even at loose thresholds
    loose_overlap_frac = args.loose_min_overlap_pct / 100.0

    for idx, (img_i, iid) in enumerate(singletons):
        found_partner = False
        for j in range(N):
            if j == img_i or masks[j] is None:
                continue
            ids_j = get_instance_ids(masks[j])
            if not ids_j:
                continue

            corr_ij = compute_correspondence(
                point_map[img_i], point_map[j], masks[img_i], masks[j],
                H, W, args.loose_corr_thresh
            )
            best_frac, best_id = compute_overlap(
                masks[img_i], masks[j], corr_ij, iid, ids_j
            )
            if best_frac >= loose_overlap_frac and best_frac > 0:
                found_partner = True
                break

        if found_partner:
            type_a += 1
        else:
            type_b += 1

        if (idx + 1) % 50 == 0 or idx == len(singletons) - 1:
            print(f"  [{idx+1:4d}/{len(singletons)}]  "
                  f"type_a={type_a}  type_b={type_b}")

    total_checked = type_a + type_b
    print(f"\n{'='*60}")
    print(f"  SINGLETON DIAGNOSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Total singletons checked: {total_checked}")
    print(f"  Type (a) rescuable (partner exists at loose thresh): "
          f"{type_a}  ({100*type_a/total_checked:.1f}%)")
    print(f"  Type (b) unfixable (no partner at any relaxation):   "
          f"{type_b}  ({100*type_b/total_checked:.1f}%)")
    print(f"{'='*60}")

    if type_b / total_checked > 0.7:
        print("\n  -> Type (b) dominates: graph correspondence is structurally")
        print("     starved of edges for most singletons. The iterative-graph")
        print("     redesign would mostly NOT help. Global sphere-position")
        print("     approach is the better investment.")
    elif type_a / total_checked > 0.7:
        print("\n  -> Type (a) dominates: most singletons have a rescuable")
        print("     partner that current thresholds are gating out. The")
        print("     iterative/relaxed-threshold graph redesign is well")
        print("     targeted at the dominant failure mode.")
    else:
        print("\n  -> Mixed result: both failure modes are significant.")
        print("     A combined approach (graph for type (a), global position")
        print("     for type (b)) is likely the right architecture.")


if __name__ == "__main__":
    main()
