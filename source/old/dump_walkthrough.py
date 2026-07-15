"""
dump_walkthrough.py  --  Dump all intermediate algorithm data for presentation.

Prints:
  - Instance IDs per image
  - Per-pair correspondence quality
  - Cost matrices (rounded, human-readable)
  - Hungarian matching results
  - Edges added
  - Final connected components

Usage:
    python3 dump_walkthrough.py \
        --pointmap  ~/ba/output_vggt/old_room_naive_3v_pointmap.npy \
        --filenames ~/ba/output_vggt/old_room_naive_3v_filenames.txt \
        --masks     ~/ba/output_sam/old_room/semantics_sam3 \
        --corr_thresh     0.018 \
        --min_overlap_pct 5
"""

import os
import argparse
import numpy as np
from collections import Counter
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


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
        return np.full(H*W, -1, dtype=np.int32), 0.0
    tree = cKDTree(pts2_flat[valid2])
    apple1_flat = (mask1.reshape(-1) > 0) if mask1 is not None else np.ones(H*W, bool)
    valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
    apple1_idx = np.where(apple1_flat & valid1_flat)[0]
    if apple1_idx.size == 0:
        return np.full(H*W, -1, dtype=np.int32), 0.0
    distances, nn = tree.query(pm1.reshape(-1, 3)[apple1_idx], workers=-1)
    nn_flat = valid2_idx[nn]
    pct_good = float((distances < corr_thresh).mean() * 100)
    corr = np.full(H*W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = nn_flat[good]
    return corr, pct_good

def compute_cost_matrix(mask1, mask2, corr, ids1, ids2, min_overlap_pct):
    if not ids1 or not ids2:
        return np.ones((max(1,len(ids1)), max(1,len(ids2))), dtype=np.float32), {}
    mask1_flat = mask1.reshape(-1)
    mask2_flat = mask2.reshape(-1)
    C = np.ones((len(ids1), len(ids2)), dtype=np.float32)
    min_frac = min_overlap_pct / 100.0
    overlaps = {}
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
            overlaps[(a, b)] = frac
            if frac < min_frac:
                C[ai, bi] = 1.0
            else:
                C[ai, bi] = 1.0 - frac
    return C, overlaps


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
        return len(set(self.find(x) for x in self.parent))
    def component_map(self):
        from collections import Counter
        roots = {x: self.find(x) for x in self.parent}
        root_counts = Counter(roots.values())
        sorted_roots = [r for r, _ in root_counts.most_common()]
        root_to_id = {r: i for i, r in enumerate(sorted_roots)}
        return {x: root_to_id[roots[x]] for x in self.parent}


def print_cost_matrix(C, ids1, ids2, overlaps, min_overlap_pct):
    col_w = 10
    header = f"{'':12s}" + "".join(f"  inst_{b:<4d}" for b in ids2)
    print(header)
    print("  " + "-" * (12 + col_w * len(ids2)))
    for ai, a in enumerate(ids1):
        row = f"  inst_{a:<6d}"
        for bi, b in enumerate(ids2):
            c = C[ai, bi]
            frac = overlaps.get((a, b), None)
            if c >= 1.0:
                cell = "  [  -  ]"
            else:
                cell = f"  [{frac*100:4.1f}% ]"
            row += cell
        print(row)
    print(f"  (cells show overlap %; '-' = below {min_overlap_pct}% threshold or no correspondence)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",        required=True)
    parser.add_argument("--filenames",       required=True)
    parser.add_argument("--masks",           required=True)
    parser.add_argument("--corr_thresh",     type=float, default=0.018)
    parser.add_argument("--min_overlap_pct", type=float, default=5.0)
    args = parser.parse_args()

    print("=" * 70)
    print("ALGORITHM WALKTHROUGH DUMP")
    print("=" * 70)

    point_map = np.load(args.pointmap)
    N, H, W, _ = point_map.shape
    filenames = load_filenames(args.filenames)

    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        m = load_mask(mpath, W, H)
        masks.append(m)

    # ── STEP 1: Nodes ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: GRAPH NODES")
    print("─" * 70)
    uf = UnionFind()
    total_nodes = 0
    for i, fname in enumerate(filenames):
        ids = get_instance_ids(masks[i])
        print(f"  Image {i} ({fname}): {len(ids)} instances  →  IDs: {ids}")
        for iid in ids:
            uf.find((i, iid))
            total_nodes += 1
    print(f"\n  Total nodes: {total_nodes}")

    # ── STEP 2-5: Per pair ────────────────────────────────────────────────────
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    all_edges = []

    for i, j in pairs:
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])
        fname_i = filenames[i]
        fname_j = filenames[j]

        print("\n" + "─" * 70)
        print(f"PAIR: Image {i} ({fname_i})  ↔  Image {j} ({fname_j})")
        print("─" * 70)

        if not ids_i or not ids_j:
            print("  Skipped: one image has no instances")
            continue

        # Step 2: Correspondence
        corr, pct_good = compute_correspondence(
            point_map[i], point_map[j], masks[i], masks[j], H, W, args.corr_thresh
        )
        print(f"\nSTEP 2: Pixel Correspondence")
        print(f"  corr_thresh = {args.corr_thresh}")
        print(f"  {pct_good:.1f}% of instance pixels in image {i} found a match within threshold")

        # Step 3: Cost matrix
        C, overlaps = compute_cost_matrix(
            masks[i], masks[j], corr, ids_i, ids_j, args.min_overlap_pct
        )
        print(f"\nSTEP 3: Cost Matrix  (min_overlap_pct = {args.min_overlap_pct}%)")
        print_cost_matrix(C, ids_i, ids_j, overlaps, args.min_overlap_pct)

        # Step 4: Hungarian
        row_ind, col_ind = linear_sum_assignment(C)
        print(f"\nSTEP 4: Hungarian Matching")
        edges_this_pair = []
        for ai, bi in zip(row_ind, col_ind):
            a = ids_i[ai]
            b = ids_j[bi]
            cost = C[ai, bi]
            if cost < 1.0:
                frac = overlaps.get((a, b), 0)
                print(f"  ✓ MATCH:  (img{i}, inst{a}) ↔ (img{j}, inst{b})  "
                      f"overlap={frac*100:.1f}%  cost={cost:.3f}")
                edges_this_pair.append(((i, a), (j, b), frac))
            else:
                print(f"  ✗ NO MATCH: (img{i}, inst{a}) → no valid counterpart")

        # Step 5: Add edges
        print(f"\nSTEP 5: Add Edges")
        for (n1, n2, frac) in edges_this_pair:
            uf.union(n1, n2)
            all_edges.append((n1, n2, frac))
            print(f"  Edge added: {n1} ↔ {n2}  (overlap {frac*100:.1f}%)")
        if not edges_this_pair:
            print("  No edges added for this pair")

    # ── STEP 6: Components ────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 6: CONNECTED COMPONENTS")
    print("─" * 70)

    comp_map = uf.component_map()
    from collections import defaultdict
    components = defaultdict(list)
    for node, comp_id in comp_map.items():
        components[comp_id].append(node)

    for comp_id in sorted(components.keys()):
        nodes = sorted(components[comp_id])
        print(f"  Component {comp_id}: {nodes}")

    count = uf.components()
    print(f"\n  Total edges:      {len(all_edges)}")
    print(f"  Total nodes:      {total_nodes}")
    print(f"  Components:       {count}  →  COUNT = {count}")

    # ── Summary for slides ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY FOR SLIDES")
    print("=" * 70)
    print(f"  Images:    {N}")
    print(f"  Nodes:     {total_nodes}  (one per instance per image)")
    print(f"  Edges:     {len(all_edges)}")
    print(f"  Result:    {count} unique objects")
    print(f"\n  All edges:")
    for (n1, n2, frac) in all_edges:
        print(f"    {n1} ↔ {n2}  ({frac*100:.1f}% overlap)")


if __name__ == "__main__":
    main()
