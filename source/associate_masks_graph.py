"""
associate_masks_graph.py  --  Count apples using graph-based mask association.

Pipeline:
  - Nodes  = (image_index, instance_id)  one per SAM3 instance per image
  - Edges  = Hungarian matches between instances across image pairs
  - Apple-to-apple correspondence: KD-tree built from apple pixels only
  - Connected components of the graph = unique physical apples

Usage:
    python associate_masks_graph.py \
        --pointmap   ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --filenames  ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks      ~/ba/output_sam/tree_02/semantics_sam3 \
        --transforms ~/ba/data/FruitNeRF_Real/FruitNeRF_Dataset/tree_02/transforms.json \
        --out        ~/ba/output_vggt/t02_360_64v_graph_count.txt

Optional:
    --cam_dist_thresh    max camera distance to consider a pair (default: 999)
    --corr_thresh        max 3D distance for valid pixel correspondence (default: 0.38)
    --min_overlap_pct    min % apple pixels within corr_thresh to run Hungarian (default: 5)
    --min_match_overlap  min overlap fraction to accept a Hungarian match (default: 0.01)
    --ground_truth       ground truth apple count for error reporting (default: 113)
    --save_colored_ply   save colored PLY where each unique apple has a distinct color
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import Counter
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


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


def compute_correspondence(pm1, pm2, mask1, mask2, H, W, corr_thresh):
    """
    Apple-to-apple correspondence: for each apple pixel in image 1,
    find nearest apple pixel in image 2 via KD-tree.
    """
    pts2_flat = pm2.reshape(-1, 3)

    if mask2 is not None:
        apple2_flat = (mask2.reshape(-1) > 0)
        valid2 = apple2_flat & ~np.isnan(pts2_flat).any(axis=1)
    else:
        valid2 = ~np.isnan(pts2_flat).any(axis=1)

    valid2_idx = np.where(valid2)[0]
    if valid2_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32), 0.0

    tree = cKDTree(pts2_flat[valid2])

    if mask1 is not None:
        apple1_flat = (mask1.reshape(-1) > 0)
        valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
        apple1_idx = np.where(apple1_flat & valid1_flat)[0]
    else:
        apple1_idx = np.where(~np.isnan(pm1.reshape(-1, 3)).any(axis=1))[0]

    if apple1_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32), 0.0

    distances, nn = tree.query(pm1.reshape(-1, 3)[apple1_idx], workers=-1)
    nn_flat = valid2_idx[nn]
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
            if min(size_a, size_b) > 0:
                C[ai, bi] = 1.0 - overlap / min(size_a, size_b)

    return C


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

    def component_sizes(self):
        roots = [self.find(x) for x in self.parent]
        return sorted(Counter(roots).values(), reverse=True)

    def get_component_map(self):
        """Returns (node_to_comp, root_counts) sorted by size descending."""
        roots = {x: self.find(x) for x in self.parent}
        root_counts = Counter(roots.values())
        sorted_roots = [r for r, _ in root_counts.most_common()]
        root_to_id = {r: i for i, r in enumerate(sorted_roots)}
        node_to_comp = {x: root_to_id[roots[x]] for x in self.parent}
        return node_to_comp, root_counts


def make_colors(n):
    """Generate n visually distinct colors using bit-reversal permutation of hue."""
    colors = []
    for comp_id in range(n):
        # Bit-reversal permutation for maximum hue separation
        idx = 0
        bit = comp_id
        for b in range(8):
            idx = (idx << 1) | (bit & 1)
            bit >>= 1
        hue = (idx % max(n, 1)) / max(n, 1)
        h = hue * 6
        i = int(h)
        f = h - i
        palette = [
            [255, int(255*f), 0],
            [int(255*(1-f)), 255, 0],
            [0, 255, int(255*f)],
            [0, int(255*(1-f)), 255],
            [int(255*f), 0, 255],
            [255, 0, int(255*(1-f))],
        ]
        colors.append(palette[i % 6])
    return colors


def save_colored_ply(point_map, masks, filenames, node_to_comp, apple_count, path):
    """Save apple PLY where each unique apple component gets a distinct color."""
    colors_by_comp = make_colors(apple_count)
    N, H, W, _ = point_map.shape

    all_points = []
    all_colors = []

    for i, fname in enumerate(filenames):
        if masks[i] is None:
            continue
        mask_flat = masks[i].reshape(-1)
        pts_flat = point_map[i].reshape(-1, 3)

        for iid in get_instance_ids(masks[i]):
            node = (i, iid)
            if node not in node_to_comp:
                continue
            comp_id = node_to_comp[node]
            color = colors_by_comp[comp_id]

            pixel_idx = np.where(mask_flat == iid)[0]
            pts = pts_flat[pixel_idx]
            valid = ~np.isnan(pts).any(axis=1)
            pts = pts[valid]

            if len(pts) == 0:
                continue

            all_points.append(pts)
            all_colors.append(np.tile(color, (len(pts), 1)))

    if not all_points:
        print("  No points to save!")
        return

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(all_points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(all_points.astype(np.float32), all_colors.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(c))

    print(f"  Colored PLY saved: {path}  ({len(all_points):,} points, {apple_count} colors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",          required=True)
    parser.add_argument("--filenames",         required=True)
    parser.add_argument("--masks",             required=True)
    parser.add_argument("--transforms",        required=True)
    parser.add_argument("--out",               default=None)
    parser.add_argument("--ground_truth",      type=int, default=113)
    parser.add_argument("--cam_dist_thresh",   type=float, default=999.0)
    parser.add_argument("--corr_thresh",       type=float, default=0.38)
    parser.add_argument("--min_overlap_pct",   type=float, default=5.0)
    parser.add_argument("--min_match_overlap", type=float, default=0.01)
    parser.add_argument("--save_colored_ply",  action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Apple Counter -- Graph-based Association")
    print("=" * 60)
    print(f"  cam_dist_thresh={args.cam_dist_thresh}  corr_thresh={args.corr_thresh}")
    print(f"  min_overlap_pct={args.min_overlap_pct}%  min_match_overlap={args.min_match_overlap}")

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

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < args.cam_dist_thresh:
                pairs.append((i, j))

    print(f"\nCandidate pairs (camera dist < {args.cam_dist_thresh}): {len(pairs)}")

    uf = UnionFind()
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            uf.find((i, iid))

    total_nodes = len(uf.parent)
    print(f"Total instance nodes: {total_nodes}")

    print(f"\nProcessing {len(pairs)} pairs...")
    edges_added = 0
    pairs_skipped = 0

    for pair_idx, (i, j) in enumerate(pairs):
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])

        if not ids_i or not ids_j:
            continue

        corr_ij, pct_ij = compute_correspondence(
            point_map[i], point_map[j], masks[i], masks[j], H, W, args.corr_thresh
        )

        if pct_ij < args.min_overlap_pct:
            pairs_skipped += 1
            continue

        C = compute_cost_matrix(masks[i], masks[j], corr_ij, ids_i, ids_j)
        row_ind, col_ind = linear_sum_assignment(C)

        matched = 0
        for ai, bi in zip(row_ind, col_ind):
            if C[ai, bi] < (1.0 - args.min_match_overlap):
                uf.union((i, ids_i[ai]), (j, ids_j[bi]))
                edges_added += 1
                matched += 1

        if (pair_idx + 1) % 200 == 0 or pair_idx == len(pairs) - 1:
            print(f"  [{pair_idx+1:4d}/{len(pairs)}] edges: {edges_added}  skipped: {pairs_skipped}")

    apple_count = uf.components()
    gt = args.ground_truth
    error = apple_count - gt

    # Component size distribution
    sizes = uf.component_sizes()
    print(f"\n  Component size distribution:")
    print(f"    Total components:        {len(sizes)}")
    print(f"    Largest component:       {sizes[0]} nodes")
    print(f"    2nd largest:             {sizes[1] if len(sizes)>1 else 0} nodes")
    print(f"    Median component:        {sizes[len(sizes)//2]} nodes")
    print(f"    Singletons (1 node):     {sum(1 for s in sizes if s==1)}")
    print(f"    Mean nodes/component:    {np.mean(sizes):.1f}")
    print(f"    Max nodes/component:     {sizes[0]}")

    print(f"\n  Pairs processed: {len(pairs) - pairs_skipped}/{len(pairs)}")
    print(f"  Pairs skipped:   {pairs_skipped}")
    print(f"  Edges added:     {edges_added}")
    print(f"  Total nodes:     {total_nodes}")

    print(f"\n{'='*60}")
    print(f"  APPLE COUNT:   {apple_count}")
    print(f"  GROUND TRUTH:  {gt}")
    print(f"  Error: {error:+d}  ({100*error/gt:+.1f}%)")
    print(f"{'='*60}")

    if args.save_colored_ply:
        node_to_comp, root_counts = uf.get_component_map()
        ply_path = (os.path.splitext(args.out)[0] + "_colored.ply"
                    if args.out else "apples_colored.ply")
        print(f"\nSaving colored PLY...")
        save_colored_ply(point_map, masks, filenames, node_to_comp, apple_count, ply_path)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"apple_count: {apple_count}\n")
            f.write(f"ground_truth: {gt}\n")
            f.write(f"error: {error:+d}\n")
            f.write(f"error_pct: {100*error/gt:+.1f}\n")
            f.write(f"cam_dist_thresh: {args.cam_dist_thresh}\n")
            f.write(f"corr_thresh: {args.corr_thresh}\n")
            f.write(f"min_overlap_pct: {args.min_overlap_pct}\n")
            f.write(f"min_match_overlap: {args.min_match_overlap}\n")
            f.write(f"pairs_processed: {len(pairs) - pairs_skipped}\n")
            f.write(f"edges_added: {edges_added}\n")
            f.write(f"largest_component: {sizes[0]}\n")
            f.write(f"singletons: {sum(1 for s in sizes if s==1)}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
