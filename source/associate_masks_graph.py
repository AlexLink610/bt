"""
associate_masks_graph.py  --  Count apples using graph-based mask association.

Pipeline:
  - Nodes  = (image_index, instance_id)  one per SAM3 instance per image
  - Edges  = Hungarian matches between instances across image pairs
  - Apple-to-apple correspondence: KD-tree built from apple pixels only
  - Connected components of the graph = unique physical apples
  - Optional: VGGT confidence filtering to exclude low-quality 3D points

Key changes vs previous version:
  - min_overlap_pct is now checked PER INSTANCE PAIR, not per image pair.
  - --transforms is now optional. If not provided, all image pairs are used.

Usage:
    python associate_masks_graph.py \
        --pointmap   ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --filenames  ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks      ~/ba/output_sam/tree_02/semantics_sam3 \
        --transforms ~/ba/data/FruitNeRF_Real/FruitNeRF_Dataset/tree_02/transforms.json \
        --out        ~/ba/output_vggt/t02_360_64v_graph_count.txt

Optional:
    --transforms         path to transforms.json for camera-distance filtering
                         (if omitted, all pairs are used)
    --confmap            path to _confmap.npy (N,H,W) normalised 0-1
    --conf_thresh        min confidence to include a pixel (default: 0.3)
    --cam_dist_thresh    max camera distance to consider a pair (default: 999)
    --corr_thresh        max 3D distance for valid pixel correspondence (default: 0.010)
    --min_overlap_pct    min % of an instance's OWN pixels that must land on the
                         matched instance in image j (per-instance gate, default: 5)
    --min_match_overlap  min overlap fraction to accept a Hungarian match (default: 0.01)
    --ground_truth       ground truth count for error reporting (default: 113)
    --save_colored_ply   save colored PLY where each unique instance has a distinct color
"""

import os
import json
import argparse
import numpy as np
from collections import Counter
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import pyransac3d as pyrsc

def load_pointmap(path):
    pm = np.load(path)
    print(f"  Pointmap shape: {pm.shape}  ({pm.nbytes / 1e6:.1f} MB)")
    return pm


def load_confmap(path):
    cm = np.load(path)
    print(f"  Confmap shape:  {cm.shape}  ({cm.nbytes / 1e6:.1f} MB)")
    pcts = np.percentile(cm, [10, 25, 50, 75, 90])
    print(f"  Conf percentiles [10,25,50,75,90]: {pcts.round(3)}")
    return cm


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


def compute_correspondence(pm1, pm2, mask1, mask2, conf1, conf2,
                           H, W, corr_thresh, conf_thresh):
    """
    For each object pixel in image 1, find nearest object pixel in image 2
    via KD-tree. Returns a flat corr array mapping pixel index -> pixel index
    in image 2 (or -1 if no match within corr_thresh).
    """
    pts2_flat = pm2.reshape(-1, 3)

    apple2_flat = (mask2.reshape(-1) > 0) if mask2 is not None else np.ones(H * W, bool)
    valid2 = apple2_flat & ~np.isnan(pts2_flat).any(axis=1)
    if conf2 is not None:
        valid2 &= (conf2.reshape(-1) >= conf_thresh)

    valid2_idx = np.where(valid2)[0]
    if valid2_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32)

    tree = cKDTree(pts2_flat[valid2])

    apple1_flat = (mask1.reshape(-1) > 0) if mask1 is not None else np.ones(H * W, bool)
    valid1_flat = ~np.isnan(pm1.reshape(-1, 3)).any(axis=1)
    if conf1 is not None:
        valid1_flat &= (conf1.reshape(-1) >= conf_thresh)
    apple1_idx = np.where(apple1_flat & valid1_flat)[0]

    if apple1_idx.size == 0:
        return np.full(H * W, -1, dtype=np.int32)

    distances, nn = tree.query(pm1.reshape(-1, 3)[apple1_idx], workers=-1)
    nn_flat = valid2_idx[nn]

    corr = np.full(H * W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = nn_flat[good]

    return corr


def compute_cost_matrix(mask1, mask2, corr, ids1, ids2, min_overlap_pct):
    """
    Build cost matrix for Hungarian matching between instances in image 1 and 2.

    min_overlap_pct: per-instance gate — if the fraction of instance a's pixels
    that land on instance b is below this threshold, the pair is treated as
    no-match (cost=1.0).
    """
    if not ids1 or not ids2:
        return np.ones((max(1, len(ids1)), max(1, len(ids2))), dtype=np.float32)

    mask1_flat = mask1.reshape(-1)
    mask2_flat = mask2.reshape(-1)
    C = np.ones((len(ids1), len(ids2)), dtype=np.float32)
    min_overlap_frac = min_overlap_pct / 100.0

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
            if frac < min_overlap_frac:
                C[ai, bi] = 1.0
            else:
                C[ai, bi] = 1.0 - frac

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
        roots = {x: self.find(x) for x in self.parent}
        root_counts = Counter(roots.values())
        sorted_roots = [r for r, _ in root_counts.most_common()]
        root_to_id = {r: i for i, r in enumerate(sorted_roots)}
        node_to_comp = {x: root_to_id[roots[x]] for x in self.parent}
        return node_to_comp, root_counts

    def get_component_nodes(self, root):
        """Return all nodes belonging to the component with the given root."""
        return [x for x in self.parent if self.find(x) == root]


def make_colors(n):
    """Generate n maximally distinct colors using golden ratio hue stepping
    with alternating brightness levels."""
    colors = []
    golden = 0.618033988749895
    h = 0.0
    for i in range(n):
        h = (h + golden) % 1.0
        v = 1.0 if i % 2 == 0 else 0.6
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
        colors.append([int(r*255), int(g*255), int(b*255)])
    return colors

def fit_sphere_to_component(points, thresh=0.008, max_iter=500):
    """
    Fit a sphere to a set of 3D points using RANSAC.
    Returns (center, radius, inlier_ratio) or None if too few points.
    """
    if len(points) < 4:
        return None
    sph = pyrsc.Sphere()
    try:
        center, radius, inliers = sph.fit(points, thresh=thresh, maxIteration=max_iter)
        inlier_ratio = len(inliers) / len(points)
        return center, radius, inlier_ratio
    except Exception:
        return None

def save_colored_ply(point_map, masks, filenames, node_to_comp, instance_count, path):
    colors_by_comp = make_colors(instance_count)
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

    print(f"  Colored PLY saved: {path}  ({len(all_points):,} points, {instance_count} colors)")

def save_sphere_ply(sphere_results, comp_to_nodes, point_map, masks, instance_count, path):
    colors = make_colors(instance_count)

    def sample_sphere(center, radius, n=500):
        phi   = np.random.uniform(0, 2 * np.pi, n)
        costh = np.random.uniform(-1, 1, n)
        sinth = np.sqrt(1 - costh**2)
        return np.stack([
            center[0] + radius * sinth * np.cos(phi),
            center[1] + radius * sinth * np.sin(phi),
            center[2] + radius * costh,
        ], axis=1)

    all_pts, all_colors = [], []
    for comp_id, res in sphere_results.items():
        if res is None:
            continue
        center, radius, inlier_ratio = res
        color = np.array(colors[comp_id], dtype=np.uint8)
        pts = sample_sphere(center, radius)
        all_pts.append(pts)
        all_colors.append(np.tile(color, (len(pts), 1)))

    if not all_pts:
        print("  No spheres to save.")
        return

    all_pts    = np.concatenate(all_pts,    axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(all_pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(all_pts.astype(np.float32), all_colors.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(c))
    print(f"  Sphere PLY saved: {path}  ({len(all_pts):,} points, {len(sphere_results)} spheres)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",          required=True)
    parser.add_argument("--confmap",           default=None,
                        help="Path to _confmap.npy (N,H,W) normalised 0-1.")
    parser.add_argument("--conf_thresh",       type=float, default=0.3,
                        help="Min confidence for a pixel to participate (default: 0.3).")
    parser.add_argument("--filenames",         required=True)
    parser.add_argument("--masks",             required=True)
    parser.add_argument("--transforms",        default=None,
                        help="Path to transforms.json for camera-distance filtering. "
                             "If omitted, all image pairs are used.")
    parser.add_argument("--out",               default=None)
    parser.add_argument("--ground_truth",      type=int, default=113)
    parser.add_argument("--cam_dist_thresh",   type=float, default=999.0)
    parser.add_argument("--corr_thresh",       type=float, default=0.010)
    parser.add_argument("--min_overlap_pct",   type=float, default=5.0,
                        help="Min %% of an instance's OWN pixels that must correspond "
                             "to the matched instance (per-instance gate, default: 5).")
    parser.add_argument("--min_match_overlap", type=float, default=0.01,
                        help="Min overlap fraction for Hungarian match to be accepted "
                             "(default: 0.01).")
    parser.add_argument("--save_colored_ply",  action="store_true")
    parser.add_argument("--sphere_thresh",     type=float, default=0.008,
                    help="RANSAC inlier distance for sphere fitting (default: 0.008 units).")
    parser.add_argument("--sphere_min_inliers", type=float, default=0.0,
                    help="Min inlier ratio to keep a component (0.0 = disabled, default).")
    parser.add_argument("--save_sphere_ply", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Instance Counter -- Graph-based Association")
    print("=" * 60)
    print(f"  cam_dist_thresh={args.cam_dist_thresh}  corr_thresh={args.corr_thresh}")
    print(f"  min_overlap_pct={args.min_overlap_pct}% (per-instance)  "
          f"min_match_overlap={args.min_match_overlap}")
    if args.confmap:
        print(f"  conf_thresh={args.conf_thresh}")
    else:
        print(f"  conf filtering: DISABLED")
    if args.transforms is None:
        print(f"  transforms: NOT PROVIDED — using all pairs")

    print("\nLoading inputs...")
    point_map = load_pointmap(args.pointmap)

    conf_map = None
    if args.confmap is not None:
        conf_map = load_confmap(args.confmap)

    filenames = load_filenames(args.filenames)
    N, H, W, _ = point_map.shape

    # Build candidate pairs
    if args.transforms is not None:
        positions = load_camera_positions(args.transforms, filenames)
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < args.cam_dist_thresh:
                    pairs.append((i, j))
        print(f"\nCandidate pairs (camera dist < {args.cam_dist_thresh}): {len(pairs)}")
    else:
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        print(f"\nCandidate pairs (all): {len(pairs)}")

    print(f"Loading {N} masks...")
    masks = []
    for fname in filenames:
        mpath = mask_path_for(args.masks, fname)
        masks.append(load_mask(mpath, W, H))
    loaded = sum(1 for m in masks if m is not None)
    print(f"  Loaded {loaded}/{N} masks")

    uf = UnionFind()
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            uf.find((i, iid))

    total_nodes = len(uf.parent)
    print(f"Total instance nodes: {total_nodes}")

    print(f"\nProcessing {len(pairs)} pairs...")
    edges_added = 0

    for pair_idx, (i, j) in enumerate(pairs):
        ids_i = get_instance_ids(masks[i])
        ids_j = get_instance_ids(masks[j])

        if not ids_i or not ids_j:
            continue

        conf_i = conf_map[i] if conf_map is not None else None
        conf_j = conf_map[j] if conf_map is not None else None

        corr_ij = compute_correspondence(
            point_map[i], point_map[j], masks[i], masks[j],
            conf_i, conf_j, H, W, args.corr_thresh, args.conf_thresh
        )

        C = compute_cost_matrix(
            masks[i], masks[j], corr_ij, ids_i, ids_j, args.min_overlap_pct
        )
        row_ind, col_ind = linear_sum_assignment(C)

        for ai, bi in zip(row_ind, col_ind):
            if C[ai, bi] < (1.0 - args.min_match_overlap):
                # One-node-per-image-per-component cap:
                # only merge if neither component already contains a node from the other's image
                root_i = uf.find((i, ids_i[ai]))
                root_j = uf.find((j, ids_j[bi]))
                if root_i == root_j:
                    continue
                comp_images_i = {img for (img, _) in uf.get_component_nodes(root_i)}
                comp_images_j = {img for (img, _) in uf.get_component_nodes(root_j)}
                if comp_images_i & comp_images_j:
                    continue  # merging would create two nodes from the same image
                uf.union((i, ids_i[ai]), (j, ids_j[bi]))
                edges_added += 1

        if (pair_idx + 1) % 200 == 0 or pair_idx == len(pairs) - 1:
            print(f"  [{pair_idx+1:4d}/{len(pairs)}] edges: {edges_added}")

    instance_count = uf.components()
    gt = args.ground_truth
    error = instance_count - gt

    sizes = uf.component_sizes()
    # ── Sphere fitting (post-processing) ─────────────────────────────────────
    node_to_comp_sf, _ = uf.get_component_map()
    comp_to_nodes = {}
    for node, comp_id in node_to_comp_sf.items():
        comp_to_nodes.setdefault(comp_id, []).append(node)

    sphere_results = {}   # comp_id -> (center, radius, inlier_ratio) or None
    filtered_out = 0

    if args.sphere_min_inliers > 0.0 or True:   # always run for diagnostics
        print(f"\nFitting spheres to {len(comp_to_nodes)} components "
              f"(thresh={args.sphere_thresh})...")
        for comp_id, nodes in comp_to_nodes.items():
            pts_list = []
            for (img_idx, iid) in nodes:
                if masks[img_idx] is None:
                    continue
                mask_flat = masks[img_idx].reshape(-1)
                pts_flat  = point_map[img_idx].reshape(-1, 3)
                px = np.where(mask_flat == iid)[0]
                pts = pts_flat[px]
                valid = ~np.isnan(pts).any(axis=1)
                pts_list.append(pts[valid])
            if not pts_list:
                continue
            all_pts = np.concatenate(pts_list, axis=0)
            sphere_results[comp_id] = fit_sphere_to_component(
                all_pts, thresh=args.sphere_thresh)

        ratios = [r[2] for r in sphere_results.values() if r is not None]
        if ratios:
            print(f"  Sphere inlier ratio — "
                  f"mean: {np.mean(ratios):.2f}  "
                  f"median: {np.median(ratios):.2f}  "
                  f"min: {np.min(ratios):.2f}  "
                  f"max: {np.max(ratios):.2f}")

    if args.sphere_min_inliers > 0.0:
        kept = {cid for cid, res in sphere_results.items()
                if res is not None and res[2] >= args.sphere_min_inliers}
        filtered_out = len(comp_to_nodes) - len(kept)
        instance_count = len(kept)
        print(f"  Filtered {filtered_out} components below "
              f"inlier ratio {args.sphere_min_inliers:.2f}")
        print(f"  Remaining after sphere filter: {instance_count}")
    print(f"\n  Component size distribution:")
    print(f"    Total components:        {len(sizes)}")
    print(f"    Largest component:       {sizes[0]} nodes")
    print(f"    2nd largest:             {sizes[1] if len(sizes)>1 else 0} nodes")
    print(f"    Median component:        {sizes[len(sizes)//2]} nodes")
    print(f"    Singletons (1 node):     {sum(1 for s in sizes if s==1)}")
    print(f"    Mean nodes/component:    {np.mean(sizes):.1f}")
    print(f"    Max nodes/component:     {sizes[0]}")

    print(f"\n  Edges added:     {edges_added}")
    print(f"  Total nodes:     {total_nodes}")

    print(f"\n{'='*60}")
    print(f"  INSTANCE COUNT: {instance_count}")
    print(f"  GROUND TRUTH:   {gt}")
    print(f"  Error: {error:+d}" + (f"  ({100*error/gt:+.1f}%)" if gt > 0 else ""))
    print(f"{'='*60}")

    if args.save_colored_ply:
        node_to_comp, root_counts = uf.get_component_map()
        ply_path = (os.path.splitext(args.out)[0] + "_colored.ply"
                    if args.out else "instances_colored.ply")
        print(f"\nSaving colored PLY...")
        save_colored_ply(point_map, masks, filenames, node_to_comp, instance_count, ply_path)

    if args.save_sphere_ply:
        sphere_ply_path = (os.path.splitext(args.out)[0] + "_spheres.ply"
                           if args.out else "instances_spheres.ply")
        print(f"\nSaving sphere PLY...")
        save_sphere_ply(sphere_results, comp_to_nodes, point_map, masks, instance_count, sphere_ply_path)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"instance_count: {instance_count}\n")
            f.write(f"ground_truth: {gt}\n")
            f.write(f"error: {error:+d}\n")
            f.write(f"error_pct: {100*error/gt:+.1f}\n" if gt > 0 else "error_pct: N/A\n")
            f.write(f"cam_dist_thresh: {args.cam_dist_thresh}\n")
            f.write(f"corr_thresh: {args.corr_thresh}\n")
            f.write(f"min_overlap_pct: {args.min_overlap_pct}\n")
            f.write(f"min_match_overlap: {args.min_match_overlap}\n")
            f.write(f"conf_thresh: {args.conf_thresh if args.confmap else 'disabled'}\n")
            f.write(f"edges_added: {edges_added}\n")
            f.write(f"largest_component: {sizes[0]}\n")
            f.write(f"singletons: {sum(1 for s in sizes if s==1)}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
