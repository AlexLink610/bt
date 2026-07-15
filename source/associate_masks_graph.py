"""
associate_masks_graph.py  --  Count apples using graph-based mask association + DBSCAN on spheres

Usage:
    python associate_masks_graph.py \
        --pointmap   ~/ba/output_vggt/t02_360_64v_pointmap.npy \
        --filenames  ~/ba/output_vggt/t02_360_64v_filenames.txt \
        --masks      ~/ba/output_sam/tree_02/semantics_sam3 \
        --out        ~/ba/output_vggt/t02_360_64v_graph_count.txt

Key optional flags:
    --confmap / --conf_thresh      confidence filtering (usually counterproductive)
    --corr_thresh                  max 3D distance for a valid pixel correspondence
    --dbscan_merge / --dbscan_eps  merge components via DBSCAN on sphere centers
    --max_radius / --min_radius    sphere radius bounds (pre-DBSCAN filter)
    --bilateral                    reciprocal-correspondence down-weighting
    --save_stages_ply PREFIX       write 4 PLYs showing the cloud at each stage:
                                     <PREFIX>_1_nodes.ply     raw (image,instance) nodes
                                     <PREFIX>_2_graph.ply     after graph association
                                     <PREFIX>_3_filtered.ply  after sphere/radius filter
                                                              (dropped = gray)
                                     <PREFIX>_4_merged.ply    after DBSCAN (final clusters)
    --save_merged_ply PATH         only the final-cluster-colored PLY
"""

import os
import json
import argparse
import random
import numpy as np
from collections import Counter
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyransac3d")


# ───────────────────────────── IO helpers ──────────────────────────────────

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
        name_to_pos[os.path.basename(frame["file_path"])] = np.array([M[0, 3], M[1, 3], M[2, 3]])
    return np.array([name_to_pos[f] for f in fnames])


# ─────────────────────────── correspondence ────────────────────────────────

def compute_correspondence(pm1, pm2, mask1, mask2, conf1, conf2,
                           H, W, corr_thresh, conf_thresh):
    """For each apple pixel in image 1, find nearest apple pixel in image 2
    via KD-tree. Returns flat array: pixel index -> pixel index in image 2 (-1 = none)."""
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


def compute_reciprocity(corr_fwd, pm1, pm2, mask1, mask2, H, W, corr_thresh,
                        conf1, conf2, conf_thresh):
    """True where a forward match i->j is reciprocal (mutual nearest neighbour)."""
    corr_bwd = compute_correspondence(pm2, pm1, mask2, mask1, conf2, conf1,
                                      H, W, corr_thresh, conf_thresh)
    reciprocal = np.zeros(H * W, dtype=bool)
    fwd_idx = np.where(corr_fwd >= 0)[0]
    if fwd_idx.size == 0:
        return reciprocal
    j_targets = corr_fwd[fwd_idx]
    back_targets = corr_bwd[j_targets]
    reciprocal[fwd_idx] = (back_targets == fwd_idx)
    return reciprocal


def compute_cost_matrix(mask1, mask2, corr, ids1, ids2, min_overlap_pct,
                        reciprocal=None, reciprocity_penalty_weight=0.3):
    """Hungarian cost matrix between instances of image 1 and image 2."""
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

        if reciprocal is not None:
            recip_a_valid = reciprocal[pixels_a[valid]]

        for bi, b in enumerate(ids2):
            overlap = int((landed == b).sum())
            size_b = int((mask2_flat == b).sum())
            if min(size_a, size_b) == 0:
                continue
            frac = overlap / min(size_a, size_b)
            if frac < min_overlap_frac:
                C[ai, bi] = 1.0
            else:
                base_cost = 1.0 - frac
                if reciprocal is not None:
                    matched_recip = recip_a_valid[landed == b]
                    if len(matched_recip) > 0:
                        recip_ratio = matched_recip.mean()
                        base_cost = min(1.0, base_cost +
                                        (1.0 - recip_ratio) * reciprocity_penalty_weight)
                C[ai, bi] = base_cost
    return C


# ───────────────────────────── union-find ──────────────────────────────────

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

    def component_sizes(self):
        roots = [self.find(x) for x in self.parent]
        return sorted(Counter(roots).values(), reverse=True)

    def get_component_map(self):
        roots = {x: self.find(x) for x in self.parent}
        root_counts = Counter(roots.values())
        sorted_roots = [r for r, _ in root_counts.most_common()]
        root_to_id = {r: i for i, r in enumerate(sorted_roots)}
        return {x: root_to_id[roots[x]] for x in self.parent}, root_counts

    def get_component_nodes(self, root):
        return [x for x in self.parent if self.find(x) == root]


# ────────────────────────────── colors / plys ──────────────────────────────

def make_colors(n):
    """n maximally distinct colors via golden-ratio hue stepping."""
    colors = []
    golden = 0.618033988749895
    h = 0.0
    for i in range(n):
        h = (h + golden) % 1.0
        v = 1.0 if i % 2 == 0 else 0.6
        hi = int(h * 6)
        f = h * 6 - hi
        p, q, t = 0.0, v * (1 - f), v * f
        hi %= 6
        if hi == 0:   r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        else:         r, g, b = v, p, q
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


def _write_ply(pts, cols, path):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(pts.astype(np.float32), cols.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(c))
    print(f"  {path}  ({len(pts):,} points)")


def save_stage_ply(point_map, masks, filenames, node_color_fn, path):
    """Generic stage writer. node_color_fn((img_idx, iid)) -> [r,g,b] or None to skip."""
    all_pts, all_cols = [], []
    for i in range(len(filenames)):
        if i >= len(masks) or masks[i] is None:
            continue
        mask_flat = masks[i].reshape(-1)
        pts_flat = point_map[i].reshape(-1, 3)
        for iid in get_instance_ids(masks[i]):
            col = node_color_fn((i, iid))
            if col is None:
                continue
            px = np.where(mask_flat == iid)[0]
            pts = pts_flat[px]
            pts = pts[~np.isnan(pts).any(axis=1)]
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_cols.append(np.tile(col, (len(pts), 1)))
    if not all_pts:
        print(f"  (no points for {path})")
        return
    _write_ply(np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), path)


def save_colored_ply(point_map, masks, filenames, node_to_comp, n_colors, path):
    colors = make_colors(max(1, n_colors))
    save_stage_ply(point_map, masks, filenames,
                   lambda n: colors[node_to_comp[n]] if n in node_to_comp else None,
                   path)


def fit_sphere_to_component(points, thresh=0.008, max_iter=500):
    """RANSAC sphere fit. Returns (center, radius, inlier_ratio) or None."""
    if len(points) < 4:
        return None
    sph = pyrsc.Sphere()
    try:
        center, radius, inliers = sph.fit(points, thresh=thresh, maxIteration=max_iter)
        return center, radius, len(inliers) / len(points)
    except Exception:
        return None


def save_sphere_ply(sphere_results, n_colors, path):
    colors = make_colors(max(1, n_colors))

    def sample_sphere(center, radius, n=500):
        phi = np.random.uniform(0, 2 * np.pi, n)
        costh = np.random.uniform(-1, 1, n)
        sinth = np.sqrt(1 - costh ** 2)
        return np.stack([
            center[0] + radius * sinth * np.cos(phi),
            center[1] + radius * sinth * np.sin(phi),
            center[2] + radius * costh,
        ], axis=1)

    all_pts, all_cols = [], []
    for comp_id, res in sphere_results.items():
        if res is None:
            continue
        center, radius, _ = res
        col = np.array(colors[comp_id % len(colors)], dtype=np.uint8)
        pts = sample_sphere(center, radius)
        all_pts.append(pts)
        all_cols.append(np.tile(col, (len(pts), 1)))
    if not all_pts:
        print("  No spheres to save.")
        return
    _write_ply(np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), path)


# ──────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",          required=True)
    parser.add_argument("--filenames",         required=True)
    parser.add_argument("--masks",             required=True)
    parser.add_argument("--confmap",           default=None)
    parser.add_argument("--conf_thresh",       type=float, default=0.3)
    parser.add_argument("--transforms",        default=None)
    parser.add_argument("--out",               default=None)
    parser.add_argument("--ground_truth",      type=int, default=113)
    parser.add_argument("--cam_dist_thresh",   type=float, default=999.0)
    parser.add_argument("--corr_thresh",       type=float, default=0.010)
    parser.add_argument("--min_overlap_pct",   type=float, default=5.0)
    parser.add_argument("--min_match_overlap", type=float, default=0.01)
    parser.add_argument("--sphere_thresh",     type=float, default=0.008)
    parser.add_argument("--sphere_min_inliers", type=float, default=0.0)
    parser.add_argument("--dbscan_merge",      action="store_true")
    parser.add_argument("--dbscan_eps",        type=float, default=0.08)
    parser.add_argument("--dbscan_min",        type=int,   default=1)
    parser.add_argument("--max_radius",        type=float, default=0.15)
    parser.add_argument("--min_radius",        type=float, default=0.01)
    parser.add_argument("--bilateral",         action="store_true")
    parser.add_argument("--reciprocity_weight", type=float, default=0.3)
    # visualization
    parser.add_argument("--save_colored_ply",  action="store_true",
                        help="PLY colored by graph component (pre-DBSCAN).")
    parser.add_argument("--save_sphere_ply",   action="store_true")
    parser.add_argument("--save_centers_ply",  default=None,
                        help="PLY of DBSCAN-merged sphere centers.")
    parser.add_argument("--save_merged_ply",   default=None,
                        help="PLY colored by FINAL DBSCAN cluster.")
    parser.add_argument("--save_stages_ply",   default=None,
                        help="Path PREFIX; writes _1_nodes / _2_graph / _3_filtered "
                             "/ _4_merged PLYs showing each pipeline stage.")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("Instance Counter -- Graph-based Association")
    print("=" * 60)
    print(f"  cam_dist_thresh={args.cam_dist_thresh}  corr_thresh={args.corr_thresh}")
    print(f"  min_overlap_pct={args.min_overlap_pct}%  min_match_overlap={args.min_match_overlap}")
    print(f"  conf filtering: {args.conf_thresh if args.confmap else 'DISABLED'}")
    print(f"  bilateral: {'ENABLED (w=%s)' % args.reciprocity_weight if args.bilateral else 'DISABLED'}")

    print("\nLoading inputs...")
    point_map = load_pointmap(args.pointmap)
    conf_map = load_confmap(args.confmap) if args.confmap else None
    filenames = load_filenames(args.filenames)
    N, H, W, _ = point_map.shape

    if args.transforms is not None:
        positions = load_camera_positions(args.transforms, filenames)
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)
                 if np.linalg.norm(positions[i] - positions[j]) < args.cam_dist_thresh]
        print(f"\nCandidate pairs (cam dist < {args.cam_dist_thresh}): {len(pairs)}")
    else:
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        print(f"\nCandidate pairs (all): {len(pairs)}")

    print(f"Loading {N} masks...")
    masks = [load_mask(mask_path_for(args.masks, f), W, H) for f in filenames]
    print(f"  Loaded {sum(1 for m in masks if m is not None)}/{N} masks")

    uf = UnionFind()
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            uf.find((i, iid))
    total_nodes = len(uf.parent)
    print(f"Total instance nodes: {total_nodes}")

    # ── graph association ───────────────────────────────────────────────────
    print(f"\nProcessing {len(pairs)} pairs...")
    edges_added = 0
    for pair_idx, (i, j) in enumerate(pairs):
        ids_i, ids_j = get_instance_ids(masks[i]), get_instance_ids(masks[j])
        if not ids_i or not ids_j:
            continue
        conf_i = conf_map[i] if conf_map is not None else None
        conf_j = conf_map[j] if conf_map is not None else None

        corr_ij = compute_correspondence(point_map[i], point_map[j], masks[i], masks[j],
                                         conf_i, conf_j, H, W,
                                         args.corr_thresh, args.conf_thresh)
        reciprocal_ij = None
        if args.bilateral:
            reciprocal_ij = compute_reciprocity(corr_ij, point_map[i], point_map[j],
                                                masks[i], masks[j], H, W,
                                                args.corr_thresh, conf_i, conf_j,
                                                args.conf_thresh)

        C = compute_cost_matrix(masks[i], masks[j], corr_ij, ids_i, ids_j,
                                args.min_overlap_pct, reciprocal=reciprocal_ij,
                                reciprocity_penalty_weight=args.reciprocity_weight)
        row_ind, col_ind = linear_sum_assignment(C)

        for ai, bi in zip(row_ind, col_ind):
            if C[ai, bi] < (1.0 - args.min_match_overlap):
                root_i = uf.find((i, ids_i[ai]))
                root_j = uf.find((j, ids_j[bi]))
                if root_i == root_j:
                    continue
                imgs_i = {img for (img, _) in uf.get_component_nodes(root_i)}
                imgs_j = {img for (img, _) in uf.get_component_nodes(root_j)}
                if imgs_i & imgs_j:
                    continue  # one node per image per component
                uf.union((i, ids_i[ai]), (j, ids_j[bi]))
                edges_added += 1

        if (pair_idx + 1) % 200 == 0 or pair_idx == len(pairs) - 1:
            print(f"  [{pair_idx+1:4d}/{len(pairs)}] edges: {edges_added}")

    instance_count = uf.components()
    gt = args.ground_truth
    sizes = uf.component_sizes()

    node_to_comp_sf, root_counts = uf.get_component_map()
    comp_to_nodes = {}
    for node, comp_id in node_to_comp_sf.items():
        comp_to_nodes.setdefault(comp_id, []).append(node)

    # ── sphere fitting ──────────────────────────────────────────────────────
    print(f"\nFitting spheres to {len(comp_to_nodes)} components (thresh={args.sphere_thresh})...")
    sphere_results = {}
    for comp_id, nodes in comp_to_nodes.items():
        pts_list = []
        for (img_idx, iid) in nodes:
            if masks[img_idx] is None:
                continue
            mask_flat = masks[img_idx].reshape(-1)
            pts_flat = point_map[img_idx].reshape(-1, 3)
            pts = pts_flat[np.where(mask_flat == iid)[0]]
            pts_list.append(pts[~np.isnan(pts).any(axis=1)])
        if not pts_list:
            continue
        sphere_results[comp_id] = fit_sphere_to_component(
            np.concatenate(pts_list, axis=0), thresh=args.sphere_thresh)

    ratios = [r[2] for r in sphere_results.values() if r is not None]
    if ratios:
        print(f"  Sphere inlier ratio — mean: {np.mean(ratios):.2f}  "
              f"median: {np.median(ratios):.2f}  min: {np.min(ratios):.2f}  "
              f"max: {np.max(ratios):.2f}")

    if args.sphere_min_inliers > 0.0:
        kept = {cid for cid, res in sphere_results.items()
                if res is not None and res[2] >= args.sphere_min_inliers}
        instance_count = len(kept)
        print(f"  Filtered {len(comp_to_nodes) - len(kept)} components below "
              f"inlier ratio {args.sphere_min_inliers:.2f}")
        print(f"  Remaining after sphere filter: {instance_count}")

    # ── DBSCAN merge ────────────────────────────────────────────────────────
    comp_ids, centers, comp_to_cluster, labels, n_clusters = [], [], {}, None, 0
    if args.dbscan_merge:
        print(f"\nRunning DBSCAN merge on sphere centers "
              f"(eps={args.dbscan_eps}, min_samples={args.dbscan_min})...")
        for comp_id, res in sphere_results.items():
            if res is None:
                continue
            center, radius, inlier_ratio = res
            if radius < args.min_radius or radius > args.max_radius:
                continue
            if args.sphere_min_inliers > 0.0 and inlier_ratio < args.sphere_min_inliers:
                continue
            comp_ids.append(comp_id)
            centers.append(center)

        print(f"  Components entering DBSCAN: {len(centers)} "
              f"(of {len(sphere_results)} total, after radius/inlier filtering)")

        if len(centers) > 0:
            centers_arr = np.array(centers)
            labels = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min).fit(centers_arr).labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            print(f"  DBSCAN clusters: {n_clusters}  (noise points: {n_noise})")
            instance_count = n_clusters
            comp_to_cluster = {cid: int(lbl) for cid, lbl in zip(comp_ids, labels)}

            if args.save_centers_ply:
                cluster_colors = make_colors(max(1, n_clusters))
                cols = np.array([[110, 110, 110] if l < 0 else cluster_colors[l]
                                 for l in labels], dtype=np.uint8)
                _write_ply(centers_arr, cols, args.save_centers_ply)
        else:
            print("  No valid sphere centers for DBSCAN — instance_count unchanged.")

    # ── stats ───────────────────────────────────────────────────────────────
    print(f"\n  Component size distribution:")
    print(f"    Total components:        {len(sizes)}")
    print(f"    Largest component:       {sizes[0]} nodes")
    print(f"    2nd largest:             {sizes[1] if len(sizes) > 1 else 0} nodes")
    print(f"    Median component:        {sizes[len(sizes)//2]} nodes")
    print(f"    Singletons (1 node):     {sum(1 for s in sizes if s == 1)}")
    print(f"    Mean nodes/component:    {np.mean(sizes):.1f}")
    print(f"\n  Edges added:     {edges_added}")
    print(f"  Total nodes:     {total_nodes}")

    error = instance_count - gt
    print(f"\n{'='*60}")
    print(f"  INSTANCE COUNT: {instance_count}")
    print(f"  GROUND TRUTH:   {gt}")
    print(f"  Error: {error:+d}" + (f"  ({100*error/gt:+.1f}%)" if gt > 0 else ""))
    print(f"{'='*60}")

    # ── visualization ───────────────────────────────────────────────────────
    comp_colors = make_colors(max(1, len(comp_to_nodes)))
    kept_comps = set(comp_ids) if args.dbscan_merge else set(sphere_results.keys())

    if args.save_stages_ply:
        pre = args.save_stages_ply
        print(f"\nSaving pipeline-stage PLYs...")

        # 1. raw nodes -- every (image, instance) its own color
        node_list = sorted(node_to_comp_sf.keys())
        node_ids = {n: k for k, n in enumerate(node_list)}
        node_colors = make_colors(max(1, len(node_list)))
        save_stage_ply(point_map, masks, filenames,
                       lambda n: node_colors[node_ids[n]] if n in node_ids else None,
                       f"{pre}_1_nodes.ply")

        # 2. after graph -- color by connected component
        save_stage_ply(point_map, masks, filenames,
                       lambda n: comp_colors[node_to_comp_sf[n]] if n in node_to_comp_sf else None,
                       f"{pre}_2_graph.ply")

        # 3. after sphere/radius filter -- dropped components gray
        def _c3(n):
            if n not in node_to_comp_sf:
                return None
            cid = node_to_comp_sf[n]
            return comp_colors[cid] if cid in kept_comps else [110, 110, 110]
        save_stage_ply(point_map, masks, filenames, _c3, f"{pre}_3_filtered.ply")

        # 4. after DBSCAN -- color by final cluster
        if args.dbscan_merge and len(centers) > 0:
            clus_colors = make_colors(max(1, n_clusters))
            def _c4(n):
                if n not in node_to_comp_sf:
                    return None
                lbl = comp_to_cluster.get(node_to_comp_sf[n])
                return [110, 110, 110] if lbl is None or lbl < 0 else clus_colors[lbl]
            save_stage_ply(point_map, masks, filenames, _c4, f"{pre}_4_merged.ply")

    if args.save_merged_ply and args.dbscan_merge and len(centers) > 0:
        clus_colors = make_colors(max(1, n_clusters))
        def _cm(n):
            if n not in node_to_comp_sf:
                return None
            lbl = comp_to_cluster.get(node_to_comp_sf[n])
            return [110, 110, 110] if lbl is None or lbl < 0 else clus_colors[lbl]
        print(f"\nSaving merged (final-cluster) PLY...")
        save_stage_ply(point_map, masks, filenames, _cm, args.save_merged_ply)

    if args.save_colored_ply:
        ply_path = (os.path.splitext(args.out)[0] + "_colored.ply"
                    if args.out else "instances_colored.ply")
        print(f"\nSaving colored PLY (graph components)...")
        save_colored_ply(point_map, masks, filenames, node_to_comp_sf,
                         len(root_counts), ply_path)

    if args.save_sphere_ply:
        sphere_ply_path = (os.path.splitext(args.out)[0] + "_spheres.ply"
                           if args.out else "instances_spheres.ply")
        print(f"\nSaving sphere PLY...")
        save_sphere_ply(sphere_results, len(comp_to_nodes), sphere_ply_path)

    # ── results file ────────────────────────────────────────────────────────
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"instance_count: {instance_count}\n")
            f.write(f"ground_truth: {gt}\n")
            f.write(f"error: {error:+d}\n")
            f.write(f"error_pct: {100*error/gt:+.1f}\n" if gt > 0 else "error_pct: N/A\n")
            f.write(f"pointmap: {args.pointmap}\n")
            f.write(f"corr_thresh: {args.corr_thresh}\n")
            f.write(f"min_overlap_pct: {args.min_overlap_pct}\n")
            f.write(f"min_match_overlap: {args.min_match_overlap}\n")
            f.write(f"sphere_thresh: {args.sphere_thresh}\n")
            f.write(f"sphere_min_inliers: {args.sphere_min_inliers}\n")
            f.write(f"dbscan_merge: {args.dbscan_merge}\n")
            f.write(f"dbscan_eps: {args.dbscan_eps}\n")
            f.write(f"dbscan_min: {args.dbscan_min}\n")
            f.write(f"min_radius: {args.min_radius}\n")
            f.write(f"max_radius: {args.max_radius}\n")
            f.write(f"conf_thresh: {args.conf_thresh if args.confmap else 'disabled'}\n")
            f.write(f"bilateral: {args.bilateral}\n")
            f.write(f"reciprocity_weight: {args.reciprocity_weight if args.bilateral else 'disabled'}\n")
            f.write(f"edges_added: {edges_added}\n")
            f.write(f"total_nodes: {total_nodes}\n")
            f.write(f"raw_components: {len(comp_to_nodes)}\n")
            f.write(f"components_into_dbscan: {len(centers)}\n")
            f.write(f"largest_component: {sizes[0]}\n")
            f.write(f"singletons: {sum(1 for s in sizes if s == 1)}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
