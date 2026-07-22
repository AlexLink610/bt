"""
associate_masks_graph.py  --  Count apples using graph-based mask association,
sphere fitting, and a final clustering merge.

Pipeline stages (visualize all four with --save_stages_ply PREFIX):
    1. raw (image, instance) nodes
    2. graph association -> connected components
    3. sphere fit + radius/inlier filter (dropped components reported & grayed)
    4. clustering merge on sphere centers -> final count

Usage:
    python associate_masks_graph.py \
        --pointmap   ~/ba/output_vggt/t02_360_32v_pointmap.npy \
        --filenames  ~/ba/output_vggt/t02_360_32v_filenames.txt \
        --masks      ~/ba/output_sam/tree_02/semantics_sam3 \
        --corr_thresh 0.020 --ground_truth 113 \
        --cluster_method agglomerative --cluster_dist 0.05 \
        --save_stages_ply ~/ba/output_vggt/t02_stages \
        --out ~/ba/output_vggt/t02_result.txt

Clustering methods:
    dbscan         density-based; merges if ANY point is within eps of ANY other
                   -> prone to CHAINING across the canopy (giant merges)
    agglomerative  complete-linkage with distance_threshold; merges only if ALL
                   member pairs are within the threshold -> cannot chain
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
from sklearn.cluster import DBSCAN, AgglomerativeClustering
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
    print(f"  Conf percentiles [10,25,50,75,90]: "
          f"{np.percentile(cm, [10,25,50,75,90]).round(3)}")
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
    corr = np.full(H * W, -1, dtype=np.int32)
    good = distances < corr_thresh
    corr[apple1_idx[good]] = valid2_idx[nn][good]
    return corr


def compute_reciprocity(corr_fwd, pm1, pm2, mask1, mask2, H, W, corr_thresh,
                        conf1, conf2, conf_thresh):
    corr_bwd = compute_correspondence(pm2, pm1, mask2, mask1, conf2, conf1,
                                      H, W, corr_thresh, conf_thresh)
    reciprocal = np.zeros(H * W, dtype=bool)
    fwd_idx = np.where(corr_fwd >= 0)[0]
    if fwd_idx.size == 0:
        return reciprocal
    reciprocal[fwd_idx] = (corr_bwd[corr_fwd[fwd_idx]] == fwd_idx)
    return reciprocal


def compute_cost_matrix(mask1, mask2, corr, ids1, ids2, min_overlap_pct,
                        reciprocal=None, reciprocity_penalty_weight=0.3):
    if not ids1 or not ids2:
        return np.ones((max(1, len(ids1)), max(1, len(ids2))), dtype=np.float32)

    mask1_flat = mask1.reshape(-1)
    mask2_flat = mask2.reshape(-1)
    C = np.ones((len(ids1), len(ids2)), dtype=np.float32)
    min_overlap_frac = min_overlap_pct / 100.0

    for ai, a in enumerate(ids1):
        pixels_a = np.where(mask1_flat == a)[0]
        if len(pixels_a) == 0:
            continue
        size_a = len(pixels_a)
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
                    mr = recip_a_valid[landed == b]
                    if len(mr) > 0:
                        base_cost = min(1.0, base_cost +
                                        (1.0 - mr.mean()) * reciprocity_penalty_weight)
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
        return sorted(Counter([self.find(x) for x in self.parent]).values(), reverse=True)

    def get_component_map(self):
        roots = {x: self.find(x) for x in self.parent}
        root_counts = Counter(roots.values())
        root_to_id = {r: i for i, (r, _) in enumerate(root_counts.most_common())}
        return {x: root_to_id[roots[x]] for x in self.parent}, root_counts

    def get_component_nodes(self, root):
        return [x for x in self.parent if self.find(x) == root]


# ──────────────────────────── colors / PLY ─────────────────────────────────

def make_colors(n):
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
            pts = pts_flat[np.where(mask_flat == iid)[0]]
            pts = pts[~np.isnan(pts).any(axis=1)]
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_cols.append(np.tile(col, (len(pts), 1)))
    if not all_pts:
        print(f"  (no points for {path})")
        return
    _write_ply(np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), path)


def fit_sphere_to_component(points, thresh=0.008, max_iter=500):
    if len(points) < 4:
        return None
    try:
        center, radius, inliers = pyrsc.Sphere().fit(points, thresh=thresh,
                                                     maxIteration=max_iter)
        return center, radius, len(inliers) / len(points)
    except Exception:
        return None


def save_sphere_ply(sphere_results, n_colors, path):
    colors = make_colors(max(1, n_colors))

    def sample_sphere(center, radius, n=500):
        phi = np.random.uniform(0, 2 * np.pi, n)
        costh = np.random.uniform(-1, 1, n)
        sinth = np.sqrt(1 - costh ** 2)
        return np.stack([center[0] + radius * sinth * np.cos(phi),
                         center[1] + radius * sinth * np.sin(phi),
                         center[2] + radius * costh], axis=1)

    all_pts, all_cols = [], []
    for comp_id, res in sphere_results.items():
        if res is None:
            continue
        center, radius, _ = res
        pts = sample_sphere(center, radius)
        all_pts.append(pts)
        all_cols.append(np.tile(colors[comp_id % len(colors)], (len(pts), 1)))
    if not all_pts:
        print("  No spheres to save.")
        return
    _write_ply(np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), path)


# ──────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",           required=True)
    parser.add_argument("--filenames",          required=True)
    parser.add_argument("--masks",              required=True)
    parser.add_argument("--confmap",            default=None)
    parser.add_argument("--conf_thresh",        type=float, default=0.3)
    parser.add_argument("--transforms",         default=None)
    parser.add_argument("--out",                default=None)
    parser.add_argument("--ground_truth",       type=int,   default=113)
    parser.add_argument("--cam_dist_thresh",    type=float, default=999.0)
    parser.add_argument("--corr_thresh",        type=float, default=0.010)
    parser.add_argument("--min_overlap_pct",    type=float, default=5.0)
    parser.add_argument("--min_match_overlap",  type=float, default=0.01,
                        help="Raise to make the GRAPH less merge-aggressive (stage 1->2).")
    parser.add_argument("--sphere_thresh",      type=float, default=0.008)
    parser.add_argument("--sphere_min_inliers", type=float, default=0.0)
    parser.add_argument("--max_radius",         type=float, default=0.15,
                        help="Stage 3: drop components with sphere radius above this.")
    parser.add_argument("--min_radius",         type=float, default=0.01,
                        help="Stage 3: drop components with sphere radius below this.")
    # ---- clustering (stage 4) ----
    parser.add_argument("--cluster_method", default="none",
                        choices=["none", "dbscan", "agglomerative"],
                        help="Stage-4 merge: 'dbscan' (chains!) or 'agglomerative' "
                             "(complete-linkage, cannot chain). Default: none.")
    parser.add_argument("--dbscan_merge", action="store_true",
                        help="Back-compat alias for --cluster_method dbscan.")
    parser.add_argument("--dbscan_eps",   type=float, default=0.08,
                        help="DBSCAN eps (used when --cluster_method dbscan).")
    parser.add_argument("--dbscan_min",   type=int,   default=1)
    parser.add_argument("--cluster_dist", type=float, default=None,
                        help="Agglomerative distance_threshold (max cluster diameter). "
                             "Defaults to --dbscan_eps if unset.")
    # ---- misc ----
    parser.add_argument("--bilateral",          action="store_true")
    parser.add_argument("--reciprocity_weight", type=float, default=0.3)
    parser.add_argument("--save_colored_ply",   action="store_true")
    parser.add_argument("--save_sphere_ply",    action="store_true")
    parser.add_argument("--save_centers_ply",   default=None)
    parser.add_argument("--save_merged_ply",    default=None)
    parser.add_argument("--save_stages_ply",    default=None,
                        help="PREFIX for _1_nodes / _2_graph / _3_filtered / _4_merged PLYs.")
    args = parser.parse_args()

    # back-compat: --dbscan_merge implies dbscan
    if args.dbscan_merge and args.cluster_method == "none":
        args.cluster_method = "dbscan"
    do_cluster = args.cluster_method != "none"
    cluster_dist = args.cluster_dist if args.cluster_dist is not None else args.dbscan_eps

    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("Instance Counter -- Graph-based Association")
    print("=" * 60)
    print(f"  corr_thresh={args.corr_thresh}  min_overlap_pct={args.min_overlap_pct}%  "
          f"min_match_overlap={args.min_match_overlap}")
    print(f"  radius bounds: [{args.min_radius}, {args.max_radius}]  "
          f"sphere_thresh={args.sphere_thresh}")
    print(f"  cluster_method={args.cluster_method}" +
          (f"  eps={args.dbscan_eps}" if args.cluster_method == "dbscan" else
           f"  cluster_dist={cluster_dist}" if args.cluster_method == "agglomerative" else ""))
    print(f"  conf filtering: {args.conf_thresh if args.confmap else 'DISABLED'}")
    print(f"  bilateral: {('ENABLED (w=%s)' % args.reciprocity_weight) if args.bilateral else 'DISABLED'}")

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

    # ── stage 1 -> 2: graph association ─────────────────────────────────────
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
                                                masks[i], masks[j], H, W, args.corr_thresh,
                                                conf_i, conf_j, args.conf_thresh)

        C = compute_cost_matrix(masks[i], masks[j], corr_ij, ids_i, ids_j,
                                args.min_overlap_pct, reciprocal=reciprocal_ij,
                                reciprocity_penalty_weight=args.reciprocity_weight)
        row_ind, col_ind = linear_sum_assignment(C)
        for ai, bi in zip(row_ind, col_ind):
            if C[ai, bi] < (1.0 - args.min_match_overlap):
                root_i, root_j = uf.find((i, ids_i[ai])), uf.find((j, ids_j[bi]))
                if root_i == root_j:
                    continue
                imgs_i = {img for (img, _) in uf.get_component_nodes(root_i)}
                imgs_j = {img for (img, _) in uf.get_component_nodes(root_j)}
                if imgs_i & imgs_j:
                    continue
                uf.union((i, ids_i[ai]), (j, ids_j[bi]))
                edges_added += 1

        if (pair_idx + 1) % 200 == 0 or pair_idx == len(pairs) - 1:
            print(f"  [{pair_idx+1:4d}/{len(pairs)}] edges: {edges_added}")

    instance_count = uf.components()
    gt = args.ground_truth
    sizes = uf.component_sizes()

    node_to_comp, root_counts = uf.get_component_map()
    comp_to_nodes = {}
    for node, comp_id in node_to_comp.items():
        comp_to_nodes.setdefault(comp_id, []).append(node)

    # ── stage 2 -> 3: sphere fit + radius/inlier filter ─────────────────────
    print(f"\nFitting spheres to {len(comp_to_nodes)} components (thresh={args.sphere_thresh})...")
    sphere_results = {}
    drop_few_pts = 0        # <4 points, RANSAC impossible
    drop_fit_fail = 0       # RANSAC raised / degenerate
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
            drop_few_pts += 1
            continue
        all_pts = np.concatenate(pts_list, axis=0)
        if len(all_pts) < 4:
            drop_few_pts += 1
            sphere_results[comp_id] = None
            continue
        res = fit_sphere_to_component(all_pts, thresh=args.sphere_thresh)
        if res is None:
            drop_fit_fail += 1
        sphere_results[comp_id] = res

    ratios = [r[2] for r in sphere_results.values() if r is not None]
    if ratios:
        print(f"  Sphere inlier ratio — mean: {np.mean(ratios):.2f}  "
              f"median: {np.median(ratios):.2f}  min: {np.min(ratios):.2f}  "
              f"max: {np.max(ratios):.2f}")

    # Select survivors + count WHY the rest were dropped
    kept_comp_ids, centers = [], []
    drop_small_r = drop_big_r = drop_inliers = 0
    radii_all = []
    for comp_id, res in sphere_results.items():
        if res is None:
            continue
        center, radius, inlier_ratio = res
        radii_all.append(radius)
        if radius < args.min_radius:
            drop_small_r += 1
            continue
        if radius > args.max_radius:
            drop_big_r += 1
            continue
        if args.sphere_min_inliers > 0.0 and inlier_ratio < args.sphere_min_inliers:
            drop_inliers += 1
            continue
        kept_comp_ids.append(comp_id)
        centers.append(center)

    n_dropped = len(comp_to_nodes) - len(kept_comp_ids)
    print(f"\n  Stage 3 filter: {len(comp_to_nodes)} -> {len(kept_comp_ids)} components "
          f"(dropped {n_dropped}, {100*n_dropped/max(1,len(comp_to_nodes)):.1f}%)")
    print(f"    dropped, <4 points (no fit possible): {drop_few_pts}")
    print(f"    dropped, RANSAC fit failed:           {drop_fit_fail}")
    print(f"    dropped, radius < {args.min_radius}:            {drop_small_r}")
    print(f"    dropped, radius > {args.max_radius}:            {drop_big_r}")
    if args.sphere_min_inliers > 0.0:
        print(f"    dropped, inlier_ratio < {args.sphere_min_inliers}:     {drop_inliers}")
    if radii_all:
        rp = np.percentile(radii_all, [5, 25, 50, 75, 95])
        print(f"    fitted radius percentiles [5,25,50,75,95]: {rp.round(4)}")

    if not do_cluster:
        instance_count = len(kept_comp_ids)

    # ── stage 3 -> 4: clustering merge ──────────────────────────────────────
    comp_to_cluster, labels, n_clusters = {}, None, 0
    if do_cluster:
        print(f"\nStage 4 merge: {args.cluster_method} on {len(centers)} sphere centers")
        if len(centers) > 0:
            centers_arr = np.array(centers)
            if args.cluster_method == "dbscan":
                print(f"  DBSCAN(eps={args.dbscan_eps}, min_samples={args.dbscan_min})")
                labels = DBSCAN(eps=args.dbscan_eps,
                                min_samples=args.dbscan_min).fit(centers_arr).labels_
            else:
                print(f"  AgglomerativeClustering(linkage=complete, "
                      f"distance_threshold={cluster_dist})")
                labels = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=cluster_dist,
                    linkage="complete", metric="euclidean").fit(centers_arr).labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum()) if -1 in labels else 0
            print(f"  Clusters: {n_clusters}" +
                  (f"  (noise: {n_noise})" if n_noise else ""))
            instance_count = n_clusters
            comp_to_cluster = {cid: int(l) for cid, l in zip(kept_comp_ids, labels)}

            # cluster size stats -- reveals over-merging
            csizes = sorted(Counter(labels[labels >= 0]).values(), reverse=True)
            if csizes:
                print(f"  Components per cluster — max: {csizes[0]}  "
                      f"median: {csizes[len(csizes)//2]}  "
                      f"singletons: {sum(1 for s in csizes if s == 1)}")

            if args.save_centers_ply:
                ccols = make_colors(max(1, n_clusters))
                cols = np.array([[110, 110, 110] if l < 0 else ccols[l] for l in labels],
                                dtype=np.uint8)
                _write_ply(centers_arr, cols, args.save_centers_ply)
        else:
            print("  No valid sphere centers -- skipping merge.")

    # ── stats ───────────────────────────────────────────────────────────────
    print(f"\n  Component size distribution (graph):")
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

    # ── stage PLYs ──────────────────────────────────────────────────────────
    comp_colors = make_colors(max(1, len(comp_to_nodes)))
    kept_set = set(kept_comp_ids)

    def _c4(n):
        if n not in node_to_comp:
            return None
        lbl = comp_to_cluster.get(node_to_comp[n])
        return [110, 110, 110] if lbl is None or lbl < 0 else _c4.colors[lbl]

    if args.save_stages_ply:
        pre = args.save_stages_ply
        print(f"\nSaving pipeline-stage PLYs...")

        node_list = sorted(node_to_comp.keys())
        node_ids = {n: k for k, n in enumerate(node_list)}
        node_colors = make_colors(max(1, len(node_list)))
        save_stage_ply(point_map, masks, filenames,
                       lambda n: node_colors[node_ids[n]] if n in node_ids else None,
                       f"{pre}_1_nodes.ply")

        save_stage_ply(point_map, masks, filenames,
                       lambda n: comp_colors[node_to_comp[n]] if n in node_to_comp else None,
                       f"{pre}_2_graph.ply")

        def _c3(n):
            if n not in node_to_comp:
                return None
            cid = node_to_comp[n]
            return comp_colors[cid] if cid in kept_set else [110, 110, 110]
        save_stage_ply(point_map, masks, filenames, _c3, f"{pre}_3_filtered.ply")

        if do_cluster and len(centers) > 0:
            _c4.colors = make_colors(max(1, n_clusters))
            save_stage_ply(point_map, masks, filenames, _c4, f"{pre}_4_merged.ply")

    if args.save_merged_ply and do_cluster and len(centers) > 0:
        _c4.colors = make_colors(max(1, n_clusters))
        print(f"\nSaving merged (final-cluster) PLY...")
        save_stage_ply(point_map, masks, filenames, _c4, args.save_merged_ply)

    if args.save_colored_ply:
        ply_path = (os.path.splitext(args.out)[0] + "_colored.ply"
                    if args.out else "instances_colored.ply")
        print(f"\nSaving colored PLY (graph components)...")
        save_stage_ply(point_map, masks, filenames,
                       lambda n: comp_colors[node_to_comp[n]] if n in node_to_comp else None,
                       ply_path)

    if args.save_sphere_ply:
        sp = (os.path.splitext(args.out)[0] + "_spheres.ply"
              if args.out else "instances_spheres.ply")
        print(f"\nSaving sphere PLY...")
        save_sphere_ply(sphere_results, len(comp_to_nodes), sp)

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
            f.write(f"min_radius: {args.min_radius}\n")
            f.write(f"max_radius: {args.max_radius}\n")
            f.write(f"cluster_method: {args.cluster_method}\n")
            f.write(f"dbscan_eps: {args.dbscan_eps}\n")
            f.write(f"cluster_dist: {cluster_dist}\n")
            f.write(f"conf_thresh: {args.conf_thresh if args.confmap else 'disabled'}\n")
            f.write(f"bilateral: {args.bilateral}\n")
            f.write(f"edges_added: {edges_added}\n")
            f.write(f"total_nodes: {total_nodes}\n")
            f.write(f"raw_components: {len(comp_to_nodes)}\n")
            f.write(f"components_kept_stage3: {len(kept_comp_ids)}\n")
            f.write(f"dropped_few_points: {drop_few_pts}\n")
            f.write(f"dropped_fit_failed: {drop_fit_fail}\n")
            f.write(f"dropped_radius_small: {drop_small_r}\n")
            f.write(f"dropped_radius_large: {drop_big_r}\n")
            f.write(f"dropped_inliers: {drop_inliers}\n")
            f.write(f"largest_component: {sizes[0]}\n")
            f.write(f"singletons: {sum(1 for s in sizes if s == 1)}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
