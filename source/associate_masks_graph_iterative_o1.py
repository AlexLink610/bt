"""
associate_masks_graph_iterative.py -- Iterative apple counting.

Over-merge detection now uses BOTH sphere radius AND point-cloud extent, because
a sphere radius alone misses two common over-merge shapes:
  * "apple + smear": one real apple plus a smeared tail -> elongated cloud, but
    RANSAC still fits a small sphere to the good apple -> radius looks fine.
  * several grouped instances sitting close -> RANSAC fits one, ignores the rest.
Extent (longest trimmed axis span) catches both.

Loop:
  1. Graph over ALL instances (initial corr_3d_thresh / min_match_overlap).
  2. Per component, fit sphere + measure extent.
       - radius in bounds AND extent in bounds -> ACCEPT
             (multi-instance: strict bounds; singleton: looser bounds)
       - too big (radius OR extent) & multi-instance -> SPLIT: re-graph that
             component's instances stricter -> sub-components next iteration.
       - too big & singleton -> TODO (keep/drop, configurable).
  3. Repeat until worklist empty or max_iter reached.
  4. Cluster accepted centers -> final count.

Diagnostics: --diag prints  inst / radius / extent / inlier / pts  per component.
Set --max_extent_* very high (e.g. 999) + --max_iter 0 to inspect raw values
before choosing thresholds.

Usage:
    python associate_masks_graph_iterative.py \
        --pointmap ...r12.npy --filenames ...txt --masks ...sam \
        --corr_3d_thresh 0.020 --min_match_overlap 0.05 \
        --corr_shrink 0.6 --overlap_grow 0.10 --max_iter 4 \
        --sphere_thresh 0.008 --min_radius 0.005 \
        --max_radius_multi 0.10 --max_radius_single 0.18 \
        --max_extent_multi 0.10 --max_extent_single 0.14 \
        --cluster_dist 0.08 --ground_truth 113 \
        --save_stages_ply ...iter --out ...result.txt
"""

import os
import argparse
import random
import numpy as np
from collections import Counter
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import pyransac3d as pyrsc
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ───────────────────────────── IO helpers ──────────────────────────────────

def load_filenames(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def mask_path_for(md, fn):
    return os.path.join(md, f"mask_{os.path.splitext(os.path.basename(fn))[0]}.png")

def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))

def get_instance_ids(mask):
    if mask is None:
        return []
    ids = np.unique(mask)
    return ids[ids != 0].tolist()


# ───────────────────────── graph over a SUBSET ─────────────────────────────

def build_graph_over_nodes(nodes, point_map, masks, H, W,
                           corr_3d_thresh, min_match_overlap):
    """Pairwise correspondence + Hungarian matching over a SET of nodes.
    Returns list of components (each a list of (image_index, instance_id))."""
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_img = {}
    for (im, iid) in nodes:
        by_img.setdefault(im, []).append(iid)
    imgs = sorted(by_img.keys())

    def comp_images(root_node):
        r = find(root_node)
        return {im for (im, _) in nodes if find((im, _)) == r} if False else \
               {im for n in nodes if find(n) == r for (im, _) in [n]}

    for a_i in range(len(imgs)):
        for b_i in range(a_i + 1, len(imgs)):
            i, j = imgs[a_i], imgs[b_i]
            ids_i, ids_j = by_img[i], by_img[j]
            if not ids_i or not ids_j:
                continue

            pj = point_map[j].reshape(-1, 3)
            mj = masks[j].reshape(-1)
            apple_j = np.isin(mj, ids_j) & ~np.isnan(pj).any(axis=1)
            idx_j = np.where(apple_j)[0]
            if idx_j.size == 0:
                continue
            tree = cKDTree(pj[apple_j])

            pi = point_map[i].reshape(-1, 3)
            mi = masks[i].reshape(-1)
            apple_i = np.isin(mi, ids_i) & ~np.isnan(pi).any(axis=1)
            idx_i = np.where(apple_i)[0]
            if idx_i.size == 0:
                continue
            dist, nn = tree.query(pi[apple_i], workers=-1)
            corr = np.full(H * W, -1, dtype=np.int64)
            good = dist < corr_3d_thresh
            corr[idx_i[good]] = idx_j[nn[good]]

            C = np.ones((len(ids_i), len(ids_j)), dtype=np.float32)
            for ai, a in enumerate(ids_i):
                pix_a = np.where(mi == a)[0]
                size_a = len(pix_a)
                if size_a == 0:
                    continue
                ca = corr[pix_a]
                val = ca >= 0
                landed = mj[ca[val]] if val.any() else np.array([], int)
                for bi, b in enumerate(ids_j):
                    size_b = int((mj == b).sum())
                    if min(size_a, size_b) == 0:
                        continue
                    ov = int((landed == b).sum())
                    frac = min(ov / min(size_a, size_b), 1.0)
                    C[ai, bi] = 1.0 - frac

            row, col = linear_sum_assignment(C)
            for ai, bi in zip(row, col):
                if C[ai, bi] < (1.0 - min_match_overlap):
                    na, nb = (i, ids_i[ai]), (j, ids_j[bi])
                    if find(na) == find(nb):
                        continue
                    if comp_images(na) & comp_images(nb):
                        continue
                    union(na, nb)

    comp_map = {}
    for n in nodes:
        comp_map.setdefault(find(n), []).append(n)
    return list(comp_map.values())


# ───────────────────────── sphere / extent / points ────────────────────────

def component_points(comp_nodes, point_map, masks):
    pts_list = []
    for (im, iid) in comp_nodes:
        mf = masks[im].reshape(-1)
        pf = point_map[im].reshape(-1, 3)
        p = pf[np.where(mf == iid)[0]]
        pts_list.append(p[~np.isnan(p).any(axis=1)])
    if not pts_list:
        return np.empty((0, 3))
    return np.concatenate(pts_list, axis=0)

def fit_sphere(points, thresh=0.008):
    if len(points) < 4:
        return None
    try:
        c, r, inl = pyrsc.Sphere().fit(points, thresh=thresh, maxIteration=500)
        return np.array(c), float(r), len(inl) / len(points)
    except Exception:
        return None

def point_extent(points):
    """Longest axis-extent (2..98 percentile trimmed) — proxy for elongated /
    multi-apple clumps that a sphere radius alone misses."""
    if len(points) < 2:
        return 0.0
    lo = np.percentile(points, 2, axis=0)
    hi = np.percentile(points, 98, axis=0)
    return float(np.max(hi - lo))


# ──────────────────────────── colors / PLY ─────────────────────────────────

def make_colors(n):
    import colorsys
    cols, golden, h = [], 0.618033988749895, 0.15
    for i in range(max(1, n)):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
        cols.append([int(r*255), int(g*255), int(b*255)])
    return cols

def write_ply(path, pts, cols):
    if len(pts) == 0:
        print(f"  (no points for {path})"); return
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(pts)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode())
        for p, c in zip(pts.astype(np.float32), cols.astype(np.uint8)):
            f.write(p.tobytes()); f.write(bytes(c))
    print(f"  saved {path}  ({len(pts):,} pts)")

def components_to_ply(components, point_map, masks, path, colors=None):
    if colors is None:
        colors = make_colors(len(components))
    all_p, all_c = [], []
    for k, comp in enumerate(components):
        pts = component_points(comp, point_map, masks)
        if len(pts) == 0:
            continue
        all_p.append(pts)
        all_c.append(np.tile(colors[k % len(colors)], (len(pts), 1)))
    if all_p:
        write_ply(path, np.concatenate(all_p), np.concatenate(all_c))
    else:
        print(f"  (no points for {path})")


# ──────────────────────────────── main ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--ground_truth", type=int, default=113)
    ap.add_argument("--out", default=None)
    # initial graph params
    ap.add_argument("--corr_3d_thresh", type=float, default=0.020)
    ap.add_argument("--min_match_overlap", type=float, default=0.05)
    # how params tighten each iteration
    ap.add_argument("--corr_shrink", type=float, default=0.6,
                    help="corr_3d_thresh *= this each iteration (stricter).")
    ap.add_argument("--overlap_grow", type=float, default=0.10,
                    help="min_match_overlap += this each iteration (stricter).")
    ap.add_argument("--max_iter", type=int, default=4)
    # sphere fit
    ap.add_argument("--sphere_thresh", type=float, default=0.008)
    ap.add_argument("--min_radius", type=float, default=0.005)
    # accept bounds -- radius
    ap.add_argument("--max_radius_multi", type=float, default=0.10)
    ap.add_argument("--max_radius_single", type=float, default=0.18)
    # accept bounds -- extent (NEW: catches elongated / clumped over-merges)
    ap.add_argument("--max_extent_multi", type=float, default=0.10,
                    help="Max point-cloud extent to ACCEPT a multi-instance component.")
    ap.add_argument("--max_extent_single", type=float, default=0.14,
                    help="Max point-cloud extent to ACCEPT a singleton component.")
    ap.add_argument("--singleton_toolarge", choices=["keep", "drop"], default="drop",
                    help="TODO placeholder: what to do with a too-large singleton.")
    ap.add_argument("--diag", action="store_true",
                    help="Print per-component radius/extent/inlier diagnostics.")
    # final clustering
    ap.add_argument("--cluster_dist", type=float, default=0.08)
    # output
    ap.add_argument("--save_stages_ply", default=None)
    args = ap.parse_args()

    np.random.seed(42); random.seed(42)
    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    fnames = load_filenames(args.filenames)
    masks = [load_mask(mask_path_for(args.masks, f), W, H) for f in fnames]

    print("="*60)
    print("Iterative Instance Counter (radius + extent)")
    print("="*60)
    print(f"  initial corr_3d_thresh={args.corr_3d_thresh}  "
          f"min_match_overlap={args.min_match_overlap}")
    print(f"  tighten: corr*={args.corr_shrink}  overlap+={args.overlap_grow}  "
          f"max_iter={args.max_iter}")
    print(f"  accept radius: multi<={args.max_radius_multi} single<={args.max_radius_single}")
    print(f"  accept extent: multi<={args.max_extent_multi} single<={args.max_extent_single}")

    all_nodes = []
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            all_nodes.append((i, iid))
    print(f"  total instance nodes: {len(all_nodes)}")

    components = build_graph_over_nodes(
        all_nodes, pm, masks, H, W,
        args.corr_3d_thresh, args.min_match_overlap)
    print(f"\n[iter 0] initial components: {len(components)}")

    accepted = []          # (comp_nodes, center, radius)
    worklist = [(c, 0) for c in components]

    pre = args.save_stages_ply
    if pre:
        components_to_ply(components, pm, masks, f"{pre}_iter0_graph.ply")

    iteration = 0
    while worklist and iteration <= args.max_iter:
        next_worklist = []
        accepted_this_iter = []
        split_this_iter = []

        corr_now = args.corr_3d_thresh * (args.corr_shrink ** iteration)
        overlap_now = min(0.99, args.min_match_overlap + args.overlap_grow * iteration)

        if args.diag:
            print(f"  --- iter {iteration} diagnostics (inst/r/ext/inl/pts) ---")

        for comp, born in worklist:
            n_inst = len(comp)
            pts = component_points(comp, pm, masks)
            res = fit_sphere(pts, thresh=args.sphere_thresh)
            if res is None:
                continue
            center, radius, inlier = res
            extent = point_extent(pts)

            is_singleton = (n_inst == 1)
            max_r = args.max_radius_single if is_singleton else args.max_radius_multi
            max_ext = args.max_extent_single if is_singleton else args.max_extent_multi

            radius_ok = (args.min_radius <= radius <= max_r)
            extent_ok = (extent <= max_ext)
            fits = radius_ok and extent_ok
            too_big = (radius > max_r) or (extent > max_ext)

            if args.diag:
                tag = "OK" if fits else ("R!" if not radius_ok else "E!")
                print(f"    [{tag}] inst={n_inst:2d} r={radius:.4f} "
                      f"ext={extent:.4f} inl={inlier:.2f} pts={len(pts)}")

            if fits:
                accepted.append((comp, center, radius))
                accepted_this_iter.append(comp)

            elif too_big and (not is_singleton) and iteration < args.max_iter:
                sub_corr = args.corr_3d_thresh * (args.corr_shrink ** (iteration + 1))
                sub_overlap = min(0.99, args.min_match_overlap +
                                  args.overlap_grow * (iteration + 1))
                subs = build_graph_over_nodes(
                    comp, pm, masks, H, W, sub_corr, sub_overlap)
                if len(subs) > 1:
                    for s in subs:
                        next_worklist.append((s, iteration + 1))
                    split_this_iter.append(comp)
                else:
                    # didn't split further -> genuine scatter, accept as-is
                    accepted.append((comp, center, radius))
                    accepted_this_iter.append(comp)

            elif too_big and is_singleton:
                # TODO: too-large singleton (smeared single apple vs junk)
                if args.singleton_toolarge == "keep":
                    accepted.append((comp, center, radius))
                    accepted_this_iter.append(comp)
                # else drop

            else:
                # radius < min_radius, or too-big-multi at final iteration
                if too_big and (not is_singleton):
                    # iterations exhausted but still too big -> accept to avoid loss
                    accepted.append((comp, center, radius))
                    accepted_this_iter.append(comp)
                # else (radius < min_radius) -> drop

        if pre:
            if accepted_this_iter:
                components_to_ply(accepted_this_iter, pm, masks,
                                  f"{pre}_iter{iteration}_accepted.ply")
            if split_this_iter:
                components_to_ply(split_this_iter, pm, masks,
                                  f"{pre}_iter{iteration}_split.ply")

        print(f"[iter {iteration}] worklist {len(worklist)} -> "
              f"accepted {len(accepted_this_iter)}, split {len(split_this_iter)}, "
              f"carry {len(next_worklist)}  (corr={corr_now:.4f}, overlap={overlap_now:.2f})")

        worklist = next_worklist
        iteration += 1

    print(f"\nTotal accepted candidates: {len(accepted)}")

    centers = np.array([c for (_, c, _) in accepted])
    if len(centers) == 0:
        print("No accepted candidates; cannot cluster.")
        return
    if len(centers) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=args.cluster_dist,
            linkage="complete", metric="euclidean").fit(centers).labels_
    n_clusters = len(set(labels))

    gt = args.ground_truth
    err = n_clusters - gt
    print(f"\n{'='*60}")
    print(f"  FINAL COUNT (clusters): {n_clusters}")
    print(f"  GROUND TRUTH:           {gt}")
    print(f"  Error: {err:+d}" + (f"  ({100*err/gt:+.1f}%)" if gt > 0 else ""))
    print(f"{'='*60}")

    radii = np.array([r for (_, _, r) in accepted])
    print(f"  accepted radius percentiles [5,50,95]: "
          f"{np.percentile(radii, [5,50,95]).round(4)}")

    if pre:
        clu_colors = make_colors(n_clusters)
        all_p, all_c = [], []
        for (comp, _, _), lab in zip(accepted, labels):
            pts = component_points(comp, pm, masks)
            if len(pts) == 0:
                continue
            all_p.append(pts)
            all_c.append(np.tile(clu_colors[lab], (len(pts), 1)))
        if all_p:
            write_ply(f"{pre}_final_clustered.ply",
                      np.concatenate(all_p), np.concatenate(all_c))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"final_count: {n_clusters}\n")
            f.write(f"ground_truth: {gt}\n")
            f.write(f"error: {err:+d}\n")
            f.write(f"error_pct: {100*err/gt:+.1f}\n" if gt > 0 else "error_pct: N/A\n")
            f.write(f"accepted_candidates: {len(accepted)}\n")
            f.write(f"iterations_used: {iteration-1}\n")
            f.write(f"corr_3d_thresh_init: {args.corr_3d_thresh}\n")
            f.write(f"min_match_overlap_init: {args.min_match_overlap}\n")
            f.write(f"corr_shrink: {args.corr_shrink}\n")
            f.write(f"overlap_grow: {args.overlap_grow}\n")
            f.write(f"max_radius_multi: {args.max_radius_multi}\n")
            f.write(f"max_radius_single: {args.max_radius_single}\n")
            f.write(f"max_extent_multi: {args.max_extent_multi}\n")
            f.write(f"max_extent_single: {args.max_extent_single}\n")
            f.write(f"cluster_dist: {args.cluster_dist}\n")
        print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
