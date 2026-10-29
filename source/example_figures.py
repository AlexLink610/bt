"""
example_figures.py -- Figures for the running example (Chapter 4).

Recomputes the association with the same rules as the pipeline (pointmap
correspondences with tau_corr, overlap normalized by the smaller mask,
rectangular Hungarian matching, edges with overlap > tau_ov, Union-Find with
the same-view constraint) and writes, into --out:

  masks_<k>.png       view k with its instance masks, one color per final
                      instance (same colors in all figures)
  corr_<i>_<j>.png    views i and j side by side with sampled accepted pixel
                      correspondences i -> j
  overlap_<i>_<j>.tex overlap matrix o_ab of the pair as a LaTeX tabular
                      (bold: accepted edge, italic: assigned but below tau_ov)
  graph.pdf/.png      mask graph: one column per view, one node per mask,
                      edges = accepted matches, node color = instance
  spheres.ply         points of every instance plus its fitted sphere
                      (sphere drawn red if its radius is rejected)
  spheres_mesh.ply    the fitted spheres as triangle meshes (show as
                      wireframe in MeshLab, on top of the point head PLY)
  instances_points.ply, instances_spheres_mesh.ply,
  clusters_points.ply,  clusters_spheres_mesh.ply
                      with --cluster_dist: points and sphere meshes of the
                      accepted instances, colored by instance and by
                      cluster (same camera in MeshLab -> before/after)
  instance_<k>.ply, center_<k>.ply, sphere_<k>.ply
                      with --cluster_dist: points of each accepted instance
                      alone, its sphere center as a small red blob, and its
                      fitted sphere as points (lighter instance color)
                      (numbering as in graph.pdf); the console lists the
                      instances that the clustering merges
  centers.pdf/.png    with --cluster_dist: top view of the accepted
                      sphere centers and outlines, colored by instance
                      (left) and by cluster after complete-linkage
                      clustering (right)

Place next to associate_masks_graph_iterative.py (its helpers are reused).

Usage:
    python example_figures.py --images <image dir> --pointmap <..._rf.npy>
        --filenames <..._filenames.txt> --masks <mask dir> --out <dir>
        --corr 0.0201 --ov 0.2 --pair 0 1
        --sphere_thresh 0.0101 --min_radius 0.0201
        --max_radius_multi 0.0604 --max_radius_single 0.0806
"""

import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from associate_masks_graph_iterative import (  # noqa: E402
    load_filenames, mask_path_for, load_mask, get_instance_ids, fit_sphere)

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
    [250, 190, 212], [0, 128, 128], [170, 110, 40], [128, 0, 0],
], dtype=np.uint8)


def color(k):
    return PALETTE[k % len(PALETTE)]


def pair_overlap(i, j, pm, masks, ids, tau):
    """Directed correspondences i -> j and overlap matrix o[a, b]."""
    mi, mj = masks[i].reshape(-1), masks[j].reshape(-1)
    pi, pj = pm[i].reshape(-1, 3), pm[j].reshape(-1, 3)
    vi = np.isin(mi, ids[i]) & ~np.isnan(pi).any(axis=1)
    vj = np.isin(mj, ids[j]) & ~np.isnan(pj).any(axis=1)
    idx_i, idx_j = np.where(vi)[0], np.where(vj)[0]
    o = np.zeros((len(ids[i]), len(ids[j])))
    if len(idx_i) == 0 or len(idx_j) == 0:
        return o, idx_i[:0], idx_j[:0]
    d, nn = cKDTree(pj[idx_j]).query(pi[idx_i], workers=-1)
    good = d < tau
    src, dst = idx_i[good], idx_j[nn[good]]
    size_i = np.bincount(mi, minlength=256)
    size_j = np.bincount(mj, minlength=256)
    hist = np.bincount(mi[src].astype(np.int64) * 256 + mj[dst],
                       minlength=256 * 256).reshape(256, 256)
    for a, ma in enumerate(ids[i]):
        for b, mb in enumerate(ids[j]):
            den = min(size_i[ma], size_j[mb])
            if den > 0:
                o[a, b] = min(hist[ma, mb] / den, 1.0)
    return o, src, dst


def associate(pm, masks, ids, tau, ov):
    nodes = [(i, m) for i in range(len(ids)) for m in ids[i]]
    parent = {n: n for n in nodes}
    views = {n: {n[0]} for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges, blocked, matches, pairs = [], [], [], {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if not ids[i] or not ids[j]:
                continue
            o, src, dst = pair_overlap(i, j, pm, masks, ids, tau)
            rows, cols = linear_sum_assignment(1.0 - o)
            pairs[(i, j)] = (o, list(zip(rows, cols)), src, dst)
            for a, b in zip(rows, cols):
                if o[a, b] <= ov:
                    continue
                na, nb = (i, ids[i][a]), (j, ids[j][b])
                ra, rb = find(na), find(nb)
                if not (views[ra] & views[rb]) or ra == rb:
                    matches.append((na, nb))       # accepted match (graph edge)
                if ra == rb:
                    continue
                if views[ra] & views[rb]:
                    blocked.append((na, nb))      # same-view constraint
                    continue
                parent[ra] = rb
                views[rb] |= views.pop(ra)
                edges.append((na, nb))
    comps = {}
    for n in nodes:
        comps.setdefault(find(n), []).append(n)
    comps = sorted(comps.values(), key=lambda c: min(c))
    label = {n: k for k, c in enumerate(comps) for n in c}
    return comps, label, edges, blocked, matches, pairs


def tinted(img, mask, ids_view, view, label, alpha=0.55):
    out = np.asarray(img, dtype=np.float32).copy()
    for m in ids_view:
        sel = mask == m
        out[sel] = (1 - alpha) * out[sel] + alpha * color(label[(view, m)])
    return Image.fromarray(out.astype(np.uint8))


def write_ply(path, pts, cols):
    v = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["red"], v["green"], v["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        f.write(v.tobytes())


def uv_sphere(c, r, nlat=10, nlon=20):
    th = np.linspace(0, np.pi, nlat + 1)
    ph = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)],
                 axis=-1).reshape(-1, 3)
    f = []
    for i in range(nlat):
        for j in range(nlon):
            a, b = i * nlon + j, i * nlon + (j + 1) % nlon
            c2, d = (i + 1) * nlon + j, (i + 1) * nlon + (j + 1) % nlon
            f += [(a, c2, b), (b, c2, d)]
    return c + r * v, np.array(f, dtype=np.int32)


def write_mesh_ply(path, verts, cols, faces):
    v = np.zeros(len(verts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    v["red"], v["green"], v["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    fc = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "i4", (3,))])
    fc["n"], fc["i"] = 3, faces
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {len(fc)}\nproperty list uchar int vertex_indices\n"
                 "end_header\n").encode())
        f.write(v.tobytes())
        f.write(fc.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corr", type=float, required=True)
    ap.add_argument("--ov", type=float, default=0.2)
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--n_lines", type=int, default=60)
    ap.add_argument("--sphere_thresh", type=float, required=True)
    ap.add_argument("--min_radius", type=float, required=True)
    ap.add_argument("--max_radius_multi", type=float, required=True)
    ap.add_argument("--max_radius_single", type=float, required=True)
    ap.add_argument("--cluster_dist", type=float, default=None)
    ap.add_argument("--views", type=int, nargs="+", default=None,
                    help="use only these views (0-based) for the association; "
                         "with --corr 0 every mask becomes its own instance")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(42)
    random.seed(42)                   # RANSAC sampling, as in the pipeline

    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    fnames = load_filenames(args.filenames)
    masks = [load_mask(mask_path_for(args.masks, f), W, H) for f in fnames]
    masks = [m if m is not None else np.zeros((H, W), np.uint8) for m in masks]
    ids = [list(get_instance_ids(m)) for m in masks]
    if args.views is not None:
        ids = [x if k in args.views else [] for k, x in enumerate(ids)]
        if not set(args.pair) <= set(args.views):
            args.pair = sorted(args.views)[:2]
    imgs = [Image.open(os.path.join(args.images, f)).convert("RGB").resize((W, H))
            for f in fnames]

    comps, label, edges, blocked, matches, pairs = associate(
        pm, masks, ids, args.corr, args.ov)
    print(f"{sum(len(x) for x in ids)} masks, {len(matches)} accepted matches "
          f"({len(edges)} needed for the components), "
          f"{len(blocked)} blocked by the same-view constraint, "
          f"{len(comps)} instances")

    # 1. masks per view
    for k in range(N):
        tinted(imgs[k], masks[k], ids[k], k, label).save(
            os.path.join(args.out, f"masks_{k}.png"))

    # 2. correspondences of the chosen pair
    i, j = sorted(args.pair)
    o, src, dst = pairs[(i, j)][0], pairs[(i, j)][2], pairs[(i, j)][3]
    gap = 20
    canvas = Image.new("RGB", (2 * W + gap, H), (255, 255, 255))
    canvas.paste(tinted(imgs[i], masks[i], ids[i], i, label, 0.35), (0, 0))
    canvas.paste(tinted(imgs[j], masks[j], ids[j], j, label, 0.35), (W + gap, 0))
    draw = ImageDraw.Draw(canvas)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(src), size=min(args.n_lines, len(src)), replace=False) \
        if len(src) else []
    mi = masks[i].reshape(-1)
    for p in pick:
        yi, xi = divmod(int(src[p]), W)
        yj, xj = divmod(int(dst[p]), W)
        c = tuple(int(v) for v in color(label[(i, int(mi[src[p]]))]))
        draw.line([(xi, yi), (W + gap + xj, yj)], fill=c, width=2)
    canvas.save(os.path.join(args.out, f"corr_{i}_{j}.png"))

    # 3. overlap table of the chosen pair
    assigned = dict(pairs[(i, j)][1])
    lines = ["\\begin{tabular}{c|" + "c" * len(ids[j]) + "}",
             " & " + " & ".join(f"$b_{{{b + 1}}}$" for b in range(len(ids[j])))
             + " \\\\ \\hline"]
    for a in range(len(ids[i])):
        cells = []
        for b in range(len(ids[j])):
            v = f"{o[a, b]:.2f}"
            if assigned.get(a) == b:
                v = f"\\textbf{{{v}}}" if o[a, b] > args.ov else f"\\textit{{{v}}}"
            cells.append(v)
        lines.append(f"$a_{{{a + 1}}}$ & " + " & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}")
    with open(os.path.join(args.out, f"overlap_{i}_{j}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\noverlap view {i} (rows a) x view {j} (cols b):")
    print(np.round(o, 2))

    # 4. mask graph: one row per final instance, one column per view
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(1.4 * N + 1.5, 0.5 * len(comps) + 1))
    pos = {n: (n[0], -label[n]) for n in label}
    for na, nb in matches:
        (xa, ya), (xb, yb) = pos[na], pos[nb]
        if xa > xb:
            (xa, ya), (xb, yb) = (xb, yb), (xa, ya)
        dx = xb - xa   # edges only join masks of the same instance (same row)
        rad = 0.0 if dx <= 1 else -(0.12 + 0.07 * dx) / dx
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-",
                                     connectionstyle=f"arc3,rad={rad}",
                                     color="0.55", lw=1, zorder=1))
    for na, nb in blocked:
        (xa, ya), (xb, yb) = pos[na], pos[nb]
        ax.plot([xa, xb], [ya, yb], color="red", lw=1, ls="--", zorder=1)
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=260, color=color(label[n]) / 255, edgecolor="k",
                   zorder=2)
        ax.text(x, y, str(n[1]), ha="center", va="center", fontsize=8, zorder=3)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"view {k + 1}" for k in range(N)])
    ax.set_yticks([-k for k in range(len(comps))])
    ax.set_yticklabels([f"instance {k + 1}" for k in range(len(comps))])
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-len(comps) + 0.4, 0.8)
    for sp in ("right", "top"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "graph.pdf"))
    fig.savefig(os.path.join(args.out, "graph.png"), dpi=200)

    # 5. instances with fitted spheres
    all_p, all_c = [], []
    mesh_v, mesh_c, mesh_f = [], [], []
    accepted = []                     # (instance index, center, radius)
    inst_pts = []
    pts_by_k = {}
    u = np.random.default_rng(1).normal(size=(2000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for k, comp in enumerate(comps):
        pts = np.concatenate([pm[v][masks[v] == m] for (v, m) in comp])
        pts = pts[~np.isnan(pts).any(axis=1)]
        if len(pts) < 4:
            continue
        all_p.append(pts)
        all_c.append(np.tile(color(k), (len(pts), 1)))
        inst_pts.append(pts)
        pts_by_k[k] = pts
        res = fit_sphere(pts, thresh=args.sphere_thresh)
        if res is None:
            continue
        c, r, _ = res
        r_max = args.max_radius_single if len(comp) == 1 else args.max_radius_multi
        ok = args.min_radius <= r <= r_max
        col = color(k) if ok else np.array([255, 0, 0], np.uint8)
        all_p.append(c + r * u)
        all_c.append(np.tile(col, (len(u), 1)))
        sv, sf = uv_sphere(c, r)
        mesh_f.append(sf + sum(len(x) for x in mesh_v))
        mesh_v.append(sv)
        mesh_c.append(np.tile(col, (len(sv), 1)))
        if ok:
            accepted.append((k, c, r))
        print(f"instance {k + 1}: {len(comp)} masks, r={r:.4f} "
              f"{'accepted' if ok else 'REJECTED'}")
    write_ply(os.path.join(args.out, "spheres.ply"),
              np.concatenate(all_p).astype(np.float32), np.concatenate(all_c))
    if mesh_v:
        write_mesh_ply(os.path.join(args.out, "spheres_mesh.ply"),
                       np.concatenate(mesh_v).astype(np.float32),
                       np.concatenate(mesh_c), np.concatenate(mesh_f))

    # 6. clustering of the sphere centers (top view)
    if args.cluster_dist is not None and accepted:
        from sklearn.cluster import AgglomerativeClustering
        C = np.array([c for _, c, _ in accepted])
        if len(C) >= 2:
            lab = AgglomerativeClustering(
                n_clusters=None, distance_threshold=args.cluster_dist,
                linkage="complete").fit_predict(C)
        else:
            lab = np.zeros(1, int)
        print(f"accepted instances: {len(C)}, clusters: {len(set(lab))}")
        P = np.concatenate(inst_pts)
        mu = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
        xy = (C - mu) @ vt[:2].T          # plane of the two main directions
        first = {}
        for (k, _, _), l in zip(accepted, lab):
            first.setdefault(l, k)        # cluster color = its first instance

        # single instances and their sphere centers (red blobs)
        rng = np.random.default_rng(2)
        red = np.array([255, 0, 0], np.uint8)
        groups = {}
        for (k, c, r), l in zip(accepted, lab):
            write_ply(os.path.join(args.out, f"instance_{k + 1}.ply"),
                      pts_by_k[k].astype(np.float32),
                      np.tile(color(k), (len(pts_by_k[k]), 1)))
            d = rng.normal(size=(400, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            blob = c + 0.1 * r * d * rng.random((400, 1)) ** (1 / 3)
            write_ply(os.path.join(args.out, f"center_{k + 1}.ply"),
                      blob.astype(np.float32), np.tile(red, (400, 1)))
            light = (0.5 * color(k) + 127).astype(np.uint8)
            write_ply(os.path.join(args.out, f"sphere_{k + 1}.ply"),
                      (c + r * u).astype(np.float32), np.tile(light, (len(u), 1)))
            groups.setdefault(l, []).append(k + 1)
        for ks in groups.values():
            if len(ks) > 1:
                cs = [c for (k, c, _) in accepted if k + 1 in ks]
                dist = max(np.linalg.norm(a - b) for a in cs for b in cs)
                desc = ", ".join(f"{k} ({len(comps[k - 1])} masks, "
                                 f"{len(pts_by_k[k - 1])} pts)" for k in ks)
                print(f"merged by clustering: {desc}; "
                      f"center distance {dist:.4f}")

        # PLYs for MeshLab: same points/spheres, colored by instance / cluster
        for tag, colfn in (("instances", lambda k, l: color(k)),
                           ("clusters", lambda k, l: color(first[l]))):
            pp, pc, mv, mc, mf = [], [], [], [], []
            for (k, c, r), l in zip(accepted, lab):
                col = colfn(k, l)
                pp.append(pts_by_k[k])
                pc.append(np.tile(col, (len(pts_by_k[k]), 1)))
                sv, sf = uv_sphere(c, r)
                mf.append(sf + sum(len(x) for x in mv))
                mv.append(sv)
                mc.append(np.tile(col, (len(sv), 1)))
            write_ply(os.path.join(args.out, f"{tag}_points.ply"),
                      np.concatenate(pp).astype(np.float32), np.concatenate(pc))
            write_mesh_ply(os.path.join(args.out, f"{tag}_spheres_mesh.ply"),
                           np.concatenate(mv).astype(np.float32),
                           np.concatenate(mc), np.concatenate(mf))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        for ax, title, cols in (
                (axes[0], "instances", [color(k) for k, _, _ in accepted]),
                (axes[1], "clusters", [color(first[l]) for l in lab])):
            for (k, _, r), (x, y), col in zip(accepted, xy, cols):
                ax.add_patch(plt.Circle((x, y), r, fill=False, lw=1.5,
                                        color=col / 255))
                ax.scatter(x, y, s=18, color=col / 255, zorder=3)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.autoscale_view()
            pad = max(r for _, _, r in accepted) * 1.3
            ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
            ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "centers.pdf"))
        fig.savefig(os.path.join(args.out, "centers.png"), dpi=200)

    print(f"\nfigures written to {args.out}")


if __name__ == "__main__":
    main()
