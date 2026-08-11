"""
visualize_clustering.py -- Visualize the final clustering step: take ALL sphere
centers (one per surviving graph component), and show them before vs after
agglomerative clustering.

Outputs:
  centers_before.ply   all sphere centers, uniform color (clustering input)
  centers_after.ply    same centers, colored by final cluster (= final count)
  centers_before_in_scene.ply / centers_after_in_scene.ply  embedded in scene

This mirrors associate_masks_graph.py's stage 3->4, but renders the CENTERS
(not the apple points) so the clustering itself is the visual.

Usage:
    python visualize_clustering.py \
        --pointmap  ~/ba/output_vggt/old/table/table_naive_5v_pointmap_r12.npy \
        --filenames ~/ba/output_vggt/old/table/table_naive_5v_filenames.txt \
        --masks     ~/ba/output_sam/table \
        --corr_thresh 0.020 --min_overlap_pct 25 --min_match_overlap 0.20 \
        --sphere_thresh 0.008 --min_radius 0.005 --max_radius 0.15 \
        --cluster_dist 0.07 \
        --scene_ply ~/ba/output_vggt/old/table/table_naive_5v.ply --scene_stride 8 \
        --out_dir ~/ba/output_vggt/old/table/cluster_vis
"""

import os
import argparse
import numpy as np
from collections import Counter
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import pyransac3d as pyrsc
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def mask_path_for(md, fn):
    return os.path.join(md, f"mask_{os.path.splitext(os.path.basename(fn))[0]}.png")


def get_ids(mask):
    if mask is None:
        return []
    u = np.unique(mask)
    return u[u != 0].tolist()


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb
    def nodes(self, root):
        return [x for x in self.p if self.find(x) == root]


def make_colors(n):
    import colorsys
    cols, golden, h = [], 0.618033988749895, 0.15
    for i in range(max(1, n)):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
        cols.append([int(r*255), int(g*255), int(b*255)])
    return cols


def sphere_marker(center, r=0.01, n=1500):
    pts = []
    for rr in np.linspace(r*0.2, r, 5):
        phi = np.random.uniform(0, 2*np.pi, n//5)
        ct = np.random.uniform(-1, 1, n//5)
        th = np.arccos(ct)
        pts.append(np.stack([
            center[0]+rr*np.sin(th)*np.cos(phi),
            center[1]+rr*np.sin(th)*np.sin(phi),
            center[2]+rr*np.cos(th)], axis=1))
    return np.concatenate(pts, axis=0)


def read_ply(path, stride=1):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"; f.readline()
        n, props = 0, []
        while True:
            l = f.readline().strip()
            if l.startswith(b"element vertex"): n = int(l.split()[-1])
            elif l.startswith(b"property"): props.append(l.split()[-1].decode())
            elif l == b"end_header": break
        rgb = "red" in props
        fl = [("x","<f4"),("y","<f4"),("z","<f4")]
        if rgb: fl += [("red","u1"),("green","u1"),("blue","u1")]
        dt = np.dtype(fl); d = np.frombuffer(f.read(n*dt.itemsize), dtype=dt, count=n)
    p = np.stack([d["x"],d["y"],d["z"]],1).astype(np.float32)
    c = (np.stack([d["red"],d["green"],d["blue"]],1).astype(np.uint8) if rgb
         else np.full((len(p),3),170,np.uint8))
    v = ~np.isnan(p).any(1); p, c = p[v], c[v]
    return (p[::stride], c[::stride]) if stride > 1 else (p, c)


def write_ply(path, pts, cols):
    h = ("ply\nformat binary_little_endian 1.0\n"
         f"element vertex {len(pts)}\n"
         "property float x\nproperty float y\nproperty float z\n"
         "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(h.encode()); 
        for p, c in zip(pts.astype(np.float32), cols.astype(np.uint8)):
            f.write(p.tobytes()); f.write(bytes(c))
    print(f"  saved {path}  ({len(pts):,} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--corr_thresh", type=float, default=0.020)
    ap.add_argument("--min_overlap_pct", type=float, default=5.0)
    ap.add_argument("--min_match_overlap", type=float, default=0.01)
    ap.add_argument("--sphere_thresh", type=float, default=0.008)
    ap.add_argument("--min_radius", type=float, default=0.005)
    ap.add_argument("--max_radius", type=float, default=0.15)
    ap.add_argument("--cluster_dist", type=float, default=0.07)
    ap.add_argument("--marker_r", type=float, default=0.012,
                    help="Radius of the center-marker blobs (visual size).")
    ap.add_argument("--scene_ply", default=None)
    ap.add_argument("--scene_stride", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]
    masks = [load_mask(mask_path_for(args.masks, f), W, H) for f in fnames]

    # --- graph association (same as main pipeline) ---
    uf = UF()
    for i in range(N):
        for iid in get_ids(masks[i]): uf.find((i, iid))

    for i in range(N):
        for j in range(i+1, N):
            ids_i, ids_j = get_ids(masks[i]), get_ids(masks[j])
            if not ids_i or not ids_j: continue
            pb = pm[j].reshape(-1,3); ab = (masks[j].reshape(-1)>0) & ~np.isnan(pb).any(1)
            ib = np.where(ab)[0]
            if ib.size == 0: continue
            tree = cKDTree(pb[ab])
            pa = pm[i].reshape(-1,3); aa = (masks[i].reshape(-1)>0) & ~np.isnan(pa).any(1)
            ia = np.where(aa)[0]
            if ia.size == 0: continue
            dist, nn = tree.query(pa[aa], workers=-1)
            corr = np.full(H*W, -1, np.int64); g = dist < args.corr_thresh
            corr[ia[g]] = ib[nn[g]]
            mfi, mfj = masks[i].reshape(-1), masks[j].reshape(-1)
            C = np.ones((len(ids_i), len(ids_j)), np.float32)
            for ai, a in enumerate(ids_i):
                pxa = np.where(mfi == a)[0]; sa = len(pxa)
                ca = corr[pxa]; val = ca >= 0; land = mfj[ca[val]] if val.any() else np.array([],int)
                for bi, b in enumerate(ids_j):
                    sb = int((mfj==b).sum()); ov = int((land==b).sum())
                    if min(sa,sb)==0: continue
                    frac = min(ov/min(sa,sb), 1.0)
                    if frac >= args.min_overlap_pct/100: C[ai,bi] = 1.0 - frac
            ri, ci = linear_sum_assignment(C)
            for ai, bi in zip(ri, ci):
                if C[ai,bi] < 1.0 - args.min_match_overlap:
                    r1, r2 = uf.find((i,ids_i[ai])), uf.find((j,ids_j[bi]))
                    if r1 == r2: continue
                    if {im for im,_ in uf.nodes(r1)} & {im for im,_ in uf.nodes(r2)}: continue
                    uf.union((i,ids_i[ai]), (j,ids_j[bi]))

    # components
    roots = {}
    for x in uf.p: roots.setdefault(uf.find(x), []).append(x)

    # --- sphere fit + filter -> centers ---
    centers = []
    for root, nodes in roots.items():
        pts = []
        for (im, iid) in nodes:
            mf = masks[im].reshape(-1); pf = pm[im].reshape(-1,3)
            p = pf[np.where(mf==iid)[0]]; pts.append(p[~np.isnan(p).any(1)])
        allp = np.concatenate(pts,0) if pts else np.empty((0,3))
        if len(allp) < 4: continue
        try:
            c, r, inl = pyrsc.Sphere().fit(allp, thresh=args.sphere_thresh, maxIteration=500)
        except Exception:
            continue
        if r < args.min_radius or r > args.max_radius: continue
        centers.append(np.array(c))
    centers = np.array(centers)
    print(f"\nSphere centers surviving filter: {len(centers)}")
    if len(centers) == 0:
        print("No centers — nothing to cluster."); return

    # --- cluster ---
    if len(centers) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=args.cluster_dist,
            linkage="complete", metric="euclidean").fit(centers).labels_
    n_clusters = len(set(labels))
    print(f"Clusters (= apple count): {n_clusters}")
    sizes = Counter(labels)
    print(f"Cluster sizes: max {max(sizes.values())}, "
          f"singletons {sum(1 for v in sizes.values() if v==1)}")

    # --- render centers as marker blobs ---
    def build(colored):
        cols_lut = make_colors(n_clusters)
        P, Cc = [], []
        for k, ctr in enumerate(centers):
            m = sphere_marker(ctr, r=args.marker_r)
            col = cols_lut[labels[k]] if colored else [255, 60, 60]
            P.append(m); Cc.append(np.tile(col, (len(m),1)))
        return np.concatenate(P,0), np.concatenate(Cc,0)

    bp, bc = build(colored=False)
    ap_, ac = build(colored=True)
    write_ply(os.path.join(args.out_dir, "centers_before.ply"), bp, bc)
    write_ply(os.path.join(args.out_dir, "centers_after.ply"), ap_, ac)

    if args.scene_ply:
        sp, sc = read_ply(args.scene_ply, stride=args.scene_stride)
        sc = (sc.astype(np.float32)*0.7).astype(np.uint8)
        write_ply(os.path.join(args.out_dir, "centers_before_in_scene.ply"),
                  np.concatenate([sp,bp],0), np.concatenate([sc,bc],0))
        write_ply(os.path.join(args.out_dir, "centers_after_in_scene.ply"),
                  np.concatenate([sp,ap_],0), np.concatenate([sc,ac],0))


if __name__ == "__main__":
    main()
