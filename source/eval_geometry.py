"""
eval_geometry.py -- Compare VGGT point maps and the pipeline result with the
Blender ground-truth geometry from render_gt_geometry.py.

For every point map (--pointmap "label=path", several allowed):
  1. pairs (VGGT point, true world position) for every apple pixel
  2. similarity transform VGGT -> Blender (Umeyama, refit on the best 90 %)
  3. point error: distance after alignment, in radii of the pixel's apple
  4. local point error: as 3, but after removing each apple's own median
     offset -> spread of the points around the apple without global
     distortions of the reconstruction (what the association sees)
  5. oracle center error: center of a sphere with the TRUE radius, fitted
     robustly to all points of each apple using the TRUE association -> the
     best center accuracy this reconstruction allows, independent of our
     association (the free-radius RANSAC fit is reported as radius ratio,
     a measure of how much the points are smeared)

With --conf "label=path" (raw confidence map (N,H,W) of that point map):
  confidence analysis: rank correlation of confidence and point error,
  error per confidence quintile, and what dropping the least confident 25 % /
  50 % of the apple pixels removes (good points < 1 r, bad points > 2 r)

With --filtered "name=path" (a filtered version of the FIRST point map, e.g.
radius or confidence filter): share of apple pixels removed, share of good
(< 1 r) and bad (> 2 r) points among them, and the error of what is left

With --dump (pipeline result computed on the point map --dump_on):
  6. F1 on pixel IoU (as in eval_gt_assoc.py) and the center error of the
     true positives
  7. distance F1: counted centers vs. true centers, one-to-one, a match
     counts if it is closer than one (and two) radii of the true apple

All errors are given in apple radii (scale-free).

Usage:
    python eval_geometry.py --run <run_dir> --filenames <..._filenames.txt>
        --pointmap "head=<..._pointmap_head.npy>" --pointmap "depth=<..._pointmap.npy>"
        [--dump <assoc.json> --dump_on head] [--csv results.csv --tag S05]
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import OpenEXR
from PIL import Image
from scipy.optimize import linear_sum_assignment, least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gt_assoc import load_gid_map, mask_pixels, f1_iou   # noqa: E402

AOV_OBJ, AOV_POS = "apple_obj", "apple_pos"
AXIS = {"R": 0, "X": 0, "G": 1, "Y": 1, "B": 2, "Z": 2}


# ── ground truth ──────────────────────────────────────────────────────────────
def read_exr(path):
    # copy the pixels while the file is open (newer OpenEXR versions release
    # the channel data when the file is closed)
    with OpenEXR.File(path, separate_channels=True) as f:
        ch = f.channels()
        obj_names = [n for n in ch if AOV_OBJ in n]
        pos_names = [n for n in ch if AOV_POS in n and n.split(".")[-1] in AXIS]
        if not obj_names or len(pos_names) < 3:
            sys.exit(f"{path}: channels {AOV_OBJ}/{AOV_POS} missing, have {list(ch)}")
        obj = np.rint(np.array(ch[obj_names[0]].pixels, np.float64)).astype(np.int64)
        pos_names = sorted(pos_names, key=lambda n: AXIS[n.split(".")[-1]])[:3]
        pos = np.stack([np.array(ch[n].pixels, np.float64) for n in pos_names], -1)
    return obj, pos


def nearest_index(Hs, Ws, H, W):
    """Source pixel for every target pixel, exactly as PIL NEAREST resizes
    the masks in the pipeline."""
    idx = np.arange(Hs * Ws, dtype=np.int32).reshape(Hs, Ws)
    return np.array(Image.fromarray(idx).resize((W, H), Image.NEAREST)).reshape(-1)


def load_gt(run, fnames, H, W):
    """Per view: object index and true position at point-map resolution,
    plus the mapping global mask ID -> object index."""
    views, votes = [], {}
    gid = load_gid_map(os.path.join(run, "masks_gt"), fnames)
    for i, fn in enumerate(fnames):
        stem = os.path.splitext(os.path.basename(fn))[0]
        obj, pos = read_exr(os.path.join(run, "geom_exr", f"{stem}.exr"))
        Hs, Ws = obj.shape
        sel = nearest_index(Hs, Ws, H, W)
        views.append((obj.reshape(-1)[sel], pos.reshape(-1, 3)[sel]))
        mpath = os.path.join(run, "masks_gt", f"mask_{stem}.png")
        if os.path.exists(mpath):
            m = np.array(Image.open(mpath))
            if m.shape == obj.shape:
                for local in np.unique(m[m > 0]):
                    g = gid.get((i, int(local)))
                    if g is None:
                        continue
                    o = obj[m == local]
                    o = o[o > 0]
                    if len(o):
                        votes.setdefault(g, []).append(int(np.bincount(o).argmax()))
    gid_obj = {g: int(np.bincount(v).argmax()) for g, v in votes.items()}
    return views, gid_obj


# ── geometry helpers ─────────────────────────────────────────────────────────
def umeyama(X, Y):
    """s, R, t minimising ||Y - (s R X + t)||."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(D) @ E) / ((Xc ** 2).sum() / len(X))
    return s, R, my - s * R @ mx


def align(P, Q, keep=0.9, max_n=200_000, seed=0):
    """Similarity VGGT -> Blender. It is estimated in the direction
    Blender -> VGGT and then inverted: noise in the source points would bias
    the Umeyama scale downwards, noise in the target points does not, and the
    noisy side is VGGT."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(P), min(len(P), max_n), replace=False)
    s, R, t = umeyama(Q[sub], P[sub])
    res = np.linalg.norm(s * Q[sub] @ R.T + t - P[sub], axis=1)
    best = sub[res <= np.quantile(res, keep)]
    s, R, t = umeyama(Q[best], P[best])
    # invert  P = s R Q + t   ->   Q = (1/s) R^T (P - t)
    return 1.0 / s, R.T, -(R.T @ t) / s


def fit_sphere(pts, thresh):
    """RANSAC sphere as in the pipeline (pyRANSAC-3D, 500 iterations), with a
    least-squares fallback."""
    try:
        import pyransac3d as pyrsc
        random.seed(42)
        np.random.seed(42)
        c, r, _ = pyrsc.Sphere().fit(pts, thresh=thresh, maxIteration=500)
        if c is not None and len(c) == 3 and np.isfinite(r) and r > 0:
            return np.asarray(c, float), float(r)
    except ImportError:
        pass
    A = np.c_[2 * pts, np.ones(len(pts))]
    sol = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 0)))


def fit_center_fixed_radius(pts, r, max_n=5000, seed=0):
    """Robust center of a sphere with known radius (soft-L1 loss)."""
    if len(pts) > max_n:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_n, replace=False)]
    res = least_squares(lambda c: np.linalg.norm(pts - c, axis=1) - r,
                        np.median(pts, axis=0), loss="soft_l1", f_scale=0.25 * r)
    return res.x


def stats(x):
    return {"median": float(np.median(x)), "mean": float(np.mean(x)),
            "p90": float(np.quantile(x, 0.9)),
            "lt_0.5r": float(np.mean(x < 0.5)), "lt_1r": float(np.mean(x < 1.0))}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--pointmap", action="append", required=True,
                    help='"label=path"; several allowed')
    ap.add_argument("--conf", action="append", default=[],
                    help='"label=path" raw confidence (N,H,W) of point map "label"')
    ap.add_argument("--filtered", action="append", default=[],
                    help='"name=path": filtered version of the first point map')
    ap.add_argument("--sphere_thresh", type=float, default=0.008,
                    help="RANSAC tolerance in point-map units (as in the pipeline)")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--dump_on", default=None,
                    help="label of the point map the pipeline ran on (frame of --dump)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]
    with open(os.path.join(args.run, "gt_geometry.json")) as f:
        geo = json.load(f)
    C = {a["index"]: np.array(a["center"]) for a in geo["apples"]}
    Rad = {a["index"]: a["radius"] for a in geo["apples"]}

    pms = []
    for spec in args.pointmap:
        label, path = spec.split("=", 1)
        pms.append((label, np.load(path)))
    N, H, W, _ = pms[0][1].shape
    views, gid_obj = load_gt(args.run, fnames, H, W)
    print(f"{len(views)} views, {len(gid_obj)} mask IDs linked to Blender objects")

    confs = {}
    for spec in args.conf:
        label, path = spec.split("=", 1)
        confs[label] = np.load(path)

    rows, transforms = [], {}
    for label, pm in pms:
        P, Q, O, K = [], [], [], []
        cf = confs.get(label)
        for i, (obj, pos) in enumerate(views):
            p = pm[i].reshape(-1, 3)
            ok = (obj > 0) & ~np.isnan(p).any(1) & np.isin(obj, list(C))
            P.append(p[ok]); Q.append(pos[ok]); O.append(obj[ok])
            if cf is not None:
                K.append(cf[i].reshape(-1)[ok])
        P, Q, O = np.concatenate(P), np.concatenate(Q), np.concatenate(O)
        s, R, t = align(P, Q)
        transforms[label] = (s, R, t)
        Pa = s * P @ R.T + t
        radius = np.array([Rad[o] for o in O])
        err = np.linalg.norm(Pa - Q, axis=1) / radius
        pt = stats(err)

        loc, oc, rr = [], [], []
        for o in np.unique(O):
            sel = O == o
            d = Pa[sel] - Q[sel]
            loc.append(np.linalg.norm(d - np.median(d, axis=0), axis=1) / Rad[o])
            pts = Pa[sel]
            if len(pts) < 50:
                continue
            c = fit_center_fixed_radius(pts, Rad[o])
            oc.append(np.linalg.norm(c - C[o]) / Rad[o])
            rr.append(fit_sphere(pts, args.sphere_thresh * s)[1] / Rad[o])
        lp = stats(np.concatenate(loc))
        oracle = stats(np.array(oc))
        row = {"tag": args.tag, "pointmap": label, "pixels": len(P), "scale": s,
               "pt_median": pt["median"], "pt_p90": pt["p90"],
               "pt_lt_0.5r": pt["lt_0.5r"], "pt_lt_1r": pt["lt_1r"],
               "local_median": lp["median"], "local_p90": lp["p90"],
               "local_lt_0.5r": lp["lt_0.5r"],
               "oracle_center_median": oracle["median"],
               "oracle_center_p90": oracle["p90"],
               "oracle_radius_ratio": float(np.median(rr))}

        print(f"\n[{label}]  {len(P):,} apple pixels, scale VGGT->Blender {s:.3f}")
        print(f"  point error (apple radii): median {pt['median']:.2f}  "
              f"mean {pt['mean']:.2f}  p90 {pt['p90']:.2f}  "
              f"< 0.5 r: {pt['lt_0.5r']:.0%}  < 1 r: {pt['lt_1r']:.0%}")
        print(f"  local error, per-apple offset removed: median {lp['median']:.2f}  "
              f"p90 {lp['p90']:.2f}  < 0.5 r: {lp['lt_0.5r']:.0%}  < 1 r: {lp['lt_1r']:.0%}")
        print(f"  oracle center error ({len(oc)} apples, true radius): median "
              f"{oracle['median']:.2f}  p90 {oracle['p90']:.2f}")
        print(f"  free RANSAC radius / true radius (smearing): median {np.median(rr):.2f}")

        if cf is not None:
            row.update(conf_analysis(np.concatenate(K), err))

        if label == pms[0][0]:
            for spec in args.filtered:
                name, path = spec.split("=", 1)
                fpm = np.load(path)
                removed = []
                for i, (obj, pos) in enumerate(views):
                    p = pm[i].reshape(-1, 3)
                    ok = (obj > 0) & ~np.isnan(p).any(1) & np.isin(obj, list(C))
                    removed.append(np.isnan(fpm[i].reshape(-1, 3)[ok]).any(1))
                removed = np.concatenate(removed)
                good, bad = err < 1.0, err > 2.0
                fr = {"removed": float(removed.mean()),
                      "good_removed": float(good[removed].sum() / max(good.sum(), 1)),
                      "bad_removed": float(bad[removed].sum() / max(bad.sum(), 1)),
                      "median_left": float(np.median(err[~removed])),
                      "lt_1r_left": float(np.mean(err[~removed] < 1.0))}
                print(f"  filter {name}: removes {fr['removed']:.1%} of apple pixels, "
                      f"{fr['good_removed']:.0%} of good (< 1 r) and "
                      f"{fr['bad_removed']:.0%} of bad (> 2 r) points; left: median "
                      f"{fr['median_left']:.2f}, < 1 r {fr['lt_1r_left']:.0%}")
                row.update({f"{name}_{k}": v for k, v in fr.items()})

        if args.dump and label == (args.dump_on or pms[0][0]):
            row.update(eval_dump(args, fnames, pm, s, R, t, C, Rad, gid_obj, H, W))
        rows.append(row)

    if args.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.4f}" if isinstance(r.get(k), float)
                                 else str(r.get(k, "")) for k in keys) + "\n")


def conf_analysis(conf, err):
    from scipy.stats import spearmanr
    rho = float(spearmanr(conf, err).correlation)
    q = np.quantile(conf, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(conf, q)
    med = [float(np.median(err[bins == b])) for b in range(5)]
    print(f"  confidence: Spearman(conf, error) {rho:+.2f}; median error per "
          f"quintile (low -> high conf): " + " ".join(f"{m:.2f}" for m in med))
    out = {"conf_spearman": rho}
    good, bad = err < 1.0, err > 2.0
    for frac in (0.25, 0.5):
        drop = conf < np.quantile(conf, frac)
        g = float(good[drop].sum() / max(good.sum(), 1))
        b = float(bad[drop].sum() / max(bad.sum(), 1))
        m = float(np.median(err[~drop]))
        print(f"  drop least confident {frac:.0%}: removes {g:.0%} of good points "
              f"(< 1 r) and {b:.0%} of bad points (> 2 r); remaining median error {m:.2f}")
        out[f"drop{int(frac * 100)}_good_removed"] = g
        out[f"drop{int(frac * 100)}_bad_removed"] = b
        out[f"drop{int(frac * 100)}_median"] = m
    return out


def eval_dump(args, fnames, pm, s, R, t, C, Rad, gid_obj, H, W):
    with open(args.dump) as f:
        d = json.load(f)
    masks_dir = os.path.join(args.run, "masks_gt")
    gid = load_gid_map(masks_dir, fnames)
    n_gt = len(set(gid.values()))

    clusters = {}
    for k, (comp, lab) in enumerate(zip(d["accepted"], d["labels"])):
        clusters.setdefault(lab, []).append(k)
    labs = sorted(clusters)
    groups = [[n for k in clusters[l] for n in d["accepted"][k]] for l in labs]

    # center of a counted apple = mean of the sphere centers of its instances
    if "centers" in d:
        cen = np.array([np.mean([d["centers"][k] for k in clusters[l]], axis=0)
                        for l in labs])
    else:   # older dumps: fit a sphere to all points of the cluster
        print("  (dump without centers -> refitting spheres per cluster)")
        cen = []
        for g in groups:
            pts = []
            for i, local in g:
                stem = os.path.splitext(os.path.basename(fnames[i]))[0]
                m = np.array(Image.open(os.path.join(masks_dir, f"mask_{stem}.png"))
                             .resize((W, H), Image.NEAREST)).reshape(-1)
                p = pm[i].reshape(-1, 3)[m == local]
                pts.append(p[~np.isnan(p).any(1)])
            cen.append(fit_sphere(np.concatenate(pts), args.sphere_thresh)[0])
        cen = np.array(cen)
    cen_a = s * cen @ R.T + t

    # F1 on pixel IoU + center error of the true positives
    pix = mask_pixels(masks_dir, fnames)
    prec, rec, f1, miou, tps = f1_iou(groups, gid, pix, n_gt)
    ce = [np.linalg.norm(cen_a[k] - C[gid_obj[g]]) / Rad[gid_obj[g]]
          for k, g, _ in tps if g in gid_obj]

    # distance F1: counted centers vs. true centers of the visible apples
    objs = sorted({gid_obj[g] for g in set(gid.values()) if g in gid_obj})
    Ct = np.array([C[o] for o in objs])
    Rt = np.array([Rad[o] for o in objs])
    D = np.linalg.norm(cen_a[:, None] - Ct[None], axis=2)
    r_, c_ = linear_sum_assignment(D)

    def dist_f1(k):
        tp = int(np.sum(D[r_, c_] < k * Rt[c_]))
        p, r = tp / len(cen_a), tp / len(Ct)
        return (2 * p * r / (p + r) if tp else 0.0), p, r
    f1_d, p_d, r_d = dist_f1(1)
    f1_d2 = dist_f1(2)[0]

    print(f"  pipeline ({len(groups)} counted, {n_gt} true):")
    print(f"    IoU F1 {f1:.3f}  (P {prec:.3f}, R {rec:.3f}, mean IoU {miou:.3f})")
    if ce:
        print(f"    center error of true positives: median {np.median(ce):.2f}  "
              f"p90 {np.quantile(ce, 0.9):.2f} apple radii")
    print(f"    distance F1 {f1_d:.3f}  (P {p_d:.3f}, R {r_d:.3f}, match < 1 radius)"
          f"   |  {f1_d2:.3f} with match < 2 radii")
    return {"count": len(groups), "gt": n_gt, "f1_iou": f1, "precision_iou": prec,
            "recall_iou": rec, "tp_center_median": float(np.median(ce)) if ce else float("nan"),
            "tp_center_p90": float(np.quantile(ce, 0.9)) if ce else float("nan"),
            "f1_dist": f1_d, "f1_dist_2r": f1_d2}


if __name__ == "__main__":
    main()
