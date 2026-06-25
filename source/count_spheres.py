"""
count_spheres.py -- Count apples by fitting spheres to instance masks and clustering centers.

Pipeline:
  - For each (image, instance) node: collect 3D points from pointmap
  - Fit a sphere via RANSAC -> get center
  - DBSCAN on sphere centers -> each cluster = one apple

Usage:
    python count_spheres.py \
        --pointmap  ~/ba/output_vggt/t02_360_122v_pointmap.npy \
        --filenames ~/ba/output_vggt/t02_360_122v_filenames.txt \
        --masks     ~/ba/output_sam/tree_02/semantics_sam3 \
        --ground_truth 113
"""

import os
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc


def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def get_instance_ids(mask):
    if mask is None:
        return []
    ids = np.unique(mask)
    return ids[ids != 0].tolist()


def fit_sphere(points, thresh, max_iter=500):
    if len(points) < 4:
        return None
    sph = pyrsc.Sphere()
    try:
        center, radius, inliers = sph.fit(points, thresh=thresh, maxIteration=max_iter)
        if any(np.isnan(center)) or np.isnan(radius) or radius <= 0:
            return None
        inlier_ratio = len(inliers) / len(points)
        return np.array(center), radius, inlier_ratio
    except Exception:
        return None

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",         required=True)
    parser.add_argument("--filenames",        required=True)
    parser.add_argument("--masks",            required=True)
    parser.add_argument("--ground_truth",     type=int, default=113)
    parser.add_argument("--sphere_thresh",    type=float, default=0.008,
                        help="RANSAC inlier distance for sphere fitting (default: 0.008)")
    parser.add_argument("--min_inlier_ratio", type=float, default=0.5,
                        help="Min inlier ratio to accept a sphere fit (default: 0.5)")
    parser.add_argument("--max_radius",       type=float, default=0.12,
                        help="Max sphere radius to accept (default: 0.12 units)")
    parser.add_argument("--min_radius",       type=float, default=0.01,
                        help="Min sphere radius to accept (default: 0.01 units)")
    parser.add_argument("--dbscan_eps",       type=float, default=0.05,
                        help="DBSCAN epsilon for clustering centers (default: 0.05)")
    parser.add_argument("--dbscan_min",       type=int,   default=1,
                        help="DBSCAN min_samples (default: 1)")
    parser.add_argument("--save_centers_ply", default=None,
                        help="Path to save sphere centers as PLY (colored by DBSCAN cluster).")
    args = parser.parse_args()

    print("=" * 60)
    print("Sphere Center Clustering -- Apple Counter")
    print("=" * 60)
    print(f"  sphere_thresh={args.sphere_thresh}  min_inlier_ratio={args.min_inlier_ratio}")
    print(f"  radius range=[{args.min_radius}, {args.max_radius}]")
    print(f"  dbscan_eps={args.dbscan_eps}  dbscan_min={args.dbscan_min}")

    point_map = np.load(args.pointmap)
    N, H, W, _ = point_map.shape
    print(f"\nPointmap: {point_map.shape}  ({point_map.nbytes/1e6:.1f} MB)")

    with open(args.filenames) as f:
        filenames = [l.strip() for l in f if l.strip()]

    # Fit sphere per (image, instance) node
    centers = []
    radii   = []
    skipped_few_pts = 0
    skipped_inliers = 0
    skipped_radius  = 0
    total_nodes     = 0

    print(f"\nFitting spheres...")
    for i, fname in enumerate(filenames):
        stem  = os.path.splitext(fname)[0]
        mpath = os.path.join(args.masks, f"mask_{stem}.png")
        mask  = load_mask(mpath, W, H)
        if mask is None:
            continue

        pts_flat  = point_map[i].reshape(-1, 3)
        mask_flat = mask.reshape(-1)

        for iid in get_instance_ids(mask):
            total_nodes += 1
            px    = np.where(mask_flat == iid)[0]
            pts   = pts_flat[px]
            valid = ~np.isnan(pts).any(axis=1)
            pts   = pts[valid]

            if len(pts) < 4:
                skipped_few_pts += 1
                continue

            result = fit_sphere(pts, thresh=args.sphere_thresh)
            if result is None:
                skipped_few_pts += 1
                continue

            center, radius, inlier_ratio = result

            if inlier_ratio < args.min_inlier_ratio:
                skipped_inliers += 1
                continue
            if radius < args.min_radius or radius > args.max_radius:
                skipped_radius += 1
                continue

            centers.append(center)
            radii.append(radius)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{N}] accepted: {len(centers)}  "
                  f"skipped: pts={skipped_few_pts} inliers={skipped_inliers} radius={skipped_radius}")

    print(f"\nTotal nodes:       {total_nodes}")
    print(f"Accepted spheres:  {len(centers)}")
    print(f"Skipped (few pts): {skipped_few_pts}")
    print(f"Skipped (inliers): {skipped_inliers}")
    print(f"Skipped (radius):  {skipped_radius}")

    if len(centers) == 0:
        print("No valid sphere centers — check sphere_thresh / min_inlier_ratio / radius range.")
        return

    centers = np.array(centers)
    radii   = np.array(radii)
    print(f"\nRadius stats: mean={radii.mean():.4f}  median={np.median(radii):.4f}  "
          f"min={radii.min():.4f}  max={radii.max():.4f}")

    # DBSCAN on sphere centers
    print(f"\nRunning DBSCAN (eps={args.dbscan_eps}, min_samples={args.dbscan_min})...")
    db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min).fit(centers)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    gt    = args.ground_truth
    error = n_clusters - gt

    print(f"\n{'='*60}")
    print(f"  INSTANCE COUNT: {n_clusters}")
    print(f"  NOISE POINTS:   {n_noise}")
    print(f"  GROUND TRUTH:   {gt}")
    print(f"  Error: {error:+d}" + (f"  ({100*error/gt:+.1f}%)" if gt > 0 else ""))
    print(f"{'='*60}")

    if args.save_centers_ply:
        cluster_colors = make_colors(max(labels) + 1) if max(labels) >= 0 else []
        noise_color = [128, 128, 128]

        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(centers)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        with open(args.save_centers_ply, "wb") as f:
            f.write(header.encode("ascii"))
            for pt, lbl in zip(centers, labels):
                color = noise_color if lbl == -1 else cluster_colors[lbl]
                f.write(pt.astype(np.float32).tobytes())
                f.write(bytes(color))
        print(f"\nCenters PLY saved: {args.save_centers_ply}  ({len(centers)} points)")


if __name__ == "__main__":
    main()