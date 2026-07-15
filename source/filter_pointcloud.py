"""
filter_pointcloud.py -- Remove noise/outliers from a point cloud (.ply) using
Open3D, supporting both radius-based and statistical outlier removal.

This is the preprocessing step FruitNeRF applies before clustering
(their config: remove_outliers_nb_points=200, remove_outliers_radius=0.01).

Methods:
  radius       remove points with < nb_points neighbours within radius
               (predictable, scale-sensitive -- FruitNeRF's choice)
  statistical  remove points whose mean distance to k nearest neighbours is
               > std_ratio standard deviations above the global mean
               (adapts to local density)
  both         run statistical first, then radius

Usage:
    # radius (FruitNeRF-style)
    python filter_pointcloud.py \
        --in  ~/ba/output_vggt/t02_360_32v_fruit.ply \
        --out ~/ba/output_vggt/t02_360_32v_fruit_filtered.ply \
        --method radius --nb_points 20 --radius 0.02

    # statistical
    python filter_pointcloud.py --in ... --out ... \
        --method statistical --nb_neighbors 20 --std_ratio 2.0

    # both, and also save the REMOVED points for inspection
    python filter_pointcloud.py --in ... --out ... --method both --save_removed

Tip: run with --analyze first (no filtering) to see the nearest-neighbour
distance distribution, which tells you what radius/nb_points make sense
for YOUR scene scale (VGGT units are scene-normalized, not metric).
"""

import os
import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d not installed. Run:  pip install open3d")


def analyze(pcd):
    """Report NN-distance stats to help choose radius / nb_points."""
    pts = np.asarray(pcd.points)
    print(f"\nPoints: {len(pts):,}")
    print(f"Extent: X[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]  "
          f"Y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]  "
          f"Z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")

    # Nearest-neighbour distances (sample for speed)
    n = len(pts)
    sample_idx = np.random.default_rng(42).choice(
        n, size=min(20000, n), replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in sample_idx:
        # k=2 because the first hit is the point itself
        _, _, d2 = tree.search_knn_vector_3d(pcd.points[i], 2)
        if len(d2) > 1:
            dists.append(np.sqrt(d2[1]))
    dists = np.array(dists)
    p = np.percentile(dists, [10, 25, 50, 75, 90, 99])
    print(f"\nNearest-neighbour distance (sampled {len(dists):,} points):")
    print(f"  mean={dists.mean():.5f}  median={p[2]:.5f}")
    print(f"  p10={p[0]:.5f}  p25={p[1]:.5f}  p75={p[3]:.5f}  "
          f"p90={p[4]:.5f}  p99={p[5]:.5f}")
    print(f"\nSuggestion: a --radius around {p[4]*3:.4f} (3x the p90 NN dist) is a")
    print(f"reasonable starting point; tune --nb_points to taste.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="inp", required=True, help="Input .ply")
    parser.add_argument("--out", default=None, help="Output .ply (filtered)")
    parser.add_argument("--method", default="radius",
                        choices=["radius", "statistical", "both"],
                        help="Outlier removal method (default: radius).")
    # radius params
    parser.add_argument("--nb_points", type=int, default=20,
                        help="radius method: min neighbours within --radius (default: 20).")
    parser.add_argument("--radius", type=float, default=0.02,
                        help="radius method: neighbourhood radius in scene units (default: 0.02).")
    # statistical params
    parser.add_argument("--nb_neighbors", type=int, default=20,
                        help="statistical method: k nearest neighbours (default: 20).")
    parser.add_argument("--std_ratio", type=float, default=2.0,
                        help="statistical method: std-dev multiplier (default: 2.0).")
    # misc
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Optional voxel downsample before filtering (0 = off).")
    parser.add_argument("--analyze", action="store_true",
                        help="Only report NN-distance stats, do not filter.")
    parser.add_argument("--save_removed", action="store_true",
                        help="Also save the removed points as *_removed.ply.")
    args = parser.parse_args()

    print(f"Loading {args.inp} ...")
    pcd = o3d.io.read_point_cloud(args.inp)
    n0 = len(pcd.points)
    if n0 == 0:
        raise SystemExit("Empty point cloud (or unreadable .ply).")
    print(f"Loaded {n0:,} points")

    if args.analyze:
        analyze(pcd)
        return

    if not args.out:
        raise SystemExit("--out is required unless --analyze is used.")

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"Voxel downsample ({args.voxel}): {n0:,} -> {len(pcd.points):,}")

    kept = pcd
    removed_clouds = []

    if args.method in ("statistical", "both"):
        before = len(kept.points)
        kept, idx = kept.remove_statistical_outlier(
            nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
        removed = before - len(kept.points)
        print(f"Statistical (k={args.nb_neighbors}, std_ratio={args.std_ratio}): "
              f"{before:,} -> {len(kept.points):,}  (removed {removed:,}, "
              f"{100*removed/before:.1f}%)")

    if args.method in ("radius", "both"):
        before = len(kept.points)
        kept, idx = kept.remove_radius_outlier(
            nb_points=args.nb_points, radius=args.radius)
        removed = before - len(kept.points)
        print(f"Radius (nb_points={args.nb_points}, radius={args.radius}): "
              f"{before:,} -> {len(kept.points):,}  (removed {removed:,}, "
              f"{100*removed/before:.1f}%)")

    n1 = len(kept.points)
    print(f"\nTotal: {n0:,} -> {n1:,}  (removed {n0-n1:,}, {100*(n0-n1)/n0:.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_point_cloud(args.out, kept)
    print(f"Saved: {args.out}")

    if args.save_removed:
        # Recompute which points survived by nearest-neighbour matching is
        # expensive; instead re-run filters tracking indices from the original.
        orig = o3d.io.read_point_cloud(args.inp)
        if args.voxel > 0:
            orig = orig.voxel_down_sample(args.voxel)
        work = orig
        keep_idx = np.arange(len(work.points))
        if args.method in ("statistical", "both"):
            work, idx = work.remove_statistical_outlier(
                nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
            keep_idx = keep_idx[idx]
        if args.method in ("radius", "both"):
            work, idx = work.remove_radius_outlier(
                nb_points=args.nb_points, radius=args.radius)
            keep_idx = keep_idx[idx]
        all_idx = np.arange(len(orig.points))
        removed_idx = np.setdiff1d(all_idx, keep_idx)
        rem = orig.select_by_index(removed_idx)
        rpath = os.path.splitext(args.out)[0] + "_removed.ply"
        o3d.io.write_point_cloud(rpath, rem)
        print(f"Removed points saved: {rpath}  ({len(rem.points):,} points)")


if __name__ == "__main__":
    main()
