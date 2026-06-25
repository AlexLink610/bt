import numpy as np
import argparse
import os
from PIL import Image
from scipy.spatial import cKDTree

import pyransac3d as pyrsc


def make_colors(n):
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


def sample_sphere_surface(center, radius, n_points=500):
    """Sample points uniformly on a sphere surface."""
    phi   = np.random.uniform(0, 2 * np.pi, n_points)
    costh = np.random.uniform(-1, 1, n_points)
    sinth = np.sqrt(1 - costh**2)
    pts = np.stack([
        center[0] + radius * sinth * np.cos(phi),
        center[1] + radius * sinth * np.sin(phi),
        center[2] + radius * costh,
    ], axis=1)
    return pts


def save_ply(points, colors, path):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(points.astype(np.float32), colors.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(bytes(c))
    print(f"Saved {len(points)} points to {path}")


def get_instance_ids(mask):
    if mask is None:
        return []
    ids = np.unique(mask)
    return ids[ids != 0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",    required=True)
    parser.add_argument("--filenames",   required=True)
    parser.add_argument("--masks",       required=True)
    parser.add_argument("--out",         required=True, help="Output PLY path")
    parser.add_argument("--sphere_thresh", type=float, default=0.008)
    parser.add_argument("--n_points",    type=int, default=500,
                        help="Points sampled per sphere surface (default: 500)")
    args = parser.parse_args()

    point_map = np.load(args.pointmap)
    N, H, W, _ = point_map.shape

    with open(args.filenames) as f:
        filenames = [l.strip() for l in f if l.strip()]

    masks = []
    for fname in filenames:
        stem = os.path.splitext(fname)[0]
        mpath = os.path.join(args.masks, f"mask_{stem}.png")
        if os.path.exists(mpath):
            masks.append(np.array(Image.open(mpath).convert("L").resize((W, H), Image.NEAREST)))
        else:
            masks.append(None)

    # Collect all points per instance node, then group by a simple
    # Union-Find replica — here we just fit per-image instance and report.
    # For full component grouping re-run associate_masks_graph logic;
    # here we fit one sphere per (image, instance) pair as a sanity check.
    all_sphere_pts = []
    all_sphere_colors = []

    node_list = []
    for i in range(N):
        for iid in get_instance_ids(masks[i]):
            node_list.append((i, iid))

    colors_map = make_colors(len(node_list))

    for node_idx, (i, iid) in enumerate(node_list):
        mask_flat = masks[i].reshape(-1)
        pts_flat  = point_map[i].reshape(-1, 3)
        px   = np.where(mask_flat == iid)[0]
        pts  = pts_flat[px]
        valid = ~np.isnan(pts).any(axis=1)
        pts  = pts[valid]

        if len(pts) < 4:
            print(f"  ({i},{iid}): too few points ({len(pts)}), skipping")
            continue

        sph = pyrsc.Sphere()
        try:
            center, radius, inliers = sph.fit(pts, thresh=args.sphere_thresh, maxIteration=500)
            inlier_ratio = len(inliers) / len(pts)
            print(f"  ({i},{iid}): center={np.round(center,3)}  r={radius:.4f}  inliers={inlier_ratio:.2f}")
        except Exception as e:
            print(f"  ({i},{iid}): fit failed ({e})")
            continue

        sphere_pts = sample_sphere_surface(center, radius, args.n_points)
        color = np.array(colors_map[node_idx], dtype=np.uint8)
        all_sphere_pts.append(sphere_pts)
        all_sphere_colors.append(np.tile(color, (len(sphere_pts), 1)))

    if not all_sphere_pts:
        print("No spheres fitted.")
        return

    all_pts    = np.concatenate(all_sphere_pts, axis=0)
    all_colors = np.concatenate(all_sphere_colors, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_ply(all_pts, all_colors, args.out)


if __name__ == "__main__":
    main()