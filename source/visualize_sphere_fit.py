"""
visualize_sphere_fit.py -- Visualize sphere-fitting + clustering for selected
(image, instance) nodes, with SEPARATE output layers so each can be shown on its
own slide:

  points_only.ply   just the instance pixels (raw 3D points) of all nodes
  spheres_only.ply  just the fitted sphere shells + center markers (no points)
  combined.ply      points + spheres + centers together
  in_scene.ply      combined, embedded in the full (dimmed) scene cloud
  node_<i>_<id>.ply  per-node combined

Node spec: "image_index:instance_id"  (I1_5 = 0:5, I2_2 = 1:2).

Usage:
    python visualize_sphere_fit.py \
        --pointmap  ...pointmap_r12.npy --filenames ...filenames.txt \
        --masks ~/ba/output_sam/table --nodes 0:5 1:2 \
        --scene_ply ~/ba/output_vggt/old/table/table_naive_5v.ply --scene_stride 8 \
        --out_dir ~/ba/output_vggt/old/table/sphere_vis
"""

import os
import argparse
import numpy as np
from PIL import Image
import pyransac3d as pyrsc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_mask(path, W, H):
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def sample_sphere_shell(center, radius, n=4000):
    phi = np.random.uniform(0, 2 * np.pi, n)
    costheta = np.random.uniform(-1, 1, n)
    theta = np.arccos(costheta)
    return np.stack([
        center[0] + radius * np.sin(theta) * np.cos(phi),
        center[1] + radius * np.sin(theta) * np.sin(phi),
        center[2] + radius * np.cos(theta)], axis=1)


def sphere_marker(center, r=0.006, n=1200):
    pts = []
    for rr in np.linspace(r * 0.2, r, 5):
        pts.append(sample_sphere_shell(center, rr, n // 5))
    return np.concatenate(pts, axis=0)


def read_ply_xyzrgb(path, stride=1):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        f.readline()  # format
        n, props = 0, []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                props.append(line.split()[-1].decode())
            elif line == b"end_header":
                break
        has_rgb = "red" in props
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if has_rgb:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        dt = np.dtype(fields)
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    cols = (np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
            if has_rgb else np.full((len(pts), 3), 170, np.uint8))
    valid = ~np.isnan(pts).any(axis=1)
    pts, cols = pts[valid], cols[valid]
    if stride > 1:
        pts, cols = pts[::stride], cols[::stride]
    return pts, cols


def write_ply(path, pts, cols):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(pts.astype(np.float32), cols.astype(np.uint8)):
            f.write(p.tobytes()); f.write(bytes(c))
    print(f"  saved {path}  ({len(pts):,} points)")


NODE_COLORS = [[0, 120, 255], [0, 200, 90], [255, 140, 0], [200, 0, 200]]
SHELL_COLORS = [[120, 160, 255], [120, 220, 160]]   # tinted per node so 2 spheres differ
CENTER_COLOR = [255, 0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointmap", required=True)
    ap.add_argument("--filenames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--nodes", nargs="+", required=True)
    ap.add_argument("--sphere_thresh", type=float, default=0.008)
    ap.add_argument("--scene_ply", default=None)
    ap.add_argument("--scene_stride", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pm = np.load(args.pointmap)
    N, H, W, _ = pm.shape
    with open(args.filenames) as f:
        fnames = [l.strip() for l in f if l.strip()]

    centers = []
    pts_layer_p, pts_layer_c = [], []      # points only
    sph_layer_p, sph_layer_c = [], []      # spheres + centers only

    for k, spec in enumerate(args.nodes):
        img_idx, iid = map(int, spec.split(":"))
        mask = load_mask(mask_path_for(args.masks, fnames[img_idx]), W, H)
        pts_flat = pm[img_idx].reshape(-1, 3)
        px = np.where(mask.reshape(-1) == iid)[0]
        pts = pts_flat[px]
        pts = pts[~np.isnan(pts).any(axis=1)]
        print(f"\nNode {spec}: {len(pts)} points")
        if len(pts) < 4:
            print("  <4 points, skipping"); continue

        center, radius, inliers = pyrsc.Sphere().fit(
            pts, thresh=args.sphere_thresh, maxIteration=500)
        center = np.array(center)
        centers.append(center)
        print(f"  center={center.round(4)}  radius={radius:.4f}")

        node_col = NODE_COLORS[k % len(NODE_COLORS)]
        shell_col = SHELL_COLORS[k % len(SHELL_COLORS)]
        shell = sample_sphere_shell(center, radius)
        marker = sphere_marker(center, r=max(0.004, radius * 0.18))

        # points-only layer
        pts_layer_p.append(pts)
        pts_layer_c.append(np.tile(node_col, (len(pts), 1)))
        write_ply(os.path.join(args.out_dir, f"points_{img_idx}_{iid}.ply"),
                  pts, np.tile(node_col, (len(pts), 1)))
        # spheres-only layer (shell + center marker)
        sph_layer_p.append(shell); sph_layer_c.append(np.tile(shell_col, (len(shell), 1)))
        sph_layer_p.append(marker); sph_layer_c.append(np.tile(CENTER_COLOR, (len(marker), 1)))

        # per-node combined
        np_pts = np.concatenate([pts, shell, marker], axis=0)
        np_cols = np.concatenate([
            np.tile(node_col, (len(pts), 1)),
            np.tile(shell_col, (len(shell), 1)),
            np.tile(CENTER_COLOR, (len(marker), 1))], axis=0)
        write_ply(os.path.join(args.out_dir, f"node_{img_idx}_{iid}.ply"), np_pts, np_cols)

    # ---- separate layers ----
    P_pts = np.concatenate(pts_layer_p, axis=0); P_cols = np.concatenate(pts_layer_c, axis=0)
    S_pts = np.concatenate(sph_layer_p, axis=0); S_cols = np.concatenate(sph_layer_c, axis=0)

    print("\n--- Separate layers ---")
    write_ply(os.path.join(args.out_dir, "points_only.ply"), P_pts, P_cols)
    write_ply(os.path.join(args.out_dir, "spheres_only.ply"), S_pts, S_cols)

    combined_p = np.concatenate([P_pts, S_pts], axis=0)
    combined_c = np.concatenate([P_cols, S_cols], axis=0)
    write_ply(os.path.join(args.out_dir, "combined.ply"), combined_p, combined_c)

    if args.scene_ply:
        print(f"\nLoading scene (stride {args.scene_stride})...")
        sp, sc = read_ply_xyzrgb(args.scene_ply, stride=args.scene_stride)
        sc = (sc.astype(np.float32) * 0.7).astype(np.uint8)
        # points-in-scene
        write_ply(os.path.join(args.out_dir, "points_in_scene.ply"),
                  np.concatenate([sp, P_pts], axis=0),
                  np.concatenate([sc, P_cols], axis=0))
        # spheres-in-scene
        write_ply(os.path.join(args.out_dir, "spheres_in_scene.ply"),
                  np.concatenate([sp, S_pts], axis=0),
                  np.concatenate([sc, S_cols], axis=0))

    if len(centers) >= 2:
        print("\nPairwise center distances:")
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                print(f"  {args.nodes[i]} <-> {args.nodes[j]}: {d:.4f} units")


if __name__ == "__main__":
    main()
