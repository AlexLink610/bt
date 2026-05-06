"""
count_apples.py  –  Count apples using SAM3 instance masks + VGGT point maps.

Pipeline:
    1. Load _pointmap.npy  (N, H, W, 3)  and  _filenames.txt
    2. For each image i:
         a. Load instance mask  mask_frame_XXXXX.png
         b. For each instance ID (non-zero pixel value):
              - Get all pixels belonging to that instance
              - Look up their 3D coordinates in point_map[i]
              - Remove NaNs
              - Compute centroid (mean x, y, z)
              - Store centroid + image index + instance ID
    3. DBSCAN on all centroids  →  each cluster = one unique apple
    4. Print count + save debug outputs

Usage:
    python count_apples.py ^
        --pointmap  C:\\BA\\output_vggt\\t02_180_32v_pointmap.npy ^
        --filenames C:\\BA\\output_vggt\\t02_180_32v_filenames.txt ^
        --masks     C:\\BA\\output_sam\\tree_02\\semantics_sam3 ^
        --out       C:\\BA\\output_vggt\\t02_180_32v_count.txt

Optional:
    --eps       DBSCAN epsilon (default: 0.15)   tune if over/undercounting
    --min_pts   DBSCAN min_samples (default: 2)
    --conf_min  minimum confidence to include a centroid (default: 0.0)
    --confmap   path to _confmap.npy (optional, needed for --conf_min filtering)
    --save_ply  save centroids as PLY for visual inspection
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pointmap(path):
    """Load (N, H, W, 3) float32 pointmap."""
    pm = np.load(path)
    print(f"  Pointmap shape: {pm.shape}  ({pm.nbytes / 1e6:.1f} MB)")
    return pm


def load_filenames(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def mask_path_for(masks_dir, fname):
    """
    Given an image filename like 'frame_00001.JPG',
    return the expected instance mask path:
        <masks_dir>/mask_frame_00001.png
    """
    stem = os.path.splitext(os.path.basename(fname))[0]  # 'frame_00001'
    return os.path.join(masks_dir, f"mask_{stem}.png")


def get_centroids_for_image(point_map_i, mask_path, conf_map_i=None, conf_min=0.0):
    """
    For one image, compute one 3D centroid per SAM3 instance.

    Returns list of (centroid_xyz, instance_id, mean_conf)
    """
    H, W = point_map_i.shape[:2]

    if not os.path.exists(mask_path):
        return []

    mask = np.array(Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST))
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]   # remove background

    centroids = []
    for iid in instance_ids:
        pixels = np.argwhere(mask == iid)            # (K, 2) row/col

        pts = point_map_i[pixels[:, 0], pixels[:, 1]]  # (K, 3)

        # Remove NaN points
        valid = ~np.isnan(pts).any(axis=1)
        pts = pts[valid]

        if len(pts) < 3:                             # too few points to trust
            continue

        centroid = pts.mean(axis=0)                  # (3,)

        # Optional confidence filtering
        mean_conf = 1.0
        if conf_map_i is not None:
            confs = conf_map_i[pixels[:, 0], pixels[:, 1]][valid]
            mean_conf = float(confs.mean())
            if mean_conf < conf_min:
                continue

        centroids.append((centroid, int(iid), mean_conf))

    return centroids


def save_centroids_ply(centroids_xyz, labels, path):
    """Save centroids as PLY — each DBSCAN cluster gets a unique color."""
    n = len(centroids_xyz)
    if n == 0:
        return

    # Assign colors by cluster label
    rng = np.random.default_rng(42)
    unique_labels = np.unique(labels)
    color_map = {
        lbl: (rng.integers(50, 255, 3).tolist() if lbl >= 0 else [50, 50, 50])
        for lbl in unique_labels
    }

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for xyz, lbl in zip(centroids_xyz, labels):
            r, g, b = color_map[lbl]
            f.write(np.array(xyz, dtype=np.float32).tobytes())
            f.write(bytes([r, g, b]))
    print(f"  Centroids PLY saved: {path}  ({n} points)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointmap",  required=True,
                        help="Path to _pointmap.npy")
    parser.add_argument("--filenames", required=True,
                        help="Path to _filenames.txt")
    parser.add_argument("--masks",     required=True,
                        help="Directory containing mask_frame_XXXXX.png files")
    parser.add_argument("--out",       default=None,
                        help="Path to save count result txt (optional)")
    parser.add_argument("--confmap",   default=None,
                        help="Path to _confmap.npy (optional)")
    parser.add_argument("--conf_min",  type=float, default=0.0,
                        help="Min mean confidence per centroid (default: 0.0 = keep all)")
    parser.add_argument("--eps",       type=float, default=0.15,
                        help="DBSCAN epsilon — max distance between same-apple centroids "
                             "(default: 0.15, tune based on coordinate scale)")
    parser.add_argument("--min_pts",   type=int,   default=2,
                        help="DBSCAN min_samples (default: 2)")
    parser.add_argument("--save_ply",  action="store_true",
                        help="Save centroids as colour-coded PLY for visual inspection")
    args = parser.parse_args()

    print("=" * 60)
    print("Apple Counter — centroid + DBSCAN")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\nLoading inputs...")
    point_map = load_pointmap(args.pointmap)
    filenames = load_filenames(args.filenames)
    N, H, W, _ = point_map.shape

    if len(filenames) != N:
        print(f"ERROR: filenames ({len(filenames)}) != pointmap N ({N})")
        sys.exit(1)

    conf_map = None
    if args.confmap and os.path.exists(args.confmap):
        conf_map = np.load(args.confmap)
        print(f"  Confmap shape: {conf_map.shape}")

    # ── Extract centroids ─────────────────────────────────────────────────────
    print(f"\nExtracting centroids from {N} images...")
    all_centroids = []
    total_instances = 0
    missing_masks   = 0

    for i, fname in enumerate(filenames):
        mpath = mask_path_for(args.masks, fname)
        conf_i = conf_map[i] if conf_map is not None else None

        if not os.path.exists(mpath):
            missing_masks += 1
            print(f"  [{i+1}/{N}] MISSING mask for {fname}")
            continue

        centroids = get_centroids_for_image(
            point_map[i], mpath, conf_i, args.conf_min
        )
        total_instances += len(centroids)
        all_centroids.extend([(c, iid, conf, i) for c, iid, conf in centroids])

        print(f"  [{i+1}/{N}] {fname}: {len(centroids)} instances")

    print(f"\nTotal centroids collected: {len(all_centroids)}")
    if missing_masks:
        print(f"WARNING: {missing_masks}/{N} masks were missing")

    if len(all_centroids) == 0:
        print("ERROR: No centroids found. Check mask directory and filenames.")
        sys.exit(1)

    # ── DBSCAN clustering ─────────────────────────────────────────────────────
    print(f"\nRunning DBSCAN  (eps={args.eps}, min_samples={args.min_pts})...")
    xyz = np.array([c[0] for c in all_centroids])   # (M, 3)

    db = DBSCAN(eps=args.eps, min_samples=args.min_pts, n_jobs=-1).fit(xyz)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())

    print(f"  Clusters (apples): {n_clusters}")
    print(f"  Noise points:      {n_noise}  "
          f"({100 * n_noise / len(labels):.1f}% of centroids)")

    # ── Cluster size distribution ─────────────────────────────────────────────
    if n_clusters > 0:
        sizes = [int((labels == lbl).sum()) for lbl in range(n_clusters)]
        print(f"\n  Cluster size distribution:")
        print(f"    Min views per apple: {min(sizes)}")
        print(f"    Max views per apple: {max(sizes)}")
        print(f"    Mean views per apple: {np.mean(sizes):.1f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  APPLE COUNT: {n_clusters}")
    print(f"  (ground truth tree_02: 113)")
    print(f"  Error: {n_clusters - 113:+d}  ({100*(n_clusters-113)/113:+.1f}%)")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"apple_count: {n_clusters}\n")
            f.write(f"ground_truth: 113\n")
            f.write(f"error: {n_clusters - 113:+d}\n")
            f.write(f"noise_centroids: {n_noise}\n")
            f.write(f"total_centroids: {len(all_centroids)}\n")
            f.write(f"eps: {args.eps}\n")
            f.write(f"min_pts: {args.min_pts}\n")
            f.write(f"conf_min: {args.conf_min}\n")
        print(f"\nResults saved: {args.out}")

    if args.save_ply:
        ply_path = (os.path.splitext(args.out)[0] + "_centroids.ply"
                    if args.out else "centroids.ply")
        save_centroids_ply(xyz, labels, ply_path)


if __name__ == "__main__":
    main()
