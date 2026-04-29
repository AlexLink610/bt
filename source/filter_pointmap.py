import os
import sys
import numpy as np
from PIL import Image
import struct

# ── Config ────────────────────────────────────────────────────────────────────
MASK_DIR = r"C:\Users\alex\BA\output_sam\tree_02\semantics_sam3_binary"
# ──────────────────────────────────────────────────────────────────────────────


def read_ply(path):
    """Read binary PLY file, returns (points Nx3, colors Nx3 or None)."""
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_points = 0
        has_color = False
        for line in header_lines:
            if line.startswith("element vertex"):
                n_points = int(line.split()[-1])
            if "red" in line:
                has_color = True

        # Read binary data
        if has_color:
            dtype = np.dtype([
                ("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("r", np.uint8), ("g", np.uint8), ("b", np.uint8)
            ])
            data = np.frombuffer(f.read(n_points * dtype.itemsize), dtype=dtype)
            points = np.stack([data["x"], data["y"], data["z"]], axis=1)
            colors = np.stack([data["r"], data["g"], data["b"]], axis=1)
        else:
            dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
            data = np.frombuffer(f.read(n_points * dtype.itemsize), dtype=dtype)
            points = np.stack([data["x"], data["y"], data["z"]], axis=1)
            colors = None

    return points, colors


def save_ply(points, colors, path):
    """Save Nx3 points and Nx3 uint8 colors as binary PLY file."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(points.astype(np.float32), colors.astype(np.uint8)):
            f.write(p.tobytes())
            f.write(c.tobytes())
    print(f"Saved {len(points)} apple points to {path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python filter_pointmap.py <input.ply> <filenames.txt>")
        sys.exit(1)

    ply_path = sys.argv[1]
    filenames_path = sys.argv[2]

    # Derive output path: tree_02_122.ply -> tree_02_122_apples.ply
    base = os.path.splitext(ply_path)[0]
    output_path = f"{base}_apples.ply"

    # Load PLY
    print(f"Loading point cloud from {ply_path}...")
    points, colors = read_ply(ply_path)
    print(f"Loaded {len(points)} points")

    # Load filenames
    with open(filenames_path, "r") as f:
        filenames = [line.strip() for line in f.readlines()]
    N = len(filenames)
    print(f"Loaded {N} filenames")

    # Reconstruct shape (N, H, W) from flat points
    # Points are stored as N*H*W rows — we need to know H and W
    # They are saved per-image sequentially, total = N * H * W
    total = len(points)
    # Read H and W from first image mask to determine resolution
    first_mask_path = os.path.join(MASK_DIR, filenames[0])
    if not os.path.exists(first_mask_path):
        print(f"Mask not found: {first_mask_path}")
        sys.exit(1)
    first_mask = Image.open(first_mask_path)
    orig_W, orig_H = first_mask.size
    pixels_per_image = total // N
    H, W = 350, 518
    print(f"Using image resolution: {H}x{W}")

    point_map = points.reshape(N, H, W, 3)
    color_map = colors.reshape(N, H, W, 3) if colors is not None else None

    apple_points = []
    apple_colors = []

    for i, fname in enumerate(filenames):
        mask_path = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mask_path):
            print(f"  [{i+1}/{N}] Mask not found for {fname}, skipping")
            continue

        mask = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        mask_np = np.array(mask) > 0

        pts = point_map[i][mask_np]
        # Remove NaN
        valid = ~np.isnan(pts).any(axis=1)
        pts = pts[valid]
        apple_points.append(pts)

        if color_map is not None:
            cols = color_map[i][mask_np][valid]
            apple_colors.append(cols)

        print(f"  [{i+1}/{N}] {fname}: {len(pts)} apple points")

    if len(apple_points) == 0:
        print("No apple points found!")
        sys.exit(1)

    all_points = np.concatenate(apple_points, axis=0)
    all_colors = np.concatenate(apple_colors, axis=0) if apple_colors else np.full((len(all_points), 3), [255, 0, 0], dtype=np.uint8)
    print(f"\nTotal apple points: {len(all_points)}")
    save_ply(all_points, all_colors, output_path)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()