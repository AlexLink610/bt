import os
import sys
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MASK_DIR = r"C:\Users\alex\BA\output_sam\tree_02\semantics_sam3_binary"
# ──────────────────────────────────────────────────────────────────────────────


def read_ply(path):
    """Read binary PLY file, returns (points Nx3, colors Nx3, quality N or None)."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_points  = 0
        has_color = False
        has_qual  = False
        for line in header_lines:
            if line.startswith("element vertex"):
                n_points = int(line.split()[-1])
            if "red"     in line: has_color = True
            if "quality" in line: has_qual  = True

        if has_color and has_qual:
            dtype = np.dtype([
                ("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
                ("quality", np.float32)
            ])
        elif has_color:
            dtype = np.dtype([
                ("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8)
            ])
        else:
            dtype = np.dtype([
                ("x", np.float32), ("y", np.float32), ("z", np.float32)
            ])

        data   = np.frombuffer(f.read(n_points * dtype.itemsize), dtype=dtype)
        points = np.stack([data["x"], data["y"], data["z"]], axis=1)
        colors = np.stack([data["r"], data["g"], data["b"]], axis=1) if has_color else None
        qual   = data["quality"].copy() if has_qual else None

    return points, colors, qual


def save_ply(points, colors, quality, path):
    """Save Nx3 points, Nx3 uint8 colors, optional N float32 quality."""
    has_qual = quality is not None
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
        + ("property float quality\n" if has_qual else "") +
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if has_qual:
            for p, c, q in zip(points.astype(np.float32),
                               colors.astype(np.uint8),
                               quality.astype(np.float32)):
                f.write(p.tobytes())
                f.write(c.tobytes())
                f.write(q.tobytes())
        else:
            for p, c in zip(points.astype(np.float32), colors.astype(np.uint8)):
                f.write(p.tobytes())
                f.write(c.tobytes())
    print(f"Saved {len(points)} apple points to {path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python filter_pointmap.py <input.ply> <filenames.txt>")
        sys.exit(1)

    ply_path       = sys.argv[1]
    filenames_path = sys.argv[2]
    base           = os.path.splitext(ply_path)[0]
    output_path    = f"{base}_apples.ply"
    shape_path     = f"{base}_shape.txt"
    valid_path     = f"{base}_valid.npy"

    # ── Load PLY ──────────────────────────────────────────────────────────────
    print(f"Loading point cloud from {ply_path}...")
    points, colors, quality = read_ply(ply_path)
    print(f"  {len(points):,} points  |  quality channel: {quality is not None}")

    # ── Load filenames ────────────────────────────────────────────────────────
    with open(filenames_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
    N = len(filenames)
    print(f"  {N} views")

    # ── Load shape ────────────────────────────────────────────────────────────
    if not os.path.exists(shape_path):
        print(f"ERROR: shape file not found: {shape_path}")
        print("Re-run run_vggt.py to regenerate outputs with the new format.")
        sys.exit(1)
    with open(shape_path) as f:
        N_file, H, W = map(int, f.read().split())
    assert N_file == N, f"Shape file says {N_file} views but filenames.txt has {N}"
    print(f"  Image resolution: {H}x{W}")

    # ── Reconstruct full (N, H, W) grid from valid mask ───────────────────────
    if not os.path.exists(valid_path):
        print(f"ERROR: valid mask not found: {valid_path}")
        print("Re-run run_vggt.py to regenerate outputs with the new format.")
        sys.exit(1)

    valid_mask = np.load(valid_path)  # (N, H, W) bool
    assert valid_mask.shape == (N, H, W), \
        f"Valid mask shape {valid_mask.shape} doesn't match ({N},{H},{W})"
    flat_valid = valid_mask.reshape(-1)

    # Scatter PLY points back into full N*H*W grid
    full_points = np.full((N * H * W, 3), np.nan, dtype=np.float32)
    full_colors = np.zeros((N * H * W, 3),         dtype=np.uint8)
    full_points[flat_valid] = points
    full_colors[flat_valid] = colors if colors is not None else 0

    full_qual = None
    if quality is not None:
        full_qual = np.zeros(N * H * W, dtype=np.float32)
        full_qual[flat_valid] = quality

    point_map = full_points.reshape(N, H, W, 3)
    color_map = full_colors.reshape(N, H, W, 3)
    qual_map  = full_qual.reshape(N, H, W) if full_qual is not None else None

    # ── Apply SAM3 masks per frame ────────────────────────────────────────────
    apple_points  = []
    apple_colors  = []
    apple_quality = []

    for i, fname in enumerate(filenames):
        mask_path = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mask_path):
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(MASK_DIR, stem + ".png")
        if not os.path.exists(mask_path):
            print(f"  [{i+1}/{N}] MISSING mask for {fname}, skipping")
            continue

        mask    = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        mask_np = np.array(mask) > 0

        pts  = point_map[i][mask_np]
        cols = color_map[i][mask_np]
        ql   = qual_map[i][mask_np] if qual_map is not None else None

        # Remove NaN points
        valid = ~np.isnan(pts).any(axis=1)
        pts   = pts[valid]
        cols  = cols[valid]
        if ql is not None: ql = ql[valid]

        apple_points.append(pts)
        apple_colors.append(cols)
        if ql is not None: apple_quality.append(ql)

        print(f"  [{i+1}/{N}] {fname}: {len(pts):,} apple points")

    if not apple_points:
        print("No apple points found!")
        sys.exit(1)

    all_points  = np.concatenate(apple_points,  axis=0)
    all_colors  = np.concatenate(apple_colors,  axis=0)
    all_quality = np.concatenate(apple_quality, axis=0) if apple_quality else None

    print(f"\nTotal apple points: {len(all_points):,}")
    save_ply(all_points, all_colors, all_quality, output_path)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()
