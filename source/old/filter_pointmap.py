import os
import sys
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MASK_DIR  = "/home/alex/ba/output_sam/tree_02/semantics_sam3_binary"
IMAGE_DIR = "/home/alex/ba/data/FruitNeRF_Real/FruitNeRF_Dataset/tree_02/images"
# ──────────────────────────────────────────────────────────────────────────────


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
    print(f"Saved {len(points)} apple points -> {path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python filter_pointmap.py <input.ply> <filenames.txt>")
        sys.exit(1)

    ply_path       = sys.argv[1]
    filenames_path = sys.argv[2]
    base           = os.path.splitext(ply_path)[0]
    output_path    = f"{base}_apples.ply"
    pointmap_path  = f"{base}_pointmap.npy"
    confmap_path   = f"{base}_confmap.npy"

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not os.path.exists(pointmap_path):
        print(f"ERROR: pointmap not found: {pointmap_path}")
        print("Re-run run_vggt.py to regenerate outputs with the new format.")
        sys.exit(1)

    # ── Load pointmap ─────────────────────────────────────────────────────────
    print(f"Loading point map from {pointmap_path} ...")
    point_map = np.load(pointmap_path)          # (N, H, W, 3) float32, NaNs present
    N, H, W, _ = point_map.shape
    print(f"  Point map shape: {point_map.shape}")

    # ── Load confmap (optional) ───────────────────────────────────────────────
    has_confmap = os.path.exists(confmap_path)
    if has_confmap:
        conf_map = np.load(confmap_path)        # (N, H, W) float32 0-1
        print(f"  Conf map loaded")
    else:
        conf_map = None
        print(f"  No confmap found - quality channel will be absent in output")

    # ── Load filenames ────────────────────────────────────────────────────────
    with open(filenames_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
    print(f"  {len(filenames)} views")

    if len(filenames) != N:
        print(f"ERROR: filenames.txt has {len(filenames)} entries but pointmap has N={N}")
        sys.exit(1)

    # ── Sample RGB colors from original images ────────────────────────────────
    print("Sampling RGB colors from original images...")
    colors_map = np.zeros((N, H, W, 3), dtype=np.uint8)
    missing_images = 0
    for i, fname in enumerate(filenames):
        img_path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(img_path):
            missing_images += 1
            colors_map[i] = 128
            continue
        img = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        colors_map[i] = np.array(img)

    if missing_images:
        print(f"  WARNING: {missing_images}/{N} images not found in IMAGE_DIR")

    # ── Apply SAM masks ───────────────────────────────────────────────────────
    apple_points  = []
    apple_colors  = []
    apple_quality = []

    for i, fname in enumerate(filenames):
        # Try both .JPG and .png extensions for mask
        stem      = os.path.splitext(fname)[0]
        mask_path = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(MASK_DIR, stem + ".png")
        if not os.path.exists(mask_path):
            print(f"  [{i+1}/{N}] MISSING mask for {fname}, skipping")
            continue

        mask    = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        mask_np = np.array(mask) > 0          # (H, W) bool

        pts  = point_map[i][mask_np]          # (K, 3)
        cols = colors_map[i][mask_np]         # (K, 3)

        # Remove NaN points
        valid = ~np.isnan(pts).any(axis=1)
        pts   = pts[valid]
        cols  = cols[valid]

        apple_points.append(pts)
        apple_colors.append(cols)

        if conf_map is not None:
            qual = conf_map[i][mask_np][valid]
            apple_quality.append(qual)

        print(f"  [{i+1}/{N}] {fname}: {len(pts):,} apple points")

    if not apple_points:
        print("No apple points found! Check MASK_DIR and filenames.")
        sys.exit(1)

    all_points  = np.concatenate(apple_points,  axis=0)
    all_colors  = np.concatenate(apple_colors,  axis=0)
    all_quality = np.concatenate(apple_quality, axis=0) if apple_quality else None

    print(f"\nTotal apple points: {len(all_points):,}")
    save_ply(all_points, all_colors, all_quality, output_path)
    print(f"Done! -> {output_path}")


if __name__ == "__main__":
    main()
