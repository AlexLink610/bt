import os
import numpy as np
from PIL import Image

POINT_MAPS  = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_vggt\tree_02\point_maps.npy"
FILENAMES   = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_vggt\tree_02\filenames.txt"
MASK_DIR    = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_sam\tree_02\semantics_sam3"
OUTPUT      = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_vggt\tree_02\apple_pointcloud.npy"

def main():
    # Load point maps
    print("Loading point maps...")
    point_maps = np.load(POINT_MAPS)
    N, H, W, _ = point_maps.shape
    print(f"Point map shape: {point_maps.shape}")

    # Load filenames
    with open(FILENAMES, "r") as f:
        filenames = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(filenames)} filenames")

    assert len(filenames) == N, f"Mismatch: {N} point maps but {len(filenames)} filenames"

    apple_points = []

    for i, fname in enumerate(filenames):
        #mask_path = os.path.join(MASK_DIR, fname)
        base = os.path.splitext(fname)[0]  # "frame_00001"
        mask_path = os.path.join(MASK_DIR, mask_fname)
        mask_fname = f"mask_{base}.png"

        if not os.path.exists(mask_path):
            print(f"  [{i+1}/{N}] Mask not found for {fname}, skipping")
            continue

        # Load binary mask and resize to match point map resolution
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((W, H), Image.NEAREST)
        mask_np = np.array(mask) > 0  # True where apple

        # Get apple points from point map
        apple_pts = point_maps[i][mask_np]  # shape: (K, 3)
        apple_points.append(apple_pts)
        print(f"  [{i+1}/{N}] {fname}: {apple_pts.shape[0]} apple points")

    if len(apple_points) == 0:
        print("No apple points found! Check mask paths.")
        return

    # Combine all apple points
    all_apple_points = np.concatenate(apple_points, axis=0)
    print(f"\nTotal apple points: {all_apple_points.shape[0]}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.save(OUTPUT, all_apple_points)
    print(f"Saved apple point cloud to {OUTPUT}")


if __name__ == "__main__":
    main()