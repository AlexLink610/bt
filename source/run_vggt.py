import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

MODEL_PATH = "/home/woody/iwi9/iwi9146h/vggt_weights"


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
    print(f"Saved {len(points)} colored points to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Path to folder containing images")
    parser.add_argument("--output", required=True, help="Base output path e.g. /path/to/tree_02 (num views appended automatically)")
    parser.add_argument("--num_views", type=int, default=None, help="Number of images to use (evenly sampled). Default: use all.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    # Load model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Get all image paths
    all_paths = []
    for ext in ["JPG", "jpg", "png", "PNG", "jpeg", "JPEG"]:
        all_paths.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    all_paths = sorted(all_paths)
    print(f"Found {len(all_paths)} images")

    if args.num_views is not None:
        indices = np.linspace(0, len(all_paths) - 1, args.num_views, dtype=int)
        image_paths = [all_paths[i] for i in indices]
        print(f"Selected {len(image_paths)} evenly spaced images")
    else:
        image_paths = all_paths

    if len(image_paths) == 0:
        print("No images found! Check --image_dir.")
        return

    # Build output path with view count
    num_views = len(image_paths)
    output = f"{args.output}_{num_views}"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"Output will be saved as: {output}.*")

    print(f"\nRunning VGGT on {num_views} images...")
    images = load_and_preprocess_images(image_paths).to(device)
    images = images[None]  # add batch dimension: (1, N, H, W, 3)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )

    point_map_np = np.array(point_map)  # (N, H, W, 3)
    N, H, W, _ = point_map_np.shape
    print(f"Point map shape: {point_map_np.shape}")

    # Sample RGB colors from original images at VGGT resolution
    print("Sampling RGB colors from images...")
    colors = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        colors[i] = np.array(img)

    # Save colored PLY (valid points only)
    points_flat = point_map_np.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    valid = ~np.isnan(points_flat).any(axis=1)
    save_ply(points_flat[valid], colors_flat[valid], f"{output}.ply")

    # Save filenames
    with open(f"{output}_filenames.txt", "w") as f:
        for p in image_paths:
            f.write(os.path.basename(p) + "\n")

    print(f"\nDone! Outputs saved to {output}.*")


if __name__ == "__main__":
    main()