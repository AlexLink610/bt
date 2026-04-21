import os
import glob
import argparse
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

MODEL_PATH = "/home/woody/iwi9/iwi9146h/vggt_weights"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Path to folder containing images")
    parser.add_argument("--output_dir", required=True, help="Path to save outputs")
    parser.add_argument("--subsample", type=int, default=1, help="Use every Nth image")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    image_paths = []
    for ext in ["JPG", "jpg", "png", "PNG", "jpeg", "JPEG"]:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")

    if args.subsample > 1:
        image_paths = image_paths[::args.subsample]
        print(f"Subsampled to every {args.subsample}th image: {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found! Check --image_dir.")
        return

    print(f"\nRunning VGGT on {len(image_paths)} images...")
    images = load_and_preprocess_images(image_paths).to(device)
    images = images[None]  # add batch dimension: (1, N, H, W, 3)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            # Construct accurate 3D points from depth + cameras
            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )

    # Save point maps: shape (N, H, W, 3)
    #point_map_np = point_map.cpu().numpy()
    np.save(os.path.join(args.output_dir, "point_maps.npy"), point_map)
    print(f"Point map shape: {point_map.shape}")

    # Save filenames for reference
    with open(os.path.join(args.output_dir, "filenames.txt"), "w") as f:
        for p in image_paths:
            f.write(os.path.basename(p) + "\n")

    print(f"\nDone! Saved point_maps.npy to {args.output_dir}")


if __name__ == "__main__":
    main()