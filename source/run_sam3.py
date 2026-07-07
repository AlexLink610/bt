"""
run_sam3_instances.py -- Generate per-instance apple masks with SAM3.

Each output mask is a single grayscale PNG named  mask_<stem>.png  where each
detected apple is a distinct integer value (1, 2, 3, ...), matching the format
expected by count_spheres.py / associate_masks_graph.py (get_instance_ids reads
np.unique(mask), ignoring 0 = background).

Usage (Windows, full python path since conda activate fails in PowerShell):
    C:\\Users\\alex\\miniconda3\\envs\\sam3\\python.exe ^
        C:\\Users\\alex\\BA\\source\\run_sam3_instances.py ^
        --image_dir  C:\\Users\\alex\\BA\\data\\test_rgb_5 ^
        --output_dir C:\\Users\\alex\\BA\\data\\test_rgb_5\\semantics_sam3

Optional:
    --prompt     text prompt for SAM3 (default: "apple")
    --min_size   min mask size in pixels (default: 100)
    --ext        image extension(s), comma-separated (default: png,jpg,JPG,jpeg,PNG)
    --save_binary_dir  also save merged binary masks (all apples = 255) to this dir
"""

import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from samgeo import SamGeo3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  required=True,
                        help="Folder with input images.")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to write mask_<stem>.png instance masks.")
    parser.add_argument("--prompt",     default="apple",
                        help='SAM3 text prompt (default: "apple").')
    parser.add_argument("--min_size",   type=int, default=100,
                        help="Minimum mask size in pixels (default: 100).")
    parser.add_argument("--ext",        default="png,jpg,JPG,jpeg,PNG",
                        help="Comma-separated image extensions to include.")
    parser.add_argument("--save_binary_dir", default=None,
                        help="Optional: also save merged binary masks (all apples=255) here.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_binary_dir:
        os.makedirs(args.save_binary_dir, exist_ok=True)

    # Collect images across the requested extensions, de-duplicated and sorted
    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(args.image_dir, f"*.{e}")))
    image_paths = sorted(set(paths))
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    if not image_paths:
        print("No images found -- check --image_dir and --ext.")
        return

    sam3 = SamGeo3(backend="meta", load_from_HF=True)  # init model

    total_counts = {}
    for i, image_path in enumerate(image_paths):
        fname = os.path.basename(image_path)
        stem  = os.path.splitext(fname)[0]
        print(f"[{i+1}/{len(image_paths)}] Processing {fname}...")
        try:
            sam3.set_image_batch([image_path])
            sam3.generate_masks_batch(args.prompt, min_size=args.min_size)
            masks = sam3.batch_results[0]['masks']
            count = len(masks)
            total_counts[fname] = count
            print(f"  -> {count} {args.prompt}(s)")

            # Determine mask shape (from first mask, or from image if none)
            if count > 0:
                shape = np.array(masks[0]).squeeze().shape
            else:
                shape = np.array(Image.open(image_path).convert("L")).shape

            # Instance mask: each detection gets a distinct integer id
            instance = np.zeros(shape, dtype=np.uint8)
            for idx, mask in enumerate(masks, start=1):
                if idx > 255:
                    print(f"  WARNING: >255 instances, id {idx} clipped in uint8")
                instance[np.array(mask).squeeze().astype(bool)] = idx
            Image.fromarray(instance).save(
                os.path.join(args.output_dir, f"mask_{stem}.png"))

            # Optional binary mask
            if args.save_binary_dir:
                binary = (instance > 0).astype(np.uint8) * 255
                Image.fromarray(binary).save(
                    os.path.join(args.save_binary_dir, f"mask_{stem}.png"))

        except Exception as e:
            print(f"  Error: {e}")
            total_counts[fname] = -1
        torch.cuda.empty_cache()

    print(f"\nDone! Instance masks saved to {args.output_dir}")
    print(f"Per-image instance counts: {total_counts}")
    valid = [c for c in total_counts.values() if c >= 0]
    if valid:
        print(f"Mean instances/image: {np.mean(valid):.1f}  "
              f"(min {min(valid)}, max {max(valid)})")


if __name__ == "__main__":
    main()
