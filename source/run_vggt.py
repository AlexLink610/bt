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


def save_ply(points, colors, confidence, path):
    """
    Save Nx3 points, Nx3 uint8 colors, and N float32 confidence values
    as a binary PLY file. Confidence is stored as the 'quality' property
    which MeshLab can visualise natively via Render > Color > Per Vertex Quality.
    """
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
        "property float quality\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p, c, q in zip(points.astype(np.float32),
                           colors.astype(np.uint8),
                           confidence.astype(np.float32)):
            f.write(p.tobytes())
            f.write(c.tobytes())
            f.write(q.tobytes())
    print(f"Saved {len(points)} points to {path}")


def build_tag(args, num_views):
    """
    Build a short output tag.
    Examples:
        --image_list selected_16views_180deg.txt  ->  180_16v
        --image_list selected_32views_360deg.txt  ->  360_32v
        --num_views 64                            ->  naive_64v
        (all views)                               ->  naive_366v
    """
    if args.image_list is not None:
        name  = os.path.splitext(os.path.basename(args.image_list))[0]
        parts = name.split("_")  # ['selected', 'Nviews', 'ARCdeg']
        try:
            arc = parts[2].replace("deg", "")   # e.g. '180'
            return f"{arc}_{num_views}v"
        except IndexError:
            return name
    else:
        return f"naive_{num_views}v"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  required=True,
                        help="Path to folder containing images")
    parser.add_argument("--output",     required=True,
                        help="Base output path e.g. /path/to/t02 (tag appended automatically)")
    parser.add_argument("--num_views",  type=int, default=None,
                        help="Number of images to use (evenly sampled by filename order). "
                             "Ignored if --image_list is set.")
    parser.add_argument("--image_list", default=None,
                        help="Path to a .txt file with one image filename per line. "
                             "Overrides --num_views.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (torch.bfloat16
               if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
               else torch.float16)
    print(f"Using device: {device}, dtype: {dtype}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading VGGT model...")
    model = VGGT.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Collect all image paths ───────────────────────────────────────────────
    all_paths = []
    for ext in ["JPG", "jpg", "png", "PNG", "jpeg", "JPEG"]:
        all_paths.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    all_paths = sorted(all_paths)
    print(f"Found {len(all_paths)} images in {args.image_dir}")

    # ── Select images ─────────────────────────────────────────────────────────
    if args.image_list is not None:
        with open(args.image_list, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
        name_to_path = {os.path.basename(p): p for p in all_paths}
        image_paths = []
        for fname in filenames:
            if fname in name_to_path:
                image_paths.append(name_to_path[fname])
            else:
                print(f"  WARNING: {fname} not found in image_dir, skipping")
        print(f"Loaded {len(image_paths)} images from list: {args.image_list}")
    elif args.num_views is not None:
        indices     = np.linspace(0, len(all_paths) - 1, args.num_views, dtype=int)
        image_paths = [all_paths[i] for i in indices]
        print(f"Selected {len(image_paths)} evenly spaced images (filename order)")
    else:
        image_paths = all_paths

    if len(image_paths) == 0:
        print("No images selected! Check --image_dir / --image_list.")
        return

    # ── Output path ───────────────────────────────────────────────────────────
    num_views = len(image_paths)
    tag       = build_tag(args, num_views)
    output    = f"{args.output}_{tag}"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    print(f"Output: {output}.*   ({num_views} views)")

    # ── Run VGGT ──────────────────────────────────────────────────────────────
    print(f"\nRunning VGGT on {num_views} images...")
    images = load_and_preprocess_images(image_paths).to(device)
    images = images[None]   # (1, N, H, W, 3)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )

    point_map_np  = np.array(point_map)                    # (N, H, W, 3)
    depth_conf_np = np.array(depth_conf.squeeze(0).cpu())  # (N, H, W)
    N, H, W, _    = point_map_np.shape

    print(f"Point map shape:   {point_map_np.shape}")
    print(f"Conf range:        {depth_conf_np.min():.4f} – {depth_conf_np.max():.4f}")
    print(f"Conf mean/median:  {depth_conf_np.mean():.4f} / {np.median(depth_conf_np):.4f}")

    # ── Sample RGB colors ─────────────────────────────────────────────────────
    print("Sampling RGB colors from images...")
    colors = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i, img_path in enumerate(image_paths):
        img       = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        colors[i] = np.array(img)

    # ── Normalise confidence to 0-1 ───────────────────────────────────────────
    conf_flat       = depth_conf_np.reshape(-1)
    conf_min, conf_max = conf_flat.min(), conf_flat.max()
    conf_norm       = (conf_flat - conf_min) / (conf_max - conf_min + 1e-8)

    # ── Filter NaN points ─────────────────────────────────────────────────────
    points_flat = point_map_np.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    valid       = ~np.isnan(points_flat).any(axis=1)
    print(f"Valid points: {valid.sum():,} / {len(valid):,}")

    # ── Save shape txt (N H W) for filter_pointmap.py ─────────────────────────
    with open(f"{output}_shape.txt", "w") as f:
        f.write(f"{N} {H} {W}\n")
    print(f"Shape saved: {output}_shape.txt  ({N} {H} {W})")

    # ── Save valid mask (N, H, W) for filter_pointmap.py ─────────────────────
    valid_map = ~np.isnan(point_map_np).any(axis=-1)  # (N, H, W) bool
    np.save(f"{output}_valid.npy", valid_map)
    print(f"Valid mask saved: {output}_valid.npy")

    # ── Save PLY (with confidence as quality channel) ─────────────────────────
    save_ply(points_flat[valid], colors_flat[valid], conf_norm[valid], f"{output}.ply")

    # ── Save filenames ────────────────────────────────────────────────────────
    with open(f"{output}_filenames.txt", "w") as f:
        for p in image_paths:
            f.write(os.path.basename(p) + "\n")

    print(f"\nDone! Outputs: {output}.ply / _filenames.txt / _shape.txt / _valid.npy")


if __name__ == "__main__":
    main()
