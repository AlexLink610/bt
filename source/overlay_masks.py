"""
overlay_masks.py  –  SAM3 mask QA: show RGB through mask, black elsewhere

Output per image is a side-by-side:
    [ Original RGB ]  |  [ RGB through mask (black background) ]

This makes it immediately obvious whether SAM3 is correctly picking apples:
masked regions show the actual apple texture, everything else is black.

Usage:
    python overlay_masks.py

Edit CONFIG below, then run with:
    C:\\Users\\alex\\miniconda3\\envs\\sam3\\python.exe overlay_masks.py
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = r"C:\Users\alex\BA\data\FruitNeRF_Real\FruitNeRF_Dataset\tree_02\images"
MASK_DIR   = r"C:\Users\alex\BA\output_sam\tree_02\semantics_sam3_binary"
OUTPUT_DIR = r"C:\Users\alex\BA\output_sam\tree_02\overlay_qa"

# Max long-edge for thumbnails (keeps output files manageable)
THUMB_SIZE = 1000

# None = process all images. Set e.g. 20 for a quick test run.
MAX_IMAGES = None
# ──────────────────────────────────────────────────────────────────────────────


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def rgb_through_mask(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Where mask is white  → keep original RGB pixel
    Where mask is black  → paint black
    """
    rgb_np   = np.array(rgb.convert("RGB"),         dtype=np.uint8)
    mask_np  = np.array(mask.convert("L").resize(rgb.size, Image.NEAREST)) > 0

    result = np.zeros_like(rgb_np)          # all black
    result[mask_np] = rgb_np[mask_np]       # paste RGB where mask is active

    return Image.fromarray(result, "RGB")


def make_side_by_side(rgb: Image.Image, mask: Image.Image, fname: str) -> Image.Image:
    rgb_small  = resize_keep_aspect(rgb, THUMB_SIZE)
    W, H       = rgb_small.size
    mask_resized = mask.resize((W, H), Image.NEAREST)

    through    = rgb_through_mask(rgb_small, mask_resized)

    GAP    = 10
    HEADER = 24
    canvas = Image.new("RGB", (W * 2 + GAP, H + HEADER), (20, 20, 20))
    canvas.paste(rgb_small, (0,       HEADER))
    canvas.paste(through,   (W + GAP, HEADER))

    draw = ImageDraw.Draw(canvas)
    draw.text((4,            4), "Original RGB",        fill=(200, 200, 200))
    draw.text((W + GAP + 4,  4), "SAM3 mask (RGB fill)", fill=(200, 200, 200))
    draw.text((W * 2 + GAP - len(fname) * 6, 4), fname, fill=(120, 120, 120))

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_paths = []
    for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*.{ext}")))
    img_paths = sorted(img_paths)

    if MAX_IMAGES is not None:
        img_paths = img_paths[:MAX_IMAGES]

    print(f"Found {len(img_paths)} images in {IMAGE_DIR}")

    if len(img_paths) == 0:
        print("\nNo images found! Check IMAGE_DIR in the CONFIG block.")
        print(f"  IMAGE_DIR = {IMAGE_DIR}")
        # List what IS in the data folder to help debug
        parent = os.path.dirname(IMAGE_DIR)
        if os.path.exists(parent):
            print(f"\nContents of {parent}:")
            for entry in os.listdir(parent):
                print(f"  {entry}")
        return

    missing = 0
    for i, img_path in enumerate(img_paths):
        fname = os.path.basename(img_path)
        stem  = os.path.splitext(fname)[0]

        # Try same extension first, then .png fallback
        mask_path = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(MASK_DIR, stem + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(MASK_DIR, stem + ".jpg")

        if not os.path.exists(mask_path):
            print(f"  [{i+1}/{len(img_paths)}] MISSING mask for {fname}")
            missing += 1
            continue

        rgb  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        result   = make_side_by_side(rgb, mask, fname)
        out_path = os.path.join(OUTPUT_DIR, stem + "_qa.jpg")
        result.save(out_path, quality=88)

        if (i + 1) % 20 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1}/{len(img_paths)}] {fname}")

    print(f"\nDone. Saved to: {OUTPUT_DIR}")
    if missing:
        print(f"WARNING: {missing} images had no matching mask.")


if __name__ == "__main__":
    main()