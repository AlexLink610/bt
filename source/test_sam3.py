"""
overlay_masks.py  –  Overlay SAM3 binary masks on RGB images for visual QA

Produces side-by-side images: original | mask | overlay
so you can quickly spot where SAM3 over- or under-segments apples.

Usage:
    python overlay_masks.py

Edit the CONFIG block below to point at your paths.
Output images are saved to OUTPUT_DIR.
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = r"C:\Users\alex\BA\data\FruitNeRF_Real\FruitNeRF_Dataset\tree_02\images"
MASK_DIR   = r"C:\Users\alex\BA\output_sam\tree_02\semantics_sam3_binary"
OUTPUT_DIR = r"C:\Users\alex\BA\output_sam\tree_02\overlay_qa"

# Overlay colour for masked pixels (R, G, B, A) — vivid red at 50% opacity
OVERLAY_COLOR = (255, 30, 30, 128)

# Max long-edge size for the output thumbnails (keeps files small)
THUMB_SIZE = 1000

# How many images to process. None = all. Set to e.g. 50 for a quick sample.
MAX_IMAGES = None
# ──────────────────────────────────────────────────────────────────────────────


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def make_overlay(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    """Blend a red overlay onto rgb wherever mask is white (>0)."""
    rgb_rgba = rgb.convert("RGBA")
    mask_gray = mask.convert("L").resize(rgb.size, Image.NEAREST)
    mask_np = np.array(mask_gray) > 0

    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    overlay_np = np.array(overlay)
    overlay_np[mask_np] = OVERLAY_COLOR
    overlay = Image.fromarray(overlay_np, "RGBA")

    blended = Image.alpha_composite(rgb_rgba, overlay)
    return blended.convert("RGB")


def make_mask_visual(mask: Image.Image, target_size) -> Image.Image:
    """Return mask resized to target_size as an RGB image."""
    return mask.convert("L").resize(target_size, Image.NEAREST).convert("RGB")


def make_side_by_side(rgb: Image.Image, mask: Image.Image, fname: str) -> Image.Image:
    rgb_small   = resize_keep_aspect(rgb,  THUMB_SIZE)
    W, H        = rgb_small.size
    mask_small  = mask.convert("L").resize((W, H), Image.NEAREST).convert("RGB")
    overlay_img = make_overlay(rgb_small, mask.resize((W, H), Image.NEAREST))

    # Label each panel
    panel_w = W * 3 + 20   # 10px gap between panels
    canvas  = Image.new("RGB", (panel_w, H + 30), (30, 30, 30))

    canvas.paste(rgb_small,   (0,        30))
    canvas.paste(mask_small,  (W + 10,   30))
    canvas.paste(overlay_img, (W*2 + 20, 30))

    draw = ImageDraw.Draw(canvas)
    draw.text((5,            5), "RGB",     fill=(200, 200, 200))
    draw.text((W + 15,       5), "SAM mask",fill=(200, 200, 200))
    draw.text((W*2 + 25,     5), "Overlay", fill=(200, 200, 200))

    # filename in top-right
    draw.text((panel_w - 300, 5), fname, fill=(160, 160, 160))

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect image paths
    img_paths = []
    for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*.{ext}")))
    img_paths = sorted(img_paths)

    if MAX_IMAGES is not None:
        img_paths = img_paths[:MAX_IMAGES]

    print(f"Found {len(img_paths)} images in {IMAGE_DIR}")
    missing_masks = 0

    for i, img_path in enumerate(img_paths):
        fname = os.path.basename(img_path)
        mask_path = os.path.join(MASK_DIR, fname)

        # SAM3 masks might be saved as .png even if source is .jpg
        if not os.path.exists(mask_path):
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(MASK_DIR, stem + ".png")

        if not os.path.exists(mask_path):
            print(f"  [{i+1}/{len(img_paths)}] MISSING mask for {fname}")
            missing_masks += 1
            continue

        rgb  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        result = make_side_by_side(rgb, mask, fname)

        # Save as jpg for smaller output
        out_name = os.path.splitext(fname)[0] + "_overlay.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        result.save(out_path, quality=88)

        if (i + 1) % 10 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1}/{len(img_paths)}] {fname}")

    print(f"\nDone. Overlays saved to: {OUTPUT_DIR}")
    if missing_masks:
        print(f"WARNING: {missing_masks} images had no matching mask.")


if __name__ == "__main__":
    main()