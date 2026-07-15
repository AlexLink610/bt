"""
visualize_instance_masks.py  --  Save colorized versions of instance masks.

Each unique instance ID gets a vibrant distinct color.
Background (0) is black.
Instance IDs are drawn as numbered circles at the centroid of each instance.

Usage:
    python3 visualize_instance_masks.py \
        --masks_dir ~/ba/output_sam/old_room/semantics_sam3 \
        --out_dir   ~/ba/output_sam/old_room/semantics_sam3_vis
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_colors(n):
    """Golden ratio hue stepping with alternating brightness."""
    colors = []
    golden = 0.618033988749895
    h = 0.0
    for i in range(n):
        h = (h + golden) % 1.0
        v = 1.0 if i % 2 == 0 else 0.7
        hi = int(h * 6)
        f = h * 6 - hi
        p = 0.0
        q = v * (1 - f)
        t = v * f
        hi = hi % 6
        if hi == 0: r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        else:         r, g, b = v, p, q
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors


def colorize_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    ids = [i for i in np.unique(mask) if i != 0]
    colors = make_colors(len(ids))
    id_to_color = {iid: colors[k] for k, iid in enumerate(ids)}

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for iid, color in id_to_color.items():
        rgb[mask == iid] = color

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # Font size relative to image height
    font_size = max(16, mask.shape[0] // 25)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for iid in ids:
        ys, xs = np.where(mask == iid)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(iid)

        # Black filled circle background
        r = font_size // 2 + 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0))

        # White text centered on circle
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), label,
                  fill=(255, 255, 255), font=font)

    return img, len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--out_dir",   required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(args.masks_dir, "*.png")))
    print(f"Found {len(mask_paths)} masks in {args.masks_dir}")

    for path in mask_paths:
        fname = os.path.basename(path)
        out_path = os.path.join(args.out_dir, fname)
        img, count = colorize_mask(path)
        img.save(out_path)
        print(f"  {fname}  →  {count} instances  →  {out_path}")

    print(f"\nDone! Visualizations saved to {args.out_dir}")


if __name__ == "__main__":
    main()
