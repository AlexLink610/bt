"""
time_sam3.py -- runtime of SAM3 (samgeo) on the tree_02 views used by VGGT.

Same calls as the segmentation run (prompt "apple", min_size 100).
Model loading is timed separately, masks are not saved.
"""

import os
import time

import torch
from samgeo import SamGeo3

IMAGE_DIRS = [
    r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\Data"
    r"\FruitNeRF_Real\FruitNeRF_Dataset\tree_02\images",
    r"C:\Users\alex\BA\real\tree_02\images",
]
FILENAMES = (r"C:\Users\alex\BA\real\tree_02\vggt"
             r"\T02_360_64v_fp32_naive_64v_filenames.txt")
PROMPT = "apple"
MIN_SIZE = 100


def segment(sam3, path):
    sam3.set_image_batch([path])
    try:
        sam3.generate_masks_batch(PROMPT, min_size=MIN_SIZE)
    except RuntimeError:
        # samgeo fails in its size filter (after inference) when an
        # image has no detection at all
        torch.cuda.synchronize()
        return 0
    torch.cuda.synchronize()
    return len(sam3.batch_results[0]["masks"])


def main():
    with open(FILENAMES) as f:
        names = [os.path.basename(l.strip()) for l in f if l.strip()]
    image_dir = next((d for d in IMAGE_DIRS
                      if os.path.exists(os.path.join(d, names[0]))), None)
    assert image_dir, f"{names[0]} not found in any of {IMAGE_DIRS}"
    paths = [os.path.join(image_dir, n) for n in names]
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[0]}"
    print(f"images from {image_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}   images: {len(paths)}")

    t = time.perf_counter()
    sam3 = SamGeo3(backend="meta", load_from_HF=True)
    torch.cuda.synchronize()
    print(f"model loading: {time.perf_counter() - t:.1f} s")

    segment(sam3, paths[0])  # warm-up, not timed
    times = []
    for i, p in enumerate(paths):
        t = time.perf_counter()
        n = segment(sam3, p)
        times.append(time.perf_counter() - t)
        print(f"[{i + 1}/{len(paths)}] {os.path.basename(p)}: "
              f"{times[-1]:.2f} s, {n} masks")
        torch.cuda.empty_cache()

    s = sorted(times)
    print(f"\nsegmentation of {len(times)} images: {sum(times):.1f} s "
          f"(mean {sum(times) / len(times):.2f} s, "
          f"median {s[len(s) // 2]:.2f} s per image)")


if __name__ == "__main__":
    main()
