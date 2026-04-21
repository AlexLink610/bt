import os
import glob
import torch
import numpy as np
from PIL import Image
from samgeo import SamGeo3
# Config
IMAGE_DIR = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\Data\FruitNeRF_Real\FruitNeRF_Dataset\tree_02\images"
BINARY_DIR = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\Data\FruitNeRF_Real\FruitNeRF_Dataset\tree_02\semantics_sam3_binary"

os.makedirs(BINARY_DIR, exist_ok=True)
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPG")))
print(f"Found {len(image_paths)} images")

sam3 = SamGeo3(backend="meta", load_from_HF=True) #init model

total_counts = {}
for i, image_path in enumerate(image_paths):
    fname = os.path.basename(image_path)
    print(f"[{i+1}/{len(image_paths)}] Processing {fname}...")
    try:
        sam3.set_image_batch([image_path])
        sam3.generate_masks_batch("apple", min_size=100)
        masks = sam3.batch_results[0]['masks']
        count = len(masks)
        total_counts[fname] = count
        print(f"  -> {count} apples")
        if count > 0:
            first_mask = np.array(masks[0]).squeeze()
            binary = np.zeros(first_mask.shape, dtype=np.uint8)
            for mask in masks:
                binary[np.array(mask).squeeze().astype(bool)] = 255
            Image.fromarray(binary).save(os.path.join(BINARY_DIR, fname))
    except Exception as e:
        print(f"  Error: {e}")
        total_counts[fname] = -1
    torch.cuda.empty_cache()
print(f"\nDone! Binary masks saved to {BINARY_DIR}")