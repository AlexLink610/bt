import os
import glob
import torch
import numpy as np
from PIL import Image
from samgeo import SamGeo3

IMAGE_DIR  = r"C:\Users\alex\BA\data\table"
OUTPUT_DIR = r"C:\Users\alex\BA\data\table\semantics_sam3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPG")))
print(f"Found {len(image_paths)} images")

sam3 = SamGeo3(backend="meta", load_from_HF=True)

for i, image_path in enumerate(image_paths):
    fname = os.path.basename(image_path)
    print(f"[{i+1}/{len(image_paths)}] Processing {fname}...")
    try:
        sam3.set_image_batch([image_path])
        sam3.generate_masks_batch("apple", min_size=100)
        masks = sam3.batch_results[0]['masks']
        count = len(masks)
        print(f"  -> {count} apples")
        if count > 0:
            first_mask = np.array(masks[0]).squeeze()
            instance_mask = np.zeros(first_mask.shape, dtype=np.uint8)
            for idx, mask in enumerate(masks, start=1):
                m = np.array(mask).squeeze().astype(bool)
                instance_mask[m] = idx
            stem = os.path.splitext(fname)[0]
            save_path = os.path.join(OUTPUT_DIR, f"mask_{stem}.png")
            Image.fromarray(instance_mask).save(save_path)
            print(f"  -> Saved: {save_path}")
        else:
            print(f"  -> No apples found, no mask saved")
    except Exception as e:
        print(f"  Error: {e}")
    torch.cuda.empty_cache()

print(f"\nDone! Instance masks saved to {OUTPUT_DIR}")
