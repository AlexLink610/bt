import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os

POINT_MAPS = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_vggt\tree_02\point_maps.npy"
FILENAMES  = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_vggt\tree_02\filenames.txt"
IMAGE_DIR  = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\Data\FruitNeRF_Real\FruitNeRF_Dataset\tree_02\images"

print("Loading point maps...")
point_maps = np.load(POINT_MAPS)
N, H, W, _ = point_maps.shape

with open(FILENAMES, "r") as f:
    filenames = [line.strip() for line in f.readlines()]

all_points = []
all_colors = []

for i, fname in enumerate(filenames):
    img_path = os.path.join(IMAGE_DIR, fname)
    img = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
    img_np = np.array(img)  # (H, W, 3)

    pts = point_maps[i].reshape(-1, 3)
    colors = img_np.reshape(-1, 3)

    all_points.append(pts)
    all_colors.append(colors)

points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)

# Subsample
max_points = 300000
idx = np.random.choice(len(points), size=min(max_points, len(points)), replace=False)
pts = points[idx]
cols = colors[idx]

# Convert RGB to hex strings for plotly
hex_colors = [f'rgb({r},{g},{b})' for r, g, b in cols]

fig = go.Figure(data=[go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode='markers',
    marker=dict(size=1, color=hex_colors, opacity=0.8)
)])

fig.update_layout(title="VGGT Point Cloud - tree_02 (RGB)", scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
))

fig.show()