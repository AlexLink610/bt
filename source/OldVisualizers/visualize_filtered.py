import numpy as np
import plotly.graph_objects as go

APPLE_POINTCLOUD = r"C:\Users\alex\OneDrive\Documents\Uni\SS26\BA\output_vggt\tree_02\apple_pointcloud.npy"

points = np.load(APPLE_POINTCLOUD)
print(f"Total apple points: {len(points)}")

# Subsample for performance
idx = np.random.choice(len(points), size=min(500000, len(points)), replace=False)
pts = points[idx]
print(f"Loaded apple points: {len(pts)}")


fig = go.Figure(data=[go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode='markers',
    marker=dict(size=1, color='red', opacity=0.8)
)])

fig.update_layout(title="Apple Point Cloud - tree_02", scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
))

fig.show()