import open3d as o3d
import sys

ply_path = sys.argv[1]

pcd = o3d.io.read_point_cloud(ply_path)
print(f"Loaded {len(pcd.points)} points")
o3d.visualization.draw_geometries([pcd])