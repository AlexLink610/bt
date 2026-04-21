import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <tree.ply> [apples.ply]")
        sys.exit(1)

    tree_path = sys.argv[1]
    apple_path = sys.argv[2] if len(sys.argv) > 2 else None
    title = os.path.basename(tree_path)

    # Load point clouds
    tree_pcd = o3d.io.read_point_cloud(tree_path)
    print(f"Loaded {len(tree_pcd.points)} tree points")

    apple_pcd = None
    if apple_path and os.path.exists(apple_path):
        apple_pcd = o3d.io.read_point_cloud(apple_path)
        print(f"Loaded {len(apple_pcd.points)} apple points")

    # Materials
    mat_tree = rendering.MaterialRecord()
    mat_tree.shader = "defaultUnlit"
    mat_tree.point_size = 2.0

    mat_tree_transparent = rendering.MaterialRecord()
    mat_tree_transparent.shader = "defaultUnlit"
    mat_tree_transparent.point_size = 2.0
    mat_tree_transparent.base_color = [1.0, 1.0, 1.0, 0.03]
    mat_apple = rendering.MaterialRecord()
    mat_apple.shader = "defaultUnlit"
    mat_apple.point_size = 3.0

    # App and window
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(title, 1400, 800)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    scene.scene.add_geometry("tree", tree_pcd, mat_tree)
    if apple_pcd:
        scene.scene.add_geometry("apples", apple_pcd, mat_apple)

    bounds = scene.scene.bounding_box
    scene.setup_camera(60, bounds, bounds.get_center())

    # Control panel
    panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))
    panel.add_child(gui.Label("Controls"))

    if apple_pcd:
        def toggle_apples(checked):
            scene.scene.show_geometry("apples", checked)
        chk_apples = gui.Checkbox("Show apple overlay")
        chk_apples.checked = True
        chk_apples.set_on_checked(toggle_apples)
        panel.add_child(chk_apples)

    def toggle_transparent(checked):
        if checked:
            scene.scene.modify_geometry_material("tree", mat_tree_transparent)
        else:
            scene.scene.modify_geometry_material("tree", mat_tree)
    chk_transp = gui.Checkbox("Tree transparent")
    chk_transp.checked = False
    chk_transp.set_on_checked(toggle_transparent)
    panel.add_child(chk_transp)

    window.add_child(panel)

    def on_layout(ctx):
        r = window.content_rect
        pw = 200
        panel.frame = gui.Rect(r.x, r.y, pw, r.height)
        scene.frame = gui.Rect(r.x + pw, r.y, r.width - pw, r.height)

    window.set_on_layout(on_layout)
    app.run()


if __name__ == "__main__":
    main()