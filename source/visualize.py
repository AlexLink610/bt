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

    # -----------------------------
    # Materials
    # -----------------------------

    # Tree: normal
    mat_tree_normal = rendering.MaterialRecord()
    mat_tree_normal.shader = "defaultUnlit"
    mat_tree_normal.point_size = 2.0
    mat_tree_normal.base_color = [1.0, 1.0, 1.0, 1.0]

    # Tree: faded / transparent
    mat_tree_faded = rendering.MaterialRecord()
    mat_tree_faded.shader = "defaultUnlit"
    mat_tree_faded.point_size = 2.0
    mat_tree_faded.base_color = [1.0, 1.0, 1.0, 0.08]

    mat_tree_ultra_faint = rendering.MaterialRecord()
    mat_tree_ultra_faint.shader = "defaultUnlit"
    mat_tree_ultra_faint.point_size = 1.0
    mat_tree_ultra_faint.base_color = [1.0, 1.0, 1.0, 0.05]
    
    mat_apple_normal = rendering.MaterialRecord()
    mat_apple_normal.shader = "defaultUnlit"
    mat_apple_normal.point_size = 3.0

    mat_apple_highlight = rendering.MaterialRecord()
    mat_apple_highlight.shader = "defaultUnlit"
    mat_apple_highlight.point_size = 6.0

    # -----------------------------
    # App and window
    # -----------------------------
    app = gui.Application.instance
    app.initialize()

    window = app.create_window(title, 1400, 800)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # Optional: dark background makes apples easier to see
    scene.scene.set_background([0.05, 0.05, 0.05, 1.0])

    # Add geometry
    scene.scene.add_geometry("tree", tree_pcd, mat_tree_normal)
    if apple_pcd is not None:
        scene.scene.add_geometry("apples", apple_pcd, mat_apple_normal)

    bounds = scene.scene.bounding_box
    scene.setup_camera(60, bounds, bounds.get_center())

    # -----------------------------
    # Control panel
    # -----------------------------
    panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))
    panel.add_child(gui.Label("Controls"))

    if apple_pcd is not None:
        def toggle_apples_visible(checked):
            # Only let this checkbox control visibility when not in highlight mode
            if not chk_highlight.checked:
                scene.scene.show_geometry("apples", checked)

        chk_apples = gui.Checkbox("Show apple overlay")
        chk_apples.checked = True
        chk_apples.set_on_checked(toggle_apples_visible)
        panel.add_child(chk_apples)

        def toggle_highlight_apples(checked):
            if checked:
                scene.scene.show_geometry("apples", True)
                scene.scene.modify_geometry_material("tree", mat_tree_ultra_faint)
                scene.scene.modify_geometry_material("apples", mat_apple_highlight)
            else:
                scene.scene.modify_geometry_material("tree", mat_tree_normal)
                scene.scene.modify_geometry_material("apples", mat_apple_normal)
                scene.scene.show_geometry("apples", chk_apples.checked)

        chk_highlight = gui.Checkbox("Highlight apples")
        chk_highlight.checked = False
        chk_highlight.set_on_checked(toggle_highlight_apples)
        panel.add_child(chk_highlight)
    def toggle_tree_transparent(checked):
        if checked:
            scene.scene.modify_geometry_material("tree", mat_tree_faded)
        else:
            scene.scene.modify_geometry_material("tree", mat_tree_normal)

    chk_transp = gui.Checkbox("Tree transparent")
    chk_transp.checked = False
    chk_transp.set_on_checked(toggle_tree_transparent)
    panel.add_child(chk_transp)

    window.add_child(panel)

    def on_layout(ctx):
        r = window.content_rect
        pw = 220
        panel.frame = gui.Rect(r.x, r.y, pw, r.height)
        scene.frame = gui.Rect(r.x + pw, r.y, r.width - pw, r.height)

    window.set_on_layout(on_layout)
    app.run()


if __name__ == "__main__":
    main()