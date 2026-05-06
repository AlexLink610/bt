import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import sys
import os


def read_ply_with_quality(path):
    """
    Read a PLY file and return (o3d.PointCloud, conf_array or None).
    Parses the binary PLY manually to extract the 'quality' float channel
    that Open3D does not expose through its standard API.
    """
    pcd = o3d.io.read_point_cloud(path)

    conf = None
    try:
        with open(path, "rb") as f:
            has_quality = False
            props = []
            n_pts = 0
            while True:
                line = f.readline().decode("ascii").strip()
                if line == "end_header":
                    break
                if line.startswith("element vertex"):
                    n_pts = int(line.split()[-1])
                if line.startswith("property"):
                    parts = line.split()
                    props.append((parts[1], parts[2]))
                    if parts[2] == "quality":
                        has_quality = True

            if has_quality:
                type_map = {
                    "float": np.float32, "float32": np.float32,
                    "uchar": np.uint8,   "uint8": np.uint8,
                    "double": np.float64,"float64": np.float64,
                }
                dt   = np.dtype([(name, type_map[t]) for t, name in props])
                data = np.frombuffer(f.read(n_pts * dt.itemsize), dtype=dt)
                conf = data["quality"].copy()
                print(f"  Confidence range: {conf.min():.3f} – {conf.max():.3f}")

    except Exception as e:
        print(f"  Could not read quality channel: {e}")

    return pcd, conf


def make_mat(point_size=2.0):
    m = rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = point_size
    return m


def apply_conf_filter(pcd, conf, threshold):
    """Return a new PointCloud with only points where conf >= threshold."""
    if conf is None or threshold <= 0.0:
        return pcd
    mask = conf >= threshold
    pts  = np.asarray(pcd.points)[mask]
    cols = np.asarray(pcd.colors)[mask]
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts)
    filtered.colors = o3d.utility.Vector3dVector(cols)
    return filtered


def make_red_pcd(pcd):
    """Return a copy of pcd with all points coloured bright red."""
    pts = np.asarray(pcd.points)
    red = o3d.geometry.PointCloud()
    red.points = o3d.utility.Vector3dVector(pts)
    red.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(pts), 1))
    )
    return red


def make_ghost_pcd(source_pcd, fade=0.85):
    """Blend original colors toward white by fade factor (0=original, 1=white)."""
    pts  = np.asarray(source_pcd.points)
    cols = np.asarray(source_pcd.colors)
    faded_cols = cols + (1.0 - cols) * fade
    g = o3d.geometry.PointCloud()
    g.points = o3d.utility.Vector3dVector(pts)
    g.colors = o3d.utility.Vector3dVector(faded_cols)
    return g


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <tree.ply> [apples.ply]")
        sys.exit(1)

    tree_path  = sys.argv[1]
    apple_path = sys.argv[2] if len(sys.argv) > 2 else None
    title      = os.path.basename(tree_path)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading tree: {tree_path}")
    tree_pcd, tree_conf = read_ply_with_quality(tree_path)
    print(f"  {len(tree_pcd.points):,} points")

    apple_pcd, apple_conf = None, None
    if apple_path and os.path.exists(apple_path):
        print(f"Loading apples: {apple_path}")
        apple_pcd, apple_conf = read_ply_with_quality(apple_path)
        print(f"  {len(apple_pcd.points):,} points")

    # ── State ─────────────────────────────────────────────────────────────────
    state = {
        "apples_visible": False,
        "tree_faded":     False,
        "apples_red":     False,
        "conf_threshold": 0.0,
        "point_size":     2.0,
    }

    # ── App / window ──────────────────────────────────────────────────────────
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(title, 1400, 800)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0.85, 0.85, 0.85, 1.0])
    window.add_child(scene)

    # ── Initial geometries ────────────────────────────────────────────────────
    scene.scene.add_geometry("tree", tree_pcd, make_mat(state["point_size"]))

    if apple_pcd is not None:
        scene.scene.add_geometry("apples", apple_pcd,
                                 make_mat(state["point_size"] * 1.5))
        scene.scene.show_geometry("apples", False)

    bounds = scene.scene.bounding_box
    scene.setup_camera(60, bounds, bounds.get_center())

    # ── Rebuild helpers ────────────────────────────────────────────────────────
    def rebuild_tree():
        sz    = state["point_size"]
        t_pcd = apply_conf_filter(tree_pcd, tree_conf, state["conf_threshold"])
        if state["tree_faded"]:
            t_pcd = make_ghost_pcd(t_pcd, fade=0.85)
        scene.scene.remove_geometry("tree")
        scene.scene.add_geometry("tree", t_pcd, make_mat(sz))

    def rebuild_apples():
        if apple_pcd is None:
            return
        sz    = state["point_size"]
        a_pcd = apply_conf_filter(apple_pcd, apple_conf, state["conf_threshold"])
        if state["apples_red"]:
            a_pcd = make_red_pcd(a_pcd)
        scene.scene.remove_geometry("apples")
        scene.scene.add_geometry("apples", a_pcd, make_mat(sz * 1.5))
        scene.scene.show_geometry("apples", state["apples_visible"])

    # ── Control panel ─────────────────────────────────────────────────────────
    panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))
    panel.add_child(gui.Label("-- Apple controls --"))
    if apple_pcd is not None:

        # Button 1: Show / Hide apples
        btn_apples = gui.Button("Show apples")
        def on_btn_apples():
            state["apples_visible"] = not state["apples_visible"]
            btn_apples.text = "Hide apples" if state["apples_visible"] else "Show apples"
            scene.scene.show_geometry("apples", state["apples_visible"])
        btn_apples.set_on_clicked(on_btn_apples)
        panel.add_child(btn_apples)

        # Button 2: Fade / Restore tree
        btn_fade = gui.Button("Fade tree")
        def on_btn_fade():
            state["tree_faded"] = not state["tree_faded"]
            btn_fade.text = "Restore tree" if state["tree_faded"] else "Fade tree"
            rebuild_tree()
        btn_fade.set_on_clicked(on_btn_fade)
        panel.add_child(btn_fade)

        # Button 3: Mark apples red / restore colors
        btn_red = gui.Button("Mark apples red")
        def on_btn_red():
            state["apples_red"] = not state["apples_red"]
            btn_red.text = "Restore apple colors" if state["apples_red"] else "Mark apples red"
            rebuild_apples()
        btn_red.set_on_clicked(on_btn_red)
        panel.add_child(btn_red)

    # ── Point size slider ─────────────────────────────────────────────────────
    panel.add_fixed(8)
    panel.add_child(gui.Label("-- Point size --"))
    lbl_size = gui.Label(f"Size: {state['point_size']:.1f}")
    panel.add_child(lbl_size)

    sld_size = gui.Slider(gui.Slider.DOUBLE)
    sld_size.set_limits(0.5, 8.0)
    sld_size.double_value = state["point_size"]

    def on_size(val):
        state["point_size"] = val
        lbl_size.text = f"Size: {val:.1f}"
        rebuild_tree()
        rebuild_apples()

    sld_size.set_on_value_changed(on_size)
    panel.add_child(sld_size)

    # ── Confidence slider ─────────────────────────────────────────────────────
    if tree_conf is not None or apple_conf is not None:
        panel.add_fixed(8)
        panel.add_child(gui.Label("-- Confidence filter --"))
        panel.add_child(gui.Label("Min confidence (0 = keep all)"))

        lbl_conf = gui.Label("Threshold: 0.00")
        panel.add_child(lbl_conf)

        sld_conf = gui.Slider(gui.Slider.DOUBLE)
        sld_conf.set_limits(0.0, 1.0)
        sld_conf.double_value = 0.0

        def on_conf(val):
            state["conf_threshold"] = val
            lbl_conf.text = f"Threshold: {val:.2f}"
            rebuild_tree()
            if apple_pcd is not None:
                rebuild_apples()

        sld_conf.set_on_value_changed(on_conf)
        panel.add_child(sld_conf)

    # ── Info ──────────────────────────────────────────────────────────────────
    panel.add_fixed(8)
    panel.add_child(gui.Label(f"Tree pts:  {len(tree_pcd.points):,}"))
    if apple_pcd is not None:
        panel.add_child(gui.Label(f"Apple pts: {len(apple_pcd.points):,}"))

    window.add_child(panel)

    def on_layout(ctx):
        r  = window.content_rect
        pw = 220
        panel.frame = gui.Rect(r.x, r.y, pw, r.height)
        scene.frame = gui.Rect(r.x + pw, r.y, r.width - pw, r.height)

    window.set_on_layout(on_layout)
    app.run()


if __name__ == "__main__":
    main()