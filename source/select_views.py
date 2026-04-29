"""
select_views.py  –  Pick N images evenly spaced across a given azimuth arc
                    using camera poses from transforms.json

The script reads camera positions from transforms.json, computes the azimuth
angle of each camera around the tree centre, then selects N frames that are
as evenly spread as possible across the requested arc (180° or 360°).

Selected filenames are printed and saved to a text file that run_vggt.py can
use directly (or you can pass --output_filelist to run_vggt.py).

Usage:
    python select_views.py --tree C:\\Users\\alex\\BA\\data\\tree_02 --n 8 --arc 180
    python select_views.py --tree C:\\Users\\alex\\BA\\data\\tree_02 --n 32 --arc 360

Output:
    selected_views_N_ARCdeg.txt   (filenames, one per line)
    selected_views_N_ARCdeg.png   (top-down plot of camera positions)
"""

import os
import json
import argparse
import numpy as np


def load_poses(transforms_path):
    """Return list of (filename, azimuth_deg, (x,y,z)) sorted by azimuth."""
    with open(transforms_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    positions = []
    for frame in frames:
        M = np.array(frame["transform_matrix"])
        # Camera position = last column of 4x4 transform (translation)
        x, y, z = M[0, 3], M[1, 3], M[2, 3]
        fname = frame["file_path"]
        positions.append((fname, x, y, z))

    # Compute tree centre as mean of all XY camera positions
    xs = np.array([p[1] for p in positions])
    ys = np.array([p[2] for p in positions])
    cx, cy = xs.mean(), ys.mean()

    # Azimuth = angle of camera around tree centre in XY plane
    results = []
    for fname, x, y, z in positions:
        az = np.degrees(np.arctan2(y - cy, x - cx))  # -180 to +180
        results.append((fname, az, (x, y, z)))

    results.sort(key=lambda r: r[1])  # sort by azimuth
    return results, (cx, cy)


def select_evenly_spaced(poses, n, arc_deg):
    """
    Pick N cameras evenly spaced across arc_deg.

    Strategy:
    1. Find the arc_deg-wide window that contains the most cameras
       (i.e. the densest arc — where the photographer walked).
    2. Within that window, pick N cameras at evenly spaced azimuth targets.
    """
    azimuths = np.array([p[1] for p in poses])

    if arc_deg >= 360:
        # Full circle — just pick evenly from all cameras
        targets = np.linspace(azimuths[0], azimuths[0] + 360, n, endpoint=False)
        # Wrap targets into -180..180
        targets = ((targets + 180) % 360) - 180
    else:
        # Find the densest arc window of width arc_deg
        best_start = azimuths[0]
        best_count = 0
        for start in azimuths:
            end = start + arc_deg
            # Count cameras in [start, end], handling wrap
            if end <= 180:
                count = np.sum((azimuths >= start) & (azimuths <= end))
            else:
                end_wrapped = end - 360
                count = np.sum((azimuths >= start) | (azimuths <= end_wrapped))
            if count > best_count:
                best_count = count
                best_start = start

        targets = np.linspace(best_start, best_start + arc_deg, n, endpoint=False)
        targets = ((targets + 180) % 360) - 180

    # For each target azimuth, pick the closest actual camera
    selected = []
    used_indices = set()
    for target in targets:
        # Angular distance (handle wrap)
        diffs = np.abs(azimuths - target)
        diffs = np.minimum(diffs, 360 - diffs)
        # Mask already-used cameras
        for idx in used_indices:
            diffs[idx] = 9999
        best = int(np.argmin(diffs))
        selected.append(poses[best])
        used_indices.add(best)

    selected.sort(key=lambda r: r[1])  # sort by azimuth for readability
    return selected


def plot_cameras(all_poses, selected_poses, centre, out_path, n, arc_deg):
    """Save a top-down scatter plot of camera positions."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    cx, cy = centre
    fig, ax = plt.subplots(figsize=(7, 7))

    # All cameras
    xs = [p[2][0] for p in all_poses]
    ys = [p[2][1] for p in all_poses]
    ax.scatter(xs, ys, s=12, c="lightgray", zorder=1, label="All cameras")

    # Selected cameras
    sxs = [p[2][0] for p in selected_poses]
    sys = [p[2][1] for p in selected_poses]
    ax.scatter(sxs, sys, s=60, c="crimson", zorder=3, label=f"Selected ({n})")

    # Number the selected cameras in order
    for i, (fname, az, (x, y, z)) in enumerate(selected_poses):
        ax.annotate(str(i + 1), (x, y), fontsize=7, ha="center", va="bottom",
                    color="darkred", zorder=4)

    # Tree centre
    ax.scatter([cx], [cy], s=120, c="green", marker="*", zorder=5, label="Tree centre")

    ax.set_aspect("equal")
    ax.set_title(f"Camera positions — {n} views, {arc_deg}° arc")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree",    required=True,
                        help=r"Path to tree folder, e.g. C:\Users\alex\BA\data\tree_02")
    parser.add_argument("--n",       type=int,   required=True,
                        help="Number of views to select")
    parser.add_argument("--arc",     type=float, default=180,
                        help="Arc in degrees to spread views across (default: 180)")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save output files (default: same as --tree)")
    args = parser.parse_args()

    transforms_path = os.path.join(args.tree, "transforms.json")
    if not os.path.exists(transforms_path):
        print(f"ERROR: transforms.json not found at {transforms_path}")
        return

    out_dir = args.out_dir or args.tree
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading poses from {transforms_path}...")
    all_poses, centre = load_poses(transforms_path)
    print(f"  {len(all_poses)} total frames")
    print(f"  Azimuth range: {all_poses[0][1]:.1f}° to {all_poses[-1][1]:.1f}°")
    print(f"  Tree centre (XY): ({centre[0]:.3f}, {centre[1]:.3f})")

    print(f"\nSelecting {args.n} views across {args.arc}°...")
    selected = select_evenly_spaced(all_poses, args.n, args.arc)

    print(f"\nSelected frames:")
    for i, (fname, az, pos) in enumerate(selected):
        print(f"  {i+1:2d}. az={az:+7.1f}°  {fname}")

    # Save filenames txt
    tag      = f"{args.n}views_{int(args.arc)}deg"
    txt_path = os.path.join(out_dir, f"selected_{tag}.txt")
    with open(txt_path, "w") as f:
        for fname, az, pos in selected:
            # Write just the image filename (basename), matching run_vggt.py expectation
            f.write(os.path.basename(fname) + "\n")
    print(f"\nFilename list saved: {txt_path}")

    # Save plot
    plot_path = os.path.join(out_dir, f"selected_{tag}.png")
    plot_cameras(all_poses, selected, centre, plot_path, args.n, args.arc)

    print(f"\nDone! Pass this file to run_vggt.py with --image_list {txt_path}")


if __name__ == "__main__":
    main()