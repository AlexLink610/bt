"""
select_views.py  –  Pick N images evenly spaced across a given azimuth arc,
                    stratified across three height tiers (ground / chest / ladder).

Pipeline:
    1. Load all camera poses from transforms.json
    2. Split into 3 height tiers using Z-percentiles (p33 / p66) — adapts to
       any tree dataset automatically
    3. For each tier independently: select N//3 cameras evenly spread across
       the requested azimuth arc
    4. Combine all three tier selections into one final list

This guarantees both even azimuthal coverage AND even vertical coverage,
regardless of how the photographer walked around the tree.

Usage:
    python select_views.py --tree C:\\Users\\alex\\BA\\data\\tree_02 --n 8  --arc 180
    python select_views.py --tree C:\\Users\\alex\\BA\\data\\tree_02 --n 32 --arc 360

    # Custom output directory (e.g. for cluster view txts):
    python select_views.py --tree C:\\Users\\alex\\BA\\data\\tree_02 --n 32 --arc 360 ^
        --out_dir C:\\Users\\alex\\BA\\viewtxts

Output (all saved to --out_dir or --tree folder):
    selected_Nviews_ARCdeg.txt        filenames, one per line
    selected_Nviews_ARCdeg.png        top-down XY scatter plot
    selected_Nviews_ARCdeg_az.png     azimuth distribution polar plot
"""

import os
import json
import argparse
import numpy as np


# ── Pose loading ──────────────────────────────────────────────────────────────

def load_poses(transforms_path):
    """
    Load all frames from transforms.json.
    Returns:
        poses  : list of (filename, azimuth_deg, x, y, z)  sorted by azimuth
        centre : (cx, cy)  mean XY position of all cameras
    """
    with open(transforms_path, "r") as f:
        data = json.load(f)

    raw = []
    for frame in data["frames"]:
        M     = np.array(frame["transform_matrix"])
        x, y, z = M[0, 3], M[1, 3], M[2, 3]
        fname = frame["file_path"]
        raw.append((fname, x, y, z))

    xs = np.array([r[1] for r in raw])
    ys = np.array([r[2] for r in raw])
    cx, cy = xs.mean(), ys.mean()

    poses = []
    for fname, x, y, z in raw:
        az = np.degrees(np.arctan2(y - cy, x - cx))   # –180 … +180
        poses.append((fname, az, x, y, z))

    poses.sort(key=lambda r: r[1])
    return poses, (cx, cy)


# ── Height-tier splitting ─────────────────────────────────────────────────────

def split_into_tiers(poses):
    """
    Split poses into 3 height tiers using Z-percentiles (p33 / p66).
    This is fully data-driven — works for any tree dataset.

    Returns:
        tiers      : [tier0_poses, tier1_poses, tier2_poses]  (low → high Z)
        boundaries : (p33, p66)
    """
    zvals = np.array([p[4] for p in poses])
    p33   = np.percentile(zvals, 33)
    p66   = np.percentile(zvals, 66)

    tier0 = [p for p in poses if p[4] <= p33]           # ladder / low
    tier1 = [p for p in poses if p33 < p[4] <= p66]     # chest  / mid
    tier2 = [p for p in poses if p[4] > p66]            # ground / high

    print(f"  Height tiers (Z percentile split):")
    print(f"    Tier 0 (ladder/low,  Z ≤ {p33:.2f}): {len(tier0)} cameras")
    print(f"    Tier 1 (chest/mid,   {p33:.2f} < Z ≤ {p66:.2f}): {len(tier1)} cameras")
    print(f"    Tier 2 (ground/high, Z > {p66:.2f}): {len(tier2)} cameras")

    return [tier0, tier1, tier2], (p33, p66)


# ── Even azimuth selection within one tier ────────────────────────────────────

def select_evenly_spaced(poses, n, arc_deg):
    """
    Pick n cameras from poses that are as evenly spread as possible
    across arc_deg of azimuth.

    Strategy:
        1. Find the arc_deg-wide window with the most cameras (densest arc).
        2. Place n evenly-spaced azimuth targets across that window.
        3. For each target pick the closest unused camera.

    Returns selected poses (sorted by azimuth).
    """
    if n == 0:
        return []

    azimuths = np.array([p[1] for p in poses])

    if arc_deg >= 360:
        targets = np.linspace(azimuths[0], azimuths[0] + 360, n, endpoint=False)
        targets = ((targets + 180) % 360) - 180
    else:
        # Find densest arc window
        best_start = azimuths[0]
        best_count = 0
        for start in azimuths:
            end = start + arc_deg
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

    selected    = []
    used_indices = set()
    for target in targets:
        diffs = np.abs(azimuths - target)
        diffs = np.minimum(diffs, 360 - diffs)
        for idx in used_indices:
            diffs[idx] = 9999
        best = int(np.argmin(diffs))
        selected.append(poses[best])
        used_indices.add(best)

    selected.sort(key=lambda r: r[1])
    return selected


# ── Stratified selection ──────────────────────────────────────────────────────

def select_stratified(poses, n, arc_deg):
    """
    Split into 3 height tiers, select N//3 from each, combine.

    If n is not divisible by 3, the remainder is distributed to the
    lower tiers (tier0 gets +1 first, then tier1).

    Returns (selected_poses, tier_boundaries).
    """
    tiers, boundaries = split_into_tiers(poses)

    # Distribute n across 3 tiers
    base      = n // 3
    remainder = n % 3
    counts    = [base + (1 if i < remainder else 0) for i in range(3)]

    print(f"\n  Views per tier: {counts[0]} / {counts[1]} / {counts[2]}  (total {sum(counts)})")

    selected = []
    for i, (tier, count) in enumerate(zip(tiers, counts)):
        tier_sel = select_evenly_spaced(tier, count, arc_deg)
        print(f"    Tier {i}: selected {len(tier_sel)}/{len(tier)} cameras")
        selected.extend(tier_sel)

    selected.sort(key=lambda r: r[1])   # sort combined by azimuth
    return selected, boundaries


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cameras_xy(all_poses, selected_poses, centre, boundaries, out_path, n, arc_deg):
    """Top-down XY scatter — colour-coded by height tier."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping XY plot)")
        return

    p33, p66 = boundaries
    cx, cy   = centre

    tier_colors = {0: "steelblue", 1: "darkorange", 2: "forestgreen"}
    tier_labels = {0: "Ladder (low Z)", 1: "Chest (mid Z)", 2: "Ground (high Z)"}

    fig, ax = plt.subplots(figsize=(8, 8))

    # All cameras (grey)
    xs_all = [p[2] for p in all_poses]
    ys_all = [p[3] for p in all_poses]
    ax.scatter(xs_all, ys_all, s=10, c="lightgray", zorder=1, label="All cameras")

    # Selected cameras coloured by tier
    for tier_idx, (z_lo, z_hi) in enumerate(
            [(-9999, p33), (p33, p66), (p66, 9999)]):
        tier_sel = [p for p in selected_poses if z_lo < p[4] <= z_hi]
        if tier_sel:
            xs = [p[2] for p in tier_sel]
            ys = [p[3] for p in tier_sel]
            ax.scatter(xs, ys, s=70, c=tier_colors[tier_idx],
                       zorder=3, label=tier_labels[tier_idx])

    # Number selected cameras in azimuth order
    for i, (fname, az, x, y, z) in enumerate(selected_poses):
        ax.annotate(str(i + 1), (x, y), fontsize=6, ha="center", va="bottom",
                    color="black", zorder=4)

    # Tree centre
    ax.scatter([cx], [cy], s=140, c="red", marker="*", zorder=5, label="Tree centre")

    ax.set_aspect("equal")
    ax.set_title(f"Camera positions (XY) — {n} views, {arc_deg}° arc\n"
                 f"(stratified: {n//3}+{n//3}+{n - 2*(n//3)} per height tier)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  XY plot saved:      {out_path}")


def plot_cameras_azimuth(all_poses, selected_poses, boundaries, out_path, n, arc_deg):
    """Polar azimuth distribution — colour-coded by height tier."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping azimuth plot)")
        return

    p33, p66 = boundaries
    tier_colors = {0: "steelblue", 1: "darkorange", 2: "forestgreen"}
    tier_labels = {0: "Ladder", 1: "Chest", 2: "Ground"}

    all_az = np.radians([p[1] for p in all_poses])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    # All cameras (grey ticks)
    ax.scatter(all_az, np.ones_like(all_az), s=12, c="lightgray",
               zorder=1, label="All cameras")

    # Selected cameras by tier, slightly outside unit circle
    for tier_idx, (z_lo, z_hi) in enumerate(
            [(-9999, p33), (p33, p66), (p66, 9999)]):
        tier_sel = [p for p in selected_poses if z_lo < p[4] <= z_hi]
        if tier_sel:
            az = np.radians([p[1] for p in tier_sel])
            r  = 1.10 + tier_idx * 0.12   # stack tiers at different radii
            ax.scatter(az, np.full_like(az, r), s=80,
                       c=tier_colors[tier_idx], zorder=3,
                       label=f"{tier_labels[tier_idx]} ({len(tier_sel)})")

    # Number labels (azimuth order, outermost ring)
    for i, (fname, az, x, y, z) in enumerate(selected_poses):
        ax.annotate(str(i + 1), (np.radians(az), 1.42),
                    fontsize=6, ha="center", va="center", color="black")

    ax.set_ylim(0, 1.55)
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Azimuth distribution — {n} views, {arc_deg}° arc\n"
                 f"(rings: inner=ladder, mid=chest, outer=ground)", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Azimuth plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select N views stratified by height tier and azimuth arc."
    )
    parser.add_argument("--tree",    required=True,
                        help="Path to tree folder containing transforms.json")
    parser.add_argument("--n",       type=int, required=True,
                        help="Total number of views to select")
    parser.add_argument("--arc",     type=float, default=360,
                        help="Azimuth arc in degrees (default: 360)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same as --tree)")
    args = parser.parse_args()

    transforms_path = os.path.join(args.tree, "transforms.json")
    if not os.path.exists(transforms_path):
        print(f"ERROR: transforms.json not found at {transforms_path}")
        return

    out_dir = args.out_dir or args.tree
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading poses from {transforms_path} ...")
    all_poses, centre = load_poses(transforms_path)
    print(f"  {len(all_poses)} total frames")
    print(f"  Azimuth range: {all_poses[0][1]:.1f}° to {all_poses[-1][1]:.1f}°")
    print(f"  Tree centre (XY): ({centre[0]:.3f}, {centre[1]:.3f})")

    print(f"\nSelecting {args.n} views across {args.arc}° (stratified by height) ...")
    selected, boundaries = select_stratified(all_poses, args.n, args.arc)

    print(f"\nSelected frames ({len(selected)} total):")
    for i, (fname, az, x, y, z) in enumerate(selected):
        print(f"  {i+1:3d}. az={az:+7.1f}°  z={z:+6.2f}  {fname}")

    # ── Save filenames txt ────────────────────────────────────────────────────
    tag      = f"{args.n}views_{int(args.arc)}deg"
    txt_path = os.path.join(out_dir, f"selected_{tag}.txt")
    with open(txt_path, "w") as f:
        for fname, az, x, y, z in selected:
            f.write(os.path.basename(fname) + "\n")
    print(f"\nFilename list saved: {txt_path}")

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_cameras_xy(
        all_poses, selected, centre, boundaries,
        os.path.join(out_dir, f"selected_{tag}.png"),
        args.n, args.arc
    )
    plot_cameras_azimuth(
        all_poses, selected, boundaries,
        os.path.join(out_dir, f"selected_{tag}_az.png"),
        args.n, args.arc
    )

    print(f"\nDone! Pass to run_vggt.py with --image_list {txt_path}")


if __name__ == "__main__":
    main()