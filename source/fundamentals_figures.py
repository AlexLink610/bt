"""
fundamentals_figures.py -- Generic figures for the Fundamentals chapter.

Writes vector PDFs (+ PNG previews) into the output directory:
  vggt_architecture.pdf   tokenization -> alternating attention -> heads
  graph_components.pdf    small graph with three connected components
  ransac_circle.pdf       RANSAC circle fit vs. least-squares fit
  clustering_chaining.pdf DBSCAN chaining vs. complete-linkage clustering

Fonts: Computer Modern (cmr10, shipped with matplotlib), close to the
thesis font. Usage:  python fundamentals_figures.py [out_dir]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "cmr10",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "pdf.fonttype": 42,
})

C_BLUE, C_ORANGE, C_GREEN = "#4C72B0", "#DD8452", "#55A868"
C_RED, C_GRAY, C_LIGHT = "#C44E52", "#8C8C8C", "#E8E8E8"
C_PURPLE = "#8172B3"


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec="k", fs=9, lw=0.8, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color="k", shrinkA=0, shrinkB=0))


# ───────────────────────────── VGGT architecture ────────────────────────────
def vggt_architecture(out):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.set_xlim(0, 17.4)
    ax.set_ylim(0.9, 5.0)
    ax.axis("off")

    # input images
    for k in range(3):
        ax.add_patch(Rectangle((0.2 + 0.18 * k, 2.2 + 0.18 * k), 1.3, 1.0,
                               fc="#DCE6F2", ec="k", lw=0.8))
    ax.text(1.05, 1.75, r"images $I_1,\dots,I_N$", ha="center", fontsize=8)

    # DINOv2 patch encoding
    arrow(ax, (1.95, 2.9), (2.45, 2.9))
    box(ax, 2.5, 2.25, 1.5, 1.3, "DINOv2\npatch\nencoding", "#F2F2F2", fs=8)
    arrow(ax, (4.05, 2.9), (4.55, 2.9))

    # tokens per image: camera token, register tokens, patch tokens
    s = 0.24
    for r in range(3):
        y = 3.45 - r * 0.62
        x = 4.6
        ax.add_patch(Rectangle((x, y), s, s, fc=C_ORANGE, ec="k", lw=0.5))
        for q in range(2):
            ax.add_patch(Rectangle((x + (q + 1) * (s + 0.04), y), s, s,
                                   fc="#AFC8E7", ec="k", lw=0.5))
        for q in range(5):
            ax.add_patch(Rectangle((x + (q + 3) * (s + 0.04) + 0.08, y), s, s,
                                   fc=C_LIGHT, ec="k", lw=0.5))
    ax.text(5.75, 3.95, "tokens per image", ha="center", fontsize=8)
    lx = 5.0
    for fc, lab, ly in ((C_ORANGE, "camera token", 1.85),
                        ("#AFC8E7", "register tokens", 1.5),
                        (C_LIGHT, "patch tokens", 1.15)):
        ax.add_patch(Rectangle((lx, ly), 0.2, 0.2, fc=fc, ec="k", lw=0.5))
        ax.text(lx + 0.32, ly + 0.1, lab, va="center", fontsize=7)

    # alternating attention
    arrow(ax, (7.35, 2.9), (7.85, 2.9))
    ax.add_patch(FancyBboxPatch((7.9, 1.55), 3.3, 2.75, boxstyle="round,pad=0.03",
                                fc="#FAFAFA", ec="k", lw=0.8, ls="--"))
    box(ax, 8.15, 3.05, 2.8, 0.85, "frame-wise\nself-attention", "#E3EFE3", fs=8)
    box(ax, 8.15, 1.85, 2.8, 0.85, "global\nself-attention", "#F6E3D3", fs=8)
    arrow(ax, (9.55, 3.03), (9.55, 2.73))
    ax.text(9.55, 4.5, r"alternating attention, $24\times$", ha="center", fontsize=8)

    # heads, fed by camera tokens and patch tokens
    arrow(ax, (11.25, 3.3), (12.45, 4.05))
    arrow(ax, (11.25, 2.5), (12.45, 1.75))
    ax.text(11.75, 3.95, "camera\ntokens", ha="center", fontsize=6.5)
    ax.text(11.75, 1.55, "patch\ntokens", ha="center", fontsize=6.5)
    box(ax, 12.5, 3.55, 1.7, 0.95, "camera\nhead", "#EDE7F6", fs=8)
    box(ax, 12.5, 1.2, 1.7, 0.95, "DPT\nhead", "#EDE7F6", fs=8)
    arrow(ax, (14.25, 4.02), (14.65, 4.02))
    ax.text(14.7, 4.02, "extrinsics,\nintrinsics", va="center", fontsize=8)
    arrow(ax, (14.25, 1.67), (14.65, 1.67))
    ax.text(14.7, 1.67, "depth maps,\npoint maps,\ntracking features\n(+ confidence)",
            va="center", fontsize=8)
    save(fig, out, "vggt_architecture")


# ─────────────────────────── connected components ───────────────────────────
def graph_components(out):
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ax.axis("off")
    comps = [
        # (nodes {name: (x, y)}, edges, color)
        ({"u": (0.0, 0.0), "v": (1.0, 0.6), "w": (2.0, 0.0), "x": (1.0, -0.6)},
         [("u", "v"), ("v", "w"), ("w", "x")], C_BLUE),
        ({"a": (3.4, 0.5), "b": (4.3, 0.5), "c": (3.85, -0.35)},
         [("a", "b"), ("b", "c"), ("c", "a")], C_GREEN),
        ({"d": (5.6, 0.25), "e": (6.4, -0.25)}, [("d", "e")], C_ORANGE),
    ]
    for nodes, edges, col in comps:
        for p, q in edges:
            (x1, y1), (x2, y2) = nodes[p], nodes[q]
            ax.plot([x1, x2], [y1, y2], color="k", lw=1.0, zorder=1)
        for name, (x, y) in nodes.items():
            ax.add_patch(Circle((x, y), 0.2, fc=matplotlib.colors.to_rgba(col, 1.0),
                                ec="k", lw=0.8, zorder=2))
            label = f"${name}$" if name in "uvw" else ""
            ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=3)
        xs = [p[0] for p in nodes.values()]
        ys = [p[1] for p in nodes.values()]
        ax.add_patch(FancyBboxPatch((min(xs) - 0.4, min(ys) - 0.4),
                                    max(xs) - min(xs) + 0.8, max(ys) - min(ys) + 0.8,
                                    boxstyle="round,pad=0.05", fc="none", ec=col,
                                    lw=1.0, ls="--", zorder=0))
    ax.set_xlim(-0.6, 7.0)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    save(fig, out, "graph_components")


# ─────────────────────────────── RANSAC circle ──────────────────────────────
def circle_from_3(p1, p2, p3):
    A = np.array([[2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
                  [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])]])
    b = np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1])
    c = np.linalg.solve(A, b)
    return c, np.linalg.norm(p1 - c)


def ransac_circle(out):
    rng = np.random.default_rng(4)
    ang = rng.uniform(0, 2 * np.pi, 40)
    inl = np.stack([np.cos(ang), np.sin(ang)], 1) * 1.0 + rng.normal(0, 0.03, (40, 2))
    outl = np.array([[1.9, 1.5], [2.2, 1.1], [1.7, 1.9], [2.4, 1.7], [2.0, 0.6],
                     [-1.9, -1.6], [1.4, -1.8], [-1.6, 1.7]])
    pts = np.vstack([inl, outl])

    # least squares (algebraic) fit to all points
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    sol = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
    c_ls, r_ls = sol[:2], np.sqrt(sol[2] + sol[:2] @ sol[:2])

    # RANSAC
    tol, best = 0.08, None
    for _ in range(300):
        idx = rng.choice(len(pts), 3, replace=False)
        try:
            c, r = circle_from_3(*pts[idx])
        except np.linalg.LinAlgError:
            continue
        cons = np.abs(np.linalg.norm(pts - c, axis=1) - r) <= tol
        if best is None or cons.sum() > best[2].sum():
            best = (c, r, cons, idx)
    c, r, cons, idx = best

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    t = np.linspace(0, 2 * np.pi, 400)
    band_o = c + (r + tol) * np.stack([np.cos(t), np.sin(t)], 1)
    band_i = c + (r - tol) * np.stack([np.cos(t), np.sin(t)], 1)
    ax.fill(np.r_[band_o[:, 0], band_i[::-1, 0]], np.r_[band_o[:, 1], band_i[::-1, 1]],
            color=C_BLUE, alpha=0.15, lw=0, label="tolerance band")
    ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=C_BLUE, lw=1.3,
            label="RANSAC fit")
    ax.plot(c_ls[0] + r_ls * np.cos(t), c_ls[1] + r_ls * np.sin(t), color=C_RED,
            lw=1.2, ls="--", label="least-squares fit")
    ax.scatter(*pts[cons].T, s=12, color=C_BLUE, zorder=3, label="consensus set")
    ax.scatter(*pts[~cons].T, s=18, color=C_GRAY, marker="x", zorder=3,
               label="outliers")
    ax.scatter(*pts[idx].T, s=60, facecolors="none", edgecolors="k", lw=1.2,
               zorder=4, label="minimal sample")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False)
    save(fig, out, "ransac_circle")


# ─────────────────────── DBSCAN vs. complete linkage ────────────────────────
def clustering_chaining(out):
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    rng = np.random.default_rng(1)
    g1 = rng.normal([0.0, 0.0], 0.15, (25, 2))
    g2 = rng.normal([1.6, 0.0], 0.15, (25, 2))
    chain = np.stack([np.linspace(0.35, 1.25, 5), rng.normal(0, 0.03, 5)], 1)
    g3 = rng.normal([0.8, 1.1], 0.15, (25, 2))
    pts = np.vstack([g1, g2, chain, g3])

    lab_db = DBSCAN(eps=0.28, min_samples=3).fit_predict(pts)
    lab_cl = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0,
                                     linkage="complete").fit_predict(pts)
    cols = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, "#937860", "#DA8BC3"]

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
    for ax, lab, title in ((axes[0], lab_db, "DBSCAN"),
                           (axes[1], lab_cl, "complete linkage")):
        for l in np.unique(lab):
            m = lab == l
            col = C_GRAY if l < 0 else cols[l % len(cols)]
            ax.scatter(*pts[m].T, s=10, color=col)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, out, "clustering_chaining")
    print("DBSCAN clusters:", len(set(lab_db) - {-1}),
          " complete linkage clusters:", len(set(lab_cl)))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "figures"
    os.makedirs(out, exist_ok=True)
    vggt_architecture(out)
    graph_components(out)
    ransac_circle(out)
    clustering_chaining(out)
    print("written to", out)
