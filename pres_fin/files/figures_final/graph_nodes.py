import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SAM3 mask colors by in-view id (as in the mask images)
PAL = {1: "#a124f2", 2: "#00f24c", 3: "#f22924", 4: "#0040f2",
       5: "#97f224", 6: "#f200ce", 7: "#24f2d4"}
# components of Fig. 4.4: view -> in-view mask id
INST = [{1: 1, 2: 3, 3: 7}, {1: 2, 2: 1, 3: 6, 4: 3},
        {1: 3, 2: 4, 3: 1, 4: 2}, {1: 4, 2: 6, 3: 2},
        {1: 5, 2: 2, 3: 5}, {2: 5, 3: 4, 4: 1, 5: 1},
        {2: 7, 3: 3, 4: 4, 5: 2}]

def node(ax, x, y, mid):
    c = PAL[mid]
    tc = "white" if mid in (1, 4) else "black"
    ax.scatter(x, y, s=260, c=c, edgecolors="black",
               linewidths=1.0, zorder=3)
    ax.text(x, y, str(mid), ha="center", va="center",
            fontsize=8, color=tc, zorder=4)

def frame(ax, nrows):
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(nrows + 0.6, 0.4)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"view {v}" for v in range(1, 6)])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

# Variant A: same layout as Fig. 4.4, no edges, no row labels
fig, ax = plt.subplots(figsize=(8, 4))
for r, inst in enumerate(INST, start=1):
    for v, mid in inst.items():
        node(ax, v, r, mid)
frame(ax, 7)
fig.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"graph_nodes_fig44layout.{ext}", dpi=300,
                transparent=True)

# Variant B: per view sorted by mask id (no grouping visible)
fig, ax = plt.subplots(figsize=(8, 4))
for v in range(1, 6):
    ids = sorted(inst[v] for inst in INST if v in inst)
    for r, mid in enumerate(ids, start=1):
        node(ax, v, r, mid)
frame(ax, 7)
fig.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"graph_nodes_sorted.{ext}", dpi=300,
                transparent=True)
