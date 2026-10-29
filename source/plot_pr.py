"""
plot_pr.py -- Precision-recall curves of pointmap correspondences.

Each curve comes from a CSV written by corr_precision.py (--out) and is
traced over the threshold tau, so scenes reconstructed at different scales
can share one plot. The operating point of the pipeline (the tau actually
used) can be marked per curve.

Usage:
    python plot_pr.py --curve "tree, point map head=head.csv"
                      --curve "tree, depth + camera=depth.csv"
                      --curve "table scene=table.csv"
                      --mark "tree, point map head=0.02" --out pr_curves.pdf
"""

import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "cmr10", "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9,
                     "pdf.fonttype": 42})
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
MARKERS = ["o", "s", "^", "D", "v"]


def read(path):
    with open(path) as f:
        rows = [(float(r["tau"]), float(r["precision"]), float(r["recall"]))
                for r in csv.DictReader(f)]
    return np.array(sorted(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", action="append", required=True,
                    help='"label=path.csv"')
    ap.add_argument("--mark", action="append", default=[],
                    help='"label=tau": highlight the row closest to tau')
    ap.add_argument("--out", default="pr_curves.pdf")
    args = ap.parse_args()

    marks = {}
    for m in args.mark:
        label, tau = m.rsplit("=", 1)
        marks[label] = float(tau)

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for k, spec in enumerate(args.curve):
        label, path = spec.rsplit("=", 1)
        d = read(path)
        col = COLORS[k % len(COLORS)]
        ax.plot(d[:, 2], d[:, 1], marker=MARKERS[k % len(MARKERS)], ms=3.5,
                lw=1.2, color=col, label=label)
        if label in marks:
            i = int(np.argmin(np.abs(d[:, 0] - marks[label])))
            ax.scatter(d[i, 2], d[i, 1], s=90, marker="*", color=col,
                       edgecolor="k", lw=0.6, zorder=5)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print("written", args.out)


if __name__ == "__main__":
    main()
