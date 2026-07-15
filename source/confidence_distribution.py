"""
confidence_distribution.py -- Check whether VGGT apple-pixel confidence is
Gaussian (normalverteilt). Produces a histogram + Q-Q plot, reports skewness,
kurtosis, and normality-test results, and also tests a log-transform (skewed
positive data is often log-normal rather than normal).

Usage:
    python confidence_distribution.py \
        --confmap  ~/ba/output_vggt/old/t02_180_32v_confmap.npy \
        --filenames ~/ba/output_vggt/old/t02_180_32v_filenames.txt \
        --masks    ~/ba/output_sam/tree_02/semantics_sam3 \
        --out      ~/ba/output_vggt/conf_dist_180_32v.png

    # whole-scene (no mask restriction):
    python confidence_distribution.py --confmap ... --out ...

Notes:
  - Confmaps are per-scene min-max normalized; distribution SHAPE is still valid
    to inspect (normalization is a linear rescale, doesn't change normality).
  - Normality tests on millions of points will almost always REJECT normality
    for tiny deviations, so weight the visual (histogram/Q-Q) and the
    skew/kurtosis magnitudes more than the p-value.
"""

import os
import argparse
import numpy as np
from PIL import Image


def load_mask(path, W, H):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST))


def mask_path_for(masks_dir, fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(masks_dir, f"mask_{stem}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confmap",   required=True)
    parser.add_argument("--filenames", default=None)
    parser.add_argument("--masks",     default=None,
                        help="Restrict to apple pixels (needs --filenames).")
    parser.add_argument("--out",       default=None, help="Save plot PNG here.")
    parser.add_argument("--max_sample", type=int, default=200000,
                        help="Subsample for tests/plots (default: 200k).")
    args = parser.parse_args()

    conf = np.load(args.confmap)
    if conf.ndim == 2:
        conf = conf[None]
    N, H, W = conf.shape

    # Gather values (apple-only or all)
    if args.masks:
        if args.filenames is None:
            raise SystemExit("--masks requires --filenames")
        with open(args.filenames) as f:
            fnames = [l.strip() for l in f if l.strip()]
        vals = []
        for i, fn in enumerate(fnames):
            m = load_mask(mask_path_for(args.masks, fn), W, H)
            if m is None:
                continue
            cv = conf[i][m > 0]
            vals.append(cv[~np.isnan(cv)])
        c = np.concatenate(vals) if vals else np.array([])
        scope = "apple pixels"
    else:
        c = conf.reshape(-1)
        c = c[~np.isnan(c)]
        scope = "all pixels"

    if c.size == 0:
        print("No values found.")
        return

    print(f"Scope: {scope}   N = {c.size:,}")
    print(f"mean={c.mean():.4f}  median={np.median(c):.4f}  std={c.std():.4f}")
    print(f"min={c.min():.4f}  max={c.max():.4f}")

    # Skewness & kurtosis (excess). Gaussian => skew 0, excess kurtosis 0.
    from scipy import stats
    skew = stats.skew(c)
    kurt = stats.kurtosis(c)  # excess (Fisher)
    print(f"\nskewness      = {skew:+.3f}   (0 = symmetric; >0 = right tail)")
    print(f"excess kurtosis= {kurt:+.3f}   (0 = Gaussian tails)")

    # Subsample for tests/plots
    rng = np.random.default_rng(42)
    cs = c if c.size <= args.max_sample else rng.choice(c, args.max_sample, replace=False)

    # Normality tests (note: huge N almost always rejects; report anyway)
    try:
        W_stat, p_sw = stats.shapiro(cs[:5000])  # shapiro caps ~5000
        print(f"\nShapiro-Wilk (n=5000 subsample): W={W_stat:.4f}  p={p_sw:.2e}")
    except Exception as e:
        print(f"Shapiro-Wilk failed: {e}")
    ks_stat, p_ks = stats.kstest(
        (cs - cs.mean()) / cs.std(), "norm")
    print(f"Kolmogorov-Smirnov vs normal: D={ks_stat:.4f}  p={p_ks:.2e}")

    # Log-transform check (for positive skewed data -> maybe log-normal)
    pos = cs[cs > 0]
    if pos.size > 100:
        logc = np.log(pos)
        lskew = stats.skew(logc)
        print(f"\nlog-transform skewness = {lskew:+.3f}  "
              f"(closer to 0 than {skew:+.3f} => more log-normal than normal)")

    if p_ks < 0.05:
        print("\n=> KS rejects normality. Given the skew above, the distribution "
              "is NOT Gaussian (likely right-skewed / log-normal-ish).")
    else:
        print("\n=> KS does not reject normality at 0.05.")

    # Plot
    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.hist(cs, bins=80, density=True, alpha=0.7, color="steelblue",
                     label="confidence")
            mu, sd = cs.mean(), cs.std()
            xs = np.linspace(cs.min(), cs.max(), 200)
            gauss = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax1.plot(xs, gauss, "r--", lw=2, label=f"Gaussian(mu={mu:.3f},sd={sd:.3f})")
            ax1.set_title(f"Confidence histogram ({scope})")
            ax1.set_xlabel("confidence"); ax1.set_ylabel("density"); ax1.legend()

            stats.probplot(cs, dist="norm", plot=ax2)
            ax2.set_title("Q-Q plot vs normal")

            fig.tight_layout()
            fig.savefig(args.out, dpi=120)
            print(f"\nPlot saved: {args.out}")
        except ImportError:
            print("matplotlib not available -- skipping plot.")


if __name__ == "__main__":
    main()
