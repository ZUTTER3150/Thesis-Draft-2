# residual_diagnostics.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# helpers

def freedman_diaconis_bins(x: np.ndarray) -> int:
    """Return a robust bin count using the Freedman–Diaconis rule."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return min(30, max(5, int(np.sqrt(n))))
    bin_width = 2 * iqr * n ** (-1/3)
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(10, min(80, bins))

def read_predictions(csv_path: str) -> pd.DataFrame:
    """Read the exported ZINB predictions."""
    df = pd.read_csv(csv_path)
    required = ["E_total", "std_resid", "Totalmedals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    # Fallback label column for annotation
    if "Team" not in df.columns and "Code" not in df.columns:
        df["Team"] = [f"NOC_{i}" for i in range(len(df))]
    if "Team" not in df.columns:
        df["Team"] = df["Code"].astype(str)
    return df

def annotate_top_outliers(ax, x, y, labels, k=3):
    """Annotate top-|y| points on an axes."""
    order = np.argsort(-np.abs(y))[:k]
    for idx in order:
        ax.annotate(str(labels[idx]),
                    (x[idx], y[idx]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8)

# plotting 

def plot_residual_hist(std_resid: np.ndarray, out_path="fig_resid_hist.png"):
    """Figure 3A: Histogram + kernel density + reference lines."""
    std_resid = np.asarray(std_resid)
    bins = freedman_diaconis_bins(std_resid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(std_resid, bins=bins, density=True, alpha=0.7)
    # Kernel density estimate
    kde = stats.gaussian_kde(std_resid[np.isfinite(std_resid)])
    xs = np.linspace(np.nanmin(std_resid), np.nanmax(std_resid), 400)
    ax.plot(xs, kde(xs), linestyle="-", linewidth=1.5, label="KDE density")

    # Optional normal density with same mean/std (for reference only)
    m, s = np.nanmean(std_resid), np.nanstd(std_resid, ddof=1)
    if np.isfinite(s) and s > 1e-8:
        ax.plot(xs, stats.norm.pdf(xs, loc=m, scale=s),
                linestyle="--", linewidth=1.2, label="Normal ref.")

    for y in [0]:
        ax.axvline(y, linestyle="-", linewidth=1.0)
    for thr in [2, -2, 3, -3]:
        ax.axvline(thr, linestyle=":", linewidth=1.0)

    ax.set_xlabel("standardized residual $r_i$")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_resid_vs_pred(pred: np.ndarray, std_resid: np.ndarray,
                       labels, out_path="fig_resid_vs_pred.png"):
    """Figure 3B: Standardized residual vs Predicted E[Y]."""
    pred = np.asarray(pred)
    std_resid = np.asarray(std_resid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(pred, std_resid, s=18, alpha=0.8)

    # reference lines
    ax.axhline(0, linestyle="-", linewidth=1.0)
    for thr in [2, -2, 3, -3]:
        ax.axhline(thr, linestyle=":", linewidth=1.0)

    annotate_top_outliers(ax, pred, std_resid, labels, k=3)

    ax.set_xlabel("predicted $\\mathbb{E}[Y_i]$")
    ax.set_ylabel("standardized residual $r_i$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_resid_qq(std_resid: np.ndarray, out_path="fig_resid_qq.png"):
    """Figure 3C: Normal QQ plot for standardized residuals."""
    std_resid = np.asarray(std_resid)
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    stats.probplot(std_resid[np.isfinite(std_resid)], dist="norm", plot=ax)
    ax.set_title("Normal QQ Plot of Standardized Residuals")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("sample quantiles")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

#  main 

if __name__ == "__main__":
    csv_path = "zinb_predictions.csv"   # change if needed
    df = read_predictions(csv_path)

    # Extract series
    pred = df["E_total"].values
    std_resid = df["std_resid"].values
    labels = df["Team"].astype(str).values

    # Quick summary in console (optional)
    print("[Summary] std_resid: mean={:.3f}, sd={:.3f}, min={:.3f}, max={:.3f}".format(
        np.nanmean(std_resid), np.nanstd(std_resid, ddof=1),
        np.nanmin(std_resid), np.nanmax(std_resid)))
    print("[Info] Top-|r| NOCs:")
    top_idx = np.argsort(-np.abs(std_resid))[:3]
    for i in top_idx:
        print(f"  {labels[i]:25s}  r = {std_resid[i]: .3f},  Pred = {pred[i]: .2f},  Actual = {df['Totalmedals'].iloc[i]}")

    # Make figures
    os.makedirs(".", exist_ok=True)
    plot_residual_hist(std_resid, out_path="fig_resid_hist.png")
    plot_resid_vs_pred(pred, std_resid, labels, out_path="fig_resid_vs_pred.png")
    plot_resid_qq(std_resid, out_path="fig_resid_qq.png")

    print("[Saved] fig_resid_hist.png, fig_resid_vs_pred.png, fig_resid_qq.png")
