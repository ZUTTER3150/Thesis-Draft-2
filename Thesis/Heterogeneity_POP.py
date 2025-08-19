
# Heterogeneity by Population Size: Grouped residuals

import pandas as pd
import numpy as np
from pathlib import Path

# 1) Load data
PATH = Path("zinb_predictions.csv") 
df = pd.read_csv(PATH)

required_cols = {"Totalmedals", "E_total", "std_resid", "log_Pop"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in data: {missing}. "
                     f"Expected at least: {sorted(required_cols)}")

# 2) Residuals & population-size groups 
# Residual = Actual - Predicted
df["residual"] = df["Totalmedals"] - df["E_total"]

# Quartile-based groups on log_Pop (data-driven, balanced bins)
# Labels ordered from smallest to largest population
labels = ["Small", "Medium-Small", "Medium-Large", "Large"]
df["Pop_group"] = pd.qcut(df["log_Pop"], q=4, labels=labels)

# sanity check: ensure no NA groups
if df["Pop_group"].isna().any():
    # If ties at cut points produce NA, slightly jitter log_Pop and retry
    rng = np.random.default_rng(42)
    df["_log_Pop_jitter"] = df["log_Pop"] + rng.normal(0, 1e-9, size=len(df))
    df["Pop_group"] = pd.qcut(df["_log_Pop_jitter"], q=4, labels=labels)
    df.drop(columns=["_log_Pop_jitter"], inplace=True)

# 3) Grouped summary 
group_stats = (
    df.groupby("Pop_group", observed=True)
      .agg(
          n=("Team", "count"),
          mean_residual=("residual", "mean"),
          median_residual=("residual", "median"),
          sd_residual=("residual", "std"),
          mean_std_resid=("std_resid", "mean"),
          over_share=("residual", lambda x: np.mean(x > 0)),
          under_share=("residual", lambda x: np.mean(x < 0)),
      )
      .reset_index()
)

# Order rows to match our labels order
group_stats["Pop_group"] = pd.Categorical(group_stats["Pop_group"], categories=labels, ordered=True)
group_stats = group_stats.sort_values("Pop_group")

# 4) Pretty print 
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== Prediction residuals by population-size groups ===\n")
print(group_stats.to_string(index=False))

# 5) Save outputs 
out_csv = PATH.with_name("pop_groups_residuals.csv")
group_stats.to_csv(out_csv, index=False)

# LaTeX export 
out_tex = PATH.with_name("pop_groups_residuals.tex")
try:
    group_stats.rename(columns={
        "Pop_group": "Population Group",
        "n": "N",
        "mean_residual": "Mean Residual",
        "median_residual": "Median Residual",
        "sd_residual": "SD Residual",
        "mean_std_resid": "Mean Std. Residual",
        "over_share": "Share Over-Perf.",
        "under_share": "Share Under-Perf."
    }).to_latex(
        out_tex,
        index=False,
        float_format="%.3f",
        caption="Prediction residuals by population-size groups (quartiles of log population).",
        label="tab:pop_groups_residuals"
    )
    print(f"\nLaTeX table saved to: {out_tex}")
except Exception as e:
    print(f"\n[Warn] Could not write LaTeX table: {e}")

# 6) (Optional) 
print("\nNotes:")
print("- Residual = Actual - Predicted; positive means the model under-predicted medals.")
print("- 'Mean Std. Residual' summarizes dispersion: larger values indicate more heterogeneous deviations.")
print("- 'Share Over-Perf.' / 'Under-Perf.' are the proportions with residual > 0 / < 0 within each group.")
