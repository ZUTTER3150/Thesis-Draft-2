# Heterogeneity by HDI: Grouped residuals

import pandas as pd
import numpy as np
from pathlib import Path

# 1) Load data 
PATH = Path("zinb_predictions.csv")  
df = pd.read_csv(PATH)

required_cols = {"Totalmedals", "E_total", "std_resid", "HDI"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in data: {missing}. "
                     f"Expected at least: {sorted(required_cols)}")

# 2) Residuals & HDI groups 
# Residual = Actual - Predicted
df["residual"] = df["Totalmedals"] - df["E_total"]

# Quartile-based groups on HDI (balanced bins)
labels = ["Low HDI", "Lower-Middle HDI", "Upper-Middle HDI", "High HDI"]
df["HDI_group"] = pd.qcut(df["HDI"], q=4, labels=labels)

# Optional: handle NA due to ties at cut points
if df["HDI_group"].isna().any():
    rng = np.random.default_rng(42)
    df["_HDI_jitter"] = df["HDI"] + rng.normal(0, 1e-9, size=len(df))
    df["HDI_group"] = pd.qcut(df["_HDI_jitter"], q=4, labels=labels)
    df.drop(columns=["_HDI_jitter"], inplace=True)

# 3) Grouped summary 
group_stats = (
    df.groupby("HDI_group", observed=True)
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

# Order rows as per labels
group_stats["HDI_group"] = pd.Categorical(group_stats["HDI_group"], categories=labels, ordered=True)
group_stats = group_stats.sort_values("HDI_group")

# 4) Pretty print 
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== Prediction residuals by HDI groups ===\n")
print(group_stats.to_string(index=False))

# 5) Save outputs 
out_csv = PATH.with_name("hdi_groups_residuals.csv")
group_stats.to_csv(out_csv, index=False)

# LaTeX export (for paper)
out_tex = PATH.with_name("hdi_groups_residuals.tex")
try:
    group_stats.rename(columns={
        "HDI_group": "HDI Group",
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
        caption="Prediction residuals by HDI groups (quartiles of HDI).",
        label="tab:hdi_groups_residuals"
    )
    print(f"\nLaTeX table saved to: {out_tex}")
except Exception as e:
    print(f"\n[Warn] Could not write LaTeX table: {e}")

# 6) (Optional) Simple interpretation hints 
print("\nNotes:")
print("- Residual = Actual - Predicted; positive means the model under-predicted medals.")
print("- 'Mean Std. Residual' summarizes dispersion: larger values = more heterogeneous deviations.")
print("- 'Share Over-Perf.' / 'Under-Perf.' are proportions with residual > 0 / < 0 within each group.")
