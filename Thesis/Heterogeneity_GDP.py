# Heterogeneity by GDP level: Grouped residuals

import pandas as pd
import numpy as np
from pathlib import Path

#  Settings 
COMPACT = True  # if True, drop "Mean Std. Residual" from the LaTeX table (to match your POP table)

# 1) Load data 
PATH = Path("zinb_predictions.csv")  
df = pd.read_csv(PATH)

base_required = {"Totalmedals", "E_total", "std_resid", "Team"}
missing = base_required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in data: {missing}. "
                     f"Expected at least: {sorted(base_required)}")

# 2) Residuals
# Residual = Actual - Predicted
df["residual"] = df["Totalmedals"] - df["E_total"]

# 3) Find/construct GDP variable 
# Preference order: log_GDP, GDP (log-transform), per-capita variants (log-transform)
gdp_col = None
if "log_GDP" in df.columns:
    gdp_col = "log_GDP"
    df["_log_GDP_used"] = df["log_GDP"]
else:
    # try total GDP columns
    for cand in ["GDP", "gdp", "gdp_total", "GDP_total"]:
        if cand in df.columns:
            df["_log_GDP_used"] = np.log(df[cand].clip(lower=1e-12))
            gdp_col = cand
            break
# try per-capita if still not found
if gdp_col is None:
    for cand in ["GDP_per_capita", "gdp_per_capita", "GDPpc", "gdp_pc", "GDP_pc"]:
        if cand in df.columns:
            df["_log_GDP_used"] = np.log(df[cand].clip(lower=1e-12))
            gdp_col = cand
            break

if gdp_col is None:
    raise ValueError(
        "No GDP variable found. Please include one of: "
        "'log_GDP', 'GDP', 'gdp', 'gdp_total', 'GDP_total', "
        "'GDP_per_capita', 'gdp_per_capita', 'GDPpc', 'gdp_pc', 'GDP_pc'."
    )

# 4) Quartile-based GDP groups 
labels = ["Low GDP", "Lower-Middle GDP", "Upper-Middle GDP", "High GDP"]
df["GDP_group"] = pd.qcut(df["_log_GDP_used"], q=4, labels=labels)

# handle ties at cut points producing NA
if df["GDP_group"].isna().any():
    rng = np.random.default_rng(42)
    df["_log_GDP_jitter"] = df["_log_GDP_used"] + rng.normal(0, 1e-9, size=len(df))
    df["GDP_group"] = pd.qcut(df["_log_GDP_jitter"], q=4, labels=labels)
    df.drop(columns=["_log_GDP_jitter"], inplace=True)

# 5) Grouped summary
group_stats = (
    df.groupby("GDP_group", observed=True)
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
group_stats["GDP_group"] = pd.Categorical(group_stats["GDP_group"], categories=labels, ordered=True)
group_stats = group_stats.sort_values("GDP_group")

# 6) Pretty print 
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== Prediction residuals by GDP groups ===\n")
print(group_stats.to_string(index=False))

# 7) Save outputs 
out_csv = PATH.with_name("gdp_groups_residuals.csv")
group_stats.to_csv(out_csv, index=False)

# Prepare LaTeX-friendly column names
latex_df = group_stats.rename(columns={
    "GDP_group": "GDP Group",
    "n": "N",
    "mean_residual": "Mean Residual",
    "median_residual": "Median Residual",
    "sd_residual": "SD Residual",
    "mean_std_resid": "Mean Std. Residual",
    "over_share": "Share Over-Perf.",
    "under_share": "Share Under-Perf."
})

if COMPACT:
    # match your POP compact table (remove Mean Std. Residual)
    latex_df = latex_df.drop(columns=["Mean Std. Residual"], errors="ignore")

# LaTeX export (for the paper)
out_tex = PATH.with_name("gdp_groups_residuals.tex")
try:
    latex_df.to_latex(
        out_tex,
        index=False,
        float_format="%.3f",
        caption="Prediction residuals by GDP groups (quartiles of log GDP or a logged GDP proxy).",
        label="tab:gdp_groups_residuals"
    )
    print(f"\nLaTeX table saved to: {out_tex}")
except Exception as e:
    print(f"\n[Warn] Could not write LaTeX table: {e}")

print("\nNotes:")
print("- Residual = Actual - Predicted; positive means the model under-predicted medals.")
print("- Groups are quartiles of log(GDP) or a logged GDP proxy (total or per-capita, depending on availability).")
print("- 'Share Over-Perf.' / 'Under-Perf.' are the proportions with residual > 0 / < 0 within each group.")
if not COMPACT:
    print("- 'Mean Std. Residual' summarizes dispersion in standardized units (closer to 0 indicates less systematic bias).")
