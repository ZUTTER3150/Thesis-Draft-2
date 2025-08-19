# EDA_2_3_6_interactions.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


DATA_PATH = "dataset1.xlsx"    # <- your Excel file
COUNTRY_COL = "Country"        # change if your country/NOC name column differs

# Output figures
FIG_FACET_GDP_POP = "interaction_logGDP_by_logPOP_facets.png"
FIG_FACET_ATH_HDI = "interaction_logATH_by_HDI_facets.png"
FIG_COLOR_GDP_POP = "interaction_logGDP_by_logPOP_color.png"
FIG_COLOR_ATH_HDI = "interaction_logATH_by_HDI_color.png"
FIG_FOUR_PANELS   = "scatter_four_panels.png"

# 1) Utilities

def clean_to_numeric(series: pd.Series) -> pd.Series:
    """Convert possibly string-typed numeric column to float."""
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.apply(lambda x: re.sub(r"[^0-9eE\.\-+]", "", x))
    return pd.to_numeric(s, errors="coerce")

def assert_positive(series: pd.Series, name: str):
    bad = (series <= 0) | series.isna()
    if bad.any():
        raise ValueError(
            f"Column '{name}' has {int(bad.sum())} non-positive/NaN values; "
            f"log requires strictly positive values."
        )

def quantile_bins(series: pd.Series, q=4, precision=2, label_prefix=""):
    bins = pd.qcut(series, q=q, duplicates="drop")
    labels = [f"{label_prefix}{cat.left:.{precision}f}–{cat.right:.{precision}f}" for cat in bins.cat.categories]
    return bins, labels

def annotate_top5(ax, sub_df, xcol, ycol, country_col):
    top = sub_df.nlargest(5, ycol)
    for _, r in top.iterrows():
        ax.annotate(str(r[country_col]),
                    (r[xcol], r[ycol]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)


# 2) Load & prepare data

p = Path(DATA_PATH)
if not p.exists():
    raise FileNotFoundError(f"File not found: {p.resolve()}")

df = pd.read_excel(p)

# Required columns
required = ["GDP", "Population", "Athletes", "HDI", "Totalmedals"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

# Numeric cleaning
for col in required:
    df[col] = clean_to_numeric(df[col])

# Country column for labels
if COUNTRY_COL not in df.columns:
    df[COUNTRY_COL] = np.arange(len(df))

# Drop rows with NA in required numeric columns
df = df.dropna(subset=required).copy()

# Create logs (ensure positivity)
assert_positive(df["GDP"], "GDP")
assert_positive(df["Population"], "Population")
assert_positive(df["Athletes"], "Athletes")

df["log_GDP"] = np.log(df["GDP"])
df["log_Population"] = np.log(df["Population"])
df["log_Athletes"] = np.log(df["Athletes"])


# 3) Four-panel scatter (Top-5 by medals labeled)

panels = [
    ("log_GDP", "log(GDP)"),
    ("log_Population", "log(Population)"),
    ("log_Athletes", "log(Athletes)"),
    ("HDI", "HDI"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (xcol, xlabel) in zip(axes, panels):
    ax.scatter(df[xcol], df["Totalmedals"], alpha=0.75)
    annotate_top5(ax, df, xcol, "Totalmedals", COUNTRY_COL)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total Medals")
    ax.set_title(f"Total Medals vs {xlabel}")

fig.suptitle("Total Medals vs Key Predictors (log-transformed where noted)", y=0.98, fontsize=13)
plt.tight_layout()
plt.savefig(FIG_FOUR_PANELS, dpi=300)
plt.close()


# 4) Facet: log(GDP) vs medals by log(Population) quartiles

bins_pop, labels_pop = quantile_bins(df["log_Population"], q=4, precision=2, label_prefix="log(Pop) ")
df["logPop_bin"] = bins_pop

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
axes = axes.flatten()
for ax, (cat, label) in zip(axes, zip(bins_pop.cat.categories, labels_pop)):
    sub = df[df["logPop_bin"] == cat]
    ax.scatter(sub["log_GDP"], sub["Totalmedals"], alpha=0.8)
    annotate_top5(ax, sub, "log_GDP", "Totalmedals", COUNTRY_COL)
    ax.set_title(label)
    ax.set_xlabel("log(GDP)")
    ax.set_ylabel("Total Medals")
fig.suptitle("Interaction: log(GDP) vs Total Medals, faceted by log(Population) quartiles", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(FIG_FACET_GDP_POP, dpi=300)
plt.close()


# 5) Facet: log(Athletes) vs medals by HDI quartiles

bins_hdi, labels_hdi = quantile_bins(df["HDI"], q=4, precision=3, label_prefix="HDI ")
df["HDI_bin"] = bins_hdi

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
axes = axes.flatten()
for ax, (cat, label) in zip(axes, zip(bins_hdi.cat.categories, labels_hdi)):
    sub = df[df["HDI_bin"] == cat]
    ax.scatter(sub["log_Athletes"], sub["Totalmedals"], alpha=0.8)
    annotate_top5(ax, sub, "log_Athletes", "Totalmedals", COUNTRY_COL)
    ax.set_title(label)
    ax.set_xlabel("log(Athletes)")
    ax.set_ylabel("Total Medals")
fig.suptitle("Interaction: log(Athletes) vs Total Medals, faceted by HDI quartiles", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(FIG_FACET_ATH_HDI, dpi=300)
plt.close()


# 6) Single-panel color-coded interaction plots 

# (A) log(GDP) colored by log(Pop) quartiles
colors = plt.cm.viridis(np.linspace(0, 1, len(bins_pop.cat.categories)))
color_map = {cat: colors[i] for i, cat in enumerate(bins_pop.cat.categories)}

plt.figure(figsize=(8, 6))
for cat in bins_pop.cat.categories:
    sub = df[df["logPop_bin"] == cat]
    plt.scatter(sub["log_GDP"], sub["Totalmedals"], alpha=0.8, color=color_map[cat],
                label=f"log(Pop) {cat.left:.2f}–{cat.right:.2f}")
annotate_top5(plt.gca(), df, "log_GDP", "Totalmedals", COUNTRY_COL)
plt.xlabel("log(GDP)")
plt.ylabel("Total Medals")
plt.title("Interaction: log(GDP) vs Total Medals (colored by log(Population) quartiles)")
plt.legend(title="log(Population) quartiles", fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(FIG_COLOR_GDP_POP, dpi=300)
plt.close()

# (B) log(Athletes) colored by HDI quartiles
colors = plt.cm.coolwarm(np.linspace(0, 1, len(bins_hdi.cat.categories)))
color_map = {cat: colors[i] for i, cat in enumerate(bins_hdi.cat.categories)}

plt.figure(figsize=(8, 6))
for cat in bins_hdi.cat.categories:
    sub = df[df["HDI_bin"] == cat]
    plt.scatter(sub["log_Athletes"], sub["Totalmedals"], alpha=0.8, color=color_map[cat],
                label=f"HDI {cat.left:.3f}–{cat.right:.3f}")
annotate_top5(plt.gca(), df, "log_Athletes", "Totalmedals", COUNTRY_COL)
plt.xlabel("log(Athletes)")
plt.ylabel("Total Medals")
plt.title("Interaction: log(Athletes) vs Total Medals (colored by HDI quartiles)")
plt.legend(title="HDI quartiles", fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(FIG_COLOR_ATH_HDI, dpi=300)
plt.close()

print("Saved figures:")
print(f" - {FIG_FOUR_PANELS}")
print(f" - {FIG_FACET_GDP_POP}")
print(f" - {FIG_FACET_ATH_HDI}")
print(f" - {FIG_COLOR_GDP_POP}")
print(f" - {FIG_COLOR_ATH_HDI}")
