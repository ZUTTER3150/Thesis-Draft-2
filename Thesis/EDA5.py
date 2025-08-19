import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re

# Load dataset
df = pd.read_excel("dataset1.xlsx")

# Helpers 
def clean_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a potentially string-typed numeric column to float.
    - Strips spaces
    - Removes thousands separators (commas)
    - Removes any non-numeric trailing characters (e.g., units)
    - Coerces errors to NaN
    """
    s = series.astype(str).str.strip()
    # remove common thousands separators
    s = s.str.replace(",", "", regex=False)
    # remove any characters except digits, dot, minus, and scientific notation marks
    s = s.apply(lambda x: re.sub(r"[^0-9eE\.\-+]", "", x))
    return pd.to_numeric(s, errors="coerce")

def assert_positive(series: pd.Series, name: str):
    """Raise an informative error if non-positive values exist before log."""
    nonpos = (series <= 0) | series.isna()
    if nonpos.any():
        count = int(nonpos.sum())
        sample = series[nonpos].head(5).tolist()
        raise ValueError(
            f"{name} contains {count} non-positive/NaN values; log requires positive values. "
            f"Example problematic entries: {sample}. "
            f"Please fix upstream or filter rows before logging."
        )

# Clean numeric columns 
for col in ["GDP", "Population", "Athletes", "HDI", "Totalmedals"]:
    if col in df.columns:
        df[col] = clean_to_numeric(df[col])


df = df.dropna(subset=["GDP", "Population", "Athletes", "HDI", "Totalmedals"]).copy()

# Ensure strictly positive for log
assert_positive(df["GDP"], "GDP")
assert_positive(df["Population"], "Population")
assert_positive(df["Athletes"], "Athletes")

# Log transforms 
df["log_GDP"] = np.log(df["GDP"])
df["log_Population"] = np.log(df["Population"])
df["log_Athletes"] = np.log(df["Athletes"])

# Correlation analysis 
variables_for_analysis = ["log_GDP", "log_Population", "HDI", "log_Athletes", "Totalmedals"]

pearson_corr = df[variables_for_analysis].corr(method="pearson")
spearman_corr = df[variables_for_analysis].corr(method="spearman")

pearson_corr.to_csv("pearson_correlation_matrix.csv")
spearman_corr.to_csv("spearman_correlation_matrix.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="Blues", square=True)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig("pearson_correlation_matrix.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="Blues", square=True)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.savefig("spearman_correlation_matrix.png", dpi=300)
plt.close()

# VIF 
X = df[["log_GDP", "log_Population", "HDI", "log_Athletes"]].copy()
X_const = add_constant(X)

vif_table = pd.DataFrame({
    "Variable": X_const.columns,
    "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
})
vif_table.to_csv("vif_results.csv", index=False)

print("Saved outputs:")
print(" - pearson_correlation_matrix.csv / .png")
print(" - spearman_correlation_matrix.csv / .png")
print(" - vif_results.csv")
