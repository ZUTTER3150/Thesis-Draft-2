import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import Normalize

# Load dataset
path = Path("dataset1.xlsx")  # <-- set your own path
df = pd.read_excel(path)

# Columns
medal_col = "Totalmedals" if "Totalmedals" in df.columns else "TotalMedals"
df[medal_col] = pd.to_numeric(df[medal_col], errors="coerce").fillna(0).astype(int)
name_candidates = ["Country", "Nation", "NOC", "Team", "NOC_Name", "NOC name"]
name_col = next((c for c in name_candidates if c in df.columns), "__Name__")
if name_col == "__Name__":
    df[name_col] = [f"NOC {i+1}" for i in range(len(df))]

# Top 20 (highest first)
top20 = df[[name_col, medal_col]].sort_values(medal_col, ascending=False).head(20).copy()

# Color mapping: more medals -> darker
norm = Normalize(vmin=top20[medal_col].min(), vmax=top20[medal_col].max())
cmap = sns.color_palette("crest", as_cmap=True)
colors = cmap(norm(top20[medal_col]))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top20[name_col], top20[medal_col], color=colors)
ax.invert_yaxis()  # highest at top

# Value labels
max_val = top20[medal_col].max()
for i, val in enumerate(top20[medal_col]):
    ax.text(val + max_val * 0.01, i, str(int(val)), va="center", fontsize=9)

# Labels & title
ax.set_xlabel("Total Medals Won")
ax.set_ylabel("NOC")
ax.set_title("Top 20 NOCs by Total Medals — Tokyo 2020", pad=14)

# Keep axes; remove all grids
ax.grid(False)
# Ensure left & bottom spines visible (thicker); hide top/right
for spine in ["left", "bottom"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.2)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Make sure tick marks show
ax.tick_params(axis="x", which="both", length=4, width=1)
ax.tick_params(axis="y", which="both", length=4, width=1)

plt.tight_layout()
plt.savefig("top20_total_medals_axes_visible.png", dpi=300, bbox_inches="tight")
plt.show()
