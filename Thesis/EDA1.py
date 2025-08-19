import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths 
DATA_PATH = r"C:\Users\zutte\OneDrive\桌面\new-dissertation\dataset1.xlsx"
SAVE_DIR  = r"C:\Users\zutte\OneDrive\桌面\new-dissertation"  
os.makedirs(SAVE_DIR, exist_ok=True)

# Load data 
df = pd.read_excel(DATA_PATH)

# Ensure numeric type for Totalmedals
df["Totalmedals"] = pd.to_numeric(df["Totalmedals"], errors="coerce")

#  Boxplot of Totalmedals 
plt.figure(figsize=(6, 2.8))
plt.boxplot(df["Totalmedals"].dropna(), vert=False)
plt.xlabel("Total Medals")
plt.tight_layout()
box_path = os.path.join(SAVE_DIR, "totalmedals_box.png")
plt.savefig(box_path, dpi=300, bbox_inches="tight")
plt.close()

# Zero-value proportion (zero inflation check) 
zero_count = int((df["Totalmedals"] == 0).sum())
total_teams = int(df["Totalmedals"].notna().sum())
zero_ratio = zero_count / total_teams if total_teams > 0 else 0.0
print(f"Zero-value proportion: {zero_ratio:.2%} ({zero_count} out of {total_teams} teams)")

# Extreme values via IQR rule 
q1 = df["Totalmedals"].quantile(0.25)
q3 = df["Totalmedals"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

extreme_df = df[df["Totalmedals"] > upper_bound].copy()
extreme_df = extreme_df[["Team", "Totalmedals"]].sort_values("Totalmedals", ascending=False)

print("\nExtreme values (IQR rule, above upper bound):")
if extreme_df.empty:
    print("None detected.")
else:
    print(extreme_df.to_string(index=False))

# Save a CSV of extreme teams 
extreme_csv = os.path.join(SAVE_DIR, "extreme_totalmedals_teams.csv")
extreme_df.to_csv(extreme_csv, index=False)

print(f"\nSaved figures:\n - {hist_path}\n - {box_path}")
print(f"Extreme teams CSV: {extreme_csv}")
