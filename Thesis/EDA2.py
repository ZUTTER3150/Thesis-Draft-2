import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("dataset1.xlsx")

# Ensure medals are numeric
df["Totalmedals"] = pd.to_numeric(df["Totalmedals"], errors="coerce").fillna(0).astype(int)

# Binary flag: True = zero medals, False = non-zero medals
df["ZeroMedals"] = df["Totalmedals"] == 0

# Compute counts explicitly to control the order (Non-Zero first, then Zero)
count_nonzero = (~df["ZeroMedals"]).sum()  # >= 1 medal
count_zero = (df["ZeroMedals"]).sum()      # 0 medals
counts = [count_nonzero, count_zero]

# Plot
colors = ["steelblue", "indianred"]  # Non-zero, Zero medals
labels = ["non-zero-medal NOCs", "zero-medal NOCs"]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, counts, color=colors)

# Annotate values
for bar, val in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val),
             ha="center", va="bottom", fontsize=10)

# No title
plt.ylabel("Number of National Olympic Committees")
plt.ylim(0, max(counts) + 10)

plt.tight_layout()
plt.savefig(r"C:\Users\zutte\OneDrive\桌面\new-dissertation\zero_vs_nonzero_medals.png", dpi=300)
plt.show()
