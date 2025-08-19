# ZINB on dataset1.xlsx 
# Mean model (count part):
#   log(u_i) = a + b*log(GDP_i) + c*log(Pop_i) + d*HDI_i + e*log(Athletes_i)
#   where u_i = E(total_i)
# Zero-inflation (inflation part): uses the same set by default (logit link).
# Requirements: pip install pandas numpy statsmodels openpyxl
# All variables

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Coerce to numeric safely: strip spaces, remove commas, set errors to NaN."""
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce"
    )

# 1) load & clean
file_path = "dataset1.xlsx"
df = pd.read_excel(file_path)

needed = ["Totalmedals", "GDP", "Population", "HDI", "Athletes"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# cast to numeric
for c in needed:
    df[c] = to_numeric_clean(df[c])

# drop NAs in key fields
before = len(df)
df = df.dropna(subset=needed).copy()
print(f"[Info] Dropped {before - len(df)} rows with NA in key fields.")

# require positive values for logs
for c in ["GDP", "Population", "Athletes"]:
    bad = (df[c] <= 0).sum()
    if bad:
        print(f"[Warn] {bad} rows have non-positive {c}; removed.")
        df = df[df[c] > 0].copy()

# 2) features 
df["log_GDP"] = np.log(df["GDP"])
df["log_Pop"] = np.log(df["Population"])
df["log_Athletes"] = np.log(df["Athletes"])

y = df["Totalmedals"].astype(int).values

# Count part 
X = sm.add_constant(df[["log_GDP", "log_Pop", "HDI", "log_Athletes"]])

# Inflation part
Z = sm.add_constant(df[["log_GDP", "log_Pop", "HDI", "log_Athletes"]])

# 3) fit ZINB
print("[Fit] ZINB with full inflation set...")
model = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=Z, inflation="logit")

try:
    res = model.fit(method="lbfgs", maxiter=600, disp=False)
except Exception as e1:
    print(f"[Warn] lbfgs failed ({e1}). Retrying with bfgs...")
    try:
        res = model.fit(method="bfgs", maxiter=600, disp=False)
    except Exception as e2:
        print(f"[Warn] bfgs failed ({e2}). Retrying with reduced inflation set...")
        # fallback: simplify inflation part to improve stability
        Z_small = sm.add_constant(df[["log_GDP", "log_Athletes"]])
        model_small = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=Z_small, inflation="logit")
        res = model_small.fit(method="lbfgs", maxiter=600, disp=False)

print(res.summary())

# 4) extract coefficients
coef = res.params
a = coef.get("const", np.nan)
b = coef.get("log_GDP", np.nan)
c = coef.get("log_Pop", np.nan)
d = coef.get("HDI", np.nan)
e = coef.get("log_Athletes", np.nan)
alpha = coef.get("alpha", np.nan)

print("\n=== Count-part coefficients (your a, b, c, d, e) ===")
print(f"a (const)       = {a:.6f}")
print(f"b (log_GDP)     = {b:.6f}")
print(f"c (log_Pop)     = {c:.6f}")
print(f"d (HDI)         = {d:.6f}")
print(f"e (log_Athletes)= {e:.6f}")
print(f"alpha (NB dispersion) = {alpha:.6f}")

# Incidence Rate Ratios (IRR) for interpretability
irr = np.exp([a, b, c, d, e])
print("\nIRR (exp(coeff)) for count part [const, log_GDP, log_Pop, HDI, log_Athletes]:")
print(irr)

# 5) prediction: mu (count mean), pi (zero prob), overall E[Y], std residual
# mu = E[Y | count process] from your count equation
mu = np.exp(a + b*df["log_GDP"].values + c*df["log_Pop"].values
            + d*df["HDI"].values + e*df["log_Athletes"].values)

# Zero probability pi from model (statsmodels supports which='prob-zero')
pi = res.predict(exog=X, exog_infl=Z, which="prob-zero")

# Overall expected medals:
EY = (1.0 - pi) * mu

# ZINB total variance: Var(Y) = (1-pi)*(mu + alpha*mu^2) + pi*(1-pi)*mu^2
alpha_hat = float(alpha) if np.isfinite(alpha) else float(res.params.get("alpha"))
var_nb = mu + alpha_hat * (mu**2)
var_y = (1.0 - pi) * var_nb + pi * (1.0 - pi) * (mu**2)

# Standardized (Pearson) residual: (y - E[Y]) / sqrt(Var(Y))
# add a tiny epsilon to avoid division by zero
eps = 1e-12
std_resid = (y - EY) / np.sqrt(np.maximum(var_y, eps))

out = pd.DataFrame({
    "Code": df.get("Code", pd.Series(index=df.index, dtype=object)),
    "Team": df.get("Team", pd.Series(index=df.index, dtype=object)),
    "Totalmedals": y,
    "log_GDP": df["log_GDP"],
    "log_Pop": df["log_Pop"],
    "HDI": df["HDI"],
    "log_Athletes": df["log_Athletes"],
    "pi_zero": pi,
    "mu_count": mu,
    "E_total": EY,
    "std_resid": std_resid,           
})

out.to_csv("zinb_predictions.csv", index=False)
print("\n[Saved] Predictions (incl. standardized residual) to 'zinb_predictions.csv'")


# 6) quick functions 
def predict_country(gdp, pop, hdi, athletes):
    """Return (pi_zero, mu_count, E_total) for one country."""
    lg, lp, la = np.log(gdp), np.log(pop), np.log(athletes)
    eta_infl = res.params.get("inflate_const", 0.0)
    for name, val in [("inflate_log_GDP", lg), ("inflate_log_Pop", lp),
                      ("inflate_HDI", hdi), ("inflate_log_Athletes", la)]:
        if name in res.params:
            eta_infl += res.params[name] * val
    pi0 = np.exp(eta_infl) / (1.0 + np.exp(eta_infl))
    mu_c = np.exp(a + b*lg + c*lp + d*hdi + e*la)
    return pi0, mu_c, (1.0 - pi0) * mu_c


# 7) four-panel plots: Predicted medals vs each covariate =====
import matplotlib.pyplot as plt

# If E_total not in df (should be), compute it from above objects
if "E_total" not in df.columns:
    mu = np.exp(a + b*df["log_GDP"].values + c*df["log_Pop"].values
                + d*df["HDI"].values + e*df["log_Athletes"].values)
    pi = res.predict(exog=X, exog_infl=Z, which="prob-zero")
    df["E_total"] = (1.0 - pi) * mu

# Hold-others-constant (medians) partial-effect curve
med = {
    "log_GDP": float(df["log_GDP"].median()),
    "log_Pop": float(df["log_Pop"].median()),
    "HDI": float(df["HDI"].median()),
    "log_Athletes": float(df["log_Athletes"].median()),
}

def add_const(dfX: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(dfX, has_constant="add")

def pct_grid(series: pd.Series, lo=2.5, hi=97.5, n=120):
    qlo, qhi = np.percentile(series, [lo, hi])
    return np.linspace(qlo, qhi, n)

def partial_curve(var_name: str, grid_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (x_grid, E[Y] grid) holding other covariates at medians."""
    tmp = pd.DataFrame({
        "log_GDP": np.full_like(grid_vals, med["log_GDP"], dtype=float),
        "log_Pop": np.full_like(grid_vals, med["log_Pop"], dtype=float),
        "HDI": np.full_like(grid_vals, med["HDI"], dtype=float),
        "log_Athletes": np.full_like(grid_vals, med["log_Athletes"], dtype=float),
    })
    tmp[var_name] = grid_vals
    Xg = add_const(tmp[["log_GDP", "log_Pop", "HDI", "log_Athletes"]])
    Zg = add_const(tmp[["log_GDP", "log_Pop", "HDI", "log_Athletes"]])

    # count mean via your fitted coefficients; zero prob via model prediction
    mu_g = np.exp(
        a + b*Xg["log_GDP"].values + c*Xg["log_Pop"].values
        + d*Xg["HDI"].values + e*Xg["log_Athletes"].values
    )
    pi_g = res.predict(exog=Xg, exog_infl=Zg, which="prob-zero")
    EY_g = (1.0 - pi_g) * mu_g
    return grid_vals, EY_g

# Build grids for each axis
g_logGDP = pct_grid(df["log_GDP"])
g_logPop = pct_grid(df["log_Pop"])
g_HDI    = pct_grid(df["HDI"])
g_logAth = pct_grid(df["log_Athletes"])

x1, y1 = partial_curve("log_GDP",      g_logGDP)
x2, y2 = partial_curve("log_Pop",      g_logPop)
x3, y3 = partial_curve("HDI",          g_HDI)
x4, y4 = partial_curve("log_Athletes", g_logAth)

# Plot 2x2 panels
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes = axes.ravel()

def scatter_panel(ax, x, y, xlabel, title):
    ax.scatter(x, y, s=22, alpha=0.85)      # data points
    ax.plot(*title_to_curve[title], linestyle="--", linewidth=2,
            label="Partial effect (others at medians)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted Medal Count")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)

# Map titles to curves for convenient plotting
title_to_curve = {
    "A) Predicted vs log(GDP)":        (x1, y1),
    "B) Predicted vs log(Population)": (x2, y2),
    "C) Predicted vs HDI":             (x3, y3),
    "D) Predicted vs log(Athletes)":   (x4, y4),
}

scatter_panel(axes[0], df["log_GDP"],       df["E_total"], "log(GDP)",        "A) Predicted vs log(GDP)")
scatter_panel(axes[1], df["log_Pop"],       df["E_total"], "log(Population)", "B) Predicted vs log(Population)")
scatter_panel(axes[2], df["HDI"],           df["E_total"], "HDI",             "C) Predicted vs HDI")
scatter_panel(axes[3], df["log_Athletes"],  df["E_total"], "log(Athletes)",   "D) Predicted vs log(Athletes)")

# Optional: annotate top-5 predicted NOCs if "Team" exists
if "Team" in df.columns:
    topk = df.nlargest(5, "E_total")
    for _, r in topk.iterrows():
        for ax, xname in zip(axes, ["log_GDP","log_Pop","HDI","log_Athletes"]):
            ax.annotate(str(r["Team"]), (r[xname], r["E_total"]),
                        xytext=(3,3), textcoords="offset points", fontsize=8)

plt.savefig("four_panels_zinb.png", dpi=300, bbox_inches="tight")
plt.show()
print("[Saved] four_panels_zinb.png")


