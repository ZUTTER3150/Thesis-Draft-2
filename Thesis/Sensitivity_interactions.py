
# ZINB Interaction Sensitivity (manual design matrices)

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

DATA_PATH = Path("zinb_predictions.csv")
OUT_CSV   = Path("zinb_interactions_comparison.csv")
OUT_TEX   = Path("zinb_interactions_comparison.tex")

pd.set_option("display.float_format", "{:.3f}".format)

# 1) Load & basic checks 
df = pd.read_csv(DATA_PATH)
need = ["Totalmedals","log_GDP","log_Pop","HDI","log_Athletes"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only needed cols to avoid accidental NA propagation
df = df[["Team","Totalmedals","log_GDP","log_Pop","HDI","log_Athletes"]].copy()

# Numeric cleaning
for c in ["Totalmedals","log_GDP","log_Pop","HDI","log_Athletes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Totalmedals","log_GDP","log_Pop","HDI","log_Athletes"])

# Outcome as non-negative integer
df["Totalmedals"] = df["Totalmedals"].clip(lower=0).round().astype(int)

if len(df) == 0:
    raise ValueError("No rows available after cleaning; check data.")

# 2) Center variables & build interactions 
for v in ["log_GDP","log_Pop","log_Athletes","HDI"]:
    df[f"c_{v}"] = df[v] - df[v].mean()

df["int_GDPxPop"] = df["c_log_GDP"] * df["c_log_Pop"]
df["int_AthxHDI"] = df["c_log_Athletes"] * df["c_HDI"]

# 3) Function to fit ZINB with manual matrices
def fit_zinb_manual(data, count_vars, infl_vars, label):
    """
    count_vars: list of column names for the count component
    infl_vars : list of column names for the inflation component (logit)
    Both components will include an explicit constant.
    """
    y = data["Totalmedals"].values

    X = data[count_vars].copy()
    X = sm.add_constant(X, has_constant="add")     # -> column name 'const'

    Z = data[infl_vars].copy()
    Z = sm.add_constant(Z, has_constant="add")     # -> column name 'const'
    # rename inflation constant to avoid confusion in output
    Z = Z.rename(columns={"const":"inflate_const"})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Try several optimizers for robustness
        for m in ("bfgs","lbfgs","newton"):
            try:
                model = ZeroInflatedNegativeBinomialP(
                    endog=y,
                    exog=X,
                    exog_infl=Z,
                    inflation="logit",
                    p=2,            # NB2
                    maxiter=500
                )
                res = model.fit(disp=False, method=m)
                return res, X.columns.tolist(), Z.columns.tolist()
            except Exception as e:
                last_err = e
    raise RuntimeError(f"[{label}] ZINB fit failed. Last error: {type(last_err).__name__}: {last_err}")

def collect_stats(label, res, Xcols):
    # Predictions of mean
    mu = res.predict(which="mean")
    rmse = float(np.sqrt(np.mean((df["Totalmedals"].values - mu)**2)))

    params = pd.Series(res.params, index=res.params.index)

    # Count-part coefficients are exactly the columns in X (including 'const')
    irr_rows = {}
    for name in Xcols:
        if name == "const":
            continue
        if name in params:
            b = float(params[name])
            irr_rows[f"b[{name}]"] = b
            irr_rows[f"IRR[{name}]"] = float(np.exp(b))

    out = {
        "Model": label,
        "LogLik": float(res.llf),
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "RMSE": rmse,
        "alpha": float(params.get("alpha", np.nan)),
    }
    out.update(irr_rows)
    return out

# 4) Specifications 
base_count_vars = ["log_GDP","log_Pop","HDI","log_Athletes"]
base_infl_vars  = ["log_GDP","log_Pop","HDI","log_Athletes"]

specs = [
    ("Baseline (no interactions)", base_count_vars, base_infl_vars),
    ("Add GDPxPop", base_count_vars + ["int_GDPxPop"], base_infl_vars),
    ("Add AthletesxHDI", base_count_vars + ["int_AthxHDI"], base_infl_vars),
    ("Add both interactions", base_count_vars + ["int_GDPxPop","int_AthxHDI"], base_infl_vars),
]

# 5) Fit & collect
rows = []
for label, cvars, ivars in specs:
    try:
        res, Xcols, Zcols = fit_zinb_manual(df, cvars, ivars, label)
        rows.append(collect_stats(label, res, Xcols))
    except Exception as e:
        rows.append({"Model": label, "error": str(e)})

cmp = pd.DataFrame(rows)

# Order columns for readability
front = ["Model","LogLik","AIC","BIC","RMSE","alpha"]
irr_cols = [c for c in cmp.columns if c.startswith("IRR[")]
other = [c for c in cmp.columns if c not in front + irr_cols]
cmp = cmp[[c for c in front if c in cmp.columns] + sorted(irr_cols) + other]

if "AIC" in cmp.columns:
    cmp = cmp.sort_values("AIC", na_position="last").reset_index(drop=True)

print("\n=== ZINB Interaction Sensitivity (manual design) ===\n")
print(cmp.to_string(index=False))

cmp.to_csv(OUT_CSV, index=False)
print(f"\n[Saved] CSV -> {OUT_CSV.resolve()}")

# 6) LaTeX 
try:
    tex = cmp.to_latex(index=False, float_format="%.3f",
                       caption="ZINB comparison with interaction terms (manual design matrices).",
                       label="tab:zinb_interactions_manual")
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"[Saved] LaTeX -> {OUT_TEX.resolve()}")
except Exception as e:
    print(f"[Warn] LaTeX save failed: {e}")
