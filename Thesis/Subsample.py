
# Subsample Robustness for ZINB / NB2 medal model

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, NegativeBinomialP

# 0) Settings
COUNT_FORMULA = "Totalmedals ~ log_GDP + log_Pop + HDI + log_Athletes"
INFL_FORMULA  = "~ log_GDP + log_Pop + HDI + log_Athletes"   # one-sided for inflation via dmatrix()

RESULTS_CSV = Path("robustness_subsamples_results.csv")
RESULTS_TEX = Path("robustness_subsamples_results.tex")

# 1) Load and basic checks
PATH = Path("zinb_predictions.csv")
if not PATH.exists():
    raise FileNotFoundError(f"Data file not found: {PATH.resolve()}")

df_raw = pd.read_csv(PATH)

required = {"Team", "Totalmedals", "log_GDP", "log_Pop", "HDI", "log_Athletes"}
missing = required - set(df_raw.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}. "
                     f"Expected at least: {sorted(required)}")

# 1.1) Cleaning tailored to this dataset
df = df_raw.copy()
df["Team"] = df["Team"].astype(str)

numeric_cols = ["Totalmedals", "log_GDP", "log_Pop", "HDI", "log_Athletes"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

before_n = len(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=numeric_cols)

# Ensure nonnegative integers for the count outcome
df = df[df["Totalmedals"] >= 0].copy()
df["Totalmedals"] = df["Totalmedals"].round().astype(int)

after_n = len(df)

has_region = "Region" in df.columns

print("=== Data diagnostics ===")
print(f"Rows in raw file: {before_n}")
print(f"Rows after cleaning (finite & non-missing in key columns): {after_n}")
print(f"Zero-medal rows: {(df['Totalmedals'] == 0).sum()}")
if has_region:
    print(f"Regions detected: {sorted(pd.Series(df['Region'].dropna().unique()).astype(str).tolist())}")
print()


# 2) Model helpers
def _common_sanity_checks(data: pd.DataFrame):
    if len(data) < 20:
        raise RuntimeError("Too few observations (<20) in this subset for a stable fit.")
    if data["Totalmedals"].nunique() < 2:
        raise RuntimeError("Outcome has no variation in this subset.")

def _collect_common_stats(res, data: pd.DataFrame, label: str, model_name: str) -> dict:
    """Collect likelihood-based fit stats and selected coefficients."""
    mu_hat = res.predict()  # mean of Y
    rmse   = float(np.sqrt(np.mean((data["Totalmedals"].values - mu_hat)**2)))
    coef = pd.Series(res.params, index=res.params.index)

    out = {
        "Subset": label,
        "Model": model_name,
        "n": int(len(data)),
        "LogLik": float(res.llf),
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "RMSE": rmse,
        "alpha": float(coef.get("alpha", np.nan)),
    }

    # Count-part (NB) coefficients and IRRs if names match
    count_prefix = "x"        # statsmodels names NB coefficients like x_log_GDP etc.
    infl_prefix  = "inflate"  # only present for ZINB

    def pick(prefix: str):
        return {k.replace(prefix + "_", ""): v
                for k, v in coef.items() if k.startswith(prefix + "_")}

    count_coefs = pick(count_prefix)
    infl_coefs  = pick(infl_prefix)

    for nm in ["Intercept", "log_GDP", "log_Pop", "HDI", "log_Athletes"]:
        if nm in count_coefs:
            b = float(count_coefs[nm])
            out[f"b[{nm}]"]   = b
            out[f"IRR[{nm}]"] = float(np.exp(b))

    if "Intercept" in infl_coefs:
        out["inflate_Intercept"] = float(infl_coefs["Intercept"])

    return out

def fit_zinb_and_collect(data: pd.DataFrame, label: str) -> dict:
    """Fit ZINB (NB2 count + logit inflation) and collect stats."""
    _common_sanity_checks(data)
    import patsy
    y, X = patsy.dmatrices(COUNT_FORMULA, data, return_type="dataframe")
    Z    = patsy.dmatrix(INFL_FORMULA, data, return_type="dataframe")

    ok = y.notna().all(axis=1) & X.notna().all(axis=1) & Z.notna().all(axis=1)
    y, X, Z = y.loc[ok], X.loc[ok], Z.loc[ok]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedNegativeBinomialP(
            endog=y, exog=X, exog_infl=Z, inflation="logit", p=2, maxiter=400
        )
        try:
            res = model.fit(disp=False, method="bfgs")
        except Exception:
            res = model.fit(disp=False, method="lbfgs")

    return _collect_common_stats(res, data, label, model_name="ZINB")

def fit_nb2_and_collect(data: pd.DataFrame, label: str) -> dict:
    """Fit NB2 (no inflation) and collect stats."""
    _common_sanity_checks(data)
    import patsy
    y, X = patsy.dmatrices(COUNT_FORMULA, data, return_type="dataframe")

    ok = y.notna().all(axis=1) & X.notna().all(axis=1)
    y, X = y.loc[ok], X.loc[ok]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NegativeBinomialP(endog=y, exog=X, p=2, maxiter=400)  # NB2
        try:
            res = model.fit(disp=False, method="bfgs")
        except Exception:
            res = model.fit(disp=False, method="lbfgs")

    return _collect_common_stats(res, data, label, model_name="NB2")

# 3) Define subsamples
subsets = []
subsets.append(("Full sample", df.copy()))

TOP_N = 3
topN_teams = (
    df.sort_values("Totalmedals", ascending=False)
      .head(TOP_N)["Team"].astype(str).tolist()
)
subsets.append((f"Drop top {TOP_N} medal countries", df[~df["Team"].isin(topN_teams)].copy()))

try:
    q25 = df["log_Athletes"].quantile(0.25)
    subsets.append(("Drop small (bottom 25% by Athletes)", df[df["log_Athletes"] > q25].copy()))
except Exception:
    pass

if (df["Totalmedals"] == 0).any():
    subsets.append(("Drop zero-medal nations", df[df["Totalmedals"] > 0].copy()))

if has_region:
    for reg in sorted(pd.Series(df["Region"].dropna().unique()).astype(str).tolist()):
        subsets.append((f"Drop region: {reg}", df[df["Region"].astype(str) != reg].copy()))

# 4) Run all fits with NB2 fallback logic
rows = []
for label, dsub in subsets:
    try:
        # Prefer NB2 for the "Drop zero-medal nations" subset; otherwise try ZINB first
        if label.startswith("Drop zero-medal nations"):
            try:
                row = fit_nb2_and_collect(dsub, label)
            except Exception:
                row = fit_zinb_and_collect(dsub, label)  # rare: still allow ZINB if NB2 fails
        else:
            try:
                row = fit_zinb_and_collect(dsub, label)
            except Exception:
                row = fit_nb2_and_collect(dsub, f"{label}")  # NB2 (fallback)
                row["Model"] = "NB2 (fallback)"
        rows.append(row)
    except Exception as e:
        rows.append({
            "Subset": label,
            "Model": "—",
            "n": int(len(dsub)),
            "LogLik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "RMSE": np.nan,
            "alpha": np.nan,
            "error": str(e),
        })

summary = pd.DataFrame(rows)

# 5) Arrange columns and export
front = ["Subset", "Model", "n", "LogLik", "AIC", "BIC", "RMSE", "alpha"]
coef_cols = [c for c in summary.columns if c.startswith("b[") or c.startswith("IRR[")]
front_present = [c for c in front if c in summary.columns]
other = [c for c in summary.columns if c not in set(front_present + coef_cols)]
summary = summary[front_present + sorted(coef_cols) + other]

summary_sorted = summary.sort_values("AIC", na_position="last").reset_index(drop=True)

pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== ZINB / NB2 Subsample Robustness (sorted by AIC) ===\n")
print(summary_sorted.to_string(index=False))

summary_sorted.to_csv(RESULTS_CSV, index=False)
print(f"\n[Saved] CSV -> {RESULTS_CSV.resolve()}")

try:
    latex_tbl = summary_sorted.rename(columns={
        "Subset":"Subset",
        "Model":"Model",
        "n":"N",
        "LogLik":"LogLik",
        "AIC":"AIC",
        "BIC":"BIC",
        "RMSE":"RMSE",
        "alpha":"alpha"
    }).to_latex(index=False, float_format="%.3f",
                caption="ZINB and NB2 robustness checks across subsamples.",
                label="tab:zinb_nb2_robustness_subsamples")
    Path(RESULTS_TEX).write_text(latex_tbl, encoding="utf-8")
    print(f"[Saved] LaTeX -> {RESULTS_TEX.resolve()}")
except Exception as e:
    print(f"[Warn] Could not save LaTeX: {e}")
