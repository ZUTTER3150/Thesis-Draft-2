
# ZINB Robustness: Alternative Covariate Specifications
# This script fits multiple Zero-Inflated Negative Binomial (ZINB)
# specifications that vary the set of covariates, then compares
# model fit and the stability of key effects (IRR).

import re
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix

import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

# 1) Load & clean data
# Adjust the path if needed
file_path = "dataset1.xlsx"
df = pd.read_excel(file_path)

def to_numeric_clean(x):
    # remove thousand separators/spaces, convert to numeric
    if isinstance(x, str):
        x = re.sub(r"[,\s]", "", x)
    return pd.to_numeric(x, errors="coerce")

for col in ["GDP", "Population", "HDI", "Athletes", "Totalmedals"]:
    df[col] = df[col].apply(to_numeric_clean)

# Drop rows with missing values in the required columns
df = df.dropna(subset=["GDP","Population","HDI","Athletes","Totalmedals"]).copy()

# Target and transforms
df["y"] = df["Totalmedals"].astype(int)
eps = 1e-9
df["log_GDP"]      = np.log(df["GDP"] + eps)
df["log_Pop"]      = np.log(df["Population"] + eps)
df["log_Athletes"] = np.log(df["Athletes"] + eps)
# HDI is kept in levels

# 2) Define alternative specifications 
# Each entry defines the count-part formula  and we will
# use the SAME set of covariates for the inflation equation (logit).
SPECS = {
    "Full model":
        "y ~ log_GDP + log_Pop + HDI + log_Athletes",
    "Drop HDI":
        "y ~ log_GDP + log_Pop + log_Athletes",
    "Drop GDP":
        "y ~ log_Pop + HDI + log_Athletes",
    "Drop Population":
        "y ~ log_GDP + HDI + log_Athletes",
    "GDP + Athletes only":
        "y ~ log_GDP + log_Athletes",
}

# 3) Helper functions 
def fit_zinb_formula(count_formula: str, data: pd.DataFrame):
    """
    Fit a ZINB model using the same RHS covariates for the inflation part (logit).
    Uses explicit design matrices so it works across statsmodels versions.
    """
    # count design matrices
    y_mat, X_mat = dmatrices(count_formula, data=data, return_type="dataframe")
    # inflation uses the same RHS as the count equation
    rhs = count_formula.split("~", 1)[1].strip()
    infl_formula = "~ " + rhs
    X_infl = dmatrix(infl_formula, data=data, return_type="dataframe")

    model = ZeroInflatedNegativeBinomialP(
        endog=y_mat, exog=X_mat, exog_infl=X_infl, inflation="logit"
    )
    res = model.fit(method="bfgs", maxiter=500, disp=False)
    return res

def extract_alpha(res):
    # return alpha if present, else NaN
    for key in getattr(res, "params", pd.Series()).index:
        if key.lower().startswith("alpha"):
            return float(res.params[key])
    return np.nan

def count_part_irr(res, infl_prefix="inflate_"):
    """Return IRR = exp(coef) for the count-part only (exclude inflation params)."""
    params = res.params
    idx = [k for k in params.index if not k.startswith(infl_prefix)]
    cp = params[idx]
    irr = np.exp(cp)
    out = pd.DataFrame({"coef": cp, "IRR": irr})
    return out

# 4) Fit all specs
results = {}
for name, formula in SPECS.items():
    res = fit_zinb_formula(formula, df)
    results[name] = res
    print(f"\n=== {name} ===")
    print(res.summary())

# 5) Build comparison table 
def model_row(name, res, n_obs):
    # unified BIC in case attribute missing
    try:
        bic = res.bic
    except Exception:
        k = len(res.params)
        bic = k * np.log(n_obs) - 2 * res.llf
    converged = getattr(res, "mle_retvals", {}).get("converged", getattr(res, "converged", np.nan))
    return {
        "Model": name,
        "Converged": converged,
        "n_params": len(res.params),
        "LogLik": res.llf,
        "AIC": res.aic,
        "BIC": bic,
        "alpha": extract_alpha(res),
    }

n = df.shape[0]
cmp_table = pd.DataFrame([model_row(k, v, n) for k, v in results.items()]) \
              .set_index("Model") \
              .sort_values("AIC")

print("\n=== Model comparison across alternative ZINB specifications (sorted by AIC) ===\n")
pd.set_option("display.float_format", "{:.3f}".format)
print(cmp_table)

# Save comparison table
cmp_table.to_csv("zinb_alternative_specs_comparison.csv")
try:
    cmp_table.to_latex(
        "zinb_alternative_specs_comparison.tex",
        float_format="%.3f",
        caption="ZINB robustness check: alternative covariate specifications.",
        label="tab:zinb_specs_compare",
        bold_rows=True,
    )
except Exception:
    pass

# 6) Collect IRR for key covariates across models 
# We align IRR by variable name, placing NaN where a variable is not included.
irr_panels = []
for name, res in results.items():
    irr = count_part_irr(res)
    irr["Model"] = name
    irr_panels.append(irr.reset_index().rename(columns={"index": "term"}))

irr_long = pd.concat(irr_panels, ignore_index=True)
# Keep only count-part terms (Intercept + covariates)
irr_long = irr_long[~irr_long["term"].str.startswith("inflate_")]

# Wide IRR table: rows = term, columns = model names
irr_wide = irr_long.pivot_table(index="term", columns="Model", values="IRR", aggfunc="first")
irr_wide = irr_wide.loc[[t for t in irr_wide.index if t != "alpha"]]  # just in case

print("\n=== Count-part IRR (exp(coef)) across specifications ===\n")
print(irr_wide)

# Save IRR tables
irr_long.to_csv("zinb_specs_irr_long.csv", index=False)
irr_wide.to_csv("zinb_specs_irr_wide.csv")

try:
    irr_wide.to_latex(
        "zinb_specs_irr_wide.tex",
        float_format="%.3f",
        caption="IRR (count component) across alternative ZINB specifications.",
        label="tab:zinb_specs_irr",
    )
except Exception:
    pass

# 7) (Optional) Minimal textual hints 
print("\nNotes:")
print("- Compare AIC/BIC/LogLik: smaller AIC/BIC indicates better fit.")
print("- Inspect alpha: positive values indicate overdispersion handled by NB2 component.")
print("- Check stability: signs/magnitudes of IRR for log_Athletes, log_GDP, etc.,")
print("  should be qualitatively consistent across specifications if findings are robust.")
