
# Fit and compare Poisson, NB2, ZIP, ZINB on dataset1.xlsx

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices, dmatrix
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

# 1) Load & clean data 
file_path = "dataset1.xlsx"
df = pd.read_excel(file_path)

def to_numeric_clean(x):
    # remove commas and whitespace, then convert
    if isinstance(x, str):
        x = re.sub(r"[,\s]", "", x)
    return pd.to_numeric(x, errors="coerce")

for col in ["GDP", "Population", "HDI", "Athletes", "Totalmedals"]:
    df[col] = df[col].apply(to_numeric_clean)

df = df.dropna(subset=["GDP","Population","HDI","Athletes","Totalmedals"]).copy()

# target & transforms
df["y"] = df["Totalmedals"].astype(int)
eps = 1e-9
df["log_GDP"] = np.log(df["GDP"] + eps)
df["log_Pop"] = np.log(df["Population"] + eps)
df["log_Athletes"] = np.log(df["Athletes"] + eps)
# HDI already in [0,1]; keep as-is

# 2) Formulas 
count_formula = "y ~ log_GDP + log_Pop + HDI + log_Athletes"
infl_formula  = "~ log_GDP + log_Pop + HDI + log_Athletes"   # patsy formula for inflation part

# 3) Fit models 
# Poisson (GLM)
poisson_res = smf.glm(formula=count_formula, data=df, family=sm.families.Poisson()).fit()

# NB2 (Negative Binomial, variance = mu + alpha*mu^2)
nb2_res = smf.negativebinomial(formula=count_formula, data=df).fit()

# ZIP/ZINB with explicit design matrices (works across statsmodels versions)
y_mat, X_mat = dmatrices(count_formula, data=df, return_type='dataframe')
X_infl = dmatrix(infl_formula, data=df, return_type='dataframe')

zip_mod = ZeroInflatedPoisson(endog=y_mat, exog=X_mat, exog_infl=X_infl, inflation='logit')
zip_res = zip_mod.fit(method='bfgs', maxiter=500, disp=False)

zinb_mod = ZeroInflatedNegativeBinomialP(endog=y_mat, exog=X_mat, exog_infl=X_infl, inflation='logit')
zinb_res = zinb_mod.fit(method='bfgs', maxiter=500, disp=False)

# 4) Helpers 
def mcfadden_pseudo_r2(res):
    try:
        return 1.0 - (res.llf / res.llnull)
    except Exception:
        return np.nan

def extract_alpha(res):
    for key in getattr(res, "params", pd.Series()).index:
        if key.lower().startswith("alpha"):
            return res.params[key]
    return np.nan

def model_row(name, res, n_obs):
    # unified BIC if attribute missing
    try:
        bic = res.bic
    except Exception:
        k = len(res.params)
        bic = k * np.log(n_obs) - 2 * res.llf
    return {
        "Model": name,
        "Converged": getattr(res, "mle_retvals", {}).get("converged", getattr(res, "converged", np.nan)),
        "LogLik": res.llf,
        "AIC": res.aic,
        "BIC": bic,
        "PseudoR2_McFadden": mcfadden_pseudo_r2(res),
        "alpha": extract_alpha(res),
    }

def count_part_irr(res, infl_prefix="inflate_"):
    params = res.params if hasattr(res, "params") else pd.Series()
    # exclude inflation params (prefixed by 'inflate_') for ZI models
    idx = [k for k in params.index if not k.startswith(infl_prefix)]
    cp = params[idx]
    return pd.DataFrame({"coef": cp, "IRR": np.exp(cp)})

# 5) Comparison table 
n = df.shape[0]
comparison = pd.DataFrame([
    model_row("Poisson (GLM)", poisson_res, n),
    model_row("NB2 (Negative Binomial)", nb2_res, n),
    model_row("ZIP (Zero-Inflated Poisson)", zip_res, n),
    model_row("ZINB (Zero-Inflated NB2)", zinb_res, n),
]).set_index("Model").sort_values("AIC")

print("\n=== Model comparison (sorted by AIC) ===\n")
print(comparison)

# 6) IRR for count parts 
print("\n=== IRR (exp(coef)) — Count component ===\n")
print("\nPoisson IRR:\n", np.exp(poisson_res.params))
print("\nNB2 IRR:\n", np.exp(nb2_res.params))
print("\nZIP IRR (count-part):\n", count_part_irr(zip_res))
print("\nZINB IRR (count-part):\n", count_part_irr(zinb_res))

# 7) Save summaries (optional) 
with open("summary_poisson.txt","w") as f: f.write(str(poisson_res.summary()))
with open("summary_nb2.txt","w") as f: f.write(str(nb2_res.summary()))
with open("summary_zip.txt","w") as f: f.write(str(zip_res.summary()))
with open("summary_zinb.txt","w") as f: f.write(str(zinb_res.summary()))
