This repository contains the supplementary materials for the dissertation “Olympic Medal Prediction under a Zero-Inflated Model: Quantitative Analysis of Socioeconomic Development and Demographic Factors”.
It includes the datasets, preprocessing scripts, modeling code, figures, and tables referenced in the thesis.
Thesis-Draft-2/
├─ README.md                 # Project overview (this file)
├─ data/                     # Raw and processed datasets
│   ├─ raw-Athletes.xlsx     # Raw data — number of athletes per NOC (Tokyo 2020)
│   ├─ raw-GDP.xlsx          # Raw data — GDP (World Bank)
│   ├─ raw-HDI.xlsx          # Raw data — Human Development Index (UNDP)
│   ├─ raw-population.xlsx   # Raw data — population size (World Bank)
│   └─ dataset1.xlsx         # Main processed dataset (merged & cleaned)
├─ code/                     # Python scripts for preprocessing, modeling, and visualization
│   ├─ modelselect.py        # Model selection
│   ├─ ZINB01.py             # Main ZINB model implementation + four-panel plots
│   ├─ residual_diagnostics.py   # Residual distribution visualization
│   ├─ EDA1.py               # Box plots
│   ├─ EDA2.py               # Medal distribution plots
│   ├─ EDA3.py               # Top 20 medal-winning countries visualization
│   ├─ EDA4.py               # Descriptive statistics — outlier countries by indicator
│   ├─ EDA5.py               # Correlation analysis & VIF calculation
│   ├─ EDA6.py               # Exploring relationships between predictors and response
│   ├─ Heterogeneity_Ath.py  # Heterogeneity analysis by athlete size
│   ├─ Heterogeneity_GDP.py  # Heterogeneity analysis by GDP
│   ├─ Heterogeneity_HDI.py  # Heterogeneity analysis by HDI
│   ├─ Heterogeneity_POP.py  # Heterogeneity analysis by population size
│   ├─ Othermodels.py        # Sensitivity analysis — ZINB vs alternative models
│   ├─ Sensitivity_interactions.py   # Variable interaction results
│   ├─ sensitivity_figures.py        # Visualization of interaction effects
│   ├─ Alternativevariate.py         # Models with reduced variable sets
│   └─ Subsample.py                  # Subsample robustness checks
├─ results/                  # Model outputs and statistical results
│   ├─ spearman_correlation_matrix.xlsx   # Spearman correlation matrix (numeric results)
│   ├─ spearman_correlation_matrix.png    # Spearman correlation matrix (heatmap figure)
│   ├─ vif_results.xlsx                 # Variance Inflation Factor results
│   ├─ summary_poisson.txt              # Poisson model summary
│   ├─ summary_nb2.txt                  # Negative Binomial 2 model summary
│   ├─ summary_zip.txt                  # Zero-Inflated Poisson model summary
│   ├─ summary_zinb.txt                 # Zero-Inflated Negative Binomial model summary
│   ├─ zinb_predictions.xlsx            # Predicted medal counts (ZINB model)
│   ├─ zinb_specs_irr_long.xlsx         # Incidence Rate Ratios (long format)
│   ├─ zinb_specs_irr_wide.xlsx         # Incidence Rate Ratios (wide format)
│   ├─ zinb_alternative_specs_comparison.xlsx   # ZINB with alternative specifications
│   ├─ zinb_interactions_comparison.xlsx        # ZINB interaction terms results
│   └─ univariate_descriptive_stats_with_medals.xlsx   # Descriptive statistics with medal counts
├─ figures/                  # Figures used in the dissertation
├─ tables/                   # Statistical output tables (LaTeX/CSV/Excel format)
-
