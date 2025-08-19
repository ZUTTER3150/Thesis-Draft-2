import pandas as pd


df = pd.read_excel("dataset1.xlsx")  # Change to your file path if needed


name_col = "Team"

# Variables to analyze
vars_of_interest = ["GDP", "Population", "HDI", "Athletes", "Totalmedals"]

# Ensure numeric dtypes
for col in vars_of_interest:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Descriptive statistics table 
desc_stats = pd.DataFrame({
    "Mean": df[vars_of_interest].mean(),
    "Median": df[vars_of_interest].median(),
    "Std Dev": df[vars_of_interest].std(),
    "Q1 (25%)": df[vars_of_interest].quantile(0.25),
    "Q3 (75%)": df[vars_of_interest].quantile(0.75),
    "Min": df[vars_of_interest].min(),
    "Max": df[vars_of_interest].max(),
    "Count (non-missing)": df[vars_of_interest].count()
})

# Round for readability
desc_stats = desc_stats.round(3)

# Save as CSV
desc_stats.to_csv("univariate_descriptive_stats_with_medals.csv")

print("Descriptive statistics table saved as 'univariate_descriptive_stats_with_medals.csv'")
print(desc_stats)

# Outlier detection using IQR method 
outliers = {}
for col in vars_of_interest:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers[col] = df[(df[col] < lower) | (df[col] > upper)][[name_col, col]].sort_values(col, ascending=False)
    if not outliers[col].empty:
        outliers[col].to_csv(f"outliers_{col}.csv", index=False)
        print(f"\nIQR outliers for {col} saved as 'outliers_{col}.csv'")
        print(outliers[col])

# Special case: High GDP & Low Population 
gdp_hi_threshold = df["GDP"].quantile(0.90)
pop_lo_threshold = df["Population"].quantile(0.10)
high_gdp_low_pop = df[
    (df["GDP"] >= gdp_hi_threshold) & (df["Population"] <= pop_lo_threshold)
][[name_col, "GDP", "Population"]]

if not high_gdp_low_pop.empty:
    high_gdp_low_pop.to_csv("high_GDP_low_population.csv", index=False)
    print("\nHigh GDP & Low Population (>=90th GDP & <=10th Pop) saved as 'high_GDP_low_population.csv'")
    print(high_gdp_low_pop)
