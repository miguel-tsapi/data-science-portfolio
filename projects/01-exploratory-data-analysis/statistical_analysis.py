"""
PROJECT 01: TITANIC STATISTICAL ANALYSIS

Objective:
Calculate conditional survival probabilities
and analyze relationships between variables.
"""

import pandas as pd
import seaborn as sns


# 1. LOAD DATASET

df = sns.load_dataset("titanic")


# 2. CONDITIONAL PROBABILITIES BY SEX

print("\n===== SURVIVAL PROBABILITY BY SEX =====")

survival_by_sex = df.groupby("sex", observed=True)["survived"].mean()

for sex, probability in survival_by_sex.items():
    print(f"P(Survival | {sex}) = {probability:.4f} ({probability:.2%})")


# 3. CONDITIONAL PROBABILITIES BY CLASS

print("\n===== SURVIVAL PROBABILITY BY CLASS =====")

survival_by_class = df.groupby("class", observed=True)["survived"].mean()

for passenger_class, probability in survival_by_class.items():
    print(
        f"P(Survival | {passenger_class}) = "
        f"{probability:.4f} ({probability:.2%})"
    )


# 4. CONTINGENCY TABLE: SEX AND SURVIVAL

print("\n===== CONTINGENCY TABLE: SEX AND SURVIVAL =====")

sex_table = pd.crosstab(
    df["sex"],
    df["survived"],
    margins=True
)

print(sex_table)


# 5. CONDITIONAL FREQUENCIES BY SEX

print("\n===== SURVIVAL DISTRIBUTION BY SEX =====")

sex_distribution = pd.crosstab(
    df["sex"],
    df["survived"],
    normalize="index"
)

print((sex_distribution * 100).round(2))


# 6. COMBINED ANALYSIS: SEX AND CLASS

print("\n===== SURVIVAL BY SEX AND CLASS =====")

combined_analysis = (
    df.groupby(["sex", "class"], observed=True)["survived"]
    .agg(["count", "mean"])
)

combined_analysis["mean"] = combined_analysis["mean"] * 100

combined_analysis = combined_analysis.rename(
    columns={"mean": "survival_rate_percent"}
)

print(combined_analysis.round(2))


print("\nStatistical analysis completed.")


# ==========================================
# CHI-SQUARE TEST OF INDEPENDENCE
# ==========================================

from scipy.stats import chi2_contingency
import numpy as np

# Create contingency table
contingency_table = pd.crosstab(
    df["sex"],
    df["survived"]
)

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(
    contingency_table
)

# Calculate Cramer's V
n = contingency_table.to_numpy().sum()

cramers_v = np.sqrt(
    chi2 / (n * min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1
    ))
)

# Display results
print("\n===== CHI-SQUARE TEST =====")

print("\nObserved frequencies:")
print(contingency_table)

print("\nExpected frequencies:")
print(pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
).round(2))

print("\nChi-square statistic:", round(chi2, 4))
print("Degrees of freedom:", dof)
print("P-value:", p_value)
print("Cramer's V:", round(cramers_v, 4))

if p_value < 0.05:
    print("\nConclusion: Reject H0.")
    print("There is a statistically significant association")
    print("between sex and survival.")
else:
    print("\nConclusion: Do not reject H0.")
    print("There is insufficient evidence of an association.")
