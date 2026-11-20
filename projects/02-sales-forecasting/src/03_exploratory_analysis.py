
"""
03_exploratory_analysis.py

Exploratory Time Series Analysis
Project 02 - Sales Forecasting
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

print("=" * 60)
print("1. LOADING PREPARED DATA")
print("=" * 60)

df = pd.read_csv(
    PROCESSED_DIR / "daily_global_sales.csv",
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing sales: {df['sales'].isna().sum()}")

# Temporal variables
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["day_name"] = df["date"].dt.day_name()
df["day_of_year"] = df["date"].dt.dayofyear


# ============================================================
# 3. GLOBAL SALES EVOLUTION
# ============================================================

print("\n" + "=" * 60)
print("2. GLOBAL SALES EVOLUTION")
print("=" * 60)

plt.figure(figsize=(15, 6))

plt.plot(
    df["date"],
    df["sales"],
    linewidth=0.8,
    label="Daily sales"
)

plt.title("Daily Global Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales (units)")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "01_daily_sales_evolution.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. TREND ANALYSIS USING A 30-DAY MOVING AVERAGE
# ============================================================

print("\n" + "=" * 60)
print("3. TREND ANALYSIS")
print("=" * 60)

df["rolling_mean_30"] = (
    df["sales"]
    .rolling(window=30, min_periods=30)
    .mean()
)

plt.figure(figsize=(15, 6))

plt.plot(
    df["date"],
    df["sales"],
    alpha=0.25,
    linewidth=0.7,
    label="Daily sales"
)

plt.plot(
    df["date"],
    df["rolling_mean_30"],
    linewidth=2,
    label="30-day moving average"
)

plt.title("Daily Sales and 30-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Sales (units)")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "02_sales_trend_30_days.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. MONTHLY SALES ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("4. MONTHLY SALES ANALYSIS")
print("=" * 60)

monthly_sales = (
    df.dropna(subset=["sales"])
    .groupby(["year", "month"])["sales"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=monthly_sales,
    x="month",
    y="sales",
    hue="year",
    marker="o",
    palette="tab10"
)

plt.title("Average Daily Sales by Month and Year")
plt.xlabel("Month")
plt.ylabel("Average Daily Sales (units)")
plt.xticks(range(1, 13))
plt.legend(title="Year")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "03_monthly_sales_patterns.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. WEEKLY SALES PATTERN
# ============================================================

print("\n" + "=" * 60)
print("5. WEEKLY SALES PATTERN")
print("=" * 60)

weekly_sales = (
    df.dropna(subset=["sales"])
    .groupby("day_name")["sales"]
    .mean()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ])
)

print("\nAverage sales by day of the week:")
print(weekly_sales)

plt.figure(figsize=(11, 5))

sns.barplot(
    x=weekly_sales.index,
    y=weekly_sales.values
)

plt.title("Average Daily Sales by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Average Sales (units)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "04_weekly_sales_pattern.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. SALES DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("6. SALES DISTRIBUTION")
print("=" * 60)

print(df["sales"].describe())

plt.figure(figsize=(11, 5))

sns.histplot(
    data=df,
    x="sales",
    bins=50,
    kde=True
)

plt.title("Distribution of Daily Global Sales")
plt.xlabel("Sales (units)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "05_sales_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. EXTREME VALUES
# ============================================================

print("\n" + "=" * 60)
print("7. EXTREME VALUES ANALYSIS")
print("=" * 60)

valid_sales = df["sales"].dropna()

q1 = valid_sales.quantile(0.25)
q3 = valid_sales.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df["sales"] < lower_bound) |
    (df["sales"] > upper_bound)
]

print(f"Q1: {q1:,.2f}")
print(f"Q3: {q3:,.2f}")
print(f"IQR: {iqr:,.2f}")
print(f"Lower bound: {lower_bound:,.2f}")
print(f"Upper bound: {upper_bound:,.2f}")
print(f"Number of potential outliers: {len(outliers)}")

print("\nHighest sales days:")
print(
    df.nlargest(10, "sales")[["date", "sales"]]
    .to_string(index=False)
)


# ============================================================
# 9. SAVE SUMMARY TABLES
# ============================================================

monthly_sales.to_csv(
    PROCESSED_DIR / "monthly_sales_summary.csv",
    index=False
)

weekly_sales.rename("average_sales").to_csv(
    PROCESSED_DIR / "weekly_sales_summary.csv"
)

outliers[["date", "sales"]].to_csv(
    PROCESSED_DIR / "potential_outliers.csv",
    index=False
)


# ============================================================
# 10. INVESTIGATION OF EXTREME SALES DAYS
# ============================================================

print("\n" + "=" * 60)
print("8. INVESTIGATION OF EXTREME SALES DAYS")
print("=" * 60)

# Load original training data
train = pd.read_csv(
    RAW_DIR / "train.csv",
    parse_dates=["date"]
)

# Load holiday data
holidays = pd.read_csv(
    RAW_DIR / "holidays_events.csv",
    parse_dates=["date"]
)

# Identify the five highest sales days
top_dates = (
    df.nlargest(5, "sales")[["date", "sales"]]
    .copy()
)

# Aggregate original data for each selected date
daily_details = (
    train[train["date"].isin(top_dates["date"])]
    .groupby("date")
    .agg(
        total_sales=("sales", "sum"),
        total_transactions=("store_nbr", "count"),
        stores=("store_nbr", "nunique"),
        product_families=("family", "nunique"),
        promoted_records=("onpromotion", "sum")
    )
    .reset_index()
)

# Retrieve holiday information
holiday_details = holidays[
    holidays["date"].isin(top_dates["date"])
][
    ["date", "type", "locale", "description"]
]

# Combine the results
investigation = daily_details.merge(
    holiday_details,
    on="date",
    how="left"
)

print("\nInvestigation results:")
print(investigation.to_string(index=False))

# Save investigation
investigation.to_csv(
    PROCESSED_DIR / "extreme_sales_investigation.csv",
    index=False
)



# ============================================================
# 11. STORE AND PRODUCT CONTRIBUTION TO EXTREME SALES DAYS
# ============================================================

print("\n" + "=" * 60)
print("9. STORE AND PRODUCT CONTRIBUTION")
print("=" * 60)

# Retrieve the five extreme dates
extreme_dates = top_dates["date"].tolist()

# Filter original observations
extreme_details = train[
    train["date"].isin(extreme_dates)
].copy()

# Aggregate sales by date, store and product family
store_family_sales = (
    extreme_details
    .groupby(["date", "store_nbr", "family"], as_index=False)
    .agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum")
    )
)

# Display the ten highest store-family contributions
for date in extreme_dates:

    print(f"\nDate: {date.date()}")

    daily_contributions = (
        store_family_sales[
            store_family_sales["date"] == date
        ]
        .sort_values("sales", ascending=False)
        .head(10)
    )

    print(
        daily_contributions[
            ["store_nbr", "family", "sales", "onpromotion"]
        ].to_string(index=False)
    )

# Save detailed results
store_family_sales.to_csv(
    PROCESSED_DIR / "extreme_store_family_sales.csv",
    index=False
)



# ============================================================
# 12. INVESTIGATION OF THE HIGHEST INDIVIDUAL CONTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("10. INVESTIGATION OF STORE 45 - GROCERY I")
print("=" * 60)

target_date = pd.Timestamp("2016-04-18")

comparison = train[
    (train["store_nbr"] == 45) &
    (train["family"] == "GROCERY I") &
    (train["date"].between(
        target_date - pd.Timedelta(days=7),
        target_date + pd.Timedelta(days=7)
    ))
].copy()

comparison = comparison.sort_values("date")

print("\nDaily sales around April 18, 2016:")

print(
    comparison[
        ["date", "sales", "onpromotion"]
    ].to_string(index=False)
)

print("\nStatistical summary:")
print(comparison["sales"].describe())

# Save the comparison
comparison.to_csv(
    PROCESSED_DIR / "store45_grocery_investigation.csv",
    index=False
)



print("\nStore-family investigation saved successfully.")


print("\n" + "=" * 60)
print("EXPLORATORY ANALYSIS COMPLETED")
print("=" * 60)
print(f"Figures saved in: {FIGURES_DIR}")
