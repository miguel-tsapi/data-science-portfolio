
"""
02_data_preparation.py

Prepare the Store Sales dataset for time series forecasting.

Initial objectives:
1. Load the training data.
2. Aggregate global daily sales.
3. Build a complete daily calendar.
4. Identify missing dates.
5. Compare missing dates with holiday information.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("=" * 60)
print("1. LOADING DATASETS")
print("=" * 60)

train = pd.read_csv(
    RAW_DIR / "train.csv",
    parse_dates=["date"]
)

holidays = pd.read_csv(
    RAW_DIR / "holidays_events.csv",
    parse_dates=["date"]
)

print(f"Training data shape: {train.shape}")
print(f"Holiday data shape: {holidays.shape}")


# ============================================================
# 3. AGGREGATE DAILY GLOBAL SALES
# ============================================================

print("\n" + "=" * 60)
print("2. DAILY GLOBAL SALES AGGREGATION")
print("=" * 60)

daily_sales = (
    train.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Number of observed dates: {len(daily_sales)}")
print(f"Total sales: {daily_sales['sales'].sum():,.2f}")
print(f"Average daily sales: {daily_sales['sales'].mean():,.2f}")
print(f"Minimum daily sales: {daily_sales['sales'].min():,.2f}")
print(f"Maximum daily sales: {daily_sales['sales'].max():,.2f}")

print("\nFirst five observations:")
print(daily_sales.head())


# ============================================================
# 4. BUILD A COMPLETE DAILY CALENDAR
# ============================================================

print("\n" + "=" * 60)
print("3. CALENDAR CONTINUITY CHECK")
print("=" * 60)

full_calendar = pd.date_range(
    start=daily_sales["date"].min(),
    end=daily_sales["date"].max(),
    freq="D"
)

daily_sales = (
    daily_sales
    .set_index("date")
    .reindex(full_calendar)
    .rename_axis("date")
    .reset_index()
)

daily_sales["is_observed"] = daily_sales["sales"].notna()

missing_dates = daily_sales.loc[
    daily_sales["sales"].isna(), "date"
]

print(f"Expected number of calendar days: {len(full_calendar)}")
print(f"Number of missing dates: {len(missing_dates)}")

if len(missing_dates) > 0:
    print("\nMissing dates:")
    print(missing_dates.to_list())
else:
    print("No missing dates detected.")


# ============================================================
# 5. INVESTIGATE MISSING DATES USING HOLIDAY INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("4. MISSING DATES AND HOLIDAY INVESTIGATION")
print("=" * 60)

if len(missing_dates) > 0:

    holiday_matches = holidays[
        holidays["date"].isin(missing_dates)
    ]

    columns_to_display = [
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred"
    ]

    print("\nHoliday records corresponding to missing dates:")

    if not holiday_matches.empty:
        print(
            holiday_matches[columns_to_display]
            .sort_values("date")
            .to_string(index=False)
        )
    else:
        print("No matching holiday records found.")

else:
    print("No missing dates require investigation.")


# ============================================================
# 6. SAVE PREPARED DATA
# ============================================================

output_path = PROCESSED_DIR / "daily_global_sales.csv"

daily_sales.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("5. DATA SAVING")
print("=" * 60)

print(f"Prepared dataset saved to: {output_path}")
print(f"Final dataset shape: {daily_sales.shape}")

print("\nData preparation completed successfully.")
