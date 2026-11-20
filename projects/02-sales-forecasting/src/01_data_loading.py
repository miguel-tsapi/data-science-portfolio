
"""
Project 02 - Sales Forecasting
Script 01: Data Loading and Initial Inspection

This script loads and inspects the original Kaggle datasets.
No data cleaning or transformation is performed here.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. DEFINE PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# 2. DEFINE DATASET FILES
# --------------------------------------------------

FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
    "sample_submission.csv"
]


# --------------------------------------------------
# 3. LOAD AND INSPECT DATASETS
# --------------------------------------------------

for filename in FILES:

    file_path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"DATASET: {filename}")
    print("=" * 65)

    # Check whether the file exists
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        continue

    # Load dataset
    df = pd.read_csv(file_path)

    # Convert date column when available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Dataset dimensions
    print("\n1. Dataset dimensions")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # First observations
    print("\n2. First five observations")
    print(df.head())

    # Data types
    print("\n3. Data types")
    print(df.dtypes)

    # Missing values
    print("\n4. Missing values")

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values detected.")
    else:
        print(missing_values.sort_values(ascending=False))

    # Duplicate rows
    print("\n5. Duplicate rows")
    print(f"Number of duplicates: {df.duplicated().sum():,}")

    # Date coverage
    if "date" in df.columns:
        print("\n6. Date coverage")
        print(f"Minimum date: {df['date'].min()}")
        print(f"Maximum date: {df['date'].max()}")

    # Numerical descriptive statistics
    print("\n7. Numerical descriptive statistics")
    print(df.describe().T)

    print(f"\nInspection completed for {filename}.")
    
    
    # --------------------------------------------------
    # 8. ADDITIONAL DATA QUALITY CHECKS
    # --------------------------------------------------

    if filename in ["train.csv", "test.csv"]:

        print("\n8. Temporal and categorical checks")

        print(f"Unique dates: {df['date'].nunique()}")
        print(f"Unique stores: {df['store_nbr'].nunique()}")
        print(f"Unique product families: {df['family'].nunique()}")

        # Check missing calendar dates
        expected_dates = pd.date_range(
            start=df["date"].min(),
            end=df["date"].max(),
            freq="D"
        )

        missing_dates = expected_dates.difference(
            pd.DatetimeIndex(df["date"].unique())
        )

        print(f"Missing calendar dates: {len(missing_dates)}")
        print(missing_dates.tolist()[:10])

        # Check business key uniqueness
        key_duplicates = df.duplicated(
            subset=["date", "store_nbr", "family"]
        ).sum()

        print(f"Duplicate business keys: {key_duplicates:,}")

        # Analyze sales distribution
        if "sales" in df.columns:

            print("\n9. Sales distribution")

            print(f"Zero sales: {(df['sales'] == 0).sum():,}")
            print(f"Negative sales: {(df['sales'] < 0).sum():,}")
            print(f"Positive sales: {(df['sales'] > 0).sum():,}")

            print(f"Minimum sales: {df['sales'].min()}")
            print(f"Maximum sales: {df['sales'].max()}")

    # Additional checks for oil prices
    if filename == "oil.csv":

        print("\n8. Oil price missing dates")

        missing_oil = df.loc[
            df["dcoilwtico"].isna(), "date"
        ]

        print(f"Missing oil prices: {len(missing_oil)}")
        print(missing_oil.to_list())


    # Release the current dataset from memory
    del df


print("\n" + "=" * 65)
print("INITIAL DATA INSPECTION COMPLETED")
print("=" * 65)
