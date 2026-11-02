"""
Project 01: Exploratory Data Analysis

This script performs an initial exploratory analysis
of a CSV dataset.
"""

import sys
from pathlib import Path

import pandas as pd


def inspect_dataset(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print("\n===== DATASET OVERVIEW =====")

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n===== FIRST FIVE ROWS =====")
    print(df.head())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe())


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python analysis.py dataset.csv"
        )
        sys.exit(1)

    inspect_dataset(sys.argv[1])
