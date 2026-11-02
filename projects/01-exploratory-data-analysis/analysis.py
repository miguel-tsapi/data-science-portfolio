"""
PROJECT 01: TITANIC EXPLORATORY DATA ANALYSIS

Objective:
Explore passenger data and identify factors
associated with survival.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. LOAD DATASET
print("Loading Titanic dataset...")

df = sns.load_dataset("titanic")


# 2. INITIAL EXPLORATION
print("\n===== DATASET DIMENSIONS =====")
print(df.shape)

print("\n===== FIRST FIVE OBSERVATIONS =====")
print(df.head())

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe(include="all"))


# 3. DATA QUALITY
print("\n===== MISSING VALUES =====")

missing = pd.DataFrame({
    "Missing values": df.isnull().sum(),
    "Percentage": (df.isnull().mean() * 100).round(2)
})

print(missing)

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())


# 4. SURVIVAL ANALYSIS
print("\n===== OVERALL SURVIVAL RATE =====")
print(df["survived"].value_counts(normalize=True))

print("\n===== SURVIVAL BY SEX =====")
print(df.groupby("sex", observed=True)["survived"].mean())

print("\n===== SURVIVAL BY PASSENGER CLASS =====")
print(df.groupby("class", observed=True)["survived"].mean())


# 5. VISUALIZATION
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")


# Survival distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="survived")

plt.title("Titanic Passenger Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of passengers")

plt.savefig("outputs/survival_distribution.png",
            bbox_inches="tight")

plt.close()


# Survival by sex
plt.figure(figsize=(7, 5))

sns.barplot(data=df, x="sex", y="survived")

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival rate")

plt.savefig("outputs/survival_by_sex.png",
            bbox_inches="tight")

plt.close()


# Survival by passenger class
plt.figure(figsize=(7, 5))

sns.barplot(data=df, x="class", y="survived")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger class")
plt.ylabel("Survival rate")

plt.savefig("outputs/survival_by_class.png",
            bbox_inches="tight")

plt.close()


print("\nAnalysis completed successfully.")
print("Charts saved in the outputs directory.")
