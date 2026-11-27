
"""
04_baseline_forecasting.py

Baseline Forecasting
Project 02 - Sales Forecasting

This script implements and compares simple forecasting
methods using the Corporación Favorita dataset.
"""

from pathlib import Path

import numpy as np
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
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Forecasting configuration
TARGET_STORE = 45
TARGET_FAMILY = "GROCERY I"

TRAIN_END = pd.Timestamp("2017-07-30")
VALIDATION_START = pd.Timestamp("2017-07-31")
VALIDATION_END = pd.Timestamp("2017-08-15")

FORECAST_HORIZON = 16
SEASONAL_PERIOD = 7
MOVING_AVERAGE_WINDOW = 7

# Professional visualization settings
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep"
)

plt.rcParams.update({
    "figure.figsize": (13, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25
})


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("=" * 65)
print("1. LOADING TRAINING DATA")
print("=" * 65)

train = pd.read_csv(
    RAW_DIR / "train.csv",
    parse_dates=["date"]
)

print(f"Training dataset shape: {train.shape}")
print(f"Date range: {train['date'].min()} to {train['date'].max()}")


# ============================================================
# 3. SELECT THE TARGET TIME SERIES
# ============================================================

print("\n" + "=" * 65)
print("2. SELECTING TARGET TIME SERIES")
print("=" * 65)

series_data = train.loc[
    (train["store_nbr"] == TARGET_STORE) &
    (train["family"] == TARGET_FAMILY),
    ["date", "sales"]
].copy()

series_data = (
    series_data
    .groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
)

print(f"Selected store: {TARGET_STORE}")
print(f"Selected family: {TARGET_FAMILY}")
print(f"Number of observed dates: {len(series_data)}")


# ============================================================
# 4. BUILD COMPLETE DAILY TIME SERIES
# ============================================================

print("\n" + "=" * 65)
print("3. BUILDING COMPLETE DAILY TIME SERIES")
print("=" * 65)

full_calendar = pd.date_range(
    start=series_data["date"].min(),
    end=VALIDATION_END,
    freq="D"
)

series = (
    series_data
    .set_index("date")
    .reindex(full_calendar)
    .rename_axis("date")
)

series["is_observed"] = series["sales"].notna()

series = series.reset_index()

print(f"Calendar observations: {len(series)}")
print(f"Missing sales observations: {series['sales'].isna().sum()}")
print(f"Observed zero-sales days: {(series['sales'] == 0).sum()}")


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 65)
print("4. CHRONOLOGICAL TRAIN / VALIDATION SPLIT")
print("=" * 65)

train_series = series.loc[
    series["date"] <= TRAIN_END
].copy()

validation_series = series.loc[
    series["date"].between(
        VALIDATION_START,
        VALIDATION_END
    )
].copy()

print(
    f"Training period: "
    f"{train_series['date'].min().date()} to "
    f"{train_series['date'].max().date()}"
)

print(
    f"Validation period: "
    f"{validation_series['date'].min().date()} to "
    f"{validation_series['date'].max().date()}"
)

print(f"Training calendar days: {len(train_series)}")
print(f"Validation calendar days: {len(validation_series)}")

print(
    "Missing training observations:",
    train_series["sales"].isna().sum()
)

print(
    "Missing validation observations:",
    validation_series["sales"].isna().sum()
)

print("\nInitial data preparation completed.")



# ============================================================
# 6. NAIVE FORECAST
# ============================================================

print("\n" + "=" * 65)
print("5. NAIVE FORECAST")
print("=" * 65)

# Retrieve the last observed training value
last_observed_value = (
    train_series
    .dropna(subset=["sales"])
    .iloc[-1]["sales"]
)

last_observed_date = (
    train_series
    .dropna(subset=["sales"])
    .iloc[-1]["date"]
)

# Generate constant forecasts
naive_predictions = np.repeat(
    last_observed_value,
    FORECAST_HORIZON
)

# Build forecast dataframe
naive_forecast = pd.DataFrame({
    "date": validation_series["date"].values,
    "actual_sales": validation_series["sales"].values,
    "naive_forecast": naive_predictions
})

print(f"Last observed date: {last_observed_date.date()}")
print(f"Last observed sales: {last_observed_value:,.2f}")

print("\nNaive forecast results:")

print(
    naive_forecast.to_string(index=False)
)

print("\nNaive forecast generated successfully.")



# ============================================================
# 7. SEASONAL NAIVE FORECAST
# ============================================================

print("\n" + "=" * 65)
print("6. SEASONAL NAIVE FORECAST")
print("=" * 65)

# Retrieve the last complete seasonal cycle
last_season = (
    train_series
    .set_index("date")["sales"]
    .loc[
        TRAIN_END - pd.Timedelta(days=SEASONAL_PERIOD - 1):
        TRAIN_END
    ]
)

# Ensure that the seasonal reference period is complete
if (
    len(last_season) != SEASONAL_PERIOD
    or last_season.isna().any()
):
    raise ValueError(
        "The last seasonal cycle contains missing observations."
    )

# Generate recursive seasonal forecasts
seasonal_predictions = []

for step in range(FORECAST_HORIZON):

    reference_value = last_season.iloc[
        step % SEASONAL_PERIOD
    ]

    seasonal_predictions.append(reference_value)

# Build forecast dataframe
seasonal_forecast = pd.DataFrame({
    "date": validation_series["date"].values,
    "actual_sales": validation_series["sales"].values,
    "seasonal_naive_forecast": seasonal_predictions
})

print(f"Seasonal period: {SEASONAL_PERIOD} days")

print("\nReference seasonal observations:")
print(last_season)

print("\nSeasonal Naive forecast results:")

print(
    seasonal_forecast.to_string(index=False)
)

print("\nSeasonal Naive forecast generated successfully.")



# ============================================================
# 8. MOVING AVERAGE FORECAST
# ============================================================

print("\n" + "=" * 65)
print("7. MOVING AVERAGE FORECAST")
print("=" * 65)

# Retrieve the last complete 7-day window
last_window = (
    train_series
    .set_index("date")["sales"]
    .loc[
        TRAIN_END - pd.Timedelta(days=MOVING_AVERAGE_WINDOW - 1):
        TRAIN_END
    ]
)

# Check that the reference window is complete
if (
    len(last_window) != MOVING_AVERAGE_WINDOW
    or last_window.isna().any()
):
    raise ValueError(
        "The last moving average window contains missing observations."
    )

# Initialize the historical values
history_values = last_window.astype(float).tolist()

# Generate recursive forecasts
moving_average_predictions = []

for step in range(FORECAST_HORIZON):

    prediction = np.mean(
        history_values[-MOVING_AVERAGE_WINDOW:]
    )

    moving_average_predictions.append(prediction)

    # Append the forecast, not the actual validation value
    history_values.append(prediction)

# Build forecast dataframe
moving_average_forecast = pd.DataFrame({
    "date": validation_series["date"].values,
    "actual_sales": validation_series["sales"].values,
    "moving_average_forecast": moving_average_predictions
})

print(f"Moving average window: {MOVING_AVERAGE_WINDOW} days")

print("\nReference observations:")
print(last_window)

print("\nMoving Average forecast results:")

print(
    moving_average_forecast.to_string(index=False)
)

print("\nMoving Average forecast generated successfully.")



# ============================================================
# 9. BASELINE MODEL EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("8. BASELINE MODEL EVALUATION")
print("=" * 65)

# Retrieve actual validation values
actual_values = validation_series["sales"].to_numpy(dtype=float)

# Store predictions from all baseline models
forecast_predictions = {
    "Naive": naive_predictions,
    "Seasonal Naive": seasonal_predictions,
    "Moving Average": moving_average_predictions
}

# Define evaluation function
def evaluate_forecast(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays have different lengths.")

    if np.isnan(actual).any() or np.isnan(predicted).any():
        raise ValueError("Missing values detected during evaluation.")

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape
    }


# Evaluate all baseline models
evaluation_results = []

for model_name, predictions in forecast_predictions.items():

    metrics = evaluate_forecast(
        actual_values,
        predictions
    )

    evaluation_results.append({
        "Model": model_name,
        **metrics
    })


# Build comparison dataframe
baseline_comparison = pd.DataFrame(evaluation_results)

baseline_comparison = baseline_comparison.sort_values(
    by="WAPE (%)"
).reset_index(drop=True)


print("\nBaseline model comparison:")

print(
    baseline_comparison.to_string(
        index=False,
        formatters={
            "MAE": "{:,.2f}".format,
            "RMSE": "{:,.2f}".format,
            "WAPE (%)": "{:.2f}%".format
        }
    )
)


# Save evaluation results
comparison_path = (
    TABLES_DIR / "baseline_model_comparison.csv"
)

baseline_comparison.to_csv(
    comparison_path,
    index=False
)


# Combine actual values and forecasts
baseline_forecasts = pd.DataFrame({
    "date": validation_series["date"].values,
    "actual_sales": actual_values,
    "naive_forecast": naive_predictions,
    "seasonal_naive_forecast": seasonal_predictions,
    "moving_average_forecast": moving_average_predictions
})


# Save individual forecasts
forecasts_path = (
    TABLES_DIR / "baseline_forecasts_store45_grocery_i.csv"
)

baseline_forecasts.to_csv(
    forecasts_path,
    index=False
)


print("\nEvaluation completed successfully.")

print(f"Comparison table saved to: {comparison_path}")
print(f"Forecast table saved to: {forecasts_path}")



# ============================================================
# 10. BASELINE FORECAST VISUALIZATION
# ============================================================

print("\n" + "=" * 65)
print("9. BASELINE FORECAST VISUALIZATION")
print("=" * 65)

# ------------------------------------------------------------
# 10.1. Forecast comparison plot
# ------------------------------------------------------------

# Select the last 60 days of training history
history_plot = train_series.tail(60).copy()

# Ensure chronological ordering
history_plot = history_plot.sort_values("date")

fig, ax = plt.subplots(figsize=(15, 7))

# Historical sales
ax.plot(
    history_plot["date"],
    history_plot["sales"],
    label="Historical sales",
    color="#34495E",
    linewidth=1.8
)

# Actual validation sales
ax.plot(
    baseline_forecasts["date"],
    baseline_forecasts["actual_sales"],
    label="Actual sales",
    color="#1A936F",
    linewidth=2.5,
    marker="o",
    markersize=5
)

# Naive forecast
ax.plot(
    baseline_forecasts["date"],
    baseline_forecasts["naive_forecast"],
    label="Naive",
    color="#E76F51",
    linestyle="--",
    linewidth=2
)

# Seasonal Naive forecast
ax.plot(
    baseline_forecasts["date"],
    baseline_forecasts["seasonal_naive_forecast"],
    label="Seasonal Naive",
    color="#277DA1",
    linestyle="--",
    linewidth=2.2,
    marker="s",
    markersize=4
)

# Moving Average forecast
ax.plot(
    baseline_forecasts["date"],
    baseline_forecasts["moving_average_forecast"],
    label="Moving Average (7 days)",
    color="#9B5DE5",
    linestyle="-.",
    linewidth=2
)

# Highlight validation period
ax.axvspan(
    validation_series["date"].min(),
    validation_series["date"].max(),
    color="#F4A261",
    alpha=0.10,
    label="Validation period"
)

ax.set_title(
    "Baseline Forecast Comparison — Store 45, Grocery I",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Daily sales (units)", fontsize=11)

ax.legend(
    loc="upper left",
    frameon=True,
    fontsize=9
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.25
)

plt.xticks(rotation=30)
plt.tight_layout()

forecast_figure_path = (
    FIGURES_DIR / "06_baseline_forecast_comparison.png"
)

plt.savefig(
    forecast_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 10.2. Model performance comparison
# ------------------------------------------------------------

metrics = ["MAE", "RMSE", "WAPE (%)"]

metric_colors = [
    "#277DA1",
    "#F4A261",
    "#43AA8B"
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 6)
)

for ax, metric, color in zip(
    axes,
    metrics,
    metric_colors
):

    metric_data = baseline_comparison.sort_values(
        by=metric,
        ascending=True
    )

    bars = ax.barh(
        metric_data["Model"],
        metric_data[metric],
        color=color,
        height=0.55
    )

    # Display the best-performing model at the top
    ax.invert_yaxis()

    # Annotate each bar with its value
    for bar in bars:

        value = bar.get_width()

        label = (
            f"{value:.2f}%"
            if metric == "WAPE (%)"
            else f"{value:,.0f}"
        )

        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {label}",
            va="center",
            fontsize=9
        )

    ax.set_title(
        metric,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Error (lower is better)", fontsize=9)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Baseline Model Performance Comparison",
    fontsize=16,
    fontweight="bold",
    y=1.03
)

plt.tight_layout()

performance_figure_path = (
    FIGURES_DIR / "07_baseline_model_performance.png"
)

plt.savefig(
    performance_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\nBaseline visualizations generated successfully.")

print(f"Forecast comparison: {forecast_figure_path}")
print(f"Model performance: {performance_figure_path}")
