# SALES FORECASTING — TIME SERIES ANALYSIS AND PREDICTION

## 1. Project Overview

This project focuses on time series analysis and sales forecasting using historical retail data from Corporación Favorita, an Ecuadorian grocery retailer.

The objective is to analyze historical sales patterns, identify temporal trends and seasonal behaviors, and develop forecasting models capable of predicting future product sales.

This project explores the application of statistical analysis and machine learning techniques to a real-world retail forecasting problem.

## 2. Business Problem

Accurate sales forecasting is essential for retail businesses because it supports:

- Inventory management and replenishment.
- Demand planning.
- Supply chain optimization.
- Operational and strategic decision-making.

The main challenge is to predict future sales based on historical observations while accounting for temporal patterns, seasonality, promotional activities, and external factors.

## 3. Dataset Description

The project uses the **Store Sales — Time Series Forecasting** dataset from Kaggle.

The dataset contains historical sales information from 54 stores in Ecuador, covering multiple product families.

The main files include:

| File | Description |
|---|---|
| `train.csv` | Historical daily sales observations. |
| `test.csv` | Observations for the forecasting period. |
| `stores.csv` | Store characteristics and locations. |
| `oil.csv` | Daily oil price information. |
| `holidays_events.csv` | Holiday and event information. |
| `transactions.csv` | Daily transaction counts per store. |
| `sample_submission.csv` | Example Kaggle submission format. |

The training dataset contains approximately 3 million observations, covering January 2013 to August 2017.

The target variable is `sales`, representing the number of units sold.

**Note:** The target variable represents sales volume, not monetary revenue.

## 4. Project Objectives

The main objectives are:

1. Perform an exploratory analysis of historical retail sales.
2. Identify trends, seasonality, and temporal patterns.
3. Investigate the influence of promotions, holidays, and external variables.
4. Prepare the data for time series forecasting.
5. Develop baseline and advanced forecasting models.
6. Evaluate and compare model performance using appropriate forecasting metrics.
7. Generate predictions for future sales observations.

## 5. Methodology

The project follows a progressive analytical workflow:

1. Data loading and initial inspection.
2. Data quality assessment.
3. Data preparation and temporal aggregation.
4. Exploratory time series analysis.
5. Baseline forecasting.
6. Advanced forecasting models.
7. Model evaluation and comparison.
8. Final conclusions and recommendations.

Chronological validation will be used to preserve the temporal structure of the data and avoid information leakage.

## 6. Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook

## 7. Project Structure

```text
02-sales-forecasting/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── 01_data_loading.py
│   ├── 02_data_preparation.py
│   ├── 03_exploratory_analysis.py
│   ├── 04_baseline_forecasting.py
│   ├── 05_advanced_forecasting.py
│   └── 06_model_evaluation.py
├── outputs/
│   ├── figures/
│   └── tables/
├── README.md
├── REPORT.md
└── requirements.txt
```

## 8. Project Status

**Current status: Initial data inspection completed.**

The original datasets have been loaded and inspected. Initial data quality checks have been performed, including missing-value detection, duplicate checks, temporal coverage analysis, and sales distribution assessment.

The next step is data preparation and the construction of daily sales time series.

## 9. Dataset Source

Kaggle Competition:

[Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
