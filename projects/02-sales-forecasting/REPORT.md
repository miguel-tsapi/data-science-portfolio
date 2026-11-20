# SALES FORECASTING — TECHNICAL REPORT

## 1. Introduction

Sales forecasting is an important analytical task in the retail industry. Accurate predictions of future demand help businesses improve inventory management, optimize supply chain operations, and support strategic decision-making.

This project investigates historical retail sales data from Corporación Favorita, an Ecuadorian grocery retailer, with the objective of developing time series forecasting models.

The analysis focuses on identifying temporal patterns, understanding sales behavior, and evaluating statistical and machine learning approaches for predicting future sales.

The project follows a progressive methodology, beginning with data inspection and preparation before moving towards exploratory time series analysis and predictive modeling.

## 2. Business Problem and Forecasting Objectives

### 2.1 Business Problem

Retail sales are influenced by several factors, including historical demand, seasonal patterns, promotional activities, holidays, and external economic conditions.

Understanding these factors is essential for producing reliable sales forecasts.

The central problem addressed in this project is:

**How can historical retail sales data be used to predict future sales while accounting for temporal patterns and relevant external factors?**

### 2.2 Objectives

The main objectives are:

- Analyze historical sales behavior across stores and product families.
- Identify trends, seasonal patterns, and temporal variations.
- Investigate the potential influence of promotions, holidays, and oil prices.
- Prepare reliable datasets for time series modeling.
- Develop baseline forecasting approaches.
- Explore more advanced statistical and machine learning models.
- Evaluate forecasting performance using appropriate metrics.

The target variable is `sales`, which represents the number of units sold rather than monetary revenue.

## 3. Dataset Description

### 3.1 Data Source

The dataset used in this project comes from the Kaggle competition:

**Store Sales — Time Series Forecasting**

The dataset contains historical retail information from 54 stores in Ecuador.

It provides daily sales observations for multiple product families, together with additional information about stores, transactions, promotions, holidays, and oil prices.

### 3.2 Dataset Components

The following files were used for the initial inspection:

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| train.csv | 3,000,888 | 6 | Historical sales observations |
| test.csv | 28,512 | 5 | Forecasting observations |
| stores.csv | 54 | 5 | Store characteristics |
| oil.csv | 1,218 | 2 | Daily oil prices |
| holidays_events.csv | 350 | 6 | Holidays and events |
| transactions.csv | 83,488 | 3 | Daily transactions per store |
| sample_submission.csv | 28,512 | 2 | Example submission structure |

### 3.3 Temporal Coverage

The training dataset covers the period from January 1, 2013, to August 15, 2017.

The test dataset covers the period from August 16, 2017, to August 31, 2017.

Consequently, the forecasting horizon consists of 16 consecutive calendar days.

The training dataset contains 1,684 distinct dates, while the test dataset contains 16 distinct dates.

### 3.4 Data Granularity

Each observation in the training dataset represents the sales volume associated with a specific combination of:

- Date (`date`)
- Store (`store_nbr`)
- Product family (`family`)

The business key is therefore:

`(date, store_nbr, family)`

The target variable is `sales`, while `onpromotion` represents the number of items on promotion for the corresponding observation.

## 4. Initial Data Inspection and Quality Assessment

### 4.1 Inspection Methodology

An initial inspection was performed using Python and Pandas.

The objective was to assess the structure, completeness, consistency, and temporal coverage of the original datasets before applying any transformations.

The following checks were conducted:

- Dataset dimensions and column types.
- Missing-value detection.
- Exact duplicate detection.
- Date coverage and number of distinct dates.
- Number of distinct stores and product families.
- Business key uniqueness.
- Sales distribution and identification of zero or negative values.

No cleaning or imputation was performed during this initial inspection.

### 4.2 Structural Integrity

The training dataset contains 3,000,888 observations and six columns.

The test dataset contains 28,512 observations and five columns.

The training and test datasets both contain:

- 54 distinct stores.
- 33 distinct product families.

No exact duplicate rows were detected in either dataset.

Furthermore, no duplicate business keys were identified using the combination `(date, store_nbr, family)`.

This confirms that each available store-product-family-date combination is represented by a unique observation.

### 4.3 Missing Values

No missing values were detected in the training or test datasets.

However, the oil dataset contains 43 missing values in the `dcoilwtico` variable.

These missing observations require further investigation before any imputation strategy can be selected.

The absence of oil price observations may be related to non-trading days and market holidays. Nevertheless, this explanation should be verified before treating all missing values in the same way.

### 4.4 Missing Calendar Dates

The training dataset contains four dates that are absent from its temporal coverage:

- December 25, 2013.
- December 25, 2014.
- December 25, 2015.
- December 25, 2016.

All four dates correspond to Christmas Day.

These dates are absent from the training dataset rather than being represented by observations with zero sales.

This distinction is important because an absent date should not automatically be interpreted as a zero-sales observation.

The missing dates will therefore be considered explicitly during temporal data preparation.

No artificial observations have been introduced at this stage.

### 4.5 Sales Distribution

The initial inspection revealed the following sales distribution:

| Indicator | Number of observations |
|---|---:|
| Zero sales | 939,130 |
| Positive sales | 2,061,758 |
| Negative sales | 0 |
| Total observations | 3,000,888 |

Zero-sales observations represent approximately 31.29% of the training dataset, while positive-sales observations account for approximately 68.71%.

No negative sales values were detected.

The minimum sales value is 0, and the maximum is 124,717 units.

The relatively high proportion of zero-sales observations highlights the importance of understanding demand patterns at the store and product-family levels.

These observations will be retained during subsequent analyses unless a specific data-quality issue is identified.

The maximum observed value will also be examined during exploratory analysis to determine whether it represents an unusual but valid sales observation.

### 4.6 Initial Data Quality Summary

The initial inspection indicates that the datasets have a generally consistent structure.

The main findings are:

- No missing values in the training and test datasets.
- No exact duplicate observations.
- No duplicate business keys.
- Four missing calendar dates in the training period, all corresponding to Christmas Day.
- A significant proportion of zero-sales observations.
- Forty-three missing oil price observations requiring further investigation.

These findings provide a basis for the next stage of the project: data preparation and temporal aggregation.

## 5. Data Preparation

*To be completed.*

## 6. Exploratory Time Series Analysis

*To be completed.*

## 7. Forecasting Methodology

*To be completed.*

## 8. Model Evaluation

*To be completed.*

## 9. Results and Discussion

*To be completed.*

## 10. Conclusion

*To be completed.*