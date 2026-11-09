# TITANIC — EXPLORATORY DATA ANALYSIS AND STATISTICAL ANALYSIS

## 1. Introduction

The sinking of the RMS Titanic in April 1912 is one of the most famous maritime disasters in history.

This project aims to analyze passenger data from the Titanic dataset in order to identify patterns and relationships between passenger characteristics and survival outcomes.

The main objective is to apply data analysis and statistical methods using Python, Pandas, NumPy, Matplotlib, Seaborn and SciPy.

### Research questions

This analysis focuses on the following questions:

- What proportion of passengers survived?
- Did survival rates differ between male and female passengers?
- Was passenger class associated with survival?
- Was there a statistically significant association between sex and survival?
- How did sex and passenger class jointly relate to survival?

---

## 2. Dataset Description

The analysis uses the Titanic dataset available through the Seaborn Python library.

The dataset contains information about passengers, including:

- `survived`: survival status (0 = No, 1 = Yes).
- `sex`: passenger's sex.
- `age`: passenger's age.
- `pclass`: passenger class.
- `fare`: ticket fare.
- `embarked`: port of embarkation.

The contingency table used in this analysis contains 891 passenger records.

The dataset is loaded directly using:

```python
import seaborn as sns

df = sns.load_dataset("titanic")
```

No raw dataset file is committed to the repository.

---

## 3. Exploratory Data Analysis (EDA)

The exploratory analysis was conducted to understand the dataset's structure, examine variable distributions and identify initial patterns.

The analysis included:

- Dataset inspection.
- Statistical summaries.
- Missing-value examination.
- Duplicate-value examination.
- Survival distribution analysis.
- Survival comparisons by sex and passenger class.

### 3.1 Survival distribution by sex

The analysis produced the following results:

| Sex | Survival rate |
|---|---:|
| Female | 74.20% |
| Male | 18.89% |

**Interpretation:**

Female passengers had a substantially higher survival rate than male passengers.

The difference between the two survival rates is 55.31 percentage points.

This indicates a strong descriptive difference in survival outcomes between the two groups.

### 3.2 Survival distribution by passenger class

| Passenger class | Survival rate |
|---|---:|
| First class | 62.96% |
| Second class | 47.28% |
| Third class | 24.24% |

**Interpretation:**

The survival rate decreased from first class to third class.

The difference between first-class and third-class survival rates is 38.72 percentage points.

This suggests that passenger class was associated with survival outcomes.

### 3.3 Survival by sex and passenger class

The analysis also examined survival rates by combining sex and passenger class.

| Sex | Passenger class | Passengers | Survival rate |
|---|---|---:|---:|
| Female | First | 94 | 96.81% |
| Female | Second | 76 | 92.11% |
| Female | Third | 144 | 50.00% |
| Male | First | 122 | 36.89% |
| Male | Second | 108 | 15.74% |
| Male | Third | 347 | 13.54% |

**Interpretation:**

The results show differences in survival rates across both sex and passenger class.

Female passengers had higher survival rates than male passengers in all three classes.

Among female passengers, first and second class had particularly high survival rates.

Among male passengers, the survival rate remained comparatively low across all three classes.

These observations suggest that sex and passenger class should both be considered when studying survival outcomes.

---

## 4. Conditional Probability Analysis

Conditional probabilities were calculated to measure the probability of survival given a passenger's sex or class.

The general formula is:

\[
P(A|B)=\frac{P(A\cap B)}{P(B)}
\]

where:

- A represents the event of survival.
- B represents a passenger characteristic.

### 4.1 Survival probability given sex

\[
P(Survival|Female)=\frac{233}{314}=0.7420
\]

\[
P(Survival|Male)=\frac{109}{577}=0.1889
\]

The results confirm that female passengers had a much higher observed survival probability than male passengers.

### 4.2 Survival probability given passenger class

| Class | Conditional survival probability |
|---|---:|
| First | 0.6296 |
| Second | 0.4728 |
| Third | 0.2424 |

The probability of survival decreased across passenger classes, from first to third class.

---

## 5. Contingency Table Analysis

A contingency table was constructed to examine the relationship between sex and survival.

| Sex | Did not survive | Survived | Total |
|---|---:|---:|---:|
| Female | 81 | 233 | 314 |
| Male | 468 | 109 | 577 |
| Total | 549 | 342 | 891 |

**Interpretation:**

Among female passengers, 233 survived out of 314.

Among male passengers, 109 survived out of 577.

The table highlights a substantial difference in survival distribution between the two groups.

It also provides the observed frequencies required for the chi-square test of independence.

---

## 6. Statistical Inference: Chi-Square Test

### 6.1 Objective

The chi-square test of independence was performed to determine whether an association existed between sex and survival.

### 6.2 Hypotheses

- Null hypothesis (H0): Sex and survival are statistically independent.
- Alternative hypothesis (H1): Sex and survival are statistically associated.

The significance level was set to:

\[
\alpha=0.05
\]

### 6.3 Mathematical principle

The chi-square statistic is calculated using:

\[
\chi^2=\sum\frac{(O_{ij}-E_{ij})^2}{E_{ij}}
\]

where:

- \(O_{ij}\) represents observed frequencies.
- \(E_{ij}\) represents expected frequencies under the independence hypothesis.

The expected frequency is calculated as:

\[
E_{ij}=\frac{Row\ Total\times Column\ Total}{Grand\ Total}
\]

### 6.4 Test results

| Statistical measure | Result |
|---|---:|
| Chi-square statistic | 260.717 |
| Degrees of freedom | 1 |
| P-value | Approximately 1.20 × 10⁻⁵⁸ |
| Significance level | 0.05 |
| Cramer's V | 0.5409 |

### 6.5 Interpretation

The p-value is substantially lower than the significance threshold of 0.05.

Therefore, the null hypothesis of independence is rejected.

There is strong statistical evidence of an association between sex and survival in this dataset.

Cramer's V is approximately 0.5409, indicating a relatively strong association between the two categorical variables.

These findings support the descriptive observations made earlier.

However, statistical association does not establish causation. Other passenger characteristics and historical circumstances may also have influenced survival outcomes.

---

## 7. Visualizations

The following visualizations were generated during the exploratory analysis:

1. `survival_distribution.png`: overall survival distribution.
2. `survival_by_sex.png`: survival comparison by sex.
3. `survival_by_class.png`: survival comparison by passenger class.

These visualizations help communicate the observed patterns and support the statistical analysis.

---

## 8. Conclusion

The exploratory and statistical analyses of the Titanic dataset revealed several important patterns.

First, female passengers had a substantially higher survival rate than male passengers.

Second, passengers in first class had a higher survival rate than those in second and third class.

Third, the combined analysis of sex and passenger class showed that survival outcomes varied considerably across these groups.

Finally, the chi-square test provided strong statistical evidence of an association between sex and survival. Cramer's V indicated that this association was relatively strong.

Overall, this project demonstrates how exploratory data analysis, conditional probability, contingency tables and statistical inference can be combined to investigate relationships in a real-world dataset.

### Future Work

The next stage of the project will focus on predictive modeling using supervised machine learning.

A logistic regression model will be developed to predict passenger survival based on selected passenger characteristics.

The model will subsequently be evaluated using appropriate classification metrics.

---

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Git and GitHub
