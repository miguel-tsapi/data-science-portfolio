from pathlib import Path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = sns.load_dataset("titanic")

# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

features = [
    "sex",
    "pclass",
    "age",
    "fare",
    "sibsp",
    "parch",
    "embarked"
]

target = "survived"

X = df[features]
y = df[target]

# ==========================================
# 3. IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = [
    "age",
    "fare",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "pclass",
    "embarked"
]

# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 6. DISPLAY DATA INFORMATION
# ==========================================

print("===== DATASET INFORMATION =====")
print("Dataset shape:", df.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\n===== TARGET DISTRIBUTION =====")
print(y.value_counts(normalize=True))


# ==========================================
# 7. LOGISTIC REGRESSION MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ==========================================
# CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)

print("\n===== CROSS-VALIDATION =====")
print("ROC-AUC scores:", cv_scores)
print("Mean ROC-AUC:", round(cv_scores.mean(), 4))
print("Standard deviation:", round(cv_scores.std(), 4))

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# ==========================================
# 8. MODEL EVALUATION
# ==========================================

print("\n===== MODEL PERFORMANCE =====")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix:")
print(cm)

roc_auc = roc_auc_score(y_test, y_proba)

print("ROC-AUC:", round(roc_auc, 4))


# ==========================================
# 9. MODEL COEFFICIENTS
# ==========================================

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = model.named_steps[
    "classifier"
].coef_[0]

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Odds Ratio": np.exp(coefficients)
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n===== MODEL COEFFICIENTS =====")
print(coef_df.to_string(index=False))



# ==========================================
# 10. MODEL VISUALIZATIONS
# ==========================================

# ==========================================
# 10. PROFESSIONAL MODEL VISUALIZATIONS
# ==========================================

# Save plots in the outputs directory next to this script
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent visual style
sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})


# ------------------------------------------
# 10.1 CONFUSION MATRIX
# ------------------------------------------

cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

annotations = np.array([
    [
        f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
        for j in range(cm.shape[1])
    ]
    for i in range(cm.shape[0])
])

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=annotations,
    fmt="",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    cbar=False,
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"],
    annot_kws={"size": 13, "weight": "bold"},
    ax=ax
)

ax.set_title(
    "Confusion Matrix",
    pad=18
)

ax.set_xlabel("Predicted Class", labelpad=10)
ax.set_ylabel("Actual Class", labelpad=10)

ax.text(
    0.5,
    -0.19,
    "Values represent counts and row-wise percentages.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color="dimgray"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------
# 10.2 ROC CURVE
# ------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

ax.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

# Random classifier reference
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    color="gray",
    label="Random Classifier"
)

ax.fill_between(
    fpr,
    tpr,
    alpha=0.10
)

ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=18)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax.legend(
    loc="lower right",
    frameon=True
)

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------
# 10.3 LOGISTIC REGRESSION COEFFICIENTS
# ------------------------------------------

# Convert technical feature names into readable labels
feature_labels = {
    "cat__sex_male": "Sex: Male (vs Female)",
    "cat__pclass_2": "Passenger Class: 2 (vs 1)",
    "cat__pclass_3": "Passenger Class: 3 (vs 1)",
    "cat__embarked_Q": "Embarkation: Queenstown (vs Cherbourg)",
    "cat__embarked_S": "Embarkation: Southampton (vs Cherbourg)",
    "num__age": "Age (per standard deviation)",
    "num__fare": "Fare (per standard deviation)",
    "num__sibsp": "Siblings / Spouses (per SD)",
    "num__parch": "Parents / Children (per SD)"
}

plot_df = coef_df.copy()

plot_df["Readable Feature"] = (
    plot_df["Feature"]
    .map(feature_labels)
    .fillna(plot_df["Feature"])
)

plot_df = plot_df.sort_values(
    by="Coefficient",
    ascending=True
)

colors = [
    "#D95F59" if value < 0 else "#2878B5"
    for value in plot_df["Coefficient"]
]

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    plot_df["Readable Feature"],
    plot_df["Coefficient"],
    color=colors,
    height=0.68
)

# Zero reference line
ax.axvline(
    0,
    color="black",
    linewidth=1.2
)

# Add coefficient values
for bar, value in zip(bars, plot_df["Coefficient"]):
    offset = 0.04 if value >= 0 else -0.04

    ax.text(
        value + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.3f}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=9,
        fontweight="bold"
    )

ax.set_title(
    "Logistic Regression Coefficients",
    pad=18
)

ax.set_xlabel("Coefficient Value")
ax.set_ylabel("Predictor Variables")

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

ax.grid(axis="y", visible=False)

ax.text(
    0.5,
    -0.16,
    "Blue: positive coefficient | Red: negative coefficient",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color="dimgray"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "logistic_regression_coefficients.png",
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------
# 10.4 SUMMARY
# ------------------------------------------

print("\n===== VISUALIZATIONS SAVED SUCCESSFULLY =====")

print("Output directory:", OUTPUT_DIR)

print("1. confusion_matrix.png")
print("2. roc_curve.png")
print("3. logistic_regression_coefficients.png")
