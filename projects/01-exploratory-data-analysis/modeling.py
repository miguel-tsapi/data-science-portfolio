
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
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

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

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
