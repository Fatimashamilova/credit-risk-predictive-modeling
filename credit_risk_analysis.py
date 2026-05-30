# Credit Risk Analytics & Predictive Modeling
# Author: Fatima Shamilova
# Project: Financial Risk Analytics / Machine Learning

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATA
# ============================================================

# Download dataset from:
# https://www.kaggle.com/datasets/laotse/credit-risk-dataset
# Save the file as: data/credit_risk_dataset.csv

df = pd.read_csv("data/credit_risk_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
# For numeric columns, use median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# For categorical columns, use mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# Target variable:
# loan_status is commonly coded as:
# 0 = non-default / low risk
# 1 = default / high risk

print("\nLoan Status Distribution:")
print(df["loan_status"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="loan_status", data=df)
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status: 0 = Low Risk, 1 = High Risk")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Income distribution by loan status
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_status", y="person_income", data=df)
plt.title("Income Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Applicant Income")
plt.tight_layout()
plt.show()


# Loan amount distribution by loan status
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_status", y="loan_amnt", data=df)
plt.title("Loan Amount by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.show()


# Interest rate distribution by loan status
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_status", y="loan_int_rate", data=df)
plt.title("Interest Rate by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Interest Rate")
plt.tight_layout()
plt.show()


# Loan intent analysis
plt.figure(figsize=(10, 5))
sns.countplot(x="loan_intent", hue="loan_status", data=df)
plt.title("Loan Intent by Loan Status")
plt.xlabel("Loan Intent")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Previous default history
plt.figure(figsize=(6, 4))
sns.countplot(x="cb_person_default_on_file", hue="loan_status", data=df)
plt.title("Previous Default History by Loan Status")
plt.xlabel("Previous Default on File")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Create debt-to-income / loan-to-income ratio
df["loan_to_income_ratio"] = df["loan_amnt"] / df["person_income"]

# Replace infinite values if income is zero
df["loan_to_income_ratio"] = df["loan_to_income_ratio"].replace([np.inf, -np.inf], np.nan)
df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(df["loan_to_income_ratio"].median())

# Create credit history category
df["credit_history_category"] = pd.cut(
    df["cb_person_cred_hist_length"],
    bins=[0, 3, 7, 15, 50],
    labels=["Short", "Medium", "Long", "Very Long"]
)

print("\nNew Features Created:")
print(df[["loan_to_income_ratio", "credit_history_category"]].head())


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 7. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

# Define target variable
target = "loan_status"

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

print("\nFinal Feature Set Shape:", X.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features for Logistic Regression and KNN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. MODEL TRAINING
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

results = []

for model_name, model in models.items():

    if model_name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "="*60)
    print(model_name)
    print("="*60)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

plt.figure(figsize=(10, 5))
results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

sns.barplot(x="Model", y="Score", hue="Metric", data=results_melted)
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# 10. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

rf_model = models["Random Forest"]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(15)
)
plt.title("Top 15 Features Influencing Credit Risk")
plt.tight_layout()
plt.show()


# ============================================================
# 11. BUSINESS INSIGHTS
# ============================================================

print("\nBusiness Insights:")
print("""
1. Applicants with higher loan-to-income ratios tend to show increased credit risk.
2. Interest rate and loan amount are important indicators in predicting default probability.
3. Previous default history is a strong risk signal.
4. Credit history length provides useful information about borrower reliability.
5. Random Forest generally performs well because it captures non-linear relationships between borrower characteristics and default risk.
6. Credit risk models can support financial institutions by improving risk assessment, portfolio monitoring, and data-driven lending decisions.
""")


# ============================================================
# 12. FINAL RECOMMENDATION
# ============================================================

best_model = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]

print("\nRecommended Model:")
print(best_model)

print(f"""
Final Recommendation:
The {best_model['Model']} model is recommended based on its balance of predictive performance and classification metrics.
For credit risk use cases, recall and F1-score are especially important because failing to identify high-risk borrowers may create financial losses.
This model can support risk assessment and decision-making, but it should be used together with responsible lending policies, human review, and ongoing model monitoring.
""")
