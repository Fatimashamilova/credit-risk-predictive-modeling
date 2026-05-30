# Credit Risk Analytics & Predictive Modeling

## Project Overview

This project analyzes borrower and loan characteristics to predict credit risk using machine learning. The goal is to support data-driven lending decisions by identifying patterns associated with loan default risk and building classification models that can distinguish between low-risk and high-risk applicants.

This project was designed to demonstrate practical skills in financial analytics, exploratory data analysis, feature engineering, predictive modeling, and business decision support.

## Business Problem

Financial institutions need reliable methods to evaluate borrower risk before approving loans. Manual review can be time-consuming and inconsistent, while data-driven credit risk models can help improve decision-making, reduce default exposure, and support responsible lending practices.

The objective of this project is to answer:

- Which borrower characteristics are associated with higher default risk?
- Which loan features are most important in predicting credit risk?
- Which machine learning model performs best for credit risk classification?
- How can analytical insights support better financial decision-making?

## Dataset

Recommended dataset:

**Credit Risk Dataset**  
Kaggle: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Expected file name:

```text
credit_risk_dataset.csv
```

Place the CSV file inside a folder named:

```text
data/
```

Final project structure:

```text
credit-risk-predictive-modeling/
│
├── README.md
├── credit_risk_analysis.py
├── requirements.txt
├── .gitignore
│
└── data/
    └── credit_risk_dataset.csv
```

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

## Methods Used

- Data cleaning
- Missing value treatment
- Exploratory Data Analysis
- Feature engineering
- One-hot encoding
- Train/test split
- Logistic Regression
- K-Nearest Neighbors
- Random Forest Classifier
- Model evaluation
- Confusion matrix
- Classification report
- ROC-AUC score
- Feature importance analysis

## Key Features Engineered

- Debt-to-income ratio
- Loan amount to income ratio
- Credit history risk indicators
- Encoded categorical borrower characteristics

## Machine Learning Models

The following models were trained and compared:

1. Logistic Regression
2. K-Nearest Neighbors
3. Random Forest Classifier

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

## Business Insights

The analysis focuses on identifying factors that may increase credit risk, such as:

- Higher loan-to-income ratios
- Lower income levels
- Shorter credit history
- Higher interest rates
- Certain loan intent categories
- Prior default history

## Expected Outcome

The project provides a practical machine learning workflow for credit risk classification and demonstrates how financial institutions can use predictive analytics to support risk assessment and lending decisions.

## Resume Description

**Credit Risk Analytics & Predictive Modeling | Python, Scikit-Learn, Pandas**

- Conducted exploratory data analysis on financial risk data to identify factors influencing borrower creditworthiness and loan default risk.
- Engineered predictive features and developed machine learning classification models including Logistic Regression, KNN, and Random Forest.
- Evaluated model performance using accuracy, precision, recall, F1-score, and ROC-AUC metrics to support risk assessment decisions.
- Generated analytical insights to identify key drivers of credit risk and support data-driven financial decision-making.

## Relevance to IFC Operations Analyst Role

This project aligns with IFC's Operations Analyst / Data Scientist role by demonstrating:

- Financial analytics
- Credit risk modeling
- Predictive analytics
- Machine learning
- Structured data analysis
- Business decision support
- Data quality and governance awareness
- Communication of analytical findings to non-technical stakeholders
