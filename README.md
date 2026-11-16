# Credit Risk Prediction Using Machine Learning

## Overview

An end-to-end machine learning project that classifies applicants into higher-risk and lower-risk categories using demographic, financial, employment, and historical credit information. The project explores multiple classification algorithms, handles imbalanced data, evaluates model performance, and deploys the final Random Forest model through an interactive Streamlit application.

## Dataset

The project uses two datasets:

- `application_record.csv` – Applicant demographic, financial, and employment information.
- `credit_record.csv` – Historical credit repayment records.

The target variable, `CREDIT_RISK`, is engineered from historical credit statuses:

- **0 – Lower Risk:** No observed severe delinquency status.
- **1 – Higher Risk:** At least one historical status of 2, 3, 4, or 5.

The datasets are merged using the applicant ID.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Joblib
- Streamlit

## Project Workflow

1. Data loading and inspection
2. Data cleaning and duplicate removal
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Missing-value imputation
6. Categorical feature encoding using One-Hot Encoding
7. Stratified train-test splitting
8. Handling class imbalance
9. Training and comparing classification models
10. Hyperparameter tuning using RandomizedSearchCV
11. Classification threshold experiments
12. Feature importance and permutation importance analysis
13. Model evaluation and Streamlit deployment

## Models Explored

- Gaussian Naive Bayes
- Bagging Classifier with Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Random Forest with SMOTE
- Random Forest with SMOTENC
- Tuned Random Forest
- XGBoost

## Final Model – Random Forest

The tuned Random Forest was selected based on its performance compared with the other experimental models.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 12 |
| min_samples_split | 10 |
| min_samples_leaf | 10 |
| class_weight | balanced_subsample |
| Classification threshold | 0.5 |

## Model Performance

Final evaluation on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 96% |
| Precision (Class 1) | 0.14 |
| Recall (Class 1) | 0.30 |
| F1-score (Class 1) | 0.19 |
| ROC-AUC | 0.6934 |
| PR-AUC | 0.0959 |

PR-AUC was prioritised because the dataset contains a significant class imbalance.

## Feature Importance

The leading features identified by the Random Forest include:

1. Employment years
2. Age
3. Annual income
4. Number of family members
5. Number of children

Permutation importance was also used to examine how individual features affect predictive performance.

## Streamlit Application

An interactive Streamlit application allows users to enter applicant information and receive a predicted credit-risk classification.

The application uses the saved Random Forest model and preprocessing objects.

## Key Learnings

- Implementing multiple classification algorithms.
- Understanding class imbalance and its impact on model performance.
- Applying SMOTE and SMOTENC.
- Hyperparameter tuning with RandomizedSearchCV.
- Understanding precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Classification threshold selection.
- Feature importance and model interpretation.
- Saving and deploying machine learning models.

## Limitations

- The target is an engineered proxy derived from historical credit statuses, not actual bank loan approval outcomes.
- The dataset is highly imbalanced.
- The final model has limited precision and recall for the higher-risk class.
- Feature importance does not establish causation.
- The model has not been validated for real-world lending decisions.


