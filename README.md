Machine Learning Assignment 2 – Heart Disease Classification
a. Problem Statement

The objective of this project is to implement and evaluate multiple machine learning classification models to predict the presence of heart disease based on clinical attributes. The project also includes the development and deployment of an interactive Streamlit web application.

b. Dataset Description

The Heart Disease dataset is sourced from Kaggle (originally from the UCI Machine Learning Repository).
It contains 1025 patient records with 13 clinical features such as age, cholesterol level, resting blood pressure, maximum heart rate achieved, and exercise-induced angina.
The target variable indicates the presence (1) or absence (0) of heart disease.

The dataset is programmatically downloaded using the kagglehub library to ensure reproducibility across environments.

c. Models Used & Evaluation Metrics

The following machine learning models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

All models were evaluated using:
Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

##  Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 1.0000 | 0.9714 | 0.9855 | 0.9712 |
| KNN | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Observations
Model	Observation
Logistic Regression	High recall makes it effective for identifying positive heart disease cases
Decision Tree	Achieves very high accuracy by learning complex decision rules
KNN	Balanced performance and benefits from feature scaling
Naive Bayes	Fast and efficient despite strong independence assumptions
Random Forest	Provides excellent performance through ensemble averaging
XGBoost	Best overall performance due to boosting and regularization
Streamlit Application

The Streamlit web application provides:

CSV dataset upload option

Model selection dropdown

Display of evaluation metrics

Confusion matrix and classification report

The application is deployed using Streamlit Community Cloud.

