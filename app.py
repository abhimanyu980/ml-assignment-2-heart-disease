import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Classification – ML Assignment 2")

model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

model_name = st.selectbox("Select ML Model", list(model_map.keys()))
uploaded_file = st.file_uploader("Upload Test CSV (with target column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = joblib.load(f"models/{model_map[model_name]}")

    if model_name in ["Logistic Regression", "KNN"]:
        scaler = joblib.load("models/scaler.pkl")
        X = scaler.transform(X)

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
