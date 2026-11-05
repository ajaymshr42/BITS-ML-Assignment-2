import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Classification",
    page_icon="📧",
    layout="wide"
)

st.title("Email Spam Classification")
st.caption(
    "BITS Pilani | Machine Learning Assignment 2 | "
    "Student ID: 2026AK05228"
)

st.markdown(
    "Compare five trained machine learning classifiers "
    "and predict whether uploaded emails are spam."
)

# ------------------------------------------------------------
# LOAD SAVED MODELS
# ------------------------------------------------------------
MODEL_DIR = Path("models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        models[name] = joblib.load(MODEL_DIR / filename)

    features = joblib.load(
        MODEL_DIR / "feature_names.joblib"
    )
    return models, features

try:
    models, feature_names = load_models()
except Exception as error:
    st.error(f"Unable to load saved models: {error}")
    st.stop()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

st.sidebar.success("All five trained models loaded")

# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------
st.header("1. Model Performance Comparison")

comparison_path = Path("outputs/model_comparison.csv")

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)
else:
    st.info("Model comparison file is not available.")

# ------------------------------------------------------------
# CSV UPLOAD
# ------------------------------------------------------------
st.header("2. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing the 57 input features",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head(10), use_container_width=True)

    # Validate required features
    missing_features = [
        col for col in feature_names
        if col not in data.columns
    ]

    if missing_features:
        st.error(
            f"Missing required features: {missing_features}"
        )
        st.stop()

    X_input = data[feature_names].copy()

    # Validate numeric input
    try:
        X_input = X_input.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        st.error("All 57 input features must be numeric.")
        st.stop()

    if X_input.isnull().any().any():
        st.error("Input features contain missing values.")
        st.stop()

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------
    st.header("3. Classification Results")

    model = models[selected_model]

    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]

    results = data.copy()
    results["Predicted_Class"] = predictions
    results["Prediction_Label"] = np.where(
        predictions == 1, "Spam", "Not Spam"
    )
    results["Spam_Probability"] = probabilities

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Emails", len(results))
    col2.metric("Predicted Spam", int(sum(predictions)))
    col3.metric(
        "Predicted Not Spam",
        int(sum(predictions == 0))
    )

    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download Predictions",
        results.to_csv(index=False).encode("utf-8"),
        file_name="spam_predictions.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # EVALUATE IF ACTUAL LABELS ARE PROVIDED
    # --------------------------------------------------------
    if "spam" in data.columns:

        st.header("4. Evaluation Against Actual Labels")

        y_actual = data["spam"]

        if not y_actual.isin([0, 1]).all():
            st.error("The spam column must contain only 0 and 1.")
            st.stop()

        if y_actual.nunique() < 2:
            st.warning(
                "Both classes are required to calculate ROC AUC."
            )
            auc = np.nan
        else:
            auc = roc_auc_score(y_actual, probabilities)

        metrics = {
            "Accuracy": accuracy_score(y_actual, predictions),
            "AUC": auc,
            "Precision": precision_score(
                y_actual, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_actual, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_actual, predictions, zero_division=0
            ),
            "MCC": matthews_corrcoef(
                y_actual, predictions
            )
        }

        st.dataframe(
            pd.DataFrame([metrics]).round(4),
            use_container_width=True
        )

        cm = confusion_matrix(
            y_actual, predictions, labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Spam", "Spam"]
        ).plot(ax=ax, cmap="Blues", colorbar=False)

        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption(
    "BITS Pilani | ML Assignment 2 | "
    "Student ID: 2026AK05228"
)
