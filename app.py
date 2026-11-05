"""BITS Pilani ML Assignment 2: comparison of pre-trained Spambase classifiers.

Models are loaded from models/*.joblib. No model is fitted or retrained in this app.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(
    page_title="Spam Detection | ML Assignment 2",
    page_icon="📧",
    layout="wide",
)
st.title("Email Spam Detection | Model Evaluation")
st.caption("BITS Pilani • Machine Learning Assignment 2 • Student ID: 2026AK05228")
st.write(
    "Compare five **pre-trained** classifiers using the same labeled test CSV. "
    "Every score and chart is recalculated from the uploaded test records; "
    "the models are **never retrained**."
)


@st.cache_resource
def load_models():
    """Load previously trained models and their exact expected feature order."""
    saved_models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    return saved_models, list(feature_names)


try:
    models, feature_names = load_models()
except Exception as exc:
    st.error(f"Saved models could not be loaded: {exc}")
    st.info("Ensure the five .joblib files and feature_names.joblib are in models/.")
    st.stop()

st.sidebar.header("Evaluation settings")
selected_model = st.sidebar.selectbox("Model to inspect", list(models))
st.sidebar.success("Five trained models loaded")

st.subheader("1. Select labeled test data")
st.write(
    "Upload the **held-out test data** (57 numerical features and the `spam` "
    "column with labels 0 or 1). For your assignment, use "
    "`outputs/test_data_with_labels.csv`."
)

source_choice = st.radio(
    "Data source", ["Upload CSV", "Use included test data"], horizontal=True
)

if source_choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload labeled test CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a labeled CSV to view the metrics, graphs, and predictions.")
        st.stop()
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        st.stop()
else:
    sample_paths = [
        ROOT / "test_data.csv",
        ROOT / "outputs" / "test_data_with_labels.csv",
    ]
    sample_path = next((p for p in sample_paths if p.is_file()), None)
    if sample_path is None:
        st.warning("No included sample was found. Switch to Upload CSV.")
        st.stop()
    data = pd.read_csv(sample_path)
    st.caption(f"Loaded included test dataset: `{sample_path.relative_to(ROOT)}`")

# Validate schema BEFORE computing any scores.
missing_columns = [c for c in [*feature_names, "spam"] if c not in data.columns]
if missing_columns:
    st.error(f"Required columns are missing: {missing_columns}")
    st.stop()
if data.empty:
    st.error("The CSV contains no test records.")
    st.stop()

try:
    test_features = data.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    y_true = pd.to_numeric(data["spam"], errors="raise")
except (TypeError, ValueError) as exc:
    st.error(f"The input features and target must be numerical: {exc}")
    st.stop()

if not np.isfinite(test_features.to_numpy(dtype=float)).all():
    st.error("Input features contain missing or infinite values.")
    st.stop()
if y_true.isna().any() or not y_true.isin([0, 1]).all():
    st.error("The target column `spam` must contain only 0 and 1, without missing values.")
    st.stop()
y_true = y_true.astype(int)

st.caption(f"Validated **{len(data):,} test emails**, **{len(feature_names)} features** and labeled target `spam`.")

# Evaluate every model against THIS uploaded test dataset; do not call .fit().
evaluations = {}
rows = []
for name, model in models.items():
    try:
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)[:, 1]
    except Exception as exc:
        st.error(f"Prediction failed for {name}: {exc}")
        st.stop()

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else np.nan,
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
    }
    rows.append({"Model": name, **metrics})
    evaluations[name] = {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }

scores = pd.DataFrame(rows).set_index("Model").sort_values(
    ["F1", "MCC"], ascending=False
)
selected = evaluations[selected_model]

compare_tab, inspect_tab, predictions_tab = st.tabs(
    ["Compare all models", "Selected model diagnostics", "Predictions & downloads"]
)

with compare_tab:
    st.subheader("2. Live comparison across all five models")
    st.dataframe(scores.round(4), use_container_width=True)
    st.download_button(
        "Download comparison metrics",
        data=scores.to_csv(index_label="Model").encode("utf-8"),
        file_name="comparison_uploaded_test_data.csv",
        mime="text/csv",
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    scores[METRICS].T.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("All six evaluation metrics by classifier")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("ROC curves — all five models")
    if y_true.nunique() == 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, result in evaluations.items():
            fpr, tpr, _ = roc_curve(y_true, result["probabilities"])
            line_width = 2.8 if name == selected_model else 1.5
            ax.plot(
                fpr,
                tpr,
                linewidth=line_width,
                label=f"{name} (AUC {result['metrics']['AUC']:.4f})",
            )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC comparison")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("ROC curves require at least one spam and one non-spam observation.")

with inspect_tab:
    st.subheader(f"3. Selected classifier: {selected_model}")
    metric_cols = st.columns(3)
    for i, metric in enumerate(METRICS):
        value = selected["metrics"][metric]
        metric_cols[i % 3].metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, selected["predictions"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not spam", "Spam"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{selected_model} — confusion matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    st.caption(f"TN: {tn}   |   FP: {fp}   |   FN: {fn}   |   TP: {tp}")

    st.markdown("**Classification report**")
    report = classification_report(
        y_true,
        selected["predictions"],
        labels=[0, 1],
        target_names=["Not spam", "Spam"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with predictions_tab:
    st.subheader(f"4. Predictions from {selected_model}")
    prediction_counts = pd.Series(selected["predictions"]).value_counts().reindex(
        [0, 1], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(["Not spam", "Spam"], prediction_counts.values)
    ax.bar_label(bars, padding=3)
    ax.set(ylabel="Number of emails", title="Predicted email distribution")
    ax.set_ylim(0, max(1, prediction_counts.max()) * 1.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    output = data.copy()
    output["predicted_spam"] = selected["predictions"]
    output["predicted_label"] = np.where(selected["predictions"] == 1, "Spam", "Not spam")
    output["spam_probability"] = selected["probabilities"]
    st.dataframe(output.head(100), use_container_width=True)
    st.caption("Preview displays up to 100 rows; download includes every test record.")
    st.download_button(
        "Download all predictions",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="spam_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Source: UCI Spambase • Test observations are never used for training.")
