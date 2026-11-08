"""
Breast Cancer Diagnosis - Classifier Comparison Studio
ML Assignment 2 | Gopi | 2025NS04112 | BITS Pilani WILP M.Tech (DSE)
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))
import train_models as tm  # noqa: E402

TARGET = tm.TARGET
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]
MODEL_NAMES = list(tm.build_models().keys())

st.set_page_config(page_title="Tumour Classifier Studio | Gopi", page_icon="🩺",
                   layout="wide")


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    """Load saved pipelines; if missing or version-incompatible, retrain once."""
    try:
        return {n: joblib.load(MODEL_DIR / tm.slug(n)) for n in MODEL_NAMES}
    except Exception:
        tm.main()
        return {n: joblib.load(MODEL_DIR / tm.slug(n)) for n in MODEL_NAMES}


def score_all(model, X, y):
    y_pred = model.predict(X)
    y_score = tm.positive_scores(model, X)
    auc = roc_auc_score(y, y_score) if y.nunique() == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_score


models = load_models()
feature_cols = list(next(iter(models.values())).feature_names_in_)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🩺 Tumour Classifier Studio")
    st.caption("ML Assignment 2 · M.Tech DSE · BITS Pilani WILP")
    st.markdown("**Student:** Gopi  \n**Roll No:** 2025NS04112")
    st.divider()

    st.markdown("#### 1 · Upload test data")
    uploaded = st.file_uploader("CSV with the 30 feature columns + `diagnosis`",
                                type=["csv"])
    sample_path = ROOT / "test_data.csv"
    st.download_button("⬇️ Download sample test_data.csv",
                       sample_path.read_bytes(), "test_data.csv", "text/csv")
    use_sample = st.checkbox("No file? Use bundled test_data.csv",
                             value=uploaded is None)

    st.markdown("#### 2 · Choose a model")
    model_name = st.selectbox("Classifier", MODEL_NAMES, index=0)
    threshold_note = st.empty()

# ---------------------------------------------------------------- data
st.title("Breast Cancer Diagnosis — Classifier Comparison")
st.write("Evaluate five classifiers trained on the **Breast Cancer Wisconsin "
         "(Diagnostic)** dataset. Positive class = **malignant**.")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = f"Uploaded file: `{uploaded.name}`"
elif use_sample:
    df = pd.read_csv(sample_path)
    source = "Bundled `test_data.csv` (25 % hold-out split, seed 4112)"
else:
    st.info("👈 Upload a CSV or tick *Use bundled test_data.csv* in the sidebar.")
    st.stop()

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} feature column(s), e.g. "
             f"`{missing[0]}`. Download the sample CSV to see the expected format.")
    st.stop()

X = df[feature_cols]
has_labels = TARGET in df.columns
st.caption(f"{source} · {len(df)} rows · {len(feature_cols)} features")

with st.expander("Preview data"):
    st.dataframe(df.head(20), width="stretch")

model = models[model_name]

if not has_labels:
    st.warning(f"No `{TARGET}` column found — showing predictions only "
               "(metrics need true labels).")
    preds = model.predict(X)
    out = df.copy()
    out["predicted_diagnosis"] = preds
    st.dataframe(out, width="stretch")
    st.download_button("Download predictions", out.to_csv(index=False),
                       "predictions.csv", "text/csv")
    st.stop()

y = df[TARGET].astype(int)

tab_model, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all 5 models"])

# ---------------------------------------------------------------- single model
with tab_model:
    metrics, y_pred, y_score = score_all(model, X, y)

    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="content")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Missed malignant cases (FN): **{fn}** · "
                   f"False alarms (FP): **{fp}**")

    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y, y_score)
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
        ax2.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['AUC']:.3f}")
        ax2.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2, width="content")

    st.subheader("Classification report")
    rep = classification_report(y, y_pred, labels=[0, 1],
                                target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

    out = df.copy()
    out["predicted_diagnosis"] = y_pred
    st.download_button(f"Download {model_name} predictions",
                       out.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------------------------------------------- comparison
with tab_compare:
    rows = []
    for n, m in models.items():
        mt, _, _ = score_all(m, X, y)
        rows.append({"ML Model Name": n, **mt})
    table = pd.DataFrame(rows).set_index("ML Model Name").round(4)

    st.subheader("Metric comparison on this data")
    st.dataframe(table.style.highlight_max(axis=0, color="#c8e6c9"),
                 width="stretch")

    best = table["MCC"].idxmax()
    st.success(f"🏆 Best model by MCC on this data: **{best}** "
               f"(MCC = {table.loc[best, 'MCC']:.4f})")

    metric_pick = st.radio("Plot metric", list(table.columns), horizontal=True,
                           index=5)
    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    s = table[metric_pick].sort_values()
    ax3.barh(s.index, s.values,
             color=["#2e7d32" if i == best else "#90a4ae" for i in s.index])
    ax3.set_xlim(max(0, s.min() - 0.1), 1.0)
    for i, v in enumerate(s.values):
        ax3.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax3.set_xlabel(metric_pick)
    st.pyplot(fig3)

st.divider()
st.caption("Built by Gopi (2025NS04112) for BITS Pilani WILP · ML Assignment 2")
