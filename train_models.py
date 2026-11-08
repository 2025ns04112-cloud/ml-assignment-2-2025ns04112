"""
ML Assignment 2 - Model training script
Student : Gopi  |  Roll No: 2025NS04112
Dataset : Breast Cancer Wisconsin (Diagnostic) - sklearn.datasets (originally UCI)

Trains 5 classifiers on the same split, computes 6 metrics each,
saves fitted pipelines (*.pkl), the held-out test set (test_data.csv)
and a metrics table (model/metrics.csv).
Run from the project root:  python model/train_models.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 4112  # last 4 digits of roll number 2025NS04112
TARGET = "diagnosis"  # 1 = malignant (positive class), 0 = benign
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT


def load_data():
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # sklearn encodes 0 = malignant, 1 = benign; flip so malignant is the positive class
    df[TARGET] = 1 - df.pop("target")
    return df


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=0.5, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=4,
                                           random_state=SEED)),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", random_state=SEED)),
        ]),
    }


def positive_scores(model, X):
    """Score for the positive class: probability if available, else SVM margin."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = positive_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def slug(name):
    return name.lower().replace(" ", "_") + ".pkl"


def main():
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Dataset shape: {df.shape} | class counts: {y.value_counts().to_dict()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    test_df = X_te.copy()
    test_df[TARGET] = y_te
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        joblib.dump(model, MODEL_DIR / slug(name))
        m = evaluate(model, X_te, y_te)
        rows.append({"ML Model Name": name, **m})
        print(f"{name:<24}" + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    table = pd.DataFrame(rows).round(4)
    table.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
