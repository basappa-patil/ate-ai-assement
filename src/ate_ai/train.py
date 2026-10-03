"""Train and evaluate failure-prediction models."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.ate_ai import config
from src.ate_ai.cleaning import clean_data
from src.ate_ai.data import load_data
from src.ate_ai.features import build_preprocessor, prepare_data


def _evaluate_model(model, X_test, y_test) -> dict:
    """Return a metrics dictionary for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "confusion_matrix": cm,
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }


def train_and_evaluate(data_path: str) -> dict:
    """Load data, train three models, evaluate, and persist the best.

    Parameters
    ----------
    data_path : str
        Path to the raw ATE CSV.

    Returns
    -------
    dict
        Metrics for all three models keyed by name.
    """
    # Load and clean
    df = load_data(data_path)
    df, _ = clean_data(df)

    # Prepare features and target
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Define candidate models
    preprocessor = build_preprocessor()

    candidates = {
        "baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }

    # Train and evaluate each candidate
    metrics: dict = {}
    best_name: str = ""
    best_f1: float = -1.0
    best_model = None

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        m = _evaluate_model(pipeline, X_test, y_test)
        metrics[name] = m
        print(f"{name:25s}  F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_name = name
            best_model = pipeline

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    # Persist best model
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "failure_pipeline.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")

    # Persist metrics report
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "metrics_report.json"

    # Convert any remaining numpy types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = json.loads(json.dumps(metrics, default=_convert))
    with open(report_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved metrics to {report_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ATE failure-prediction models")
    parser.add_argument("--data", required=True, help="Path to ATE CSV file")
    args = parser.parse_args()
    train_and_evaluate(args.data)
