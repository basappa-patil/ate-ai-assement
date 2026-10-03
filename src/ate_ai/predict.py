"""Inference utilities for the trained failure-prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.ate_ai import config


def load_artifact(path: str | Path | None = None):
    """Load a persisted sklearn Pipeline from disk.

    Parameters
    ----------
    path : str | Path | None
        Path to the joblib file.  Defaults to
        ``config.MODELS_DIR / 'failure_pipeline.joblib'``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded pipeline.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if path is None:
        path = config.MODELS_DIR / "failure_pipeline.joblib"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_records(
    model,
    records: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run inference on a DataFrame of records.

    Parameters
    ----------
    model : sklearn Pipeline
        Fitted pipeline that exposes ``predict_proba``.
    records : pd.DataFrame
        Input records (must contain the feature columns the model expects).
    threshold : float
        Probability cutoff — records with fail probability >= threshold
        are labelled FAIL.

    Returns
    -------
    pd.DataFrame
        A copy of *records* with two added columns:
        ``fail_probability`` and ``prediction``.
    """
    result = records.copy()
    probas = model.predict_proba(records)
    result["fail_probability"] = probas[:, 1]
    result["prediction"] = result["fail_probability"].apply(
        lambda p: "FAIL" if p >= threshold else "PASS"
    )
    return result
