"""Anomaly detection for ATE measurement records."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive anomaly-detection features from raw measurements.

    Features
    --------
    - normalized_distance: how far the measurement is from the spec
      midpoint, scaled by spec width.
    - Temperature_C, VDD_V, Retest_Count: carried through as-is.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns Measured_Value, Lower_Limit,
        Upper_Limit, Temperature_C, VDD_V, Retest_Count.

    Returns
    -------
    pd.DataFrame
        Feature matrix with four columns.
    """
    midpoint = (df["Lower_Limit"] + df["Upper_Limit"]) / 2
    spec_width = df["Upper_Limit"] - df["Lower_Limit"]

    # Replace zero spec_width with NaN to avoid division by zero
    spec_width = spec_width.replace(0, np.nan)

    normalized_distance = (df["Measured_Value"] - midpoint) / spec_width

    features = pd.DataFrame({
        "normalized_distance": normalized_distance,
        "Temperature_C": df["Temperature_C"],
        "VDD_V": df["VDD_V"],
        "Retest_Count": df["Retest_Count"],
    })
    return features


def fit_anomaly_detector(df: pd.DataFrame) -> tuple:
    """Fit an IsolationForest anomaly detector and score the data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned ATE DataFrame.

    Returns
    -------
    tuple[Pipeline, pd.DataFrame]
        The fitted pipeline and the input DataFrame augmented with
        ``anomaly_score`` (higher = more anomalous) and ``is_anomaly``
        columns, sorted by anomaly_score descending.
    """
    features = build_anomaly_features(df)

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("detector", IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
        )),
    ])

    # Impute any remaining NaN in features before fitting
    features_filled = features.fillna(features.median())

    pipeline.fit(features_filled)

    # Score: negate decision_function so higher = more anomalous
    anomaly_score = -pipeline.decision_function(features_filled)
    is_anomaly = pipeline.predict(features_filled) == -1

    scored_df = df.copy()
    scored_df["anomaly_score"] = anomaly_score
    scored_df["is_anomaly"] = is_anomaly
    scored_df = scored_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    return pipeline, scored_df
