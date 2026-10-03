"""Tests for the anomaly detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ate_ai.anomaly import build_anomaly_features, fit_anomaly_detector


@pytest.fixture()
def synthetic_ate_data():
    """Small DataFrame with a couple of obvious outliers."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "Device_ID": [f"DEV{i:04d}" for i in range(n)],
        "Test_ID": rng.choice(["P301", "P302"], n),
        "Test_Name": rng.choice(["SCAN_CHAIN", "LEAKAGE_TEST"], n),
        "Lot_ID": rng.choice(["LOT_A01", "LOT_A02"], n),
        "Wafer_ID": rng.choice(["W01", "W02"], n),
        "VDD_V": rng.uniform(1.0, 1.2, n),
        "Temperature_C": rng.uniform(25, 85, n),
        "Measured_Value": rng.uniform(0.5, 1.5, n),
        "Lower_Limit": np.full(n, 0.4),
        "Upper_Limit": np.full(n, 1.6),
        "Result": rng.choice(["PASS", "FAIL"], n, p=[0.85, 0.15]),
        "Retest_Count": rng.randint(0, 4, n),
    })

    # Inject 3 extreme outliers
    df.loc[0, "Measured_Value"] = 10.0
    df.loc[1, "Measured_Value"] = -5.0
    df.loc[2, "Temperature_C"] = 300.0

    return df


def test_fit_anomaly_detector_returns_tuple(synthetic_ate_data):
    """fit_anomaly_detector returns (Pipeline, DataFrame)."""
    pipeline, scored_df = fit_anomaly_detector(synthetic_ate_data)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(scored_df, pd.DataFrame)


def test_scored_df_has_required_columns(synthetic_ate_data):
    """Scored DataFrame must have anomaly_score and is_anomaly."""
    _, scored_df = fit_anomaly_detector(synthetic_ate_data)
    assert "anomaly_score" in scored_df.columns
    assert "is_anomaly" in scored_df.columns


def test_scored_df_sorted_descending(synthetic_ate_data):
    """Rows should be sorted by anomaly_score descending."""
    _, scored_df = fit_anomaly_detector(synthetic_ate_data)
    scores = scored_df["anomaly_score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_is_anomaly_boolean(synthetic_ate_data):
    """is_anomaly should be boolean."""
    _, scored_df = fit_anomaly_detector(synthetic_ate_data)
    assert scored_df["is_anomaly"].dtype == bool


def test_build_anomaly_features_columns(synthetic_ate_data):
    """build_anomaly_features returns the expected four columns."""
    feats = build_anomaly_features(synthetic_ate_data)
    assert list(feats.columns) == [
        "normalized_distance",
        "Temperature_C",
        "VDD_V",
        "Retest_Count",
    ]
    assert len(feats) == len(synthetic_ate_data)


def test_outliers_score_high(synthetic_ate_data):
    """Injected outliers should appear near the top of scored output."""
    _, scored_df = fit_anomaly_detector(synthetic_ate_data)
    # The three injected outlier devices should be among the top-10
    top10_devices = set(scored_df.head(10)["Device_ID"].values)
    # At least 2 of the 3 extreme outliers should surface in top-10
    injected = {"DEV0000", "DEV0001", "DEV0002"}
    assert len(injected & top10_devices) >= 2, (
        f"Expected at least 2 injected outliers in top-10, "
        f"got {injected & top10_devices}"
    )
