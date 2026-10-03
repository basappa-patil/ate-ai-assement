"""Tests for the prediction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.ate_ai.features import FEATURE_COLUMNS, build_preprocessor
from src.ate_ai.predict import predict_records


@pytest.fixture()
def small_training_data():
    """Minimal DataFrame that mirrors real feature columns."""
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "Test_ID": rng.choice(["P301", "P302"], n),
        "Test_Name": rng.choice(["SCAN_CHAIN", "LEAKAGE_TEST"], n),
        "Lot_ID": rng.choice(["LOT_A01", "LOT_A02"], n),
        "Wafer_ID": rng.choice(["W01", "W02"], n),
        "VDD_V": rng.uniform(1.0, 1.2, n),
        "Temperature_C": rng.uniform(25, 125, n),
        "Retest_Count": rng.randint(0, 4, n),
    })
    # ~30% FAIL so both classes exist
    labels = np.array([1] * 18 + [0] * 42)
    rng.shuffle(labels)
    return df, pd.Series(labels, name="target")


@pytest.fixture()
def fitted_dummy_model(small_training_data):
    """A fitted Pipeline using DummyClassifier(strategy='most_frequent')."""
    X, y = small_training_data
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    model.fit(X, y)
    return model


def test_predict_records_columns(fitted_dummy_model, small_training_data):
    """predict_records adds fail_probability and prediction columns."""
    X, _ = small_training_data
    result = predict_records(fitted_dummy_model, X)
    assert "fail_probability" in result.columns
    assert "prediction" in result.columns
    assert len(result) == len(X)


def test_predict_records_values_range(fitted_dummy_model, small_training_data):
    """Probabilities should be in [0, 1]."""
    X, _ = small_training_data
    result = predict_records(fitted_dummy_model, X)
    assert result["fail_probability"].between(0.0, 1.0).all()


def test_threshold_zero_all_fail(fitted_dummy_model, small_training_data):
    """With threshold=0.0 every record should be FAIL."""
    X, _ = small_training_data
    result = predict_records(fitted_dummy_model, X, threshold=0.0)
    assert (result["prediction"] == "FAIL").all()


def test_threshold_one_all_pass(fitted_dummy_model, small_training_data):
    """With threshold=1.0 every record should be PASS.

    Note: predict_proba returns values strictly < 1.0 for
    DummyClassifier, so all records fall below the threshold.
    """
    X, _ = small_training_data
    result = predict_records(fitted_dummy_model, X, threshold=1.0)
    # DummyClassifier(strategy='most_frequent') assigns prob < 1.0
    # to the minority class, so all fail_probability < 1.0 → PASS.
    assert (result["prediction"] == "PASS").all()
