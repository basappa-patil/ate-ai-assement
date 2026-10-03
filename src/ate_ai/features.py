"""Feature engineering and data preparation for failure prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Leakage-safe allow-list — excludes post-test / target columns:
# Measured_Value, Failure_Mode, Yield_Impact, Result
FEATURE_COLUMNS: list[str] = [
    "Test_ID",
    "Test_Name",
    "Lot_ID",
    "Wafer_ID",
    "VDD_V",
    "Temperature_C",
    "Retest_Count",
]

NUMERIC_FEATURES: list[str] = ["VDD_V", "Temperature_C", "Retest_Count"]
CATEGORICAL_FEATURES: list[str] = ["Test_ID", "Test_Name", "Lot_ID", "Wafer_ID"]


def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for ATE feature columns.

    Returns
    -------
    ColumnTransformer
        Transformer with numeric (median impute + scale) and
        categorical (mode impute + one-hot encode) pipelines.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Encode the target and perform a stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a ``Result`` column (PASS/FAIL).
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test) where target is 0=PASS, 1=FAIL.
    """
    target = (df["Result"] == "FAIL").astype(int)
    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state, stratify=target,
    )
    return X_train, X_test, y_train, y_test
