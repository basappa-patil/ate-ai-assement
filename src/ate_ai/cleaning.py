"""Data cleaning pipeline for ATE test records."""

from __future__ import annotations

import pandas as pd


# Columns that receive median imputation
_NUMERIC_COLS: list[str] = [
    "VDD_V",
    "Temperature_C",
    "Measured_Value",
    "Lower_Limit",
    "Upper_Limit",
    "Retest_Count",
]

# Columns that receive mode imputation (fallback 'UNKNOWN')
_CATEGORICAL_COLS: list[str] = [
    "Test_Name",
    "Lot_ID",
    "Wafer_ID",
    "Failure_Mode",
    "Yield_Impact",
]


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean an ATE test DataFrame.

    Steps
    -----
    1. Normalize *Result* to uppercase PASS / FAIL.
    2. Remove exact duplicate rows.
    3. Median-impute numeric columns.
    4. Mode-impute categorical columns (fallback ``'UNKNOWN'``).
    5. Leave Device_ID and Test_ID untouched.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned DataFrame and a cleaning report dictionary.
    """
    df = df.copy()
    original_rows = len(df)
    missing_before = df.isnull().sum().to_dict()

    # --- 1. Normalize Result ---
    if "Result" in df.columns:
        df["Result"] = df["Result"].astype(str).str.upper().str.strip()
        # Anything that is not exactly PASS or FAIL becomes NaN, then fill
        df.loc[~df["Result"].isin(["PASS", "FAIL"]), "Result"] = pd.NA
        df["Result"] = df["Result"].ffill().fillna("FAIL")

    # --- 2. Remove exact duplicates ---
    df = df.drop_duplicates()
    duplicates_removed = original_rows - len(df)

    # --- 3. Median impute numeric columns ---
    for col in _NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # --- 4. Mode impute categorical columns ---
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
            df[col] = df[col].fillna(fill_val)

    missing_after = df.isnull().sum().to_dict()

    cleaning_report: dict = {
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
    }

    return df, cleaning_report
