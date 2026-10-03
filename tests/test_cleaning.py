"""Tests for src.ate_ai.cleaning module."""

import numpy as np
import pandas as pd
import pytest

from src.ate_ai.cleaning import clean_data


def _make_base_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal valid DataFrame for testing."""
    return pd.DataFrame({
        "Device_ID": [f"DEV{i:04d}" for i in range(n)],
        "Test_ID": ["P301"] * n,
        "Test_Name": ["SCAN_CHAIN"] * n,
        "Lot_ID": ["LOT_A01"] * n,
        "Wafer_ID": ["W01"] * n,
        "VDD_V": np.random.default_rng(42).uniform(1.0, 1.2, n),
        "Temperature_C": np.random.default_rng(42).uniform(25, 125, n),
        "Measured_Value": np.random.default_rng(42).uniform(0.5, 1.5, n),
        "Lower_Limit": [0.4] * n,
        "Upper_Limit": [1.6] * n,
        "Result": ["PASS"] * (n - 2) + ["FAIL"] * 2,
        "Failure_Mode": [np.nan] * (n - 2) + ["TIMING_VIOLATION"] * 2,
        "Retest_Count": [0] * n,
        "Yield_Impact": [np.nan] * (n - 2) + ["HIGH"] * 2,
    })


class TestDuplicateRemoval:
    """Verify that exact duplicate rows are removed."""

    def test_known_duplicates_removed(self):
        df = _make_base_df(5)
        # Inject 3 exact duplicates
        dups = df.iloc[:3].copy()
        df_with_dups = pd.concat([df, dups], ignore_index=True)
        assert len(df_with_dups) == 8

        cleaned, report = clean_data(df_with_dups)
        assert report["duplicates_removed"] == 3
        assert report["cleaned_rows"] == 5

    def test_no_duplicates_unchanged(self):
        df = _make_base_df(5)
        cleaned, report = clean_data(df)
        assert report["duplicates_removed"] == 0
        assert len(cleaned) == 5


class TestMissingValueHandling:
    """Verify that NaN values are imputed correctly."""

    def test_numeric_nans_filled(self):
        df = _make_base_df(10)
        # Inject NaNs in numeric columns
        df.loc[0, "VDD_V"] = np.nan
        df.loc[1, "Temperature_C"] = np.nan
        df.loc[2, "Measured_Value"] = np.nan
        df.loc[3, "Retest_Count"] = np.nan

        cleaned, report = clean_data(df)
        for col in ["VDD_V", "Temperature_C", "Measured_Value", "Retest_Count"]:
            assert cleaned[col].isnull().sum() == 0, f"{col} still has NaNs"

    def test_categorical_nans_filled(self):
        df = _make_base_df(10)
        df.loc[0, "Test_Name"] = np.nan
        df.loc[1, "Lot_ID"] = np.nan

        cleaned, _ = clean_data(df)
        assert cleaned["Test_Name"].isnull().sum() == 0
        assert cleaned["Lot_ID"].isnull().sum() == 0


class TestResultNormalization:
    """Verify that Result values are normalized to PASS or FAIL."""

    def test_lowercase_pass(self):
        df = _make_base_df(5)
        df.loc[0, "Result"] = "pass"
        cleaned, _ = clean_data(df)
        assert cleaned.loc[0, "Result"] == "PASS"

    def test_mixed_case_pass(self):
        df = _make_base_df(5)
        df.loc[0, "Result"] = "Pass"
        cleaned, _ = clean_data(df)
        assert cleaned.loc[0, "Result"] == "PASS"

    def test_whitespace_fail(self):
        df = _make_base_df(5)
        df.loc[0, "Result"] = " FAIL "
        cleaned, _ = clean_data(df)
        assert cleaned.loc[0, "Result"] == "FAIL"

    def test_only_pass_or_fail_after_cleaning(self):
        df = _make_base_df(10)
        df.loc[0, "Result"] = "pass"
        df.loc[1, "Result"] = "Pass"
        df.loc[2, "Result"] = " FAIL "
        cleaned, _ = clean_data(df)
        assert set(cleaned["Result"].unique()).issubset({"PASS", "FAIL"})


class TestCleaningReport:
    """Verify the structure of the cleaning report dict."""

    def test_report_keys(self):
        df = _make_base_df(5)
        _, report = clean_data(df)
        expected_keys = {
            "original_rows",
            "cleaned_rows",
            "duplicates_removed",
            "missing_values_before",
            "missing_values_after",
        }
        assert set(report.keys()) == expected_keys

    def test_report_types(self):
        df = _make_base_df(5)
        _, report = clean_data(df)
        assert isinstance(report["original_rows"], int)
        assert isinstance(report["cleaned_rows"], int)
        assert isinstance(report["duplicates_removed"], int)
        assert isinstance(report["missing_values_before"], dict)
        assert isinstance(report["missing_values_after"], dict)
