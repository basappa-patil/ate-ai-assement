"""Tests for src.ate_ai.analytics module."""

import numpy as np
import pandas as pd
import pytest

from src.ate_ai.analytics import compute_yield_analytics


def _make_known_df() -> pd.DataFrame:
    """Create a DataFrame with known PASS/FAIL distribution.

    10 rows: 7 PASS, 3 FAIL → 70% yield, 30% fail rate.
    """
    return pd.DataFrame({
        "Device_ID": [f"DEV{i:04d}" for i in range(10)],
        "Test_ID": ["P301"] * 5 + ["P302"] * 5,
        "Test_Name": ["SCAN_CHAIN"] * 5 + ["LEAKAGE_TEST"] * 5,
        "Lot_ID": ["LOT_A01"] * 5 + ["LOT_A02"] * 5,
        "Wafer_ID": ["W01"] * 5 + ["W02"] * 5,
        "VDD_V": [1.1] * 10,
        "Temperature_C": [75.0] * 10,
        "Measured_Value": [1.0] * 10,
        "Lower_Limit": [0.4] * 10,
        "Upper_Limit": [1.6] * 10,
        "Result": ["PASS"] * 4 + ["FAIL"] + ["PASS"] * 3 + ["FAIL"] * 2,
        "Failure_Mode": [np.nan] * 4 + ["TIMING_VIOLATION"] + [np.nan] * 3 + ["LEAKAGE_EXCESS", "VOLTAGE_DRIFT"],
        "Retest_Count": [0] * 10,
        "Yield_Impact": [np.nan] * 4 + ["HIGH"] + [np.nan] * 3 + ["MEDIUM", "LOW"],
    })


class TestOverallYield:
    """Verify overall yield and fail rate calculations."""

    def test_overall_yield(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        assert result["overall_yield"] == pytest.approx(70.0)

    def test_overall_fail_rate(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        assert result["overall_fail_rate"] == pytest.approx(30.0)


class TestYieldByTest:
    """Verify yield_by_test DataFrame structure and values."""

    def test_columns(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        ybt = result["yield_by_test"]
        for col in ["Test_ID", "total", "pass_count", "fail_count", "yield_pct", "fail_rate"]:
            assert col in ybt.columns, f"Missing column: {col}"

    def test_values_p301(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        ybt = result["yield_by_test"]
        p301 = ybt[ybt["Test_ID"] == "P301"].iloc[0]
        assert p301["total"] == 5
        assert p301["pass_count"] == 4
        assert p301["fail_count"] == 1
        assert p301["yield_pct"] == pytest.approx(80.0)

    def test_values_p302(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        ybt = result["yield_by_test"]
        p302 = ybt[ybt["Test_ID"] == "P302"].iloc[0]
        assert p302["total"] == 5
        assert p302["pass_count"] == 3
        assert p302["fail_count"] == 2
        assert p302["yield_pct"] == pytest.approx(60.0)


class TestResultKeys:
    """Verify that all expected keys are present in the result dict."""

    def test_all_keys_present(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        expected_keys = {
            "overall_yield",
            "overall_fail_rate",
            "yield_by_test",
            "yield_by_lot",
            "yield_by_wafer",
            "top_failing_tests",
            "failure_mode_distribution",
        }
        assert set(result.keys()) == expected_keys

    def test_dataframe_types(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        for key in ["yield_by_test", "yield_by_lot", "yield_by_wafer",
                     "top_failing_tests", "failure_mode_distribution"]:
            assert isinstance(result[key], pd.DataFrame), f"{key} is not a DataFrame"


class TestTopFailingTests:
    """Verify top_failing_tests structure."""

    def test_columns(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        tft = result["top_failing_tests"]
        for col in ["Test_ID", "fail_count", "fail_rate", "sample_size"]:
            assert col in tft.columns, f"Missing column: {col}"

    def test_sorted_descending(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        tft = result["top_failing_tests"]
        fail_counts = tft["fail_count"].tolist()
        assert fail_counts == sorted(fail_counts, reverse=True)


class TestFailureModeDistribution:
    """Verify failure_mode_distribution structure."""

    def test_columns(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        fmd = result["failure_mode_distribution"]
        for col in ["Failure_Mode", "count", "percentage"]:
            assert col in fmd.columns, f"Missing column: {col}"

    def test_percentages_sum_to_100(self):
        df = _make_known_df()
        result = compute_yield_analytics(df)
        fmd = result["failure_mode_distribution"]
        if len(fmd) > 0:
            assert fmd["percentage"].sum() == pytest.approx(100.0, abs=0.1)
