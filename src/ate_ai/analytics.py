"""Yield analytics computations for ATE test data."""

from __future__ import annotations

import pandas as pd


def compute_yield_analytics(df: pd.DataFrame) -> dict:
    """Compute yield analytics from a cleaned ATE test DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with at least columns: Result, Test_ID,
        Lot_ID, Wafer_ID, Failure_Mode.

    Returns
    -------
    dict
        Keys:
        - overall_yield (float 0-100)
        - overall_fail_rate (float 0-100)
        - yield_by_test (DataFrame)
        - yield_by_lot (DataFrame)
        - yield_by_wafer (DataFrame)
        - top_failing_tests (DataFrame)
        - failure_mode_distribution (DataFrame)
    """
    total = len(df)
    pass_count = (df["Result"] == "PASS").sum()
    fail_count = total - pass_count

    overall_yield = (pass_count / total * 100) if total else 0.0
    overall_fail_rate = (fail_count / total * 100) if total else 0.0

    # --- Yield by test ---
    test_groups = df.groupby("Test_ID")["Result"]
    yield_by_test = test_groups.agg(
        total="count",
        pass_count=lambda s: (s == "PASS").sum(),
    ).reset_index()
    yield_by_test["fail_count"] = yield_by_test["total"] - yield_by_test["pass_count"]
    yield_by_test["yield_pct"] = (
        yield_by_test["pass_count"] / yield_by_test["total"] * 100
    )
    yield_by_test["fail_rate"] = (
        yield_by_test["fail_count"] / yield_by_test["total"] * 100
    )

    # --- Yield by lot ---
    lot_groups = df.groupby("Lot_ID")["Result"]
    yield_by_lot = lot_groups.agg(
        total="count",
        pass_count=lambda s: (s == "PASS").sum(),
    ).reset_index()
    yield_by_lot["yield_pct"] = (
        yield_by_lot["pass_count"] / yield_by_lot["total"] * 100
    )
    yield_by_lot = yield_by_lot[["Lot_ID", "yield_pct"]]

    # --- Yield by wafer ---
    wafer_groups = df.groupby("Wafer_ID")["Result"]
    yield_by_wafer = wafer_groups.agg(
        total="count",
        pass_count=lambda s: (s == "PASS").sum(),
    ).reset_index()
    yield_by_wafer["yield_pct"] = (
        yield_by_wafer["pass_count"] / yield_by_wafer["total"] * 100
    )
    yield_by_wafer = yield_by_wafer[["Wafer_ID", "yield_pct"]]

    # --- Top failing tests ---
    top_failing = yield_by_test[["Test_ID", "fail_count", "fail_rate", "total"]].copy()
    top_failing = top_failing.rename(columns={"total": "sample_size"})
    top_failing = top_failing.sort_values("fail_count", ascending=False).reset_index(
        drop=True
    )

    # --- Failure mode distribution ---
    if "Failure_Mode" in df.columns:
        fm_counts = (
            df["Failure_Mode"]
            .value_counts()
            .reset_index()
        )
        fm_counts.columns = ["Failure_Mode", "count"]
        fm_total = fm_counts["count"].sum()
        fm_counts["percentage"] = fm_counts["count"] / fm_total * 100 if fm_total else 0.0
    else:
        fm_counts = pd.DataFrame(columns=["Failure_Mode", "count", "percentage"])

    return {
        "overall_yield": float(overall_yield),
        "overall_fail_rate": float(overall_fail_rate),
        "yield_by_test": yield_by_test,
        "yield_by_lot": yield_by_lot,
        "yield_by_wafer": yield_by_wafer,
        "top_failing_tests": top_failing,
        "failure_mode_distribution": fm_counts,
    }
