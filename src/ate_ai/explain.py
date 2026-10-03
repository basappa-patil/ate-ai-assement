"""Human-readable explanations for anomaly and failure predictions."""

from __future__ import annotations

import pandas as pd


def generate_explanation(
    record: pd.Series,
    anomaly_score: float,
    feature_importances: dict | None = None,
) -> str:
    """Generate a plain-English explanation for a single record.

    Parameters
    ----------
    record : pd.Series
        A row from the scored ATE DataFrame.
    anomaly_score : float
        The anomaly score assigned by the detector (higher = more anomalous).
    feature_importances : dict | None
        Optional mapping of feature names to importance values.

    Returns
    -------
    str
        Multi-line explanation text.
    """
    lines: list[str] = []

    # --- Observed evidence ---
    lines.append("Observed evidence:")
    if "Measured_Value" in record.index:
        lower = record.get("Lower_Limit", "N/A")
        upper = record.get("Upper_Limit", "N/A")
        lines.append(
            f"  - Measured Value: {record['Measured_Value']:.4f} "
            f"(limits: {lower} – {upper})"
        )
    if "VDD_V" in record.index:
        lines.append(f"  - VDD: {record['VDD_V']:.3f} V")
    if "Temperature_C" in record.index:
        lines.append(f"  - Temperature: {record['Temperature_C']:.1f} °C")
    if "Retest_Count" in record.index:
        lines.append(f"  - Retest Count: {int(record['Retest_Count'])}")

    # --- Anomaly score ---
    if anomaly_score > 0.5:
        level = "high"
    elif anomaly_score > 0.2:
        level = "moderate"
    else:
        level = "low"
    lines.append(f"Anomaly score: {anomaly_score:.4f} ({level})")

    # --- Possible cause ---
    lines.append("Possible cause:")
    failure_mode = record.get("Failure_Mode")
    has_failure_mode = (
        failure_mode is not None
        and not (isinstance(failure_mode, float) and pd.isna(failure_mode))
        and str(failure_mode).strip() not in ("", "nan", "NaN", "UNKNOWN")
    )

    if has_failure_mode:
        lines.append(f"  - Failure Mode: {failure_mode}")

    if feature_importances:
        sorted_feats = sorted(
            feature_importances.items(), key=lambda kv: kv[1], reverse=True,
        )
        top = sorted_feats[:3]
        lines.append("  - Top contributing features: " + ", ".join(
            f"{name} ({importance:.4f})" for name, importance in top
        ))
    elif not has_failure_mode:
        lines.append(
            "  - Insufficient evidence to determine root cause "
            "— manual investigation recommended."
        )

    return "\n".join(lines)
