"""Schema validation for ATE test data."""

import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "Device_ID",
    "Test_ID",
    "Test_Name",
    "Lot_ID",
    "Wafer_ID",
    "VDD_V",
    "Temperature_C",
    "Measured_Value",
    "Lower_Limit",
    "Upper_Limit",
    "Result",
]


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check that all required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, unchanged.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )
    return df
