"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ate_ai.schema import validate_schema


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and validate its schema.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return validate_schema(df)
