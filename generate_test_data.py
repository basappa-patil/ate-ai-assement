"""Generate synthetic ATE test data (~8000 rows) for development and testing."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NUM_DEVICES = 1600
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "ATE_AI_Engineering_Assessment.csv"

# Test definitions: (Test_ID, Test_Name, meas_low, meas_high, lower_limit, upper_limit)
TESTS = [
    ("P301", "SCAN_CHAIN",    0.5, 1.5, 0.4, 1.6),
    ("P302", "LEAKAGE_TEST",  0.01, 0.5, 0.005, 0.6),
    ("P303", "TIMING_TEST",   1.0, 5.0, 0.8, 5.5),
    ("P304", "VOLTAGE_REG",   0.9, 1.3, 0.85, 1.35),
    ("P305", "POWER_TEST",    0.5, 2.0, 0.4, 2.2),
]

LOTS = [f"LOT_A{str(i).zfill(2)}" for i in range(1, 11)]
WAFERS = [f"W{str(i).zfill(2)}" for i in range(1, 26)]

FAILURE_MODES = ["TIMING_VIOLATION", "LEAKAGE_EXCESS", "VOLTAGE_DRIFT", "POWER_OVERSHOOT"]
YIELD_IMPACTS = ["HIGH", "MEDIUM", "LOW"]


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Each device is tested with all 5 tests → ~8000 rows
    device_ids = [f"ATE3{str(i).zfill(5)}" for i in range(1, NUM_DEVICES + 1)]

    rows: list[dict] = []
    for dev_id in device_ids:
        lot_id = rng.choice(LOTS)
        wafer_id = rng.choice(WAFERS)
        for test_id, test_name, m_lo, m_hi, ll, ul in TESTS:
            vdd = rng.uniform(1.0, 1.2)
            temp = rng.uniform(25.0, 125.0)
            measured = rng.uniform(m_lo, m_hi)

            # ~85% PASS / 15% FAIL
            result = "PASS" if rng.random() < 0.85 else "FAIL"

            failure_mode = np.nan
            yield_impact = np.nan
            if result == "FAIL":
                failure_mode = rng.choice(FAILURE_MODES)
                yield_impact = rng.choice(YIELD_IMPACTS)

            retest_count = int(rng.choice([0, 0, 0, 1, 2, 3]))

            rows.append({
                "Device_ID": dev_id,
                "Test_ID": test_id,
                "Test_Name": test_name,
                "Lot_ID": lot_id,
                "Wafer_ID": wafer_id,
                "VDD_V": round(vdd, 4),
                "Temperature_C": round(temp, 2),
                "Measured_Value": round(measured, 6),
                "Lower_Limit": ll,
                "Upper_Limit": ul,
                "Result": result,
                "Failure_Mode": failure_mode,
                "Retest_Count": retest_count,
                "Yield_Impact": yield_impact,
            })

    df = pd.DataFrame(rows)

    # --- Inject ~3% outlier measurements ---
    n_outliers = int(len(df) * 0.03)
    outlier_idx = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[outlier_idx, "Measured_Value"] = df.loc[outlier_idx, "Measured_Value"] * rng.uniform(3.0, 5.0, size=n_outliers)

    # --- Inject ~5% missing values in selected columns ---
    for col in ["Measured_Value", "Temperature_C", "VDD_V"]:
        n_missing = int(len(df) * 0.05)
        missing_idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    # --- Inject ~2% exact duplicate rows ---
    n_dups = int(len(df) * 0.02)
    dup_idx = rng.choice(df.index, size=n_dups, replace=False)
    duplicates = df.loc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print(f"Generated {len(df)} rows → {OUTPUT_FILE}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Result distribution:\n{df['Result'].value_counts()}")
    print(f"  Missing values:\n{df.isnull().sum()}")
    print(f"  Duplicate rows injected: {n_dups}")


if __name__ == "__main__":
    main()
