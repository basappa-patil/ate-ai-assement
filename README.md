# ATE AI Engineering Assessment

An end-to-end AI/ML pipeline for Automated Test Equipment (ATE) semiconductor test data — from synthetic data generation to an interactive Streamlit dashboard with yield analytics, failure prediction, anomaly detection, and explainability.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic test data (~8,000 rows)
python generate_test_data.py

# 3. Train failure prediction models
python -m src.ate_ai.train --data data/raw/ATE_AI_Engineering_Assessment.csv

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py

# 5. Run tests
python -m pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit App (app/)                    │
│   Upload → Process → Dashboard → Investigate → AI → ML  │
└────────┬──────────┬──────────┬──────────┬───────────────┘
         │          │          │          │
    ┌────▼──┐  ┌───▼───┐  ┌──▼───┐  ┌──▼──────┐
    │ data  │  │ clean │  │ anal │  │ predict │
    │ .py   │  │ ing   │  │ ytic │  │ .py     │
    │       │  │ .py   │  │ s.py │  │         │
    └───┬───┘  └───┬───┘  └──┬───┘  └────┬────┘
        │          │          │           │
    ┌───▼──────────▼──────────▼───────────▼───┐
    │         schema.py / config.py           │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  features.py → train.py → models/*.joblib│
    │  anomaly.py  → IsolationForest pipeline │
    │  explain.py  → text explanations        │
    └─────────────────────────────────────────┘
```

**Data flow**: CSV → `data.load_data` (validate) → `cleaning.clean_data` (dedupe, impute) → `analytics.compute_yield_analytics` (KPIs, pivot tables) → Plotly charts in Streamlit.

**ML flow**: cleaned data → `features.prepare_data` (leakage-safe split) → `train.train_and_evaluate` (Baseline, LR, RF) → best model saved to `models/failure_pipeline.joblib` → `predict.predict_records` for inference.

**Anomaly flow**: cleaned data → `anomaly.build_anomaly_features` → `anomaly.fit_anomaly_detector` (IsolationForest) → scored DataFrame → `explain.generate_explanation`.

## Project Structure

```
ate-ai-assessment/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # 6-screen Streamlit dashboard
├── data/
│   ├── raw/                      # Generated CSV (not checked in)
│   └── processed/                # Cleaned outputs (not checked in)
├── models/                       # Trained model artifacts (.joblib)
├── reports/
│   ├── figures/                  # Saved chart images
│   ├── engineering_summary.md    # Engineering summary document
│   └── metrics_report.json       # Training metrics (generated)
├── src/
│   └── ate_ai/
│       ├── __init__.py           # Re-exports: load_data, clean_data, compute_yield_analytics
│       ├── config.py             # Path constants, RANDOM_SEED
│       ├── schema.py             # Column validation
│       ├── data.py               # CSV loading + schema check
│       ├── cleaning.py           # Dedup, imputation, normalisation
│       ├── analytics.py          # Yield KPIs and aggregations
│       ├── features.py           # Leakage-safe feature engineering
│       ├── train.py              # Model training (Baseline, LR, RF)
│       ├── predict.py            # Inference from saved model
│       ├── anomaly.py            # IsolationForest anomaly detection
│       └── explain.py            # Human-readable explanations
├── tests/
│   ├── test_cleaning.py
│   ├── test_analytics.py
│   ├── test_prediction.py
│   └── test_anomaly.py
├── generate_test_data.py         # Synthetic data generator
├── requirements.txt              # Pinned dependencies
├── pyproject.toml                # Project metadata
├── Makefile                      # Common task shortcuts
├── .gitignore
└── README.md
```

## Modeling Decisions

### Leakage-safe features
The prediction model intentionally excludes `Measured_Value`, `Failure_Mode`, `Yield_Impact`, and `Result` from its feature set. These are post-test or target columns — including them would cause data leakage and produce artificially inflated accuracy that would not generalise to pre-test prediction.

### Balanced class weights
Both Logistic Regression and Random Forest use `class_weight='balanced'` to handle the ~85/15 PASS/FAIL class imbalance. This up-weights the minority FAIL class during training.

### IsolationForest for anomaly detection
IsolationForest is well-suited for high-dimensional, unsupervised anomaly detection. It uses `RobustScaler` preprocessing to handle outliers in the feature space. The anomaly detector operates on Measured_Value (post-test) for retrospective analysis.

### Model selection by F1
The best model is selected by F1-score for the FAIL class (positive label), which balances precision and recall for the minority class.

## Assumptions

- **Synthetic data**: All records are generated programmatically with `seed=42`. The data does not represent real semiconductor fabrication.
- **Fixed seed**: All random operations use seed 42 for reproducibility.
- **Class split**: Approximately 85% PASS / 15% FAIL, assigned stochastically.
- **Failure modes**: Randomly assigned to FAIL records (no causal relationship with measurements).
- **Feature set**: Pre-test parameters only for prediction; post-test measurements for anomaly detection.

## Limitations

- Synthetic data lacks real-world patterns (lot-level correlations, temporal trends, equipment drift).
- No temporal or sequential features are used.
- SHAP-based explainability is available as a dependency but not integrated into the default app flow.
- Low ML performance on synthetic data is expected — the leakage-safe features have limited predictive power over randomly assigned labels.
- The Streamlit app is designed for local demonstration, not production deployment.

## Reproducibility

1. All random operations use `numpy.random.default_rng(42)` or `random_state=42`.
2. Dependencies are pinned in `requirements.txt`.
3. Running `python generate_test_data.py` always produces identical output.
4. Running `python -m src.ate_ai.train --data data/raw/ATE_AI_Engineering_Assessment.csv` always produces the same model and metrics.

## Demo Flow

1. **Generate data**: Run `python generate_test_data.py` to create the test CSV.
2. **Train model**: Run `python -m src.ate_ai.train --data data/raw/ATE_AI_Engineering_Assessment.csv`.
3. **Launch app**: Run `streamlit run app/streamlit_app.py`.
4. **Upload**: On the Upload screen, browse to `data/raw/ATE_AI_Engineering_Assessment.csv` and upload it.
5. **Process**: Navigate to Process to see cleaning metrics (rows before/after, duplicates removed, missing values handled).
6. **Dashboard**: View yield KPIs, bar charts by test, the Lot × Test heatmap, failure mode distribution, and the measured-value scatter plot.
7. **Investigation**: Search for a specific Device_ID or Test_ID to view matching records and their measurements vs spec limits.
8. **AI Analysis**: Select a record to see its anomaly score and a generated explanation.
9. **Prediction**: Fill in test parameters and submit to get a PASS/FAIL prediction with probability and explanation.
