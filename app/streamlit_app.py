"""ATE AI Engineering Assessment — Streamlit application."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ate_ai.data import load_data
from src.ate_ai.cleaning import clean_data
from src.ate_ai.analytics import compute_yield_analytics
from src.ate_ai.schema import validate_schema
from src.ate_ai.predict import load_artifact, predict_records
from src.ate_ai.anomaly import fit_anomaly_detector
from src.ate_ai.explain import generate_explanation
from src.ate_ai.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATE AI Assessment",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
SCREENS = [
    "Upload",
    "Process",
    "Dashboard",
    "Investigation",
    "AI Analysis",
    "Prediction",
]

screen = st.sidebar.radio("Navigation", SCREENS)


# ---------------------------------------------------------------------------
# Cached heavy-computation wrappers
# ---------------------------------------------------------------------------
@st.cache_data
def cached_yield_analytics(df_json: str) -> dict:
    """Run yield analytics with Streamlit caching.

    We serialise the DataFrame to JSON as the cache key because
    ``st.cache_data`` requires hashable arguments.
    """
    df = pd.read_json(df_json, orient="split")
    return compute_yield_analytics(df)


@st.cache_data
def cached_anomaly_detection(df_json: str):
    """Run anomaly detection with Streamlit caching."""
    df = pd.read_json(df_json, orient="split")
    _, scored_df = fit_anomaly_detector(df)
    return scored_df


# ===================================================================
# Screen 1 — Upload
# ===================================================================
if screen == "Upload":
    st.header("📂 Upload ATE Test Data")
    st.write("Upload a CSV file containing ATE test records.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = validate_schema(df)
            st.session_state["raw_data"] = df
            st.success(
                f"✅ File uploaded and validated — {len(df)} rows, "
                f"{len(df.columns)} columns."
            )
            st.dataframe(df.head(10))
        except ValueError as exc:
            st.error(f"❌ Schema validation failed: {exc}")
        except Exception as exc:
            st.error(f"❌ Error reading file: {exc}")


# ===================================================================
# Screen 2 — Process (Cleaning)
# ===================================================================
elif screen == "Process":
    st.header("🧹 Data Processing & Cleaning")

    if "raw_data" not in st.session_state:
        st.warning("Please upload a CSV file first (go to Upload screen).")
    else:
        df_raw = st.session_state["raw_data"]
        cleaned_df, report = clean_data(df_raw)
        st.session_state["cleaned_data"] = cleaned_df

        # Summary metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Original rows", report["original_rows"])
        col2.metric("Cleaned rows", report["cleaned_rows"])
        col3.metric("Duplicates removed", report["duplicates_removed"])

        # Missing-values before/after table
        st.subheader("Missing Values — Before vs After")
        mv_before = report["missing_values_before"]
        mv_after = report["missing_values_after"]
        mv_df = pd.DataFrame({
            "Column": list(mv_before.keys()),
            "Before": list(mv_before.values()),
            "After": list(mv_after.values()),
        })
        # Show only columns that had missing values
        mv_df = mv_df[mv_df["Before"] > 0].reset_index(drop=True)
        if mv_df.empty:
            st.info("No missing values detected.")
        else:
            st.dataframe(mv_df, use_container_width=True)

        st.success("✅ Cleaning complete. Proceed to Dashboard.")


# ===================================================================
# Screen 3 — Dashboard
# ===================================================================
elif screen == "Dashboard":
    st.header("📊 Yield Analytics Dashboard")

    if "cleaned_data" not in st.session_state:
        st.warning("Please process the data first (go to Process screen).")
    else:
        df = st.session_state["cleaned_data"]
        # Cache analytics via JSON serialisation
        analytics = cached_yield_analytics(df.to_json(orient="split"))

        # --- KPI row ---
        k1, k2, k3 = st.columns(3)
        k1.metric("Overall Yield", f"{analytics['overall_yield']:.2f}%")
        k2.metric("Overall Fail Rate", f"{analytics['overall_fail_rate']:.2f}%")
        k3.metric("Total Records", f"{len(df):,}")

        # --- Bar chart: yield by test ---
        st.subheader("Yield by Test")
        ybt = analytics["yield_by_test"]
        fig_ybt = px.bar(
            ybt,
            x="Test_ID",
            y="yield_pct",
            color="Test_ID",
            text_auto=".1f",
            labels={"yield_pct": "Yield %"},
            title="Yield % by Test ID",
        )
        fig_ybt.update_layout(showlegend=False)
        st.plotly_chart(fig_ybt, use_container_width=True)

        # --- Heatmap: yield by Lot × Test ---
        st.subheader("Yield Heatmap — Lot × Test")
        pivot = df.assign(
            is_pass=(df["Result"] == "PASS").astype(int)
        ).pivot_table(
            values="is_pass",
            index="Lot_ID",
            columns="Test_ID",
            aggfunc="mean",
        ) * 100
        fig_hm = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            labels={"color": "Yield %"},
            title="Yield % by Lot × Test",
            aspect="auto",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # --- Bar chart: failure mode distribution ---
        st.subheader("Failure Mode Distribution")
        fmd = analytics["failure_mode_distribution"]
        if fmd.empty:
            st.info("No failure modes recorded.")
        else:
            fig_fm = px.bar(
                fmd,
                x="Failure_Mode",
                y="count",
                text_auto=True,
                title="Failure Mode Counts",
                labels={"count": "Count"},
            )
            st.plotly_chart(fig_fm, use_container_width=True)

        # --- Scatter: Measured_Value vs limits ---
        st.subheader("Measured Value vs Spec Limits")
        test_options = sorted(df["Test_ID"].unique().tolist())
        selected_test = st.selectbox("Select Test", test_options, key="scatter_test")
        test_df = df[df["Test_ID"] == selected_test].copy()

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df["Measured_Value"],
            mode="markers",
            name="Measured Value",
            marker=dict(
                color=test_df["Result"].map({"PASS": "green", "FAIL": "red"}),
                size=5,
            ),
        ))
        # Limit lines
        if not test_df.empty:
            ll = test_df["Lower_Limit"].iloc[0]
            ul = test_df["Upper_Limit"].iloc[0]
            fig_scatter.add_hline(y=ll, line_dash="dash", line_color="blue",
                                  annotation_text=f"Lower Limit ({ll})")
            fig_scatter.add_hline(y=ul, line_dash="dash", line_color="orange",
                                  annotation_text=f"Upper Limit ({ul})")
        fig_scatter.update_layout(
            title=f"Measured Value — {selected_test}",
            xaxis_title="Record Index",
            yaxis_title="Measured Value",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ===================================================================
# Screen 4 — Investigation
# ===================================================================
elif screen == "Investigation":
    st.header("🔍 Record Investigation")

    if "cleaned_data" not in st.session_state:
        st.warning("Please process the data first (go to Process screen).")
    else:
        df = st.session_state["cleaned_data"]

        search_term = st.text_input(
            "Search by Device_ID or Test_ID",
            placeholder="e.g. ATE300001 or P301",
        )

        if search_term:
            mask = (
                df["Device_ID"].astype(str).str.contains(search_term, case=False, na=False)
                | df["Test_ID"].astype(str).str.contains(search_term, case=False, na=False)
            )
            filtered = df[mask]
            st.write(f"Found **{len(filtered)}** matching records.")
            st.dataframe(filtered, use_container_width=True)

            if not filtered.empty:
                st.subheader("Measured Value vs Spec Limits")
                fig_inv = go.Figure()
                for idx, row in filtered.iterrows():
                    fig_inv.add_trace(go.Bar(
                        x=["Lower Limit", "Measured Value", "Upper Limit"],
                        y=[row["Lower_Limit"], row["Measured_Value"], row["Upper_Limit"]],
                        name=f"{row['Device_ID']} / {row['Test_ID']}",
                    ))
                fig_inv.update_layout(
                    title="Measured Value vs Limits (per record)",
                    yaxis_title="Value",
                    barmode="group",
                )
                # Limit to first 20 records to avoid overplotting
                if len(filtered) > 20:
                    st.info("Showing chart for first 20 records.")
                    fig_inv.data = fig_inv.data[:20]
                st.plotly_chart(fig_inv, use_container_width=True)
        else:
            st.info("Enter a Device_ID or Test_ID to search.")


# ===================================================================
# Screen 5 — AI Analysis (Anomaly Detection + Explanation)
# ===================================================================
elif screen == "AI Analysis":
    st.header("🤖 AI-Powered Anomaly Analysis")

    if "cleaned_data" not in st.session_state:
        st.warning("Please process the data first (go to Process screen).")
    else:
        df = st.session_state["cleaned_data"]

        # Run anomaly detection (cached)
        scored_df = cached_anomaly_detection(df.to_json(orient="split"))

        # Build display labels
        scored_df = scored_df.reset_index(drop=True)
        labels = (
            scored_df["Device_ID"].astype(str)
            + " | "
            + scored_df["Test_ID"].astype(str)
        )
        selected_label = st.selectbox("Select a record", labels.tolist())

        if selected_label:
            sel_idx = labels[labels == selected_label].index[0]
            record = scored_df.iloc[sel_idx]

            col1, col2 = st.columns(2)
            col1.metric("Anomaly Score", f"{record['anomaly_score']:.4f}")
            col2.metric(
                "Is Anomaly",
                "⚠️ Yes" if record["is_anomaly"] else "✅ No",
            )

            st.subheader("Explanation")
            explanation = generate_explanation(
                record=record,
                anomaly_score=record["anomaly_score"],
            )
            st.text(explanation)

            st.subheader("Record Details")
            st.json(record.to_dict())


# ===================================================================
# Screen 6 — Prediction
# ===================================================================
elif screen == "Prediction":
    st.header("🎯 Failure Prediction")

    test_ids = ["P301", "P302", "P303", "P304", "P305"]
    test_names = ["SCAN_CHAIN", "LEAKAGE_TEST", "TIMING_TEST", "VOLTAGE_REG", "POWER_TEST"]
    lot_ids = [f"LOT_A{str(i).zfill(2)}" for i in range(1, 11)]
    wafer_ids = [f"W{str(i).zfill(2)}" for i in range(1, 26)]

    with st.form("prediction_form"):
        st.subheader("Enter Test Parameters")

        fc1, fc2, fc3 = st.columns(3)
        vdd = fc1.number_input("VDD_V", min_value=1.0, max_value=1.2, value=1.1, step=0.01)
        temp = fc2.number_input("Temperature_C", min_value=25.0, max_value=125.0, value=75.0, step=1.0)
        retest = fc3.number_input("Retest_Count", min_value=0, max_value=3, value=0, step=1)

        sc1, sc2, sc3, sc4 = st.columns(4)
        test_id = sc1.selectbox("Test_ID", test_ids)
        test_name = sc2.selectbox("Test_Name", test_names)
        lot_id = sc3.selectbox("Lot_ID", lot_ids)
        wafer_id = sc4.selectbox("Wafer_ID", wafer_ids)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            model = load_artifact()

            input_df = pd.DataFrame([{
                "Test_ID": test_id,
                "Test_Name": test_name,
                "Lot_ID": lot_id,
                "Wafer_ID": wafer_id,
                "VDD_V": vdd,
                "Temperature_C": temp,
                "Retest_Count": retest,
            }])

            result_df = predict_records(model, input_df)

            prediction = result_df["prediction"].iloc[0]
            probability = result_df["fail_probability"].iloc[0]

            # Display result
            st.subheader("Prediction Result")
            rc1, rc2 = st.columns(2)
            rc1.metric(
                "Prediction",
                prediction,
                delta="FAIL" if prediction == "FAIL" else None,
                delta_color="inverse",
            )
            rc2.metric("Fail Probability", f"{probability:.4f}")

            # Generate explanation for the prediction record
            record_series = input_df.iloc[0].copy()
            record_series["Measured_Value"] = 0.0
            record_series["Lower_Limit"] = 0.0
            record_series["Upper_Limit"] = 0.0
            explanation = generate_explanation(
                record=record_series,
                anomaly_score=probability,
            )
            st.subheader("Explanation")
            st.text(explanation)

        except FileNotFoundError:
            st.error(
                "❌ Model not found. Please train the model first: "
                "`python -m src.ate_ai.train --data data/raw/ATE_AI_Engineering_Assessment.csv`"
            )
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
