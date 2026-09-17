
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "industrial_calorimetric_predictive_maintenance_dataset.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR.parent / "NTI" / DATA_PATH.name
METRICS_PATH = BASE_DIR / "metrics.json"
METADATA_PATH = BASE_DIR / "metadata.json"
LOGO_PATH = BASE_DIR / "maintainx_logo.svg"
EXACT_LOGO_PATH = BASE_DIR / "Screenshot 2026-09-17 030845.png"

FEATURE_COLUMNS = [
    "ambient_temperature_c", "ambient_humidity_pct", "dsc_heat_flow_mw",
    "sample_temperature_c", "reference_temperature_c", "thermal_gradient_c",
    "heating_rate_c_min", "cooling_rate_c_min", "infrared_mean_temp_c",
    "infrared_max_temp_c", "thermal_hotspot_area_pct",
    "phase_transition_onset_c", "phase_transition_peak_c",
    "transition_enthalpy_j_g", "heat_dissipation_rate_w", "sensor_voltage_v",
    "sensor_current_ma", "vibration_level_mm_s", "mqtt_latency_ms",
    "packet_loss_pct", "edge_cpu_usage_pct", "rolling_noise_index",
]

LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}
STATE_ORDER = ["Normal", "Warning", "Critical"]
MODEL_FILES = {
    "maintenance": "maintenance_model.pkl",
    "anomaly": "anomaly_model.pkl",
    "fault_state": "fault_state_model.pkl",
    "rul": "remaining_useful_life_h.pkl",
}

NAV_ITEMS = [
    ("dashboard", "Dashboard"),
    ("single", "Single Machine"),
    ("batch", "Batch Scan"),
    ("analytics", "Analytics"),
    ("reports", "Reports"),
]

PAGE_ICON = EXACT_LOGO_PATH if EXACT_LOGO_PATH.exists() else LOGO_PATH
st.set_page_config(page_title="MaintainX", page_icon=str(PAGE_ICON), layout="wide")

# ---------------------------------------------------------------------------
# Style: dark instrumentation-panel theme, teal/cyan accents, calorimetry-inspired
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --ink:#e9f2ff; --muted:#8ea6c4; --faint:#5c7396;
        --blue:#2f9bff; --cyan:#5dd6ff; --green:#3ddc97; --amber:#ffb648; --red:#ff5d6c;
        --panel:#111b2b; --panel-2:#0d1726; --line:#1c3252;
    }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 85% 0%, #18375b 0, #0a111d 38%, #070c14 100%); color:var(--ink); }
    .block-container { max-width:1280px; padding-top:1.6rem; }
    h1, h2, h3 { font-family:'Barlow Condensed', sans-serif; letter-spacing:.01em; }
    h1 { font-size:2.6rem !important; line-height:1 !important; margin-bottom:.1rem; }
    h3 { margin-top:0; }
    p, span, label, div { color: var(--ink); }
    .brand-mark { color:var(--cyan); font-size:.78rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700; }
    .hero-copy { color:var(--muted); max-width:680px; font-size:1rem; margin-top:.2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background:#080f1b; border-right:1px solid #1b385a; }
    [data-testid="stSidebar"] .stRadio > label { display:none; }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap:.15rem; }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        padding:.55rem .7rem; border-radius:8px; font-family:'DM Sans'; color:var(--muted);
    }
    .sidebar-title { font-family:'Barlow Condensed', sans-serif; font-size:1.7rem; font-weight:700; color:var(--ink); letter-spacing:.02em; }
    .sidebar-sub { color:var(--faint); font-size:.78rem; margin-top:-.4rem; }

    /* Cards */
    [data-testid="stMetric"] {
        background:linear-gradient(180deg, rgba(24,40,64,.9), rgba(13,23,38,.9));
        border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color:var(--faint) !important; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; }
    [data-testid="stMetricValue"] { font-family:'Barlow Condensed', sans-serif; font-size:1.9rem !important; }

    .panel {
        background:linear-gradient(180deg, rgba(20,33,53,.85), rgba(11,19,32,.9));
        border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.25rem; margin-bottom:1rem;
    }
    .panel-title { font-family:'Barlow Condensed', sans-serif; font-size:1.15rem; font-weight:600; color:var(--ink); margin-bottom:.1rem; }
    .panel-sub { color:var(--faint); font-size:.8rem; margin-bottom:.7rem; }

    .status-pill { display:inline-block; padding:.15rem .65rem; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.05em; }
    .pill-normal { background:rgba(61,220,151,.15); color:var(--green); border:1px solid rgba(61,220,151,.35); }
    .pill-warning { background:rgba(255,182,72,.15); color:var(--amber); border:1px solid rgba(255,182,72,.35); }
    .pill-critical { background:rgba(255,93,108,.15); color:var(--red); border:1px solid rgba(255,93,108,.35); }

    div.stButton > button[kind="primary"] { background:#1686df; border:0; border-radius:7px; font-weight:700; }
    div.stButton > button[kind="primary"]:hover { background:#35a8ff; }

    .section-label { color:var(--cyan); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; margin:1.1rem 0 .3rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    [data-testid="stImage"] img { border-radius:24px; }
    [data-testid="stSidebar"] [data-testid="stImage"] { display:flex; justify-content:center; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_reference_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    frame = pd.read_csv(DATA_PATH)
    frame = cleaning(frame)
    return frame[[c for c in FEATURE_COLUMNS if c in frame.columns]]


def cleaning(df):
    df = df.drop_duplicates()
    df["ambient_temperature_c"] = df["ambient_temperature_c"].replace(
        0, df["ambient_temperature_c"].median()
    )
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


@st.cache_resource
def load_models():
    missing = [name for name in MODEL_FILES.values() if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files in the project folder: " + ", ".join(missing)
        )
    return {key: joblib.load(BASE_DIR / filename) for key, filename in MODEL_FILES.items()}


@st.cache_data
def load_model_analysis():
    metrics = {}
    metadata = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    if METADATA_PATH.exists():
        with METADATA_PATH.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    return metrics, metadata


def missing_columns(frame):
    return [column for column in FEATURE_COLUMNS if column not in frame.columns]


def numeric_frame(frame):
    missing = missing_columns(frame)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    values = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values


def predict(frame, models):
    values = numeric_frame(frame)
    if values.isna().any().any():
        bad = values.columns[values.isna().any()].tolist()
        raise ValueError("These columns contain missing or non-numeric values: " + ", ".join(bad))

    maintenance = models["maintenance"].predict(values).astype(int)
    probability = models["maintenance"].predict_proba(values)[:, 1]
    anomaly = models["anomaly"].predict(values)
    rul = np.clip(models["rul"].predict(values), 0, None)

    # The downloaded fault-state classifier was fitted on the raw 22-column frame.
    states = models["fault_state"].predict(values)
    fault_labels = LABELS
    model_classes = np.asarray(getattr(models["fault_state"], "classes_", []))
    if np.array_equal(model_classes, np.array([0.0, 1.0])):
        fault_labels = {0: "Warning", 1: "Critical"}
    state_names = [fault_labels.get(int(value), "Normal") for value in states]
    visible_states = [state if state in {"Warning", "Critical"} and flag else "Normal" for state, flag in zip(state_names, maintenance)]

    return pd.DataFrame({
        "maintenance_required": maintenance,
        "maintenance_probability": np.round(probability, 4),
        "anomaly_score": np.round(anomaly, 4),
        "fault_state": visible_states,
        "remaining_useful_life_h": np.round(rul, 1),
    }, index=frame.index)


def status_pill(label):
    css = {"Normal": "pill-normal", "Warning": "pill-warning", "Critical": "pill-critical"}.get(label, "pill-normal")
    return f'<span class="status-pill {css}">{label.upper()}</span>'


def display_state_chart(state_counts):
    chart_data = pd.DataFrame({
        "State": STATE_ORDER,
        "Machines": [int(state_counts.get(state, 0)) for state in STATE_ORDER],
    })
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("State:N", sort=STATE_ORDER, title=None),
        y=alt.Y("Machines:Q", title="Machines"),
        tooltip=[alt.Tooltip("State:N", title="State"), alt.Tooltip("Machines:Q", title="Machines")],
    )
    st.altair_chart(chart, use_container_width=True)


def add_prediction_history(kind, source, data):
    history = st.session_state.setdefault("prediction_history", [])
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "source": source,
        "rows": len(data),
        "data": data.copy(),
    })


# ---------------------------------------------------------------------------
# Shared panel helpers
# ---------------------------------------------------------------------------
def panel_start(title, subtitle=None):
    sub = f'<div class="panel-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>{sub}', unsafe_allow_html=True)


def panel_end():
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Single machine diagnosis
# ---------------------------------------------------------------------------
def display_single_result(result, inputs):
    row = result.iloc[0]
    needs_maintenance = int(row["maintenance_required"]) == 1
    fault_state = row["fault_state"] or "Normal"

    panel_start("Machine status", "Latest diagnosis for the entered sensor readings")
    cards = st.columns(4 if needs_maintenance else 3)
    cards[0].metric("Maintenance", "REQUIRED" if needs_maintenance else "NOT REQUIRED")
    cards[1].metric("Maintenance probability", f"{float(row['maintenance_probability']):.1%}")
    cards[2].metric("Remaining useful life", f"{float(row['remaining_useful_life_h']):,.0f} h")
    if needs_maintenance:
        cards[3].metric("Fault state", fault_state)
        st.error("Action required: schedule maintenance for this machine.")
    else:
        st.success("This machine does not currently require maintenance.")
    panel_end()

    with st.expander("Technical output"):
        st.dataframe(result, use_container_width=True, hide_index=True)

    st.session_state["last_single_result"] = result
    history_row = pd.concat([inputs.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
    add_prediction_history("Single machine", "Manual sensor input", history_row)


def render_single(models, reference):
    st.markdown('<div class="section-label">Single machine</div>', unsafe_allow_html=True)
    st.subheader("Inspect one machine")
    st.caption("Enter the 22 sensor readings. Values start at the reference dataset mean.")
    st.info("Maintenance probability is the model's estimated chance that this machine needs maintenance. It is not a guarantee.")

    panel_start("Sensor input", "Adjust readings, then run the diagnosis")
    inputs = {}
    for start in range(0, len(FEATURE_COLUMNS), 3):
        columns = st.columns(3)
        for column, feature in zip(columns, FEATURE_COLUMNS[start:start + 3]):
            series = pd.to_numeric(reference[feature], errors="coerce") if feature in reference else pd.Series([0.0])
            series = series.dropna()
            mean = float(series.mean()) if len(series) else 0.0
            data_min = float(series.min()) if len(series) else mean - 1.0
            data_max = float(series.max()) if len(series) else mean + 1.0
            data_span = data_max - data_min
            padding = max(data_span * 0.25, abs(mean) * 0.1, 1.0)
            minimum = data_min - padding
            maximum = data_max + padding
            if minimum == maximum:
                minimum -= 1.0
                maximum += 1.0
            text_key = f"{feature}_text"
            slider_key = f"{feature}_slider"
            if text_key not in st.session_state:
                st.session_state[text_key] = f"{mean:.4f}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = min(max(mean, minimum), maximum)
            with column:
                st.text_input(feature.replace("_", " ").title(), key=text_key)
                st.slider(
                    f"Adjust {feature.replace('_', ' ').title()}",
                    min_value=minimum,
                    max_value=maximum,
                    key=slider_key,
                    format="%.4f",
                    label_visibility="collapsed",
                    on_change=lambda name=feature: st.session_state.__setitem__(
                        f"{name}_text", f"{st.session_state[f'{name}_slider']:.4f}"
                    ),
                )
                try:
                    inputs[feature] = float(st.session_state[text_key])
                except ValueError:
                    inputs[feature] = st.session_state[text_key]
    run = st.button("Run machine diagnosis", type="primary", use_container_width=True)
    panel_end()

    if run:
        try:
            input_frame = pd.DataFrame([inputs])
            display_single_result(predict(input_frame, models), input_frame)
        except ValueError as error:
            st.error(str(error))


# ---------------------------------------------------------------------------
# Batch scan
# ---------------------------------------------------------------------------
def display_batch_results(output, predictions, filename):
    flagged = int(predictions["maintenance_required"].sum())
    panel_start("Batch results", f"{filename} · {len(predictions):,} rows")
    first, second, third = st.columns(3)
    first.metric("Maintenance flagged", f"{flagged:,}", f"{flagged / len(predictions):.1%} of batch")
    second.metric("Healthy records", f"{len(predictions) - flagged:,}")
    third.metric("Median useful life", f"{predictions['remaining_useful_life_h'].median():,.0f} h")
    state_counts = predictions["fault_state"].value_counts().reindex(STATE_ORDER, fill_value=0)
    display_state_chart(state_counts)
    panel_end()

    st.dataframe(output.head(300), use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored CSV",
        output.to_csv(index=False),
        "maintainx_predictions.csv",
        "text/csv",
        type="primary",
    )


def render_batch(models):
    st.markdown('<div class="section-label">Batch intelligence</div>', unsafe_allow_html=True)
    st.subheader("Scan a complete dataset")
    st.caption("Upload a CSV. Extra columns are preserved; only the required sensor columns are sent to the models.")

    panel_start("Upload machine readings", "CSV with the 22 required sensor columns")
    uploaded = st.file_uploader("Upload machine readings", type="csv", label_visibility="collapsed")
    panel_end()

    if uploaded is None:
        output = st.session_state.get("last_batch_output")
        predictions = st.session_state.get("last_batch_predictions")
        filename = st.session_state.get("last_batch_filename", "previous batch")
        if output is not None and predictions is not None:
            display_batch_results(output, predictions, filename)
        return
    try:
        frame = pd.read_csv(uploaded)
    except pd.errors.ParserError:
        uploaded.seek(0)
        frame = pd.read_csv(uploaded, on_bad_lines="skip")
        raw_rows = [line for line in uploaded.getvalue().splitlines() if line.strip()]
        skipped = max(0, len(raw_rows) - 1 - len(frame))
        st.warning(
            f"The CSV contains malformed rows. {skipped} row(s) were skipped. "
            "Fix the source CSV before relying on the complete batch."
        )
    except Exception as error:
        st.error(f"Could not read this CSV: {error}")
        return

    missing = missing_columns(frame)
    if missing:
        st.error(f"Missing {len(missing)} required column(s). No predictions were made.")
        st.code("\n".join(missing))
        st.info("Add the missing columns and upload the file again.")
        return
    if frame.empty:
        st.warning("The uploaded CSV contains no machine records.")
        return

    st.success(f"Schema check passed: {len(frame):,} machine record(s) ready.")
    try:
        with st.spinner("Scoring the batch..."):
            predictions = predict(frame, models)
    except ValueError as error:
        st.error(str(error))
        return

    output = pd.concat([frame, predictions.add_prefix("pred_")], axis=1)
    st.session_state["last_batch_output"] = output
    st.session_state["last_batch_predictions"] = predictions
    st.session_state["last_batch_filename"] = uploaded.name
    batch_signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get("last_batch_history_signature") != batch_signature:
        add_prediction_history("Batch scan", uploaded.name, output)
        st.session_state["last_batch_history_signature"] = batch_signature

    display_batch_results(output, predictions, uploaded.name)


# ---------------------------------------------------------------------------
# Dashboard (overview landing page, styled after the reference mockup)
# ---------------------------------------------------------------------------
def render_dashboard(models, reference):
    metrics, metadata = load_model_analysis()

    st.markdown('<div class="section-label">Model analysis</div>', unsafe_allow_html=True)
    st.subheader("Predictive maintenance model performance")
    st.caption("Validation metrics and training coverage for the deployed model suite.")

    if not metrics:
        st.warning("Model analysis files were not found in the project folder.")
        return

    panel_start("Classification performance", "Held-out test-set metrics")
    maintenance, fault_state = st.columns(2)
    maintenance.metric("Maintenance accuracy", f"{metrics['maintenance_required']['accuracy']:.1%}")
    maintenance.caption(f"Macro F1: {metrics['maintenance_required']['f1_macro']:.3f}")
    fault_state.metric("Fault-state accuracy", f"{metrics['fault_state']['accuracy']:.1%}")
    fault_state.caption(f"Macro F1: {metrics['fault_state']['f1_macro']:.3f}")
    panel_end()

    left, right = st.columns(2, gap="medium")
    with left:
        panel_start("Model comparison", "Accuracy and macro F1")
        comparison = pd.DataFrame({
            "Accuracy": {
                "Maintenance": metrics["maintenance_required"]["accuracy"],
                "Fault state": metrics["fault_state"]["accuracy"],
            },
            "Macro F1": {
                "Maintenance": metrics["maintenance_required"]["f1_macro"],
                "Fault state": metrics["fault_state"]["f1_macro"],
            },
        })
        st.bar_chart(comparison)
        panel_end()

    with right:
        panel_start("Regression performance", "Held-out test-set metrics")
        rul = metrics["remaining_useful_life_h"]
        anomaly = metrics.get("anomaly_score")
        first, second = st.columns(2)
        first.metric("RUL R² score", f"{rul['r2']:.3f}")
        second.metric("RUL mean absolute error", f"{rul['mae']:,.2f} h")
        if anomaly:
            st.metric("Anomaly R² score", f"{anomaly['r2']:.3f}")
            st.caption(f"Anomaly MAE: {anomaly['mae']:.4f}")
        st.caption("R² indicates explained variance; MAE is the average prediction error.")
        panel_end()

    panel_start("Training coverage", "Dataset and target configuration")
    train, test, features, equipment = st.columns(4)
    train.metric("Training rows", f"{metadata.get('n_train_rows', 0):,}")
    test.metric("Test rows", f"{metadata.get('n_test_rows', 0):,}")
    features.metric("Input features", f"{len(metadata.get('feature_cols', [])):,}")
    equipment.metric("Test equipment", f"{metadata.get('n_test_equipment', 0):,}")
    st.dataframe(
        pd.DataFrame({
            "Target": ["Maintenance", "Fault state", "Anomaly score", "Useful life"],
            "Task": ["Classification", "Classification", "Regression", "Regression"],
            "Classes": [
                "2",
                ", ".join(metadata.get("fault_state_classes", [])),
                "Continuous score",
                "Continuous hours",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )
    panel_end()


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def render_input_data_analysis(data, source_label):
    panel_start("Input data analysis", source_label)
    numeric_data = data.select_dtypes(include="number")
    missing_data = data.isna().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    data_cards = st.columns(4)
    data_cards[0].metric("Rows", f"{len(data):,}")
    data_cards[1].metric("Columns", f"{len(data.columns):,}")
    data_cards[2].metric("Numeric fields", f"{len(numeric_data.columns):,}")
    data_cards[3].metric("Missing values", f"{int(data.isna().sum().sum()):,}")

    if not numeric_data.empty:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Feature distribution**")
            selected_feature = st.selectbox(
                "Choose a numeric field",
                list(numeric_data.columns),
                key=f"{source_label}_data_analysis_feature",
            )
            distribution = alt.Chart(numeric_data[[selected_feature]]).mark_bar().encode(
                x=alt.X(f"{selected_feature}:Q", bin=alt.Bin(maxbins=18), title=selected_feature),
                y=alt.Y("count():Q", title="Records"),
                tooltip=[alt.Tooltip("count():Q", title="Records")],
            )
            st.altair_chart(distribution, use_container_width=True)
        with right:
            st.markdown("**Descriptive statistics**")
            summary = numeric_data.describe().T.reset_index().rename(columns={"index": "Feature"})
            summary = summary[["Feature", "count", "mean", "std", "min", "50%", "max"]]
            summary.columns = ["Feature", "Count", "Mean", "Std", "Min", "Median", "Max"]
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)

    if not missing_data.empty:
        st.markdown("**Missing values by field**")
        st.bar_chart(missing_data)
    else:
        st.success("This dataset contains no missing values.")
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)
    panel_end()


def render_analytics(models, reference):
    st.markdown('<div class="section-label">Analytics</div>', unsafe_allow_html=True)
    st.subheader("Fleet analytics")
    st.caption("The main dataset is analyzed first; uploaded batches replace it after scoring.")

    predictions = st.session_state.get("last_batch_predictions")
    output = st.session_state.get("last_batch_output")
    filename = st.session_state.get("last_batch_filename", "main dataset")

    if predictions is None or output is None:
        if reference.empty:
            render_input_data_analysis(reference, "Main dataset analysis")
            return
        output = reference.copy()
        predictions = predict(reference, models)
        filename = "main dataset"

    state_counts = predictions["fault_state"].value_counts().reindex(STATE_ORDER, fill_value=0)
    flagged = int(predictions["maintenance_required"].sum())
    healthy = len(predictions) - flagged

    panel_start("Dataset overview", f"{filename} · {len(predictions):,} machines")
    overview = st.columns(4)
    overview[0].metric("Machines scored", f"{len(predictions):,}")
    overview[1].metric("Normal", f"{int(state_counts['Normal']):,}")
    overview[2].metric("Warning", f"{int(state_counts['Warning']):,}")
    overview[3].metric("Critical", f"{int(state_counts['Critical']):,}")
    st.caption(
        f"{flagged:,} machines flagged for maintenance · {healthy:,} not flagged · "
        f"Average anomaly score: {predictions['anomaly_score'].mean():.3f}"
    )
    panel_end()

    left, right = st.columns(2, gap="medium")
    with left:
        panel_start("Overall machine states", "Normal → Warning → Critical")
        display_state_chart(state_counts)
        panel_end()

    with right:
        panel_start("Maintenance outcome", "Machines requiring maintenance")
        maintenance_data = pd.DataFrame({
            "Outcome": ["Not required", "Required"],
            "Machines": [healthy, flagged],
        })
        maintenance_chart = alt.Chart(maintenance_data).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("Machines:Q"),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=["Not required", "Required"], range=["#3ddc97", "#ff5d6c"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Outcome:N", "Machines:Q"],
        )
        st.altair_chart(maintenance_chart, use_container_width=True)
        panel_end()

    left, right = st.columns(2, gap="medium")
    with left:
        panel_start("Anomaly score distribution", "Higher values indicate more unusual readings")
        anomaly_data = predictions[["anomaly_score"]].copy()
        anomaly_chart = alt.Chart(anomaly_data).mark_bar().encode(
            x=alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=12), title="Anomaly score"),
            y=alt.Y("count():Q", title="Machines"),
            tooltip=[alt.Tooltip("count():Q", title="Machines")],
        )
        st.altair_chart(anomaly_chart, use_container_width=True)
        panel_end()

    with right:
        panel_start("Remaining useful life", "Predicted hours grouped into ranges")
        rul_data = predictions[["remaining_useful_life_h"]].copy()
        rul_chart = alt.Chart(rul_data).mark_bar().encode(
            x=alt.X("remaining_useful_life_h:Q", bin=alt.Bin(maxbins=12), title="Useful life (hours)"),
            y=alt.Y("count():Q", title="Machines"),
            tooltip=[alt.Tooltip("count():Q", title="Machines")],
        )
        st.altair_chart(rul_chart, use_container_width=True)
        panel_end()

    panel_start("Average sensor profile by state", "Compare selected sensor averages across the fleet")
    numeric_features = [
        feature for feature in FEATURE_COLUMNS
        if feature in output.columns and pd.api.types.is_numeric_dtype(output[feature])
    ]
    if numeric_features:
        selected_features = st.multiselect(
            "Sensors",
            numeric_features,
            default=numeric_features[:4],
            key="analytics_sensor_features",
        )
        if selected_features:
            profile = output.assign(fault_state=predictions["fault_state"])
            profile_data = profile.groupby("fault_state", sort=False)[selected_features].mean()
            profile_data = profile_data.reindex(STATE_ORDER).reset_index().melt(
                id_vars="fault_state", var_name="Sensor", value_name="Average"
            )
            profile_chart = alt.Chart(profile_data).mark_bar().encode(
                x=alt.X("Sensor:N", title=None),
                y=alt.Y("Average:Q", title="Average reading"),
                color=alt.Color(
                    "fault_state:N",
                    sort=STATE_ORDER,
                    scale=alt.Scale(domain=STATE_ORDER, range=["#3ddc97", "#ffb648", "#ff5d6c"]),
                ),
                xOffset=alt.XOffset("fault_state:N", sort=STATE_ORDER),
                tooltip=["fault_state:N", "Sensor:N", alt.Tooltip("Average:Q", format=".2f")],
            )
            st.altair_chart(profile_chart, use_container_width=True)
    panel_end()

    panel_start("Data preview", "First 100 records")
    st.dataframe(output.head(100), use_container_width=True, hide_index=True)
    panel_end()

    panel_start("Input data analysis", f"Descriptive analysis of the {filename}")
    numeric_batch = output.select_dtypes(include="number").drop(
        columns=[column for column in output.columns if column.startswith("pred_")],
        errors="ignore",
    )
    missing_data = output.isna().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    data_cards = st.columns(4)
    data_cards[0].metric("Rows", f"{len(output):,}")
    data_cards[1].metric("Columns", f"{len(output.columns):,}")
    data_cards[2].metric("Numeric fields", f"{len(numeric_batch.columns):,}")
    data_cards[3].metric("Missing values", f"{int(output.isna().sum().sum()):,}")

    if not numeric_batch.empty:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Feature distribution**")
            selected_feature = st.selectbox(
                "Choose a numeric field",
                list(numeric_batch.columns),
                key="data_analysis_feature",
            )
            distribution = alt.Chart(numeric_batch[[selected_feature]]).mark_bar().encode(
                x=alt.X(f"{selected_feature}:Q", bin=alt.Bin(maxbins=18), title=selected_feature),
                y=alt.Y("count():Q", title="Records"),
                tooltip=[alt.Tooltip("count():Q", title="Records")],
            )
            st.altair_chart(distribution, use_container_width=True)
        with right:
            st.markdown("**Descriptive statistics**")
            summary = numeric_batch.describe().T.reset_index().rename(columns={"index": "Feature"})
            summary = summary[["Feature", "count", "mean", "std", "min", "50%", "max"]]
            summary.columns = ["Feature", "Count", "Mean", "Std", "Min", "Median", "Max"]
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)

    if not missing_data.empty:
        st.markdown("**Missing values by field**")
        st.bar_chart(missing_data)
    else:
        st.success("The uploaded input data contains no missing values.")
    panel_end()

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def render_reports():
    st.markdown('<div class="section-label">Reports</div>', unsafe_allow_html=True)
    st.subheader("Prediction history")
    st.caption("Review every single-machine and batch prediction from this app session.")

    history = st.session_state.get("prediction_history", [])

    panel_start("Run history", "Newest predictions appear at the bottom")
    if history:
        history_table = pd.DataFrame([
            {
                "Timestamp": entry["timestamp"],
                "Type": entry["kind"],
                "Source": entry["source"],
                "Rows": entry["rows"],
            }
            for entry in history
        ])
        st.dataframe(history_table, use_container_width=True, hide_index=True)
        all_runs = pd.concat(
            [
                entry["data"].assign(
                    history_timestamp=entry["timestamp"],
                    history_type=entry["kind"],
                    history_source=entry["source"],
                )
                for entry in history
            ],
            ignore_index=True,
            sort=False,
        )
        st.download_button(
            "Download complete prediction history",
            all_runs.to_csv(index=False),
            "maintainx_prediction_history.csv",
            "text/csv",
            type="primary",
        )
        selected_run = st.selectbox(
            "Open a history run",
            options=range(len(history)),
            format_func=lambda index: (
                f"{index + 1}. {history[index]['kind']} | "
                f"{history[index]['timestamp']} | {history[index]['source']}"
            ),
        )
        selected_entry = history[selected_run]
        st.dataframe(selected_entry["data"], use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected run",
            selected_entry["data"].to_csv(index=False),
            f"maintainx_run_{selected_run + 1}.csv",
            "text/csv",
        )
    else:
        st.caption("No predictions have been completed yet.")
    panel_end()

    logo_path = EXACT_LOGO_PATH if EXACT_LOGO_PATH.exists() else LOGO_PATH
    if logo_path.exists():
        st.image(str(logo_path), width=180)

# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
def main():
    st.sidebar.markdown('<div class="sidebar-title">MaintainX</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-sub">Machine intelligence, made readable.</div>', unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    labels = [label for _, label in NAV_ITEMS]
    choice = st.sidebar.radio("Workspace", labels, label_visibility="collapsed")
    page = dict((label, key) for key, label in NAV_ITEMS)[choice]
    logo_path = EXACT_LOGO_PATH if EXACT_LOGO_PATH.exists() else LOGO_PATH
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=140)

    st.markdown('<div class="brand-mark">Industrial intelligence / predictive maintenance</div>', unsafe_allow_html=True)
    st.title("MaintainX")
    st.markdown('<p class="hero-copy">A clear view of machine health, maintenance risk and remaining useful life.</p>', unsafe_allow_html=True)

    reference = load_reference_data()
    models = None
    if page in {"single", "batch", "analytics"}:
        try:
            models = load_models()
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()

    if page == "dashboard":
        render_dashboard(models, reference)
    elif page == "single":
        render_single(models, reference)
    elif page == "batch":
        render_batch(models)
    elif page == "analytics":
        render_analytics(models, reference)
    elif page == "reports":
        render_reports()
if __name__ == "__main__":
    main()
