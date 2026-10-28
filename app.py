"""Streamlit field-verification dashboard for electricity-theft risk triage."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ARTIFACT_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="GridSentinel | Field Verification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
    .stApp {background: #000000;}
    [data-testid="stSidebar"] {background: #000000; border-right: 1px solid #262626;}
    .block-container {padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1400px;}
    .eyebrow {font-size: .72rem; letter-spacing: .14em; color: #888888; font-weight: 600; text-transform: uppercase;}
    .hero-title {font-size: clamp(2rem, 4vw, 3.35rem); font-weight: 650; letter-spacing: -.045em; line-height: 1.02; margin: .45rem 0; color: #ededed;}
    .hero-copy {color: #888888; max-width: 720px; font-size: .98rem; line-height: 1.6;}
    .status-pill {display:inline-block; border:1px solid #333333; background:#0a0a0a; color:#ededed; padding:.35rem .7rem; border-radius:999px; font-size:.76rem; font-weight:550;}
    .section-label {font-size: .72rem; letter-spacing: .12em; color: #737373; font-weight: 600; text-transform: uppercase; margin-top:.3rem;}
    div[data-testid="stMetric"] {background:#0a0a0a; border:1px solid #262626; border-radius:8px; padding:1rem 1.1rem; box-shadow:none;}
    div[data-testid="stMetric"] label {color:#888888 !important;}
    div[data-testid="stMetricValue"] {color:#ededed; letter-spacing:-.035em;}
    div[data-testid="stMetricDelta"] {background:#111111; border:1px solid #262626; border-radius:999px; color:#a1a1a1; padding:.1rem .45rem;}
    .risk-high {color:#ededed; font-weight:700;}
    .notice {background:#0a0a0a; border:1px solid #262626; border-radius:8px; padding:.8rem 1rem; color:#a1a1a1; font-size:.86rem;}
    .stTabs [data-baseweb="tab-list"] {gap:0; border-bottom:1px solid #262626;}
    .stTabs [data-baseweb="tab"] {background:transparent; border-radius:0; color:#737373; padding:.65rem 1rem;}
    .stTabs [aria-selected="true"] {background:transparent; color:#ededed; border-bottom:1px solid #ededed;}
    [data-testid="stDataFrame"] {border:1px solid #262626; border-radius:8px; overflow:hidden;}
    .stButton > button, .stDownloadButton > button {background:#ededed; color:#0a0a0a; border:1px solid #ededed; border-radius:6px; font-weight:600;}
    .stButton > button:hover, .stDownloadButton > button:hover {background:#ffffff; color:#000000; border-color:#ffffff;}
    [data-baseweb="tag"] {background:#171717 !important; border:1px solid #333333 !important; color:#ededed !important;}
    [data-baseweb="tag"] span, [data-baseweb="tag"] svg {color:#ededed !important; fill:#a1a1a1 !important;}
    [data-baseweb="select"] > div {background:#0a0a0a; border-color:#262626;}
    [aria-label="Selected values"] > div {background:#171717 !important; border:1px solid #333333 !important; color:#ededed !important;}
    [aria-label="Selected values"] > div *, button[aria-label^="Remove "] {color:#ededed !important; fill:#a1a1a1 !important;}
    hr {border-color:#262626 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    required = [
        "consumer_risk_ranking.csv",
        "model_comparison.csv",
        "precision_recall_curves.csv",
        "gains_curves.csv",
        "evaluation_summary.json",
        "priority_meter_readings.csv.gz",
    ]
    missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing benchmark artifacts: {', '.join(missing)}. Run `python evaluate_models.py` first."
        )
    ranking = pd.read_csv(ARTIFACT_DIR / required[0])
    metrics = pd.read_csv(ARTIFACT_DIR / required[1])
    pr_curves = pd.read_csv(ARTIFACT_DIR / required[2])
    gains = pd.read_csv(ARTIFACT_DIR / required[3])
    summary = json.loads((ARTIFACT_DIR / required[4]).read_text(encoding="utf-8"))
    readings = pd.read_csv(ARTIFACT_DIR / required[5], parse_dates=["timestamp"])
    return ranking, metrics, pr_curves, gains, summary, readings


def manifest_pdf(rows: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("GridSentinel — Field Inspection Manifest", styles["Title"]),
        Paragraph(
            "Risk ranking from the benchmark-selected model. Confirm findings on-site before enforcement action.",
            styles["BodyText"],
        ),
        Spacer(1, 7 * mm),
    ]
    columns = ["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal", "evaluation_split"]
    labels = ["Rank", "Consumer", "Risk", "Tier", "Primary signal", "Cohort"]
    body = [labels] + [
        [
            str(int(row["rank"])),
            row["consumer_id"],
            f"{row['risk_score']:.3f}",
            row["risk_tier"],
            row["primary_signal"],
            row["evaluation_split"],
        ]
        for _, row in rows[columns].iterrows()
    ]
    table = Table(body, repeatRows=1, colWidths=[16 * mm, 30 * mm, 20 * mm, 30 * mm, 55 * mm, 28 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    document.build(story)
    return buffer.getvalue()


def line_figure(consumer: pd.DataFrame, consumer_id: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=consumer.timestamp,
            y=consumer.expected_kwh,
            name="Expected baseline",
            line=dict(color="#737373", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>Expected %{y:.2f} kWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=consumer.timestamp,
            y=consumer.consumption_kwh,
            name="Observed",
            line=dict(color="#EDEDED", width=1.6),
            hovertemplate="%{x}<br>Observed %{y:.2f} kWh<extra></extra>",
        )
    )
    abnormal = consumer[consumer.is_anomaly_period.astype(str).str.lower().isin(["true", "1"])]
    figure.add_trace(
        go.Scatter(
            x=abnormal.timestamp,
            y=abnormal.consumption_kwh,
            name="Injected abnormal period",
            mode="markers",
            marker=dict(color="#EF4444", size=4, opacity=0.8),
            hovertemplate="%{x}<br>Abnormal %{y:.2f} kWh<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Observed vs expected load · {consumer_id}",
        height=460,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A0A0A",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=1.08),
        xaxis_title=None,
        yaxis_title="Consumption (kWh)",
        hovermode="x unified",
    )
    return figure


def signal_radar(consumer: pd.DataFrame) -> go.Figure:
    observed = consumer.consumption_kwh.to_numpy(float)
    expected = consumer.expected_kwh.to_numpy(float)
    hours = consumer.timestamp.dt.hour.to_numpy()
    peak = (hours >= 18) & (hours <= 23)
    recent = np.arange(len(observed)) >= int(len(observed) * 0.65)
    measures = {
        "Mean deviation": np.mean(np.abs(observed - expected)) / max(np.mean(expected), 1e-6),
        "Peak suppression": 1 - np.mean(observed[peak & recent]) / max(np.mean(expected[peak & recent]), 1e-6),
        "Low-load persistence": np.mean(observed < np.quantile(expected, 0.15) * 0.5),
        "Volatility shift": abs(np.std(observed[recent]) - np.std(expected[recent])) / max(np.std(expected[recent]), 1e-6),
        "Baseline drift": abs(np.mean(observed[recent]) - np.mean(expected[recent])) / max(np.mean(expected[recent]), 1e-6),
    }
    values = np.clip(list(measures.values()), 0, 1)
    categories = list(measures)
    figure = go.Figure(
        go.Scatterpolar(
            r=np.r_[values, values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#EDEDED",
            fillcolor="rgba(237,237,237,.10)",
            name="Signal strength",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="#0A0A0A", radialaxis=dict(range=[0, 1], showticklabels=False)),
        margin=dict(l=45, r=45, t=35, b=35),
        showlegend=False,
    )
    return figure


try:
    ranking, metrics, pr_curves, gains, summary, readings = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python evaluate_models.py\nstreamlit run app.py", language="bash")
    st.stop()

benchmark = summary["benchmark"]
selected_model = summary["selected_model"]
top_priority = ranking[ranking.priority_top_10_percent.astype(str).str.lower().isin(["true", "1"])].copy()
top_ten = ranking.head(10).copy()

with st.sidebar:
    st.markdown("### GridSentinel")
    st.caption("Field Verification Console")
    st.divider()
    tier_options = list(ranking.risk_tier.unique())
    tiers = st.multiselect("Risk tiers", tier_options, default=tier_options)
    signal_options = sorted(ranking.primary_signal.unique())
    signals = st.multiselect("Anomaly signals", signal_options, default=signal_options)
    cohort_options = ["train", "validation", "test"]
    cohorts = st.multiselect("Evaluation cohort", cohort_options, default=cohort_options)
    filtered = ranking[
        ranking.risk_tier.isin(tiers)
        & ranking.primary_signal.isin(signals)
        & ranking.evaluation_split.isin(cohorts)
    ].copy()
    st.divider()
    st.markdown('<span class="status-pill">Benchmarks verified</span>', unsafe_allow_html=True)
    st.caption(f"Selected model: {selected_model}")
    st.caption("Validation: synthetic injected attacks")

st.markdown('<div class="eyebrow">Distribution Intelligence · Field Operations</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Electricity risk, ranked<br/>for action.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Consumer-level theft and abnormal-consumption triage with calibrated risk, transparent benchmark evidence, and inspection-ready manifests.</div>',
    unsafe_allow_html=True,
)
st.write("")

kpi_cols = st.columns(4)
kpi_cols[0].metric("Meters inspected", f"{summary['dataset']['consumers']:,}", help="Consumers included in this benchmark run")
kpi_cols[1].metric("Flagged high risk", f"{len(top_priority):,}", "Top 10% priority band")
kpi_cols[2].metric("Benchmark recall", f"{benchmark['recall']:.1%}", f"Target ≥ {summary['config']['recall_target']:.0%}")
kpi_cols[3].metric("Top 20% capture", f"{benchmark['top20_capture']:.1%}", f"Target ≥ {summary['config']['top20_target']:.0%}")

overview_tab, inspector_tab, comparison_tab, operations_tab = st.tabs(
    ["Priority queue", "Load inspector", "Model evidence", "Field operations"]
)

with overview_tab:
    st.markdown("#### Top 10 field priorities")
    st.caption("Edit the checkbox to stage a consumer for the inspection manifest. Sparklines show the latest 168 hours.")
    trend_map = {
        cid: group.sort_values("timestamp").consumption_kwh.tail(168).round(3).tolist()
        for cid, group in readings.groupby("consumer_id")
    }
    display = top_ten[["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal"]].copy()
    display["trend_7d"] = display.consumer_id.map(trend_map)
    display["schedule_inspection"] = display.consumer_id.isin(st.session_state.get("scheduled_ids", []))
    edited = st.data_editor(
        display,
        hide_index=True,
        width="stretch",
        disabled=["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal", "trend_7d"],
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
            "consumer_id": st.column_config.TextColumn("Consumer ID"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f"),
            "risk_tier": st.column_config.TextColumn("Risk category"),
            "primary_signal": st.column_config.TextColumn("Primary anomaly signal", width="large"),
            "trend_7d": st.column_config.LineChartColumn("Consumption sparkline", y_min=0, width="large"),
            "schedule_inspection": st.column_config.CheckboxColumn("Schedule field inspection"),
        },
        key="priority_editor",
    )
    st.session_state["scheduled_ids"] = edited.loc[edited.schedule_inspection, "consumer_id"].tolist()
    scheduled_count = len(st.session_state["scheduled_ids"])
    st.info(f"{scheduled_count} consumer{'s' if scheduled_count != 1 else ''} staged for field inspection.")
    st.markdown(
        '<div class="notice">Risk scores prioritize investigation; they are not proof of theft. Confirm meter integrity, premises context, and billing history before action.</div>',
        unsafe_allow_html=True,
    )

with inspector_tab:
    choices = [cid for cid in filtered.consumer_id if cid in set(readings.consumer_id)]
    if not choices:
        st.warning("No priority consumers match the current sidebar filters.")
    else:
        selected_id = st.selectbox("Consumer", choices, index=0)
        selected_row = ranking.set_index("consumer_id").loc[selected_id]
        consumer = readings[readings.consumer_id == selected_id].sort_values("timestamp")
        left, right = st.columns([2.3, 1])
        with left:
            st.plotly_chart(
                line_figure(consumer, selected_id), width="stretch", config={"displaylogo": False}
            )
        with right:
            st.metric("Calibrated risk", f"{selected_row.risk_score:.3f}")
            st.metric("Priority rank", f"#{int(selected_row['rank'])}")
            st.markdown(f"**Signal:** {selected_row.primary_signal}")
            st.plotly_chart(
                signal_radar(consumer), width="stretch", config={"displayModeBar": False}
            )

with comparison_tab:
    st.markdown("#### Shared holdout benchmark")
    st.caption(
        f"Both models were evaluated on the same {summary['dataset']['test_consumers']}-consumer test cohort. "
        f"Selection used validation AUC-PR, not test performance."
    )
    formatted_metrics = metrics.copy()
    metric_cols = ["accuracy", "recall", "precision", "f1", "auc_roc", "auc_pr", "top10_capture", "top20_capture"]
    formatted_metrics[metric_cols] = formatted_metrics[metric_cols].map(lambda value: f"{value:.1%}")
    st.dataframe(formatted_metrics, hide_index=True, width="stretch")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        pr_figure = px.line(
            pr_curves,
            x="recall",
            y="precision",
            color="model",
            title="Precision–recall curves",
            color_discrete_sequence=["#EDEDED", "#737373"],
        )
        pr_figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A0A0A", height=410)
        st.plotly_chart(pr_figure, width="stretch", config={"displaylogo": False})
    with chart_right:
        gains_figure = px.line(
            gains,
            x="population_fraction",
            y="anomaly_capture",
            color="model",
            title="Cumulative gains",
            color_discrete_sequence=["#EDEDED", "#737373"],
        )
        gains_figure.add_vline(x=0.2, line_dash="dash", line_color="#EDEDED", annotation_text="Top 20%")
        gains_figure.add_hline(y=0.6, line_dash="dot", line_color="#525252", annotation_text="60% target")
        gains_figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A0A0A", height=410)
        st.plotly_chart(gains_figure, width="stretch", config={"displaylogo": False})
    st.markdown(
        '<div class="notice"><b>Scope:</b> reported metrics measure deterministic simulated attacks in this repository. A utility deployment requires labeled field outcomes, temporal/out-of-region validation, drift monitoring, and threshold governance.</div>',
        unsafe_allow_html=True,
    )

with operations_tab:
    st.markdown("#### Inspection manifest")
    manifest_mode = st.radio("Manifest source", ["Scheduled selections", "Filtered priority list"], horizontal=True)
    if manifest_mode == "Scheduled selections":
        selected_ids = st.session_state.get("scheduled_ids", [])
        manifest = ranking[ranking.consumer_id.isin(selected_ids)].copy()
    else:
        manifest = filtered.head(100).copy()
    st.dataframe(
        manifest[["rank", "consumer_id", "risk_score", "risk_tier", "primary_signal", "evaluation_split"]],
        hide_index=True,
        width="stretch",
    )
    if manifest.empty:
        st.warning("No consumers are staged. Select rows in the Priority queue or change the manifest source.")
    else:
        csv_bytes = manifest.to_csv(index=False).encode("utf-8")
        pdf_bytes = manifest_pdf(manifest)
        export_left, export_right = st.columns(2)
        export_left.download_button(
            "Download field manifest · CSV",
            csv_bytes,
            "field_inspection_manifest.csv",
            "text/csv",
            width="stretch",
        )
        export_right.download_button(
            "Download field manifest · PDF",
            pdf_bytes,
            "field_inspection_manifest.pdf",
            "application/pdf",
            width="stretch",
        )

