import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (make_reserve_grid, flag_reserve_zones, make_production_data,
                   train_shortfall_model, forecast_future, recommend, BASELINE_TONS_PER_DAY)

st.set_page_config(page_title="AstroLithos — SIH26009", page_icon="⛏️", layout="wide",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#0B0D10"
PANEL = "#151A20"
PANEL_BORDER = "#262E38"
TEXT = "#E9E4DA"
TEXT_DIM = "#8B93A0"
AMBER = "#E8871E"
AMBER_DIM = "#8A5A22"
RUST = "#D95D4A"
TEAL = "#3FB8AF"
GREEN = "#5FBF77"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="IBM Plex Sans, sans-serif", size=13),
        title=dict(font=dict(color=TEXT, size=15)),
        xaxis=dict(gridcolor="#232A32", zerolinecolor="#232A32", linecolor=PANEL_BORDER, color=TEXT_DIM),
        yaxis=dict(gridcolor="#232A32", zerolinecolor="#232A32", linecolor=PANEL_BORDER, color=TEXT_DIM),
        colorway=[AMBER, TEAL, RUST, GREEN],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=48, l=10, r=10, b=10),
    )
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}
.stApp {{
    background: {INK};
}}
section[data-testid="stSidebar"] {{
    background: #0D1013;
    border-right: 1px solid {PANEL_BORDER};
}}
#MainMenu, footer, header {{visibility: hidden;}}

/* Hero */
.ast-hero {{
    border: 1px solid {PANEL_BORDER};
    background: linear-gradient(135deg, #171B20 0%, #12151A 100%);
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}}
.ast-hero h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: {TEXT};
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}}
.ast-hero p {{
    color: {TEXT_DIM};
    margin: 0;
    font-size: 0.88rem;
}}
.ast-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.ast-badge {{
    font-size: 0.72rem;
    padding: 5px 11px;
    border-radius: 20px;
    border: 1px solid {PANEL_BORDER};
    color: {TEXT_DIM};
    background: #10131700;
    white-space: nowrap;
}}
.ast-badge.live {{ color: {AMBER}; border-color: {AMBER_DIM}; }}
.ast-badge.sim {{ color: {TEAL}; border-color: #1E5F5A; }}

/* Section label */
.ast-section-label {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.08rem;
    font-weight: 600;
    color: {TEXT};
    margin: 4px 0 14px 0;
    padding-left: 12px;
    border-left: 3px solid {AMBER};
}}

/* KPI cards */
.ast-kpi-row {{ display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }}
.ast-kpi {{
    flex: 1;
    min-width: 160px;
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 14px 18px;
}}
.ast-kpi .label {{
    font-size: 0.72rem;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}}
.ast-kpi .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: {TEXT};
}}
.ast-kpi .value.warn {{ color: {RUST}; }}
.ast-kpi .value.ok {{ color: {GREEN}; }}
.ast-kpi .sub {{ font-size: 0.75rem; color: {TEXT_DIM}; margin-top: 2px; }}

/* Panel wrapper around charts */
.ast-panel {{
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 14px 16px 4px 16px;
    margin-bottom: 16px;
}}
.ast-panel-title {{
    font-size: 0.78rem;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}}

/* Sidebar nav */
.ast-nav-title {{
    font-family: 'Space Grotesk', sans-serif;
    color: {TEXT};
    font-weight: 700;
    font-size: 1.05rem;
    padding: 4px 4px 0 4px;
}}
.ast-nav-sub {{ color: {TEXT_DIM}; font-size: 0.75rem; padding: 0 4px 14px 4px; }}
div[role="radiogroup"] label {{
    background: transparent;
    border-radius: 8px;
    padding: 2px 4px;
}}
div[role="radiogroup"] {{ gap: 2px; }}

/* Recommendation banner */
.ast-flow {{
    display: flex; align-items: center; gap: 6px; color: {TEXT_DIM};
    font-size: 0.8rem; margin-bottom: 18px; flex-wrap: wrap;
}}
.ast-flow .node {{ color: {TEXT}; background: {PANEL}; border: 1px solid {PANEL_BORDER};
    padding: 4px 10px; border-radius: 6px; }}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", tone=""):
    tone_cls = f" {tone}" if tone else ""
    return f"""<div class="ast-kpi"><div class="label">{label}</div>
    <div class="value{tone_cls}">{value}</div>
    <div class="sub">{sub}</div></div>"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="ast-nav-title">⛏️ AstroLithos</div>', unsafe_allow_html=True)
    st.markdown('<div class="ast-nav-sub">SIH26009 · Reserve & Production Intelligence</div>',
                unsafe_allow_html=True)
    section = st.radio(
        "Navigate",
        ["🛰️ Reserve Mapping", "📉 Shortfall Prediction", "✅ Recommendations"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="ast-nav-sub" style="padding-top:0;">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; line-height:2.1; color:{TEXT_DIM};">
    🛰️ Map reserves &nbsp;→<br>
    📉 Predict shortfalls &nbsp;→<br>
    ✅ Recommend action
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("What's real vs. simulated"):
        st.markdown("""
        - **Reserve mapping**: real band-ratio mineral-index technique,
          run on *simulated* multi-band pixels standing in for
          Sentinel-2 / ASTER / ISRO Bhoonidhi imagery.
        - **Production data**: real regression modeling approach, run on
          *simulated* mine-ops data (MOIL's real logs aren't public).
        - Goal: prove the pipeline end to end so real data can be
          swapped in later.
        """)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="ast-hero">
  <div>
    <h1>Manganese Reserve &amp; Production Intelligence</h1>
    <p>SIH26009 · Ministry of Steel (MOIL Ltd.) · AI/ML + Space Technology for reserve
    identification and shortfall recovery</p>
  </div>
  <div class="ast-badges">
    <span class="ast-badge sim">◆ SIMULATED DATA</span>
    <span class="ast-badge live">◆ PIPELINE ACTIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 1: Reserve Mapping
# ---------------------------------------------------------------------------
if section == "🛰️ Reserve Mapping":
    st.markdown('<div class="ast-section-label">Probable manganese-reserve zones '
                'from spectral band ratios</div>', unsafe_allow_html=True)

    col_ctrl, col_kpi = st.columns([1, 2])
    with col_ctrl:
        percentile = st.slider("Flag top X% as probable reserve zone", 70, 95, 85)
    true_signal, combined = make_reserve_grid()
    mask, threshold = flag_reserve_zones(combined, percentile)

    with col_kpi:
        st.markdown(f"""<div class="ast-kpi-row">
            {kpi("Grid resolution", "80 × 80", "px, simulated raster")}
            {kpi("Cells flagged", f"{mask.sum()}", f"of {mask.size} total", "ok")}
            {kpi("Index threshold", f"{threshold:.2f}", f"≥ top {100 - percentile}%")}
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="ast-panel"><div class="ast-panel-title">'
                    'Ferrous / iron-oxide alteration index</div>', unsafe_allow_html=True)
        fig1 = px.imshow(combined, color_continuous_scale="Inferno")
        fig1.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ast-panel"><div class="ast-panel-title">'
                    f'Flagged probable reserve zones (top {100-percentile}%)</div>',
                    unsafe_allow_html=True)
        fig2 = px.imshow(mask.astype(int), color_continuous_scale=[PANEL, AMBER])
        fig2.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: Shortfall Prediction
# ---------------------------------------------------------------------------
elif section == "📉 Shortfall Prediction":
    st.markdown('<div class="ast-section-label">Production trend &amp; predicted '
                'shortfall risk</div>', unsafe_allow_html=True)

    df = make_production_data()
    model, r2 = train_shortfall_model(df)
    future = forecast_future(model)
    at_risk_days = int(future["shortfall_risk"].sum())
    risk_tone = "warn" if at_risk_days > 0 else "ok"

    st.markdown(f"""<div class="ast-kpi-row">
        {kpi("Model fit (R²)", f"{r2:.3f}", "RandomForestRegressor")}
        {kpi("Baseline capacity", f"{BASELINE_TONS_PER_DAY}", "tons / day")}
        {kpi("Forecast days at risk", f"{at_risk_days} / {len(future)}", "next 14-day window", risk_tone)}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="ast-panel"><div class="ast-panel-title">Historical daily production</div>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["date"], y=df["tons_produced"], name="Historical (actual)",
                               line=dict(color=TEAL, width=2)))
    fig3.add_hline(y=0.8 * BASELINE_TONS_PER_DAY, line_dash="dot", line_color=RUST,
                    annotation_text="shortfall threshold", annotation_font_color=RUST)
    fig3.update_layout(template=PLOTLY_TEMPLATE, height=320, xaxis_title=None, yaxis_title="Tons")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="ast-panel"><div class="ast-panel-title">'
                    'Next 14 days — predicted production</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        colors = [RUST if r else GREEN for r in future["shortfall_risk"]]
        fig4.add_trace(go.Bar(x=future["day"], y=future["predicted_tons"], marker_color=colors,
                               name="Predicted tons"))
        fig4.add_hline(y=0.8 * BASELINE_TONS_PER_DAY, line_dash="dot", line_color=RUST)
        fig4.update_layout(template=PLOTLY_TEMPLATE, height=320, xaxis_title=None)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ast-panel"><div class="ast-panel-title">'
                    "What's driving the prediction</div>", unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_,
                         index=["downtime_hrs", "rainfall_mm", "blast_delay_days"])
        fig5 = px.bar(imp.sort_values(), orientation="h")
        fig5.update_traces(marker_color=AMBER)
        fig5.update_layout(template=PLOTLY_TEMPLATE, height=320, showlegend=False,
                            xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.session_state["future"] = future  # hand off to Recommendations section

# ---------------------------------------------------------------------------
# Tab 3: Recommendations
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="ast-section-label">Corrective actions for flagged '
                'shortfall days</div>', unsafe_allow_html=True)

    # Recompute if user jumped here directly without visiting tab 2 first
    if "future" in st.session_state:
        future = st.session_state["future"]
    else:
        df = make_production_data()
        model, r2 = train_shortfall_model(df)
        future = forecast_future(model)

    at_risk = future[future["shortfall_risk"]].copy()
    if at_risk.empty:
        st.markdown(f"""<div class="ast-panel" style="border-color:{GREEN};">
        <span style="color:{GREEN}; font-weight:600;">✅ No shortfall risk</span>
        <span style="color:{TEXT_DIM};"> in the next 14 days under current forecast inputs.</span>
        </div>""", unsafe_allow_html=True)
    else:
        at_risk["recommendation"] = at_risk.apply(recommend, axis=1)
        at_risk = at_risk.round({"downtime_hrs": 1, "rainfall_mm": 1, "predicted_tons": 0})
        st.markdown(f"""<div class="ast-kpi-row">
            {kpi("Flagged days", f"{len(at_risk)}", "require corrective action", "warn")}
            {kpi("Avg. predicted shortfall", f"{(BASELINE_TONS_PER_DAY - at_risk['predicted_tons']).mean():.0f} t", "below baseline")}
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="ast-panel">', unsafe_allow_html=True)
        st.dataframe(
            at_risk[["day", "downtime_hrs", "rainfall_mm", "blast_delay_days",
                     "predicted_tons", "recommendation"]]
            .rename(columns={"downtime_hrs": "Downtime (hrs)", "rainfall_mm": "Rainfall (mm)",
                              "blast_delay_days": "Blast delay (days)", "predicted_tons": "Predicted tons",
                              "recommendation": "Recommended action", "day": "Day"}),
            use_container_width=True, hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
