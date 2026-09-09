import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (make_reserve_grid, flag_reserve_zones, make_production_data,
                   train_shortfall_model, forecast_future, recommend, BASELINE_TONS_PER_DAY)

st.set_page_config(page_title="AstroLithos — SIH26009", layout="wide")

st.title("⛏️ AstroLithos — Manganese Reserve & Production Intelligence")
st.caption("SIH26009 · Ministry of Steel (MOIL Ltd.) · AI/ML + Space Technology to identify "
           "manganese reserves and overcome production shortfalls")

with st.expander("ℹ️ About this prototype — what's real vs. simulated", expanded=False):
    st.markdown("""
    - **Reserve mapping** uses a *simulated* multi-band raster standing in for real
      Sentinel-2 / ASTER / ISRO Bhoonidhi imagery. The band-ratio mineral index is a
      real remote-sensing technique — only the input pixels here are synthetic.
    - **Production data** is *simulated* mine-ops data (real MOIL downtime/weather/blasting
      logs aren't public). The prediction model works the same way once real data is swapped in.
    - Goal of this build: prove the three-part pipeline the PS asks for — **map reserves →
      predict shortfalls → recommend corrective action** — end to end, in one dashboard.
    """)

tab1, tab2, tab3 = st.tabs(["🛰️ Reserve Mapping", "📉 Shortfall Prediction", "✅ Recommendations"])

# ---------------- Tab 1: Reserve Mapping ----------------
with tab1:
    st.subheader("Probable manganese-reserve zones from spectral band ratios")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        percentile = st.slider("Flag top X% as probable reserve zone", 70, 95, 85)
        st.metric("Grid size", "80 × 80 px")
    true_signal, combined = make_reserve_grid()
    mask, threshold = flag_reserve_zones(combined, percentile)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.imshow(combined, color_continuous_scale="Inferno",
                          title="Ferrous / iron-oxide alteration index (band ratio)")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.imshow(mask.astype(int), color_continuous_scale=["#1a1a2e", "#f9c74f"],
                          title=f"Flagged probable reserve zones (top {100-percentile}%)")
        st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"{mask.sum()} of {mask.size} cells flagged · threshold index value ≥ {threshold:.2f}")

# ---------------- Tab 2: Shortfall Prediction ----------------
with tab2:
    st.subheader("Production trend & predicted shortfall risk")
    df = make_production_data()
    model, r2 = train_shortfall_model(df)
    future = forecast_future(model)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model fit (R²)", f"{r2:.3f}")
    m2.metric("Baseline capacity", f"{BASELINE_TONS_PER_DAY} t/day")
    m3.metric("Forecast days at risk", f"{int(future['shortfall_risk'].sum())} / {len(future)}")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["date"], y=df["tons_produced"], name="Historical (actual)",
                               line=dict(color="#577590")))
    fig3.add_hline(y=0.8 * BASELINE_TONS_PER_DAY, line_dash="dot", line_color="crimson",
                    annotation_text="shortfall threshold")
    fig3.update_layout(title="Historical daily production", xaxis_title="Date", yaxis_title="Tons")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    colors = ["crimson" if r else "#43aa8b" for r in future["shortfall_risk"]]
    fig4.add_trace(go.Bar(x=future["day"], y=future["predicted_tons"], marker_color=colors,
                           name="Predicted tons"))
    fig4.add_hline(y=0.8 * BASELINE_TONS_PER_DAY, line_dash="dot", line_color="crimson")
    fig4.update_layout(title="Next 14 days — predicted production (red = shortfall risk)")
    st.plotly_chart(fig4, use_container_width=True)

    imp = pd.Series(model.feature_importances_, index=["downtime_hrs", "rainfall_mm", "blast_delay_days"])
    st.plotly_chart(px.bar(imp.sort_values(), orientation="h", title="What's driving the prediction"),
                     use_container_width=True)

# ---------------- Tab 3: Recommendations ----------------
with tab3:
    st.subheader("Corrective actions for flagged shortfall days")
    at_risk = future[future["shortfall_risk"]].copy()
    if at_risk.empty:
        st.success("No shortfall risk in the next 14 days under current forecast inputs.")
    else:
        at_risk["recommendation"] = at_risk.apply(recommend, axis=1)
        st.dataframe(
            at_risk[["day", "downtime_hrs", "rainfall_mm", "blast_delay_days",
                     "predicted_tons", "recommendation"]]
            .rename(columns={"downtime_hrs": "Downtime (hrs)", "rainfall_mm": "Rainfall (mm)",
                              "blast_delay_days": "Blast delay (days)", "predicted_tons": "Predicted tons",
                              "recommendation": "Recommended action", "day": "Day"}),
            use_container_width=True, hide_index=True,
        )
