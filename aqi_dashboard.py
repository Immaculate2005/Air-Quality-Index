"""
Air Quality Index Analyzer Dashboard
=====================================
A Streamlit app to analyze and monitor pollution levels across Indian regions.

Requirements:
    pip install streamlit pandas plotly numpy

Run:
    streamlit run aqi_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Index Analyzer",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262d40);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .metric-val { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e5e7eb;
        margin: 20px 0 10px 0;
        border-bottom: 1px solid #374151;
        padding-bottom: 6px;
    }
    .aqi-good { color: #22c55e; }
    .aqi-moderate { color: #eab308; }
    .aqi-unhealthy { color: #f97316; }
    .aqi-hazardous { color: #ef4444; }
    div[data-testid="stSidebarContent"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", low_memory=False)

    # ── 1. Drop fully empty rows ──────────────────────────────
    df.dropna(how="all", inplace=True)

    # ── 2. Parse date ─────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")

    # ── 3. Standardise pollutant columns ─────────────────────
    pollutants = ["so2", "no2", "rspm", "spm", "pm2_5"]
    for col in pollutants:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Remove physically impossible values ────────────────
    upper = {"so2": 1000, "no2": 1000, "rspm": 2000, "spm": 5000, "pm2_5": 1000}
    for col, limit in upper.items():
        df.loc[df[col] > limit, col] = np.nan

    # ── 5. Standardise 'type' column ─────────────────────────
    type_map = {
        "Residential, Rural and other Areas": "Residential/Rural",
        "Residential and others":             "Residential/Rural",
        "RIRUO":                              "Residential/Rural",
        "Residential":                        "Residential/Rural",
        "Industrial Area":                    "Industrial",
        "Industrial Areas":                   "Industrial",
        "Industrial":                         "Industrial",
        "Sensitive Area":                     "Sensitive",
        "Sensitive Areas":                    "Sensitive",
        "Sensitive":                          "Sensitive",
    }
    df["area_type"] = df["type"].map(type_map).fillna("Other")

    # ── 6. Compute AQI (simplified US EPA for PM2.5 + RSPM) ──
    def calc_aqi_pm(pm):
        """Simple linear AQI from PM2.5 (µg/m³)."""
        breakpoints = [
            (0,    12.0,   0,   50),
            (12.1, 35.4,  51,  100),
            (35.5, 55.4, 101,  150),
            (55.5, 150.4,151,  200),
            (150.5,250.4,201,  300),
            (250.5,350.4,301,  400),
            (350.5,500.4,401,  500),
        ]
        if pd.isna(pm):
            return np.nan
        for lo, hi, alo, ahi in breakpoints:
            if lo <= pm <= hi:
                return round((ahi - alo) / (hi - lo) * (pm - lo) + alo)
        return 500

    # Use pm2_5 if available else rspm as proxy
    df["aqi_input"] = df["pm2_5"].combine_first(df["rspm"])
    df["aqi"]       = df["aqi_input"].apply(calc_aqi_pm)

    def aqi_category(aqi):
        if pd.isna(aqi):   return "Unknown"
        if aqi <= 50:      return "Good"
        if aqi <= 100:     return "Moderate"
        if aqi <= 150:     return "Unhealthy for Sensitive"
        if aqi <= 200:     return "Unhealthy"
        if aqi <= 300:     return "Very Unhealthy"
        return "Hazardous"

    df["aqi_category"] = df["aqi"].apply(aqi_category)

    # ── 7. Drop rows where ALL pollutants are null ────────────
    df = df[df[pollutants].notna().any(axis=1)].copy()

    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=72)
    st.title("🌫️ AQI Analyzer")
    st.caption("India Air Quality Monitor")
    st.divider()

    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    data_path = "data.csv"          # default – place data.csv next to this script

    st.divider()
    st.markdown("### Filters")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if uploaded:
    df_raw = pd.read_csv(uploaded, encoding="latin1", low_memory=False)
    df_raw.to_csv("/tmp/_aqi_upload.csv", index=False)
    df = load_and_clean("/tmp/_aqi_upload.csv")
else:
    try:
        df = load_and_clean(data_path)
    except FileNotFoundError:
        st.error(
            "⚠️ `data.csv` not found. Please place it next to this script "
            "or upload it using the sidebar."
        )
        st.stop()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    states = sorted(df["state"].dropna().unique())
    sel_states = st.multiselect("State(s)", states, default=states[:5])

    years = sorted(df["year"].dropna().unique().astype(int))
    yr_min, yr_max = int(min(years)), int(max(years))
    sel_years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    area_types = sorted(df["area_type"].unique())
    sel_types  = st.multiselect("Area type", area_types, default=area_types)

    pollutants_all = {"SO₂": "so2", "NO₂": "no2", "RSPM": "rspm", "SPM": "spm", "PM2.5": "pm2_5"}
    sel_pollutant_label = st.selectbox("Primary pollutant", list(pollutants_all.keys()))
    sel_pollutant       = pollutants_all[sel_pollutant_label]

    st.divider()
    st.caption("Data spans 1987–2015 across 37 Indian states.")

# ─────────────────────────────────────────────
# FILTER DATAFRAME
# ─────────────────────────────────────────────
mask = (
    df["state"].isin(sel_states if sel_states else states) &
    df["year"].between(sel_years[0], sel_years[1]) &
    df["area_type"].isin(sel_types if sel_types else area_types)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.2rem; font-weight:800;
           background: linear-gradient(90deg,#38bdf8,#818cf8);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           margin-bottom:0;'>
🌫️ Air Quality Index Analyzer
</h1>
<p style='text-align:center; color:#6b7280; margin-top:4px;'>
Monitoring pollution levels across Indian regions · {n:,} records loaded
</p>
""".format(n=len(fdf)), unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

avg_aqi  = fdf["aqi"].mean()
max_aqi  = fdf["aqi"].max()
avg_so2  = fdf["so2"].mean()
avg_no2  = fdf["no2"].mean()
n_locs   = fdf["location"].nunique()

def aqi_color(v):
    if pd.isna(v):      return "#6b7280"
    if v <= 50:         return "#22c55e"
    if v <= 100:        return "#eab308"
    if v <= 150:        return "#f97316"
    if v <= 200:        return "#ef4444"
    return "#b91c1c"

for col, label, value, unit, color in [
    (col1, "Avg AQI",    avg_aqi,  "",       aqi_color(avg_aqi)),
    (col2, "Max AQI",    max_aqi,  "",       aqi_color(max_aqi)),
    (col3, "Avg SO₂",   avg_so2,  " µg/m³", "#60a5fa"),
    (col4, "Avg NO₂",   avg_no2,  " µg/m³", "#a78bfa"),
    (col5, "Locations",  n_locs,   "",       "#34d399"),
]:
    with col:
        val_str = f"{value:,.1f}{unit}" if not pd.isna(value) else "N/A"
        st.markdown(f"""
        <div class='metric-card' style='border-color:{color}'>
            <div class='metric-val' style='color:{color}'>{val_str}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 1 – Trend + AQI Category Distribution
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Pollution Trends Over Time</div>", unsafe_allow_html=True)
r1c1, r1c2 = st.columns([2, 1])

with r1c1:
    trend = (
        fdf.groupby("year")[[sel_pollutant, "aqi"]]
        .mean()
        .reset_index()
        .rename(columns={sel_pollutant: sel_pollutant_label})
    )
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Scatter(
            x=trend["year"], y=trend[sel_pollutant_label],
            name=sel_pollutant_label, mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5),
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=trend["year"], y=trend["aqi"],
            name="Avg AQI", mode="lines+markers",
            line=dict(color="#f97316", width=2, dash="dot"),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    fig_trend.update_yaxes(title_text=f"{sel_pollutant_label} (µg/m³)", secondary_y=False, gridcolor="#1f2937")
    fig_trend.update_yaxes(title_text="AQI", secondary_y=True)
    fig_trend.update_xaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_trend, use_container_width=True)

with r1c2:
    cat_counts = fdf["aqi_category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    color_map = {
        "Good":                       "#22c55e",
        "Moderate":                   "#eab308",
        "Unhealthy for Sensitive":    "#f97316",
        "Unhealthy":                  "#ef4444",
        "Very Unhealthy":             "#b91c1c",
        "Hazardous":                  "#7f1d1d",
        "Unknown":                    "#6b7280",
    }
    fig_pie = px.pie(
        cat_counts, values="Count", names="Category",
        color="Category", color_discrete_map=color_map,
        hole=0.55,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="v", font=dict(size=11)),
        title=dict(text="AQI Categories", font=dict(size=14)),
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 2 – State Comparison + Monthly Heatmap
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>🗺️ Regional Comparison</div>", unsafe_allow_html=True)
r2c1, r2c2 = st.columns([1, 1])

with r2c1:
    state_avg = (
        fdf.groupby("state")[sel_pollutant]
        .mean()
        .dropna()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig_bar = px.bar(
        state_avg, x=sel_pollutant, y="state",
        orientation="h",
        color=sel_pollutant,
        color_continuous_scale="RdYlGn_r",
        labels={sel_pollutant: f"{sel_pollutant_label} (µg/m³)", "state": ""},
        title=f"Average {sel_pollutant_label} by State",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0),
        height=max(300, 22 * len(state_avg)),
    )
    fig_bar.update_xaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_bar, use_container_width=True)

with r2c2:
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heat = (
        fdf.groupby(["year", "month_name"])[sel_pollutant]
        .mean()
        .reset_index()
    )
    heat["month_name"] = pd.Categorical(heat["month_name"], categories=month_order, ordered=True)
    heat = heat.sort_values(["year", "month_name"])
    heat_pivot = heat.pivot(index="year", columns="month_name", values=sel_pollutant)
    # reorder columns
    heat_pivot = heat_pivot[[m for m in month_order if m in heat_pivot.columns]]

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels=dict(x="Month", y="Year", color=sel_pollutant_label),
        title=f"Monthly {sel_pollutant_label} Heatmap (µg/m³)",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 3 – Pollutant Correlation + Area Type Box
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>🔬 Pollutant Deep-Dive</div>", unsafe_allow_html=True)
r3c1, r3c2 = st.columns([1, 1])

with r3c1:
    poll_cols = ["so2", "no2", "rspm", "spm", "pm2_5"]
    corr = fdf[poll_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels=dict(color="Correlation"),
        title="Pollutant Correlation Matrix",
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with r3c2:
    fig_box = px.box(
        fdf[fdf[sel_pollutant].notna()],
        x="area_type", y=sel_pollutant,
        color="area_type",
        labels={"area_type": "Area Type", sel_pollutant: f"{sel_pollutant_label} (µg/m³)"},
        title=f"{sel_pollutant_label} Distribution by Area Type",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_box.update_xaxes(gridcolor="#1f2937")
    fig_box.update_yaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 4 – Top Polluted Cities + Scatter
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>🏙️ City-Level Insights</div>", unsafe_allow_html=True)
r4c1, r4c2 = st.columns([1, 1])

with r4c1:
    top_cities = (
        fdf.groupby(["location", "state"])[sel_pollutant]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_cities["label"] = top_cities["location"] + " (" + top_cities["state"] + ")"
    fig_top = px.bar(
        top_cities[::-1], x=sel_pollutant, y="label",
        orientation="h",
        color=sel_pollutant,
        color_continuous_scale="Reds",
        labels={sel_pollutant: f"Avg {sel_pollutant_label}", "label": ""},
        title=f"Top 15 Most Polluted Locations ({sel_pollutant_label})",
    )
    fig_top.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    fig_top.update_xaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_top, use_container_width=True)

with r4c2:
    scatter_df = fdf[["so2", "no2", "aqi", "state", "area_type"]].dropna()
    sample = scatter_df.sample(min(5000, len(scatter_df)), random_state=42)
    fig_sc = px.scatter(
        sample, x="so2", y="no2", color="aqi",
        color_continuous_scale="RdYlGn_r",
        opacity=0.6, size_max=8,
        labels={"so2": "SO₂ (µg/m³)", "no2": "NO₂ (µg/m³)", "aqi": "AQI"},
        title="SO₂ vs NO₂ (coloured by AQI)",
        hover_data=["state", "area_type"],
    )
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    fig_sc.update_xaxes(gridcolor="#1f2937")
    fig_sc.update_yaxes(gridcolor="#1f2937")
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 5 – Cleaned Data Preview + Download
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Cleaned Data Preview</div>", unsafe_allow_html=True)

display_cols = ["date", "state", "location", "area_type", "so2", "no2", "rspm", "spm", "pm2_5", "aqi", "aqi_category"]
preview = fdf[display_cols].sort_values("date", ascending=False).head(500)

st.dataframe(
    preview.style.background_gradient(subset=["aqi"], cmap="RdYlGn_r"),
    use_container_width=True, height=320,
)

# Download cleaned CSV
csv_bytes = fdf[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Cleaned Data (CSV)",
    data=csv_bytes,
    file_name="aqi_cleaned.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center; color:#4b5563; font-size:0.8rem;'>
Air Quality Index Analyzer · Data source: India CPCB monitoring stations (1987–2015)
</p>
""", unsafe_allow_html=True)
