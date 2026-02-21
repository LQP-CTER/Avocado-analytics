import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import re
import warnings

warnings.filterwarnings('ignore')

# ── IBCS COLOR SYSTEM ─────────────────────────────────────────────────────────
# IBCS mandates: black/dark for actual, grey for previous/reference,
# white/outlined for plan, accent ONLY for highlight (max 1 per chart)

C_BLACK     = "#1A1A1A"   # Current year / primary actual
C_DARK_GREY = "#5A5A5A"   # Secondary actual / labels
C_MID_GREY  = "#A0A0A0"   # Reference / previous period
C_LIGHT_GREY= "#D8D8D8"   # Grid lines, separators
C_BG        = "#FAFAFA"   # Page background
C_WHITE     = "#FFFFFF"   # Card background
C_ACCENT    = "#2D6A2D"   # Single accent — avocado dark green (use sparingly)
C_WARN      = "#B54A00"   # Negative variance / warning

# IBCS chart template
IBCS_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="DM Sans, sans-serif", size=12, color=C_BLACK),
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        xaxis=dict(showgrid=False, zeroline=False, linecolor=C_LIGHT_GREY, ticks="outside", tickcolor=C_LIGHT_GREY, ticklen=3),
        yaxis=dict(showgrid=True,  gridcolor=C_LIGHT_GREY, zeroline=False, linecolor=C_LIGHT_GREY),
        margin=dict(l=10, r=10, t=36, b=10),
        colorway=[C_BLACK, C_MID_GREY, C_ACCENT, C_WARN],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Avocado Market | IBCS Report",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
# Using DM Sans (humanist grotesque) + DM Mono for numbers — very editorial
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── RESET & BASE ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ 
    background-color: {C_BG} !important; 
    font-family: 'DM Sans', sans-serif !important;
    color: {C_BLACK};
}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
    background-color: {C_WHITE} !important;
    border-right: 1px solid {C_LIGHT_GREY} !important;
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1.25rem; }}

/* Sidebar brand strip */
.sidebar-brand {{
    background: {C_BLACK};
    color: {C_WHITE};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 12px 16px;
    margin: -1.5rem -1.25rem 1.5rem -1.25rem;
}}
.sidebar-brand span {{ color: {C_LIGHT_GREY}; }}

/* Sidebar section label */
.sidebar-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C_MID_GREY};
    margin: 1.25rem 0 0.5rem 0;
    border-bottom: 1px solid {C_LIGHT_GREY};
    padding-bottom: 0.35rem;
}}

/* ── MAIN HEADER ── */
.report-header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 3px solid {C_BLACK};
    padding-bottom: 12px;
    margin-bottom: 4px;
}}
.report-title {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {C_BLACK};
    line-height: 1.1;
}}
.report-subtitle {{
    font-size: 13px;
    color: {C_DARK_GREY};
    font-weight: 400;
    margin-top: 4px;
}}
.report-meta {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {C_MID_GREY};
    text-align: right;
    line-height: 1.6;
}}

/* ── SECTION HEADER — IBCS H1 style ── */
.section-h {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {C_WHITE};
    background: {C_BLACK};
    padding: 5px 10px;
    display: inline-block;
    margin: 28px 0 18px 0;
}}

/* ── CARD (chart wrapper) ── */
.card {{
    background: {C_WHITE};
    border: 1px solid {C_LIGHT_GREY};
    border-top: 3px solid {C_BLACK};
    padding: 16px 18px 10px 18px;
    margin-bottom: 18px;
    height: 100%;
}}
.card-title {{
    font-size: 13px;
    font-weight: 700;
    color: {C_BLACK};
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}}
.card-unit {{
    font-size: 11px;
    color: {C_MID_GREY};
    font-family: 'DM Mono', monospace;
    margin-bottom: 12px;
}}

/* ── IBCS KPI BLOCK ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: {C_LIGHT_GREY};
    border: 1px solid {C_LIGHT_GREY};
    margin-bottom: 24px;
}}
.kpi-cell {{
    background: {C_WHITE};
    padding: 18px 20px;
}}
.kpi-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C_MID_GREY};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: {C_BLACK};
    letter-spacing: -0.04em;
    line-height: 1;
}}
.kpi-delta {{
    font-size: 11px;
    margin-top: 5px;
    font-family: 'DM Mono', monospace;
}}
.kpi-delta.pos {{ color: {C_ACCENT}; }}
.kpi-delta.neg {{ color: {C_WARN}; }}

/* ── TABLE STYLE ── */
.ibcs-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.ibcs-table th {{
    background: {C_BLACK};
    color: {C_WHITE};
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.06em;
    font-size: 10px;
    text-transform: uppercase;
}}
.ibcs-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid {C_LIGHT_GREY};
    font-family: 'DM Mono', monospace;
    font-size: 11px;
}}
.ibcs-table tr:hover td {{ background: #F0F0F0; }}

/* ── VARIANCE BAR (small inline bar chart in table) ── */
.var-bar-pos {{ 
    display: inline-block; height: 8px; background: {C_ACCENT}; 
    vertical-align: middle; margin-left: 4px;
}}
.var-bar-neg {{ 
    display: inline-block; height: 8px; background: {C_WARN}; 
    vertical-align: middle; margin-left: 4px;
}}

/* ── TAB NAVIGATION ── */
div[data-testid="stTabs"] button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: {C_MID_GREY} !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_BLACK} !important;
    border-bottom-color: {C_BLACK} !important;
}}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stMetric"] {{ display: none; }}  /* hide default metrics, we use custom */
.stPlotlyChart {{ border: none !important; }}
div.stButton > button {{
    background: {C_BLACK};
    color: {C_WHITE};
    border-radius: 0;
    border: none;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 24px;
}}
div.stButton > button:hover {{
    background: {C_ACCENT};
    color: {C_WHITE};
}}
div[data-testid="stForm"] {{
    background: {C_WHITE};
    border: 1px solid {C_LIGHT_GREY};
    border-top: 3px solid {C_BLACK};
    padding: 20px;
}}
.stSelectbox label, .stSlider label, .stNumberInput label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {C_DARK_GREY} !important;
}}
.stSuccess {{
    background: {C_WHITE} !important;
    border-left: 4px solid {C_ACCENT} !important;
    border-radius: 0 !important;
    color: {C_BLACK} !important;
}}
.stInfo {{
    background: {C_WHITE} !important;
    border-left: 4px solid {C_DARK_GREY} !important;
    border-radius: 0 !important;
}}

/* ── PREDICTION RESULT BOX ── */
.pred-box {{
    background: {C_BLACK};
    color: {C_WHITE};
    padding: 24px 28px;
    margin-top: 16px;
}}
.pred-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {C_MID_GREY};
    margin-bottom: 8px;
}}
.pred-value {{
    font-family: 'DM Mono', monospace;
    font-size: 42px;
    font-weight: 500;
    color: {C_WHITE};
    letter-spacing: -0.04em;
}}
.pred-context {{
    font-size: 11px;
    color: {C_MID_GREY};
    margin-top: 6px;
}}

/* ── FOOTNOTE ── */
.footnote {{
    font-size: 10px;
    color: {C_MID_GREY};
    border-top: 1px solid {C_LIGHT_GREY};
    margin-top: 12px;
    padding-top: 6px;
    font-style: italic;
}}

/* ── DIVIDER ── */
hr {{ border: none; border-top: 1px solid {C_LIGHT_GREY}; margin: 20px 0; }}
</style>
""", unsafe_allow_html=True)


# ── DATA PROCESSING ───────────────────────────────────────────────────────────
def extract_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('avocado-cleaned.csv')
    except FileNotFoundError:
        st.error("File 'avocado-cleaned.csv' not found. Please place it in the same directory as app.py.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df['type'] = df['type'].astype(str).str.lower().str.strip()
    df['region'] = df['region'].astype(str).apply(lambda x: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', x))
    df = df[df['TotalVolume'] > 0]
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(extract_season)
    df['Bags_Ratio'] = np.where(df['TotalVolume'] > 0, df['TotalBags'] / df['TotalVolume'], 0)
    df['Small_Medium_Ratio'] = np.where(df['TotalVolume'] > 0, df['plu4046'] / df['TotalVolume'], 0)
    df['Large_Ratio']        = np.where(df['TotalVolume'] > 0, df['plu4225'] / df['TotalVolume'], 0)
    df['XLarge_Ratio']       = np.where(df['TotalVolume'] > 0, df['plu4770'] / df['TotalVolume'], 0)
    return df


@st.cache_resource
def train_model_and_evaluate(df):
    features = ['TotalVolume','Bags_Ratio','Small_Medium_Ratio','Large_Ratio','XLarge_Ratio','year','Month']
    le_type   = LabelEncoder(); df['type_encoded']   = le_type.fit_transform(df['type'])
    le_region = LabelEncoder(); df['region_encoded'] = le_region.fit_transform(df['region'])
    le_season = LabelEncoder(); df['season_encoded'] = le_season.fit_transform(df['Season'])
    features += ['type_encoded','region_encoded','season_encoded']

    X = df[features]; y = df['AveragePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {'MAE': mean_absolute_error(y_test, y_pred),
               'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
               'R2': r2_score(y_test, y_pred)}
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    return model, le_type, le_region, le_season, features, metrics, comparison_df


def run_kmeans(df, n_clusters=4):
    X = df[['AveragePrice','TotalVolume','Bags_Ratio']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X_scaled)


def get_keywords(text_series):
    text = " ".join(text_series.astype(str).tolist()).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop = {'and','of','the','us','total','city','south','north','west','east','new','san','las','los'}
    filtered = [w for w in words if w not in stop and len(w) > 3]
    return Counter(filtered).most_common(15)


def apply_ibcs_layout(fig, title="", unit="", height=380):
    """Apply consistent IBCS styling to any plotly figure."""
    fig.update_layout(
        template=None,
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        font=dict(family="DM Sans, sans-serif", size=11, color=C_BLACK),
        title=None,
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10)
        ),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=C_LIGHT_GREY,
                   ticks="outside", tickcolor=C_LIGHT_GREY, ticklen=3,
                   tickfont=dict(family="DM Mono, monospace", size=10)),
        yaxis=dict(showgrid=True, gridcolor=C_LIGHT_GREY, zeroline=False,
                   linecolor=C_LIGHT_GREY,
                   tickfont=dict(family="DM Mono, monospace", size=10)),
    )
    return fig


def card(title, unit=""):
    """Render card header HTML."""
    return f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-unit">{unit}</div>
    """


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = load_and_clean_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        AVOCADO ANALYTICS <span>· IBCS Report</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Data Filters</div>', unsafe_allow_html=True)
    region_filter = st.multiselect("Region", sorted(df['region'].unique()), placeholder="All regions")
    type_filter   = st.multiselect("Avocado Type", df['type'].unique(), placeholder="All types")
    year_filter   = st.multiselect("Year", sorted(df['year'].unique()),
                                   default=sorted(df['year'].unique())[-2:])
    
    min_p, max_p = float(df['AveragePrice'].min()), float(df['AveragePrice'].max())
    price_filter = st.slider("Price Range (USD)", min_p, max_p, (min_p, max_p), format="$%.2f")

    st.markdown('<div class="sidebar-label">Report Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px; color:{C_DARK_GREY}; line-height:1.8;">
        <b>Author</b><br>Lê Quý Phát<br>Data Scientist &amp; Analyst<br><br>
        <b>Standard</b><br>IBCS · International Business<br>Communication Standards<br><br>
        <span style="color:{C_MID_GREY};">© 2026 lequyphat</span>
    </div>
    """, unsafe_allow_html=True)

# ── FILTER ────────────────────────────────────────────────────────────────────
fdf = df.copy()
if region_filter: fdf = fdf[fdf['region'].isin(region_filter)]
if type_filter:   fdf = fdf[fdf['type'].isin(type_filter)]
if year_filter:   fdf = fdf[fdf['year'].isin(year_filter)]
fdf = fdf[(fdf['AveragePrice'] >= price_filter[0]) & (fdf['AveragePrice'] <= price_filter[1])]

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
years_str = f"{min(year_filter)}–{max(year_filter)}" if year_filter else "All Years"
st.markdown(f"""
<div class="report-header">
    <div>
        <div class="report-title">US Avocado Market — Management Report</div>
        <div class="report-subtitle">
            Price sensitivity · Consumer segmentation · Predictive pricing · {years_str}
        </div>
    </div>
    <div class="report-meta">
        IBCS-compliant | {len(fdf):,} observations<br>
        Regions: {fdf['region'].nunique()} &nbsp;|&nbsp; 
        Types: {', '.join(fdf['type'].unique()) if len(fdf) else '—'}
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "A  ·  Market Overview",
    "B  ·  Price Sensitivity",
    "C  ·  Segmentation & NLP",
    "D  ·  Predictive ML"
])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB A — MARKET OVERVIEW                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-h">A · Market Health — Key Performance Indicators</div>', unsafe_allow_html=True)

    # ── KPI block ──────────────────────────────────────────────────────────────
    tot_vol      = fdf['TotalVolume'].sum()
    avg_price    = fdf['AveragePrice'].mean()
    bag_ratio    = (fdf['TotalBags'].sum() / fdf['TotalVolume'].sum() * 100) if len(fdf) > 0 else 0
    organic_pct  = len(fdf[fdf['type']=='organic']) / len(fdf) * 100 if len(fdf) > 0 else 0
    n_markets    = fdf['region'].nunique()

    # Simple YoY delta (last year vs previous)
    if year_filter and len(year_filter) >= 2:
        y1 = sorted(year_filter)[-1]; y0 = sorted(year_filter)[-2]
        p1 = fdf[fdf['year']==y1]['AveragePrice'].mean()
        p0 = fdf[fdf['year']==y0]['AveragePrice'].mean()
        price_delta = (p1-p0)/p0*100 if p0 else 0
        delta_str = f"▲ +{price_delta:.1f}%" if price_delta >= 0 else f"▼ {price_delta:.1f}%"
        delta_cls  = "pos" if price_delta >= 0 else "neg"
        delta_label = f"vs {y0}"
    else:
        delta_str, delta_cls, delta_label = "—", "pos", ""

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-cell">
            <div class="kpi-label">Total Volume Sold</div>
            <div class="kpi-value">{tot_vol/1e6:.1f}M</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">units</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Avg. Retail Price (ADR)</div>
            <div class="kpi-value">${avg_price:.2f}</div>
            <div class="kpi-delta {delta_cls}">{delta_str} <span style="color:{C_MID_GREY};">{delta_label}</span></div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Bagged Sales Ratio</div>
            <div class="kpi-value">{bag_ratio:.1f}%</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">of total volume</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Organic Share</div>
            <div class="kpi-value">{organic_pct:.1f}%</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">across {n_markets} markets</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="card"><div class="card-title">Average Price Over Time — Conventional vs Organic</div><div class="card-unit">USD per unit · weekly data</div>', unsafe_allow_html=True)
        df_trend = fdf.groupby(['Date', 'type'])['AveragePrice'].mean().reset_index()
        
        # IBCS: actual = black solid, organic = dark grey dashed — clear without color
        fig_line = go.Figure()
        for typ, color, dash in [('conventional', C_BLACK, 'solid'), ('organic', C_MID_GREY, 'dot')]:
            d = df_trend[df_trend['type']==typ]
            fig_line.add_trace(go.Scatter(
                x=d['Date'], y=d['AveragePrice'], name=typ.capitalize(),
                mode='lines', line=dict(color=color, width=2, dash=dash)
            ))
        apply_ibcs_layout(fig_line, height=340)
        fig_line.update_yaxes(tickprefix="$", tickformat=".2f")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('<div class="footnote">IBCS: solid line = current actual · dotted line = comparative segment</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">Volume by Avocado Size</div><div class="card-unit">% share of total units sold</div>', unsafe_allow_html=True)
        size_data = pd.DataFrame({
            'Size': ['Small/Medium\n(PLU 4046)', 'Large\n(PLU 4225)', 'X-Large\n(PLU 4770)'],
            'Volume': [fdf['plu4046'].sum(), fdf['plu4225'].sum(), fdf['plu4770'].sum()]
        })
        size_data['Pct'] = size_data['Volume'] / size_data['Volume'].sum() * 100

        # IBCS: Use horizontal bar instead of pie — values readable
        fig_size = go.Figure(go.Bar(
            x=size_data['Pct'], y=size_data['Size'], orientation='h',
            marker_color=[C_BLACK, C_DARK_GREY, C_MID_GREY],
            text=[f"{v:.1f}%" for v in size_data['Pct']],
            textposition='outside', textfont=dict(family="DM Mono", size=11)
        ))
        apply_ibcs_layout(fig_size, height=340)
        fig_size.update_xaxes(ticksuffix="%", range=[0, size_data['Pct'].max()*1.2])
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)
        st.markdown('<div class="footnote">Source: avocado-cleaned.csv · PLU = Price Look-Up code</div></div>', unsafe_allow_html=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-h">A2 · Top Markets by Volume — Ranked Bar</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([3, 2])

    exclude_regions = ['Total US','West','South Central','Northeast','Southeast','Midsouth','Great Lakes','Plains']
    city_df = fdf[~fdf['region'].isin(exclude_regions)]
    top_cities = city_df.groupby('region')['TotalVolume'].sum().nlargest(12).reset_index()
    top_cities['Rank'] = range(1, len(top_cities)+1)
    top_cities['Color'] = [C_ACCENT if i == 0 else C_BLACK if i < 3 else C_DARK_GREY if i < 6 else C_MID_GREY
                           for i in range(len(top_cities))]

    with c3:
        st.markdown('<div class="card"><div class="card-title">Top 12 Markets by Total Volume</div><div class="card-unit">Total units sold · ranked descending</div>', unsafe_allow_html=True)
        fig_rank = go.Figure(go.Bar(
            x=top_cities['TotalVolume'], y=top_cities['region'],
            orientation='h', marker_color=top_cities['Color'],
            text=[f"{v/1e6:.1f}M" for v in top_cities['TotalVolume']],
            textposition='outside', textfont=dict(family="DM Mono", size=10)
        ))
        apply_ibcs_layout(fig_rank, height=400)
        fig_rank.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            xaxis=dict(tickformat=".0s"),
            showlegend=False
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        st.markdown('<div class="footnote">▪ Green bar = #1 market · Regional aggregates excluded</div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="card-title">Seasonality — Avg. Price by Month</div><div class="card-unit">USD per unit · all years combined</div>', unsafe_allow_html=True)
        monthly = fdf.groupby(['Month', 'type'])['AveragePrice'].mean().reset_index()
        month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        monthly['MonthName'] = monthly['Month'].map(month_names)

        fig_season = go.Figure()
        for typ, color, dash in [('conventional', C_BLACK, 'solid'), ('organic', C_MID_GREY, 'dot')]:
            d = monthly[monthly['type']==typ].sort_values('Month')
            fig_season.add_trace(go.Scatter(
                x=d['MonthName'], y=d['AveragePrice'], name=typ.capitalize(),
                mode='lines+markers', line=dict(color=color, width=2, dash=dash),
                marker=dict(size=5, color=color)
            ))
        apply_ibcs_layout(fig_season, height=400)
        fig_season.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_season, use_container_width=True)
        st.markdown('<div class="footnote">Seasonality analysis: monthly average across all selected years</div></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB B — PRICE SENSITIVITY                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-h">B · Price Sensitivity Analysis — Elasticity & Distribution</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        st.markdown('<div class="card"><div class="card-title">Price Distribution by Region — Top 15</div><div class="card-unit">USD per unit · box plot (median, IQR, whiskers)</div>', unsafe_allow_html=True)
        top_15 = fdf['region'].value_counts().head(15).index
        df_t15 = fdf[fdf['region'].isin(top_15)]
        
        fig_box = go.Figure()
        for typ, color in [('conventional', C_BLACK), ('organic', C_MID_GREY)]:
            d = df_t15[df_t15['type']==typ]
            fig_box.add_trace(go.Box(
                x=d['AveragePrice'], y=d['region'],
                name=typ.capitalize(), orientation='h',
                marker_color=color, line_color=color,
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                boxpoints=False
            ))
        apply_ibcs_layout(fig_box, height=480)
        fig_box.update_layout(
            boxmode='group',
            yaxis=dict(categoryorder='median ascending'),
            xaxis=dict(tickprefix="$")
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('<div class="footnote">IBCS: box = IQR · line = median · whiskers = 1.5×IQR</div></div>', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="card"><div class="card-title">Price–Volume Elasticity — Demand Curve</div><div class="card-unit">Log-scale volume vs average price · OLS trend</div>', unsafe_allow_html=True)
        q_hi = fdf["TotalVolume"].quantile(0.95)
        df_el = fdf[fdf["TotalVolume"] < q_hi]

        fig_scatter = px.scatter(
            df_el, x="TotalVolume", y="AveragePrice", color="type",
            color_discrete_map={'conventional': C_BLACK, 'organic': C_MID_GREY},
            trendline="ols", opacity=0.18, log_x=True
        )
        # Make trendlines more prominent
        for trace in fig_scatter.data:
            if hasattr(trace, 'mode') and trace.mode == 'lines':
                trace.line.width = 2.5
        apply_ibcs_layout(fig_scatter, height=480)
        fig_scatter.update_yaxes(tickprefix="$")
        fig_scatter.update_xaxes(title_text="Total Volume (log scale)", tickformat=".0s")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('<div class="footnote">OLS regression lines indicate demand elasticity · top 5% volume outliers excluded</div></div>', unsafe_allow_html=True)

    # Price premium table
    st.markdown('<div class="section-h">B2 · Organic Premium Analysis — by Region</div>', unsafe_allow_html=True)
    premium_df = fdf.groupby(['region','type'])['AveragePrice'].mean().unstack(fill_value=np.nan)
    if 'organic' in premium_df.columns and 'conventional' in premium_df.columns:
        premium_df['Premium ($)']  = premium_df['organic'] - premium_df['conventional']
        premium_df['Premium (%)']  = (premium_df['Premium ($)'] / premium_df['conventional'] * 100)
        premium_df = premium_df.dropna().sort_values('Premium (%)', ascending=False).head(15)
        
        max_prem = premium_df['Premium (%)'].max()
        rows = ""
        for region, row in premium_df.iterrows():
            bar_w = int(row['Premium (%)']/max_prem*80) if max_prem > 0 else 0
            rows += f"""
            <tr>
                <td>{region}</td>
                <td style="font-family:'DM Mono',monospace;">${row['conventional']:.2f}</td>
                <td style="font-family:'DM Mono',monospace;">${row['organic']:.2f}</td>
                <td style="font-family:'DM Mono',monospace;">${row['Premium ($)']:.2f}</td>
                <td style="font-family:'DM Mono',monospace;">
                    {row['Premium (%)']:.1f}%
                    <span class="var-bar-pos" style="width:{bar_w}px;"></span>
                </td>
            </tr>
            """
        st.markdown(f"""
        <table class="ibcs-table">
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Conv. Price</th>
                    <th>Org. Price</th>
                    <th>Premium $</th>
                    <th>Premium % ▼</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        <div class="footnote" style="margin-top:8px;">Top 15 regions by organic price premium · ▪ bar = relative premium magnitude</div>
        """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB C — SEGMENTATION & NLP                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-h">C · Market Segmentation — K-Means Clustering</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    if len(fdf) > 10:
        df_cluster = fdf.copy()
        df_cluster['Cluster'] = run_kmeans(df_cluster, n_clusters=4).astype(str)

        # IBCS cluster colors — pattern/symbol based, not rainbow
        cluster_colors = {'0': C_BLACK, '1': C_DARK_GREY, '2': C_MID_GREY, '3': C_ACCENT}

        with col1:
            st.markdown('<div class="card"><div class="card-title">Consumer Segment Map — Bubble Chart</div><div class="card-unit">X = Volume (log) · Y = Avg Price · Size = Bag Ratio</div>', unsafe_allow_html=True)
            fig_clust = px.scatter(
                df_cluster, x="TotalVolume", y="AveragePrice",
                color="Cluster", size="Bags_Ratio",
                hover_data=['region','type'],
                log_x=True,
                color_discrete_map=cluster_colors,
                opacity=0.55
            )
            apply_ibcs_layout(fig_clust, height=460)
            fig_clust.update_yaxes(tickprefix="$")
            fig_clust.update_xaxes(title_text="Total Volume (log)", tickformat=".0s")
            # Relabel clusters
            for trace in fig_clust.data:
                trace.name = f"Segment {trace.name}"
            st.plotly_chart(fig_clust, use_container_width=True)
            st.markdown('<div class="footnote">K-Means · k=4 · StandardScaler on Price, Volume, Bag Ratio · bubble size = bagged sales share</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><div class="card-title">Segment Profile</div><div class="card-unit">Centroid averages</div>', unsafe_allow_html=True)
            cluster_stats = df_cluster.groupby('Cluster')[['AveragePrice','TotalVolume','Bags_Ratio']].mean().reset_index()
            rows = ""
            for _, r in cluster_stats.iterrows():
                seg = int(r['Cluster'])
                rows += f"""
                <tr>
                    <td><span style="display:inline-block;width:10px;height:10px;background:{list(cluster_colors.values())[seg]};margin-right:6px;"></span>S{seg}</td>
                    <td>${r['AveragePrice']:.2f}</td>
                    <td>{r['TotalVolume']/1e3:.0f}K</td>
                    <td>{r['Bags_Ratio']:.1%}</td>
                </tr>"""
            st.markdown(f"""
            <table class="ibcs-table">
                <thead><tr><th>Seg</th><th>Price</th><th>Vol</th><th>Bag%</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Insufficient data — need at least 10 observations for K-Means.")

    # NLP section
    st.markdown('<div class="section-h">C2 · Region Name NLP — Keyword Frequency Analysis</div>', unsafe_allow_html=True)
    k1, k2 = st.columns(2)

    def keyword_bar(title, unit, text_series, color):
        st.markdown(f'<div class="card"><div class="card-title">{title}</div><div class="card-unit">{unit}</div>', unsafe_allow_html=True)
        if not text_series.empty:
            kw = pd.DataFrame(get_keywords(text_series), columns=['Keyword','Count'])
            fig = go.Figure(go.Bar(
                x=kw['Count'], y=kw['Keyword'], orientation='h',
                marker_color=color,
                text=kw['Count'], textposition='outside',
                textfont=dict(family="DM Mono", size=10)
            ))
            apply_ibcs_layout(fig, height=360)
            fig.update_layout(yaxis=dict(categoryorder='total ascending'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="footnote">NLP: word frequency after stopword removal · top 15 terms</div></div>', unsafe_allow_html=True)

    with k1:
        premium_regions = fdf[fdf['AveragePrice'] > fdf['AveragePrice'].quantile(0.80)]['region']
        keyword_bar("Keywords — Premium Markets (top 20% price)", "Regions where avg price > 80th percentile", premium_regions, C_BLACK)

    with k2:
        budget_regions = fdf[fdf['AveragePrice'] < fdf['AveragePrice'].quantile(0.20)]['region']
        keyword_bar("Keywords — Budget Markets (bottom 20% price)", "Regions where avg price < 20th percentile", budget_regions, C_DARK_GREY)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB D — PREDICTIVE ML                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="section-h">D · Pricing Predictive Model — Random Forest Regressor</div>', unsafe_allow_html=True)

    model, le_type, le_region, le_season, features, metrics, comparison_df = train_model_and_evaluate(df)

    # Model scorecard
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="kpi-cell">
            <div class="kpi-label">MAE — Mean Absolute Error</div>
            <div class="kpi-value" style="font-size:22px;">${metrics['MAE']:.3f}</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">average prediction error</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">RMSE — Root Mean Sq. Error</div>
            <div class="kpi-value" style="font-size:22px;">${metrics['RMSE']:.3f}</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">penalizes large errors</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">R² — Coefficient of Determination</div>
            <div class="kpi-value" style="font-size:22px; color:{'#2D6A2D' if metrics['R2']>0.85 else C_BLACK};">{metrics['R2']:.2%}</div>
            <div class="kpi-delta {'pos' if metrics['R2']>0.85 else ''}" style="{'color:#2D6A2D' if metrics['R2']>0.85 else ''}">
                {"▲ Strong fit" if metrics['R2']>0.85 else "Model fit"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown('<div class="card"><div class="card-title">Actual vs. Predicted — Goodness of Fit</div><div class="card-unit">USD · diagonal = perfect prediction · test set sample</div>', unsafe_allow_html=True)
        sample = comparison_df.sample(n=min(2000, len(comparison_df)), random_state=42)
        
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(
            x=sample['Actual'], y=sample['Predicted'],
            mode='markers', marker=dict(color=C_BLACK, size=4, opacity=0.2)
        ))
        max_v = max(sample.max())
        fig_diag.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                           line=dict(color=C_ACCENT, width=1.5, dash="dash"))
        apply_ibcs_layout(fig_diag, height=380)
        fig_diag.update_xaxes(title_text="Actual Price", tickprefix="$")
        fig_diag.update_yaxes(title_text="Predicted Price", tickprefix="$")
        st.plotly_chart(fig_diag, use_container_width=True)
        st.markdown('<div class="footnote">Green dashed line = perfect prediction baseline (y=x) · 2,000 point sample</div></div>', unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="card"><div class="card-title">Feature Importance — Variable Contribution</div><div class="card-unit">Normalized importance score · Random Forest (100 trees)</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
        
        colors = [C_ACCENT if i == len(imp_df)-1 else C_BLACK if i >= len(imp_df)-3 else C_DARK_GREY if i >= len(imp_df)-6 else C_MID_GREY
                  for i in range(len(imp_df))]
        
        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in imp_df['Importance']],
            textposition='outside', textfont=dict(family="DM Mono", size=9)
        ))
        apply_ibcs_layout(fig_imp, height=380)
        fig_imp.update_layout(showlegend=False)
        fig_imp.update_xaxes(range=[0, imp_df['Importance'].max()*1.25])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('<div class="footnote">▪ Green = highest importance feature · Darker = more influential</div></div>', unsafe_allow_html=True)

    # Simulator
    st.markdown('<div class="section-h">D2 · Pricing Simulator — Live Scenario Analysis</div>', unsafe_allow_html=True)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_region = st.selectbox("Target Market (Region)", le_region.classes_)
            inp_type   = st.selectbox("Production Type", le_type.classes_)
            inp_year   = st.selectbox("Strategy Year", sorted(df['year'].unique()), index=len(df['year'].unique())-1)
        with c2:
            inp_vol        = st.number_input("Expected Volume (units)", min_value=100, max_value=50_000_000, value=100_000, step=10_000)
            inp_bags_ratio = st.slider("Bagged Sales Ratio (%)", 0.0, 100.0, 30.0, 0.5) / 100.0
            inp_month      = st.slider("Target Month", 1, 12, 6)
        with c3:
            st.caption("SIZE MIX (% of retail volume)")
            inp_s_ratio  = st.slider("Small/Medium — PLU 4046 (%)", 0.0, 100.0, 40.0, 1.0) / 100.0
            inp_l_ratio  = st.slider("Large — PLU 4225 (%)", 0.0, 100.0, 30.0, 1.0) / 100.0
            inp_xl_ratio = st.slider("X-Large — PLU 4770 (%)", 0.0, 100.0, 0.0, 1.0) / 100.0

        submitted = st.form_submit_button("Run Pricing Model")

        if submitted:
            inp_season = extract_season(inp_month)
            input_data = pd.DataFrame({
                'TotalVolume':       [inp_vol],
                'Bags_Ratio':        [inp_bags_ratio],
                'Small_Medium_Ratio':[inp_s_ratio],
                'Large_Ratio':       [inp_l_ratio],
                'XLarge_Ratio':      [inp_xl_ratio],
                'year':              [inp_year],
                'Month':             [inp_month],
                'type_encoded':      [le_type.transform([inp_type])[0]],
                'region_encoded':    [le_region.transform([inp_region])[0]],
                'season_encoded':    [le_season.transform([inp_season])[0]]
            })[features]

            pred = model.predict(input_data)[0]
            season_name = inp_season
            month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                         7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Recommended Retail Price (ADR) — Model Output</div>
                <div class="pred-value">${pred:.2f}</div>
                <div class="pred-context">
                    {inp_type.capitalize()} · {inp_region} · {month_map[inp_month]} {inp_year} ({season_name}) · 
                    Volume: {inp_vol:,} units · Bag ratio: {inp_bags_ratio:.0%}
                </div>
            </div>
            <div class="footnote" style="margin-top:8px;">
                Random Forest prediction · MAE ±${metrics['MAE']:.3f} · R² = {metrics['R2']:.2%} · 
                Result is a statistical recommendation, not a guarantee.
            </div>
            """, unsafe_allow_html=True)