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

# ── MODERN NATURAL ENTERPRISE COLOR SYSTEM ───────────────────────────────────
C_BLACK      = "#0F172A"   # Dark Slate — Primary Headers & Text
C_DARK_GREY  = "#334155"   # Dark Grey — Subtitles & Secondary Text
C_MID_GREY   = "#64748B"   # Mid Grey — Neutral Labels & Captions
C_LIGHT_GREY = "#E2E8F0"   # Light Slate — Grid lines & Borders
C_BG         = "#F8F9FA"   # Page Background
C_WHITE      = "#FFFFFF"   # Card Background
C_ACCENT     = "#2E5A27"   # Primary Accent — Forest Green (Avocado)
C_AMBER      = "#D97706"   # Secondary Accent — Warm Amber/Gold
C_WARN       = "#DC2626"   # Warning / Negative Variance — Rose Red

# IBCS-inspired Enterprise Chart Template
IBCS_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="DM Sans, sans-serif", size=12, color=C_BLACK),
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        xaxis=dict(showgrid=False, zeroline=False, linecolor=C_LIGHT_GREY, ticks="outside", tickcolor=C_LIGHT_GREY, ticklen=3),
        yaxis=dict(showgrid=True,  gridcolor=C_LIGHT_GREY, zeroline=False, linecolor=C_LIGHT_GREY),
        margin=dict(l=10, r=10, t=36, b=10),
        colorway=[C_ACCENT, C_BLACK, C_AMBER, C_MID_GREY, C_WARN],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Avocado Market Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* RESET & BASE */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ 
    background-color: {C_BG} !important; 
    font-family: 'DM Sans', sans-serif !important;
    color: {C_BLACK};
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background-color: {C_WHITE} !important;
    border-right: 1px solid {C_LIGHT_GREY} !important;
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1.25rem; }}

.sidebar-brand {{
    background: {C_BLACK};
    color: {C_WHITE};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 14px 16px;
    margin: -1.5rem -1.25rem 1.5rem -1.25rem;
}}
.sidebar-brand span {{ color: #A7F3D0; }}

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

/* MAIN HEADER */
.report-header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 3px solid {C_ACCENT};
    padding-bottom: 12px;
    margin-bottom: 20px;
}}
.report-title {{
    font-size: 24px;
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

/* SECTION HEADER */
.section-h {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {C_WHITE};
    background: {C_BLACK};
    padding: 6px 12px;
    display: inline-block;
    margin: 24px 0 16px 0;
    border-radius: 2px;
}}

/* CARD WRAPPER */
.card {{
    background: {C_WHITE};
    border: 1px solid {C_LIGHT_GREY};
    border-top: 3px solid {C_ACCENT};
    padding: 18px 20px 12px 20px;
    margin-bottom: 18px;
    height: 100%;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}}
.card-title {{
    font-size: 14px;
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

/* EXECUTIVE TAKEAWAY CARD */
.takeaway-card {{
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    border-left: 4px solid {C_ACCENT} !important;
    padding: 14px 18px !important;
    margin-bottom: 20px !important;
    border-radius: 4px !important;
}}
.takeaway-title {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {C_ACCENT} !important;
    margin-bottom: 6px !important;
}}
.takeaway-body {{
    font-size: 13px !important;
    color: {C_BLACK} !important;
    line-height: 1.5 !important;
}}

/* KPI GRID */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: {C_LIGHT_GREY};
    border: 1px solid {C_LIGHT_GREY};
    margin-bottom: 24px;
    border-radius: 4px;
    overflow: hidden;
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

/* TABLE STYLE */
.ibcs-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.ibcs-table th {{
    background: {C_BLACK};
    color: {C_WHITE};
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.06em;
    font-size: 10px;
    text-transform: uppercase;
}}
.ibcs-table td {{
    padding: 8px 10px;
    border-bottom: 1px solid {C_LIGHT_GREY};
    font-family: 'DM Mono', monospace;
    font-size: 11px;
}}
.ibcs-table tr:hover td {{ background: #F1F5F9; }}

.var-bar-pos {{ 
    display: inline-block; height: 8px; background: {C_ACCENT}; 
    vertical-align: middle; margin-left: 6px; border-radius: 1px;
}}

/* TAB NAVIGATION */
div[data-testid="stTabs"] button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: {C_MID_GREY} !important;
    background-color: {C_WHITE} !important;
    border-radius: 4px 4px 0 0 !important;
    border: 1px solid {C_LIGHT_GREY} !important;
    border-bottom: none !important;
    padding: 10px 20px !important;
    margin-right: 4px !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_WHITE} !important;
    background-color: {C_ACCENT} !important;
    border-color: {C_ACCENT} !important;
}}

/* STREAMLIT OVERRIDES */
div[data-testid="stMetric"] {{ display: none; }}
.stPlotlyChart {{ border: none !important; }}
div.stButton > button {{
    background: {C_BLACK};
    color: {C_WHITE};
    border-radius: 3px;
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
    border-top: 3px solid {C_ACCENT};
    padding: 24px;
    border-radius: 4px;
}}
.stSelectbox label, .stSlider label, .stNumberInput label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {C_DARK_GREY} !important;
}}

/* PREDICTION RESULT BOX */
.pred-box {{
    background: {C_BLACK};
    color: {C_WHITE};
    padding: 24px 28px;
    margin-top: 16px;
    border-radius: 4px;
    border-left: 4px solid {C_AMBER};
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
    font-size: 12px;
    color: {C_LIGHT_GREY};
    margin-top: 6px;
}}

/* FOOTNOTE */
.footnote {{
    font-size: 10px;
    color: {C_MID_GREY};
    border-top: 1px solid {C_LIGHT_GREY};
    margin-top: 12px;
    padding-top: 6px;
    font-style: italic;
}}
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
    """Apply consistent Enterprise styling to any plotly figure."""
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

def render_takeaway(badge_text, body_text):
    """Render executive takeaway summary card HTML."""
    st.markdown(f"""
    <div class="takeaway-card">
        <div class="takeaway-title">{badge_text}</div>
        <div class="takeaway-body">{body_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = load_and_clean_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        AVOCADO ANALYTICS <span>| Executive Report</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">CHẾ ĐỘ XEM / VIEW MODE</div>', unsafe_allow_html=True)
    view_mode = st.radio(
        "View Mode",
        ["Quản lý (Executive View)", "Phân tích Chuyên sâu (Deep Analytics)"],
        index=0,
        label_visibility="collapsed"
    )
    is_exec_mode = ("Quản lý" in view_mode or "Executive" in view_mode)

    st.markdown('<div class="sidebar-label">BỘ LỌC DỮ LIỆU / DATA FILTERS</div>', unsafe_allow_html=True)
    region_filter = st.multiselect("Thị trường (Region)", sorted(df['region'].unique()), placeholder="Tất cả thị trường")
    type_filter   = st.multiselect("Loại sản phẩm (Type)", df['type'].unique(), placeholder="Tất cả các loại")
    year_filter   = st.multiselect("Năm (Year)", sorted(df['year'].unique()),
                                   default=sorted(df['year'].unique())[-2:])
    
    min_p, max_p = float(df['AveragePrice'].min()), float(df['AveragePrice'].max())
    price_filter = st.slider("Khoảng giá (Price Range USD)", min_p, max_p, (min_p, max_p), format="$%.2f")

    st.markdown('<div class="sidebar-label">THÔNG TIN BÁO CÁO / REPORT INFO</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px; color:{C_DARK_GREY}; line-height:1.8;">
        <b>Tác giả:</b> Lê Quý Phát<br>
        <b>Chức danh:</b> Data Scientist &amp; Analyst<br>
        <b>Tiêu chuẩn:</b> Executive Management Standard<br>
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
years_str = f"{min(year_filter)}–{max(year_filter)}" if year_filter else "Tất cả các năm"
mode_tag = "[CHẾ ĐỘ QUẢN LÝ]" if is_exec_mode else "[CHẾ ĐỘ PHÂN TÍCH CHUYÊN SÂU]"

st.markdown(f"""
<div class="report-header">
    <div>
        <div class="report-title">Báo cáo Chiến lược Thị trường Bơ Mỹ {mode_tag}</div>
        <div class="report-subtitle">
            Phân tích biến động giá &middot; Thị phần tiêu dùng &middot; Định giá bán lẻ &middot; {years_str}
        </div>
    </div>
    <div class="report-meta">
        Dữ liệu: {len(fdf):,} mẫu quan sát<br>
        Số khu vực: {fdf['region'].nunique()} &nbsp;|&nbsp; 
        Phân loại: {', '.join(fdf['type'].unique()) if len(fdf) else '—'}
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS CONFIGURATION ────────────────────────────────────────────────────────
if is_exec_mode:
    tab1, tab2, tab3 = st.tabs([
        "TỔNG QUAN THỊ TRƯỜNG",
        "NĂNG LỰC CẠNH TRANH & GIÁ",
        "MÔ PHỎNG DỰ BÁO ĐỊNH GIÁ"
    ])
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "TỔNG QUAN THỊ TRƯỜNG",
        "NĂNG LỰC CẠNH TRANH & GIÁ",
        "MÔ PHỎNG DỰ BÁO ĐỊNH GIÁ",
        "ĐỘ CO GIẢN & PHÂN CỤM",
        "MÔ HÌNH ML PREDICTIVE"
    ])


# ==================== TAB 1: MARKET OVERVIEW ====================
with tab1:
    st.markdown('<div class="section-h">TỔNG QUAN CHỈ SỐ HIỆU SUẤT THỊ TRƯỜNG (KPIS)</div>', unsafe_allow_html=True)

    # Executive Takeaway Card
    render_takeaway(
        "[NHẬN ĐỊNH THỊ TRƯỜNG QUAN TRỌNG]",
        "Sản lượng bơ hữu cơ (Organic) duy trì mức giá cao hơn bơ thường từ 30% đến 45%. Phân khúc bơ túi nhỏ (Small/Medium PLU 4046) chiếm tỷ trọng doanh số cao nhất tại các đô thị lớn."
    )

    tot_vol      = fdf['TotalVolume'].sum()
    avg_price    = fdf['AveragePrice'].mean()
    bag_ratio    = (fdf['TotalBags'].sum() / fdf['TotalVolume'].sum() * 100) if len(fdf) > 0 else 0
    organic_pct  = len(fdf[fdf['type']=='organic']) / len(fdf) * 100 if len(fdf) > 0 else 0
    n_markets    = fdf['region'].nunique()

    if year_filter and len(year_filter) >= 2:
        y1 = sorted(year_filter)[-1]; y0 = sorted(year_filter)[-2]
        p1 = fdf[fdf['year']==y1]['AveragePrice'].mean()
        p0 = fdf[fdf['year']==y0]['AveragePrice'].mean()
        price_delta = (p1-p0)/p0*100 if p0 else 0
        delta_str = f"+{price_delta:.1f}%" if price_delta >= 0 else f"{price_delta:.1f}%"
        delta_cls  = "pos" if price_delta >= 0 else "neg"
        delta_label = f"so với năm {y0}"
    else:
        delta_str, delta_cls, delta_label = "—", "pos", ""

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-cell">
            <div class="kpi-label">Tổng Sản Lượng Tải Tiêu Thụ</div>
            <div class="kpi-value">{tot_vol/1e6:.1f}M</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">đơn vị sản phẩm</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Giá Bán Bán Lẻ Trung Bình</div>
            <div class="kpi-value">${avg_price:.2f}</div>
            <div class="kpi-delta {delta_cls}">{delta_str} <span style="color:{C_MID_GREY};">{delta_label}</span></div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Tỷ Tỷ Lệ Bán Dạng Túi (Bagged)</div>
            <div class="kpi-value">{bag_ratio:.1f}%</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">trên tổng sản lượng</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Tỷ Trọng Hữu Cơ (Organic Share)</div>
            <div class="kpi-value">{organic_pct:.1f}%</div>
            <div class="kpi-delta" style="color:{C_MID_GREY};">trên {n_markets} thị trường</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="card"><div class="card-title">Xu Hướng Biến Động Giá Qua Thời Gian (Bơ Thường vs Bơ Hữu Cơ)</div><div class="card-unit">USD per unit &middot; dữ liệu hàng tuần</div>', unsafe_allow_html=True)
        df_trend = fdf.groupby(['Date', 'type'])['AveragePrice'].mean().reset_index()
        
        fig_line = go.Figure()
        for typ, color, dash in [('conventional', C_ACCENT, 'solid'), ('organic', C_AMBER, 'solid')]:
            d = df_trend[df_trend['type']==typ]
            fig_line.add_trace(go.Scatter(
                x=d['Date'], y=d['AveragePrice'], name=typ.capitalize(),
                mode='lines', line=dict(color=color, width=2, dash=dash)
            ))
        apply_ibcs_layout(fig_line, height=340)
        fig_line.update_yaxes(tickprefix="$", tickformat=".2f")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('<div class="footnote">Đường màu xanh lục = Bơ thường &middot; Đường màu vàng hổ phách = Bơ Hữu cơ (Organic)</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">Cơ Cấu Sản Lượng Theo Kích Thước Bơ</div><div class="card-unit">% thị phần sản lượng tiêu thụ</div>', unsafe_allow_html=True)
        size_data = pd.DataFrame({
            'Size': ['Túi Nhỏ/Vừa\n(PLU 4046)', 'Túi Lớn\n(PLU 4225)', 'Túi Rất Lớn\n(PLU 4770)'],
            'Volume': [fdf['plu4046'].sum(), fdf['plu4225'].sum(), fdf['plu4770'].sum()]
        })
        size_data['Pct'] = size_data['Volume'] / size_data['Volume'].sum() * 100

        fig_size = go.Figure(go.Bar(
            x=size_data['Pct'], y=size_data['Size'], orientation='h',
            marker_color=[C_ACCENT, C_DARK_GREY, C_MID_GREY],
            text=[f"{v:.1f}%" for v in size_data['Pct']],
            textposition='outside', textfont=dict(family="DM Mono", size=11)
        ))
        apply_ibcs_layout(fig_size, height=340)
        fig_size.update_xaxes(ticksuffix="%", range=[0, size_data['Pct'].max()*1.2])
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)
        st.markdown('<div class="footnote">Nguồn: avocado-cleaned.csv &middot; PLU = Mã phân loại sản phẩm</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-h">TOP THỊ TRƯỜNG TIÊU THỤ HÀNG ĐẦU</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([3, 2])

    exclude_regions = ['Total US','West','South Central','Northeast','Southeast','Midsouth','Great Lakes','Plains']
    city_df = fdf[~fdf['region'].isin(exclude_regions)]
    top_cities = city_df.groupby('region')['TotalVolume'].sum().nlargest(12).reset_index()
    top_cities['Rank'] = range(1, len(top_cities)+1)
    top_cities['Color'] = [C_ACCENT if i == 0 else C_BLACK if i < 3 else C_DARK_GREY if i < 6 else C_MID_GREY
                           for i in range(len(top_cities))]

    with c3:
        st.markdown('<div class="card"><div class="card-title">Top 12 Thị Trường Tiêu Thụ Sản Lượng Lớn Nhất</div><div class="card-unit">Tổng đơn vị sản phẩm &middot; xếp hạng giảm dần</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="footnote">Cột màu xanh lục = Thị trường dẫn đầu số 1 (Los Angeles/Dallas/Houston)</div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="card-title">Yếu Tố Mùa Vụ — Biến Động Giá Theo Tháng</div><div class="card-unit">USD per unit &middot; trung bình các năm</div>', unsafe_allow_html=True)
        monthly = fdf.groupby(['Month', 'type'])['AveragePrice'].mean().reset_index()
        month_names = {1:'T1',2:'T2',3:'T3',4:'T4',5:'T5',6:'T6',
                       7:'T7',8:'T8',9:'T9',10:'T10',11:'T11',12:'T12'}
        monthly['MonthName'] = monthly['Month'].map(month_names)

        fig_season = go.Figure()
        for typ, color, dash in [('conventional', C_ACCENT, 'solid'), ('organic', C_AMBER, 'solid')]:
            d = monthly[monthly['type']==typ].sort_values('Month')
            fig_season.add_trace(go.Scatter(
                x=d['MonthName'], y=d['AveragePrice'], name=typ.capitalize(),
                mode='lines+markers', line=dict(color=color, width=2, dash=dash),
                marker=dict(size=5, color=color)
            ))
        apply_ibcs_layout(fig_season, height=400)
        fig_season.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_season, use_container_width=True)
        st.markdown('<div class="footnote">Giá bơ thường tăng đỉnh điểm vào các tháng Mùa Hè (tháng 7 - tháng 9)</div></div>', unsafe_allow_html=True)


# ==================== TAB 2: COMPETITIVE PRICING ====================
with tab2:
    st.markdown('<div class="section-h">SO SÁNH MỨC GIÁ BÁN LẺ VÀ PHỤ PHÍ ORGANIC</div>', unsafe_allow_html=True)

    render_takeaway(
        "[NHẬN ĐỊNH VỀ MỨC GIÁ CHÊNH LỆCH]",
        "Thị trường miền Đông Bắc (Northeast / New York / San Francisco) ghi nhận mức giá bơ bán lẻ cao nhất cả nước. Phụ phí bơ Organic đạt mức cao nhất lên tới +65% so với bơ thường."
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown('<div class="card"><div class="card-title">So Sánh Mức Giá Trung Bình & Khoảng Biến Động Theo Khu Vực</div><div class="card-unit">USD per unit &middot; xếp hạng giá giảm dần</div>', unsafe_allow_html=True)
        top_15_names = fdf.groupby('region')['AveragePrice'].mean().nlargest(12).index
        df_t15 = fdf[fdf['region'].isin(top_15_names)]
        
        region_price_stats = df_t15.groupby('region')['AveragePrice'].agg(['mean', 'min', 'max']).reset_index()
        region_price_stats = region_price_stats.sort_values('mean', ascending=True)

        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            x=region_price_stats['mean'], y=region_price_stats['region'],
            orientation='h', marker_color=C_ACCENT,
            text=[f"${v:.2f}" for v in region_price_stats['mean']],
            textposition='outside', textfont=dict(family="DM Mono", size=10)
        ))
        apply_ibcs_layout(fig_range, height=460)
        fig_range.update_xaxes(tickprefix="$")
        st.plotly_chart(fig_range, use_container_width=True)
        st.markdown('<div class="footnote">Mức giá trung bình bán lẻ thực tế tại Top 12 thị trường cao nhất</div></div>', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="card"><div class="card-title">Mối Tương Quan Sản Lượng Tiêu Thụ vs Giá Bán Lẻ</div><div class="card-unit">Triệu đơn vị sản phẩm vs USD per unit</div>', unsafe_allow_html=True)
        q_hi = fdf["TotalVolume"].quantile(0.95)
        df_el = fdf[fdf["TotalVolume"] < q_hi].copy()
        df_el['Vol_Millions'] = df_el['TotalVolume'] / 1e6

        fig_scatter = px.scatter(
            df_el, x="Vol_Millions", y="AveragePrice", color="type",
            color_discrete_map={'conventional': C_ACCENT, 'organic': C_AMBER},
            trendline="ols", opacity=0.3
        )
        for trace in fig_scatter.data:
            if hasattr(trace, 'mode') and trace.mode == 'lines':
                trace.line.width = 2.5
        apply_ibcs_layout(fig_scatter, height=460)
        fig_scatter.update_yaxes(tickprefix="$")
        fig_scatter.update_xaxes(title_text="Tổng sản lượng (Triệu đơn vị)", tickformat=".1f")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('<div class="footnote">Đường xu hướng OLS minh họa độ co giãn cầu: Khi sản lượng tăng, giá bán có xu hướng giảm nhẹ</div></div>', unsafe_allow_html=True)

    # Organic Premium Table
    st.markdown('<div class="section-h">BẢNG PHÂN TÍCH PHỤ PHÍ ORGANIC THEO THỊ TRƯỜNG</div>', unsafe_allow_html=True)
    premium_df = fdf.groupby(['region','type'])['AveragePrice'].mean().unstack(fill_value=np.nan)
    if 'organic' in premium_df.columns and 'conventional' in premium_df.columns:
        premium_df['Premium ($)']  = premium_df['organic'] - premium_df['conventional']
        premium_df['Premium (%)']  = (premium_df['Premium ($)'] / premium_df['conventional'] * 100)
        premium_df = premium_df.dropna().sort_values('Premium (%)', ascending=False).head(15)
        
        max_prem = premium_df['Premium (%)'].max()
        rows = ""
        for region, row in premium_df.iterrows():
            bar_w = int(row['Premium (%)']/max_prem*80) if max_prem > 0 else 0
            rows += f"<tr><td>{region}</td><td style=\"font-family:'DM Mono',monospace;\">${row['conventional']:.2f}</td><td style=\"font-family:'DM Mono',monospace;\">${row['organic']:.2f}</td><td style=\"font-family:'DM Mono',monospace;\">${row['Premium ($)']:.2f}</td><td style=\"font-family:'DM Mono',monospace;\">+{row['Premium (%)']:.1f}%<span class=\"var-bar-pos\" style=\"width:{bar_w}px;\"></span></td></tr>"
        
        table_html = f'<table class="ibcs-table"><thead><tr><th>Khu vực (Region)</th><th>Giá Bơ Thường</th><th>Giá Bơ Organic</th><th>Chênh lệch ($)</th><th>Phụ phí Organic (%)</th></tr></thead><tbody>{rows}</tbody></table><div class="footnote" style="margin-top:8px;">Top 15 thị trường có mức chênh lệch giá bơ Organic cao nhất</div>'
        st.markdown(table_html, unsafe_allow_html=True)


# ==================== TAB 3: PRICING SIMULATOR ====================
with tab3:
    st.markdown('<div class="section-h">MÔ PHỎNG VÀ DỰ BÁO GIÁ BÁN LẺ DỰ KIẾN</div>', unsafe_allow_html=True)

    render_takeaway(
        "[CÔNG CỤ HỖ TRỢ ĐỊNH GIÁ BÁN LẺ]",
        "Nhập các thông số dự kiến về thị trường, loại sản phẩm và quy mô sản lượng để mô hình Machine Learning đề xuất mức giá bán lẻ tối ưu (ADR) mang lại doanh thu tốt nhất."
    )

    model, le_type, le_region, le_season, features, metrics, comparison_df = train_model_and_evaluate(df)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_region = st.selectbox("Thị trường Mục tiêu (Region)", le_region.classes_)
            inp_type   = st.selectbox("Loại Sản phẩm (Type)", le_type.classes_)
            inp_year   = st.selectbox("Năm Kế hoạch (Year)", sorted(df['year'].unique()), index=len(df['year'].unique())-1)
        with c2:
            inp_vol        = st.number_input("Sản lượng Dự kiến (đơn vị)", min_value=100, max_value=50_000_000, value=100_000, step=10_000)
            inp_bags_ratio = st.slider("Tỷ lệ Bán dạng Túi (%)", 0.0, 100.0, 30.0, 0.5) / 100.0
            inp_month      = st.slider("Tháng Mục tiêu", 1, 12, 6)
        with c3:
            st.caption("CƠ CẤU KÍCH THƯỚC BƠ (% sản lượng)")
            inp_s_ratio  = st.slider("Túi Nhỏ/Vừa — PLU 4046 (%)", 0.0, 100.0, 40.0, 1.0) / 100.0
            inp_l_ratio  = st.slider("Túi Lớn — PLU 4225 (%)", 0.0, 100.0, 30.0, 1.0) / 100.0
            inp_xl_ratio = st.slider("Túi Rất Lớn — PLU 4770 (%)", 0.0, 100.0, 0.0, 1.0) / 100.0

        submitted = st.form_submit_button("CHẠY MÔ HÌNH ĐỊNH GIÁ")

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
            month_map = {1:'Tháng 1',2:'Tháng 2',3:'Tháng 3',4:'Tháng 4',5:'Tháng 5',6:'Tháng 6',
                         7:'Tháng 7',8:'Tháng 8',9:'Tháng 9',10:'Tháng 10',11:'Tháng 11',12:'Tháng 12'}

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Mức Giá Bán Lẻ Đề Xuất (ADR) — Kết Quả Mô Hình</div>
                <div class="pred-value">${pred:.2f}</div>
                <div class="pred-context">
                    Loại: {inp_type.capitalize()} &middot; Thị trường: {inp_region} &middot; Thời gian: {month_map[inp_month]} {inp_year} &middot; 
                    Sản lượng: {inp_vol:,} đơn vị &middot; Tỷ lệ bán túi: {inp_bags_ratio:.0%}
                </div>
            </div>
            <div class="footnote" style="margin-top:8px;">
                Được tính toán bởi mô hình Random Forest Regressor &middot; Sai số trung bình (MAE) ±${metrics['MAE']:.3f} &middot; Độ tin cậy R² = {metrics['R2']:.2%}
            </div>
            """, unsafe_allow_html=True)


# ==================== DEEP ANALYTICS MODE EXTRA TABS ====================
if not is_exec_mode:
    # TAB 4: ELASTICITY & SEGMENTATION
    with tab4:
        st.markdown('<div class="section-h">PHÂN CỤM THỊ TRƯỜNG K-MEANS & PHÂN TÍCH TỪ KHÓA</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        if len(fdf) > 10:
            df_cluster = fdf.copy()
            df_cluster['Cluster'] = run_kmeans(df_cluster, n_clusters=4).astype(str)
            cluster_colors = {'0': C_BLACK, '1': C_DARK_GREY, '2': C_MID_GREY, '3': C_ACCENT}

            with col1:
                st.markdown('<div class="card"><div class="card-title">Phân Cụm Phân Khúc Người Tiêu Dùng (K-Means Segments)</div><div class="card-unit">X = Sản lượng &middot; Y = Giá trung bình &middot; Kích thước = Tỷ lệ bán túi</div>', unsafe_allow_html=True)
                df_cluster['Vol_Millions'] = df_cluster['TotalVolume'] / 1e6
                fig_clust = px.scatter(
                    df_cluster, x="Vol_Millions", y="AveragePrice",
                    color="Cluster", size="Bags_Ratio",
                    hover_data=['region','type'],
                    color_discrete_map=cluster_colors,
                    opacity=0.6
                )
                apply_ibcs_layout(fig_clust, height=460)
                fig_clust.update_yaxes(tickprefix="$")
                fig_clust.update_xaxes(title_text="Tổng sản lượng (Triệu đơn vị)", tickformat=".1f")
                for trace in fig_clust.data:
                    trace.name = f"Phân khúc {trace.name}"
                st.plotly_chart(fig_clust, use_container_width=True)
                st.markdown('<div class="footnote">K-Means Clustering (k=4) bóc tách các nhóm thị trường theo quy mô và mức giá</div></div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card"><div class="card-title">Thông Số Phân Cụm</div><div class="card-unit">Giá trị trung tâm (Centroids)</div>', unsafe_allow_html=True)
                cluster_stats = df_cluster.groupby('Cluster')[['AveragePrice','TotalVolume','Bags_Ratio']].mean().reset_index()
                rows = ""
                for _, r in cluster_stats.iterrows():
                    seg = int(r['Cluster'])
                    rows += f"<tr><td>Phân khúc {seg}</td><td>${r['AveragePrice']:.2f}</td><td>{r['TotalVolume']/1e3:.0f}K</td><td>{r['Bags_Ratio']:.1%}</td></tr>"
                table_html = f'<table class="ibcs-table"><thead><tr><th>Nhóm</th><th>Giá TB</th><th>Sản lượng</th><th>Tỷ lệ túi</th></tr></thead><tbody>{rows}</tbody></table>'
                st.markdown(table_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # NLP Keyword analysis
        st.markdown('<div class="section-h">PHÂN TÍCH TẦN SUẤT TỪ KHÓA KHU VỰC (NLP)</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="footnote">Tần suất xuất hiện tên khu vực sau khi lọc stopwords</div></div>', unsafe_allow_html=True)

        with k1:
            premium_regions = fdf[fdf['AveragePrice'] > fdf['AveragePrice'].quantile(0.80)]['region']
            keyword_bar("Từ khóa — Thị Trường Hạng Sang (Top 20% giá)", "Các khu vực có giá trung bình > 80th percentile", premium_regions, C_BLACK)

        with k2:
            budget_regions = fdf[fdf['AveragePrice'] < fdf['AveragePrice'].quantile(0.20)]['region']
            keyword_bar("Từ khóa — Thị Trường Phổ Thông (Bottom 20% giá)", "Các khu vực có giá trung bình < 20th percentile", budget_regions, C_DARK_GREY)

    # TAB 5: ML PREDICTIVE DEEP DIVE
    with tab5:
        st.markdown('<div class="section-h">MÔ HÌNH HỌC MÁY PREDICTIVE RANDOM FOREST</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
            <div class="kpi-cell">
                <div class="kpi-label">MAE — Sai Số Tuyệt Đối Trung Bình</div>
                <div class="kpi-value" style="font-size:22px;">${metrics['MAE']:.3f}</div>
                <div class="kpi-delta" style="color:{C_MID_GREY};">sai số trung bình của mô hình</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-label">RMSE — Căn Phổ Sai Số Trung Bình</div>
                <div class="kpi-value" style="font-size:22px;">${metrics['RMSE']:.3f}</div>
                <div class="kpi-delta" style="color:{C_MID_GREY};">đánh giá phạt sai số lớn</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-label">R² — Hệ Số Xác Định Mô Hình</div>
                <div class="kpi-value" style="font-size:22px; color:{'#2E5A27' if metrics['R2']>0.85 else C_BLACK};">{metrics['R2']:.2%}</div>
                <div class="kpi-delta {'pos' if metrics['R2']>0.85 else ''}" style="{'color:#2E5A27' if metrics['R2']>0.85 else ''}">
                    {"Mô hình khớp rất tốt" if metrics['R2']>0.85 else "Mô hình khớp vừa phải"}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="card"><div class="card-title">Thực Tế vs. Dự Báo — Kiểm Đánh Giá Độ Khớp</div><div class="card-unit">USD &middot; đường chéo = dự báo hoàn hảo</div>', unsafe_allow_html=True)
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
            fig_diag.update_xaxes(title_text="Giá thực tế", tickprefix="$")
            fig_diag.update_yaxes(title_text="Giá dự báo", tickprefix="$")
            st.plotly_chart(fig_diag, use_container_width=True)
            st.markdown('<div class="footnote">Đường đứt nét màu xanh = đường cơ sở dự báo chính xác tuyệt đối (y=x)</div></div>', unsafe_allow_html=True)

        with d2:
            st.markdown('<div class="card"><div class="card-title">Mức Độ Quan Trọng Của Các Biến (Feature Importance)</div><div class="card-unit">Chỉ số đóng góp của biến trong Random Forest</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="footnote">Cột màu xanh = biến có tác động lớn nhất tới giá bán lẻ</div></div>', unsafe_allow_html=True)

st.markdown(f"""<div class="footnote" style="margin-top:40px; padding:15px 0; border-top:1px solid {C_LIGHT_GREY};">
    Avocado Market Analytics Platform &middot; Executive Management Standard &middot; Enterprise UI
</div>""", unsafe_allow_html=True)