# ABEER BLUESTAR SOCCER FEST 2K25 — Fixed Dashboard
# Resolved CSS injection issue

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests
from datetime import datetime

# ---------------------- FIXED CSS INJECTION -------------------------
def inject_fixed_beautiful_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
      .stApp { 
        font-family: 'Inter', system-ui, -apple-system, sans-serif; 
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #1e293b;
      }
      
      .block-container { 
        padding-top: 1rem; 
        padding-bottom: 2rem; 
        max-width: 95vw; 
        width: 95vw; 
      }
      
      .main-title {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: clamp(28px, 4vw, 48px);
        font-weight: 700;
        margin: 0.5rem 0 2rem 0;
        letter-spacing: -0.02em;
        position: relative;
      }
      
      .main-title::after {
        content: '';
        display: block;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #8b5cf6);
        margin: 1rem auto;
        border-radius: 2px;
        animation: shimmer 2s infinite;
      }
      
      @keyframes shimmer {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
      }
      
      .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        position: relative;
      }
      
      .section-header::before {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6);
      }
      
      [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 25px -5px rgba(0, 0, 0, 0.15);
      }
      
      [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #8b5cf6);
      }
      
      [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
      }
      
      [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
      }
      
      .filter-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
      }
      
      .filter-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #8b5cf6);
      }
      
      .filter-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      
      .stButton > button,
      .stDownloadButton > button {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
      }
      
      .stButton > button:hover,
      .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 12px 25px rgba(14, 165, 233, 0.4);
      }
      
      .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.25rem;
        gap: 0.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
      }
      
      .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
      }
      
      .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #1e293b;
        transform: translateY(-1px);
      }
      
      .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        transform: translateY(-1px);
      }
      
      .stDataFrame {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
      }
      
      .stDataFrame:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
      }
      
      .chart-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
      }
      
      .chart-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 25px -5px rgba(0, 0, 0, 0.15);
      }
      
      .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #8b5cf6);
      }
      
      .info-card {
        background: linear-gradient(135deg, #fef3c7, #fbbf24);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        color: #92400e;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
        position: relative;
        overflow: hidden;
      }
      
      .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #f59e0b, #d97706);
      }
      
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      .stDataFrame,
      [data-testid="metric-container"],
      .chart-container {
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      @media (max-width: 768px) {
        .block-container {
          max-width: 100vw;
          width: 100vw;
          padding-left: 0.5rem;
          padding-right: 0.5rem;
        }
        
        .main-title {
          font-size: clamp(20px, 6vw, 32px);
        }
        
        [data-testid="metric-container"] {
          padding: 1rem;
        }
        
        .filter-panel {
          padding: 1rem;
        }
        
        .chart-container {
          padding: 1rem;
        }
      }
    </style>
    """, unsafe_allow_html=True)

    # Configure Altair theme
    alt.themes.register("beautiful_soccer", lambda: {
        "config": {
            "view": {"stroke": "transparent"},
            "background": "transparent",
            "title": {"font": "Inter", "fontSize": 16, "color": "#1e293b", "anchor": "start"},
            "axis": {
                "labelColor": "#64748b",
                "titleColor": "#1e293b",
                "gridColor": "#e2e8f0",
                "domainColor": "#cbd5e1",
                "tickColor": "#cbd5e1",
                "labelFont": "Inter",
                "titleFont": "Inter"
            },
            "legend": {
                "labelColor": "#64748b",
                "titleColor": "#1e293b",
                "labelFont": "Inter",
                "titleFont": "Inter"
            },
            "range": {
                "category": ["#0ea5e9", "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"],
                "heatmap": ["#f8fafc", "#0ea5e9"]
            }
        }
    })
    alt.themes.enable("beautiful_soccer")

# ========================= XLSX parsing functions ========================
def _parse_xlsx_without_openpyxl(file_bytes: bytes) -> pd.DataFrame:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                root = ET.parse(f).getroot()
                for si in root.findall(".//main:si", ns):
                    text = "".join(t.text or "" for t in si.findall(".//main:t", ns))
                    shared.append(text)
        with z.open("xl/worksheets/sheet1.xml") as f:
            root = ET.parse(f).getroot()
            rows_xml = root.find("main:sheetData", ns)
            rows, max_col_idx = [], 0
            for row in rows_xml.findall("main:row", ns):
                rdict = {}
                for c in row.findall("main:c", ns):
                    ref = c.attrib.get("r", "A1")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    col_idx = 0
                    for ch in col_letters:
                        col_idx = col_idx * 26 + (ord(ch) - 64)
                    col_idx -= 1
                    t = c.attrib.get("t")
                    v = c.find("main:v", ns)
                    val = v.text if v is not None else None
                    if t == "s" and val is not None:
                        idx = int(val)
                        if 0 <= idx < len(shared):
                            val = shared[idx]
                    rdict[col_idx] = val
                    max_col_idx = max(max_col_idx, col_idx)
                rows.append(rdict)
    if not rows:
        return pd.DataFrame()
    data = [[r.get(i) for i in range(max_col_idx + 1)] for r in rows]
    return pd.DataFrame(data)

def _read_excel_raw(file_like_or_bytes) -> pd.DataFrame:
    if isinstance(file_like_or_bytes, (str, Path)):
        p = Path(file_like_or_bytes)
        with open(p, "rb") as fh:
            b = fh.read()
        src = p
    elif isinstance(file_like_or_bytes, bytes):
        b = file_like_or_bytes
        src = BytesIO(b)
    else:
        b = file_like_or_bytes.read()
        file_like_or_bytes.seek(0)
        src = file_like_or_bytes
    try:
        return pd.read_excel(src, header=None)
    except ImportError:
        return _parse_xlsx_without_openpyxl(b)

def _find_block_start_indices(raw: pd.DataFrame) -> tuple[int | None, int | None]:
    for row_idx in (0, 1):
        if row_idx >= len(raw):
            continue
        row = raw.iloc[row_idx].astype(str).str.strip()
        b_pos, a_pos = None, None
        for idx, val in row.items():
            if val == "B Division" and b_pos is None:
                b_pos = idx
            if val == "A Division" and a_pos is None:
                a_pos = idx
        if b_pos is not None or a_pos is not None:
            return b_pos, a_pos
    return None, None

def load_and_prepare_data_from_bytes(xlsx_bytes: bytes) -> pd.DataFrame:
    raw = _read_excel_raw(xlsx_bytes)
    b_start, a_start = _find_block_start_indices(raw)
    if b_start is None and a_start is None:
        b_start = 0
        a_start = 4 if raw.shape[1] >= 7 else 3 if raw.shape[1] >= 6 else None

    header_row = 1 if (len(raw) > 1) else 0
    data_start = header_row + 1
    frames = []

    def extract_block(start_col: int, division_name: str):
        end_col = start_col + 3
        if start_col is None or end_col > raw.shape[1]:
            return
        header_vals = raw.iloc[header_row, start_col:end_col].tolist()
        labels = [(str(h).strip() if h is not None else "") for h in header_vals]
        if len(labels) != 3:
            return
        temp = raw.iloc[data_start:, start_col:end_col].copy()
        temp.columns = labels
        cols = list(temp.columns)
        if len(cols) != 3:
            return
        temp = temp.rename(columns={cols[0]: "Team", cols[1]: "Player", cols[2]: "Goals"})
        temp["Division"] = division_name
        frames.append(temp)

    if b_start is not None: extract_block(b_start, "B Division")
    if a_start is not None: extract_block(a_start, "A Division")

    if not frames:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Team", "Player", "Goals"])
    combined["Goals"] = pd.to_numeric(combined["Goals"], errors="coerce")
    combined = combined.dropna(subset=["Goals"])
    combined["Goals"] = combined["Goals"].astype(int)
    return combined[["Division", "Team", "Player", "Goals"]]

def fetch_xlsx_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content:
        raise ValueError("Downloaded file is empty.")
    return r.content

# ========================= UI Components ===========================
def display_beautiful_metrics(df: pd.DataFrame) -> None:
    total = int(df["Goals"].sum()) if not df.empty else 0
    players = df["Player"].nunique() if not df.empty else 0
    teams = df["Team"].nunique() if not df.empty else 0
    avg_goals = round(total / players, 1) if players > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="⚽ TOTAL GOALS", value=f"{total:,}")
    
    with col2:
        st.metric(label="👥 PLAYERS", value=f"{players:,}")
    
    with col3:
        st.metric(label="🏆 TEAMS", value=f"{teams:,}")
    
    with col4:
        st.metric(label="📊 AVG/PLAYER", value=f"{avg_goals}")

def create_filter_panel(full_df: pd.DataFrame):
    st.markdown("""
    <div class="filter-panel">
        <div class="filter-title">🔍 Smart Filters & Search</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        div_opts = ["All Divisions"] + sorted(full_df["Division"].unique().tolist())
        div_sel = st.selectbox("Division", div_opts, key="div_filter")
    
    with col2:
        data_div = full_df if div_sel == "All Divisions" else full_df[full_df["Division"] == div_sel]
        team_opts = sorted(data_div["Team"].unique().tolist())
        team_sel = st.multiselect("Teams (Optional)", team_opts, key="team_filter")
    
    col3, col4 = st.columns([3, 1])
    with col3:
        player_query = st.text_input(
            "🔍 Search Players", 
            value="",
            placeholder="Type player names separated by commas",
            key="player_search"
        )
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All", key="clear_filters"):
            for key in ["div_filter", "team_filter", "player_search"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    return div_sel, team_sel, player_query, data_div

def create_altair_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    return alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
        color='#0ea5e9',
        stroke='white',
        strokeWidth=1
    ).encode(
        x=alt.X(f'{x_col}:Q', title=x_col.replace('_', ' ').title(), axis=alt.Axis(format='d')),
        y=alt.Y(f'{y_col}:N', sort='-x', title=y_col.replace('_', ' ').title()),
        tooltip=[y_col, x_col]
    ).properties(
        title=title,
        height=400
    )

def main():
    st.set_page_config(
        page_title="ABEER BLUESTAR SOCCER FEST 2K25", 
        page_icon="⚽", 
        layout="wide"
    )
    
    # Inject the fixed CSS
    inject_fixed_beautiful_css()

    # Title
    st.markdown('<h1 class="main-title">⚽ ABEER BLUESTAR SOCCER FEST 2K25</h1>', 
                unsafe_allow_html=True)

    # Data URL
    XLSX_URL = ("https://docs.google.com/spreadsheets/d/e/"
                "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx")

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎮 Dashboard Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Data refreshed!")

    # Load data
    @st.cache_data(ttl=300)
    def load_data():
        data_bytes = fetch_xlsx_bytes_from_url(XLSX_URL)
        return load_and_prepare_data_from_bytes(data_bytes)

    try:
        with st.spinner("Loading data..."):
            data = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    full_df = data.copy()

    with st.sidebar:
        st.markdown("### 📋 Data Summary")
        st.write(f"📊 Total Records: {len(full_df)}")
        st.write(f"🏆 Teams: {full_df['Team'].nunique()}")
        st.write(f"👥 Players: {full_df['Player'].nunique()}")

    # Filters
    div_sel, team_sel, player_query, data_div = create_filter_panel(full_df)
    
    # Apply filters
    filtered = data_div if not team_sel else data_div[data_div["Team"].isin(team_sel)]

    if player_query.strip():
        tokens = [t.strip().lower() for t in player_query.split(",") if t.strip()]
        if tokens:
            mask = False
            for t in tokens:
                mask = mask | filtered["Player"].str.lower().str.contains(t, na=False)
            filtered = filtered[mask]

    # Main content
    st.markdown('<div class="section-header">📊 Tournament Overview</div>', unsafe_allow_html=True)
    
    # Metrics
    display_beautiful_metrics(filtered)
    
    # Data table
    st.markdown('<div class="section-header">🏅 Goal Records</div>', unsafe_allow_html=True)
    if filtered.empty:
        st.markdown('<div class="info-card">No records match your filters.</div>', unsafe_allow_html=True)
    else:
        display_df = filtered.sort_values("Goals", ascending=False).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Charts
        if len(filtered) > 0:
            st.markdown('<div class="section-header">🏆 Team Performance</div>', unsafe_allow_html=True)
            team_goals = filtered.groupby("Team", as_index=False)["Goals"].sum().sort_values("Goals", ascending=False)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_altair_chart(team_goals, "Goals", "Team", "Goals by Team")
            st.altair_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
