# ABEER BLUESTAR SOCCER FEST 2K25 — Enhanced Beautiful Dashboard
# Modern UI with cards, light theme, and enhanced filters

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------- BEAUTIFUL LIGHT THEME & CARDS -------------------------
def inject_beautiful_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
      /* Light theme base */
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
      
      /* Custom title with gradient */
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
      }
      
      /* Card styles */
      .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
      }
      
      .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px -5px rgba(0, 0, 0, 0.15);
        border-color: #cbd5e1;
      }
      
      .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #8b5cf6);
        border-radius: 16px 16px 0 0;
      }
      
      .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #0ea5e9, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      
      .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      
      /* Filter panel */
      .filter-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
      
      /* Enhanced sidebar */
      .css-1d391kg { 
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important; 
        border-right: 1px solid #e2e8f0;
      }
      .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        color: #1e293b !important;
        font-weight: 600;
      }
      
      /* Buttons */
      .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
      }
      
      .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3);
      }
      
      /* Tabs */
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
        transition: all 0.3s ease;
        border: none;
      }
      
      .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #1e293b;
      }
      
      .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
      }
      
      /* Data tables */
      .stDataFrame {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
      }
      
      /* Section headers */
      .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
      }
      
      /* Info cards */
      .info-card {
        background: linear-gradient(135deg, #fef3c7, #fbbf24);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        color: #92400e;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
      }
      
      /* Chart containers */
      .chart-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
      }
      
      /* Custom selectbox and input styling */
      .stSelectbox label, .stMultiSelect label, .stTextInput label {
        color: #1e293b !important;
        font-weight: 500 !important;
      }
      
      .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
      }
      
      /* Metrics styling for built-in st.metric */
      [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
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
      
      /* Success/error message styling */
      .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px !important;
        border: none !important;
      }
      
      /* Slider styling */
      .stSlider > div > div > div > div {
        background: #0ea5e9 !important;
      }
      
      /* Enhanced hover effects */
      .metric-card:hover, .chart-container:hover {
        transform: translateY(-2px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      /* Subtle animations */
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      .stDataFrame, .metric-card, .chart-container {
        animation: fadeIn 0.5s ease-out;
      }
    </style>
    """, unsafe_allow_html=True)

# ========================= XLSX fallback (no openpyxl) ========================
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

# ========================= Robust block parser ================================
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

# ========================= Fetch helpers ======================================
def fetch_xlsx_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content:
        raise ValueError("Downloaded file is empty.")
    return r.content

# ========================= Enhanced UI Components ===========================
def display_beautiful_metrics(df: pd.DataFrame) -> None:
    total = int(df["Goals"].sum()) if not df.empty else 0
    players = df["Player"].nunique() if not df.empty else 0
    teams = df["Team"].nunique() if not df.empty else 0
    avg_goals = round(total / players, 1) if players > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⚽ TOTAL GOALS", 
            value=f"{total:,}",
        )
    
    with col2:
        st.metric(
            label="👥 PLAYERS", 
            value=f"{players:,}",
        )
    
    with col3:
        st.metric(
            label="🏆 TEAMS", 
            value=f"{teams:,}",
        )
    
    with col4:
        st.metric(
            label="📊 AVG GOALS/PLAYER", 
            value=f"{avg_goals}",
        )

def create_filter_panel(full_df: pd.DataFrame):
    """Create a beautiful filter panel"""
    st.markdown("""
    <div class="filter-panel">
        <div class="filter-title">
            🔍 Smart Filters & Search
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        div_opts = ["🌟 All Divisions"] + [f"🏅 {div}" for div in sorted(full_df["Division"].unique().tolist())]
        div_sel = st.selectbox("Division", div_opts, key="div_filter")
        # Clean the selection
        if div_sel.startswith("🏅 "):
            div_sel = div_sel[3:]
        elif div_sel == "🌟 All Divisions":
            div_sel = "All Divisions"
    
    with col2:
        data_div = full_df if div_sel == "All Divisions" else full_df[full_df["Division"] == div_sel]
        team_opts = [f"⚽ {team}" for team in sorted(data_div["Team"].unique().tolist())]
        team_sel = st.multiselect("Teams (Optional)", team_opts, key="team_filter")
        # Clean the selections
        team_sel = [team[2:] for team in team_sel]  # Remove "⚽ " prefix
    
    # Player search with enhanced styling
    col3, col4 = st.columns([3, 1])
    with col3:
        player_query = st.text_input(
            "🔍 Search Players", 
            value="",
            placeholder="Type player names separated by commas (e.g., Ahmed, Mohammed, Ali)",
            key="player_search"
        )
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        clear_filters = st.button("🗑️ Clear All", key="clear_filters")
        if clear_filters:
            st.session_state.div_filter = "🌟 All Divisions"
            st.session_state.team_filter = []
            st.session_state.player_search = ""
            st.rerun()
    
    return div_sel, team_sel, player_query, data_div

def create_plotly_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col=None):
    """Create a beautiful Plotly bar chart with light theme"""
    fig = px.bar(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        color=color_col,
        color_discrete_sequence=['#0ea5e9', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(color='#1e293b', family='Inter'),
        title_font=dict(size=18, color='#1e293b', family='Inter'),
        xaxis=dict(
            gridcolor='#e2e8f0',
            linecolor='#cbd5e1',
            titlefont=dict(color='#64748b', family='Inter'),
            tickfont=dict(color='#475569')
        ),
        yaxis=dict(
            gridcolor='#e2e8f0',
            linecolor='#cbd5e1',
            titlefont=dict(color='#64748b', family='Inter'),
            tickfont=dict(color='#475569')
        ),
        showlegend=bool(color_col),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_traces(
        marker_line_color='rgba(255,255,255,0.8)',
        marker_line_width=1,
        hovertemplate='<b>%{y}</b><br>Goals: %{x}<extra></extra>'
    )
    
    return fig

def create_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Create a beautiful Plotly pie chart with light theme"""
    agg = df.groupby("Division")["Goals"].sum().reset_index()
    
    fig = px.pie(
        agg,
        values='Goals',
        names='Division',
        title='🏆 Goals Distribution by Division',
        color_discrete_sequence=['#0ea5e9', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(color='#1e293b', family='Inter'),
        title_font=dict(size=18, color='#1e293b', family='Inter')
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=12,
        marker=dict(
            line=dict(color='#ffffff', width=3)
        ),
        hovertemplate='<b>%{label}</b><br>Goals: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    return fig

def create_advanced_analytics_charts(df: pd.DataFrame):
    """Create advanced analytics charts"""
    if df.empty:
        return None, None
    
    # Goals distribution histogram
    fig1 = px.histogram(
        df, 
        x='Goals', 
        nbins=20,
        title='📊 Goals Distribution',
        color_discrete_sequence=['#0ea5e9']
    )
    fig1.update_layout(
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(color='#1e293b', family='Inter'),
        title_font=dict(size=16, color='#1e293b')
    )
    
    # Top teams vs players scatter
    team_stats = df.groupby('Team').agg({
        'Goals': 'sum',
        'Player': 'nunique'
    }).reset_index()
    team_stats.columns = ['Team', 'Total_Goals', 'Player_Count']
    
    fig2 = px.scatter(
        team_stats,
        x='Player_Count',
        y='Total_Goals',
        size='Total_Goals',
        hover_name='Team',
        title='🎯 Team Performance: Players vs Goals',
        color='Total_Goals',
        color_continuous_scale='Blues'
    )
    fig2.update_layout(
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(color='#1e293b', family='Inter'),
        title_font=dict(size=16, color='#1e293b')
    )
    
    return fig1, fig2

def make_reports_zip(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    team_goals = (
        filtered_df.groupby("Team", as_index=False)["Goals"]
        .sum()
        .sort_values("Goals", ascending=False)
    )
    top_scorers = (
        filtered_df.groupby(["Player", "Team"], as_index=False)["Goals"]
        .sum()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        [["Player", "Team", "Goals"]]
    )
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("records_full.csv", full_df.to_csv(index=False))
        z.writestr("records_filtered.csv", filtered_df.to_csv(index=False))
        z.writestr("team_goals_filtered.csv", team_goals.to_csv(index=False))
        z.writestr("top_scorers_filtered.csv", top_scorers.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# ========================= Enhanced App =====================================
def main():
    st.set_page_config(
        page_title="ABEER BLUESTAR SOCCER FEST 2K25", 
        page_icon="⚽", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_beautiful_css()

    # Beautiful gradient title
    st.markdown('<h1 class="main-title">⚽ ABEER BLUESTAR SOCCER FEST 2K25</h1>', unsafe_allow_html=True)

    XLSX_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎮 Dashboard Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Data refreshed!")

    @st.cache_data(ttl=300)
    def load_data():
        data_bytes = fetch_xlsx_bytes_from_url(XLSX_URL)
        return load_and_prepare_data_from_bytes(data_bytes)

    try:
        with st.spinner("🔄 Loading tournament data..."):
            data = load_data()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

    # Keep a permanent full copy for downloads
    full_df = data.copy()

    with st.sidebar:
        last_ref = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.caption(f"📅 Last refreshed: {last_ref}")
        
        # Add data summary in sidebar
        st.markdown("---")
        st.markdown("### 📋 Data Summary")
        st.caption(f"📊 Total Records: {len(full_df)}")
        st.caption(f"🏆 Divisions: {full_df['Division'].nunique()}")
        st.caption(f"⚽ Teams: {full_df['Team'].nunique()}")
        st.caption(f"👥 Players: {full_df['Player'].nunique()}")

    # Beautiful filter panel
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

    current_players = sorted(filtered["Player"].unique().tolist())
    if current_players:
        with st.sidebar:
            st.markdown("### 👥 Refine Player Selection")
            players_pick = st.multiselect("Select specific players:", options=current_players)
            if players_pick:
                filtered = filtered[filtered["Player"].isin(players_pick)]

    # Show active filters summary
    if div_sel != "All Divisions" or team_sel or player_query.strip() or (current_players and 'players_pick' in locals() and players_pick):
        with st.expander("🔍 Active Filters", expanded=False):
            st.write(f"**Division:** {div_sel}")
            if team_sel:
                st.write(f"**Teams:** {', '.join(team_sel)}")
            if player_query.strip():
                st.write(f"**Search:** {player_query}")
            if 'players_pick' in locals() and players_pick:
                st.write(f"**Selected Players:** {', '.join(players_pick[:3])}" + (f" (+{len(players_pick)-3} more)" if len(players_pick) > 3 else ""))
            st.caption(f"Showing {len(filtered)} of {len(full_df)} records")

    # ===== Enhanced Tabs with Beautiful Icons =====
    t_overview, t_teams, t_players, t_analytics, t_downloads = st.tabs([
        "🏠 OVERVIEW", 
        "🏆 TEAMS", 
        "👥 PLAYERS", 
        "📊 ANALYTICS",
        "⬇️ DOWNLOADS"
    ])

    # ---------------------- ENHANCED OVERVIEW ----------------------
    with t_overview:
        # Beautiful metrics cards
        display_beautiful_metrics(filtered)
        
        st.markdown('<div class="section-header">🏅 Latest Goal Records</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">📭 No records match your current filters. Try adjusting the filters above to see more data.</div>', unsafe_allow_html=True)
        else:
            # Enhanced dataframe display
            display_df = filtered.sort_values("Goals", ascending=False).reset_index(drop=True)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Division": st.column_config.TextColumn("🏅 Division", width="small"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Player": st.column_config.TextColumn("👤 Player", width="medium"),
                    "Goals": st.column_config.NumberColumn("⚽ Goals", format="%d", width="small")
                },
                height=400
            )

        st.markdown('<div class="section-header">🏆 Team Performance</div>', unsafe_allow_html=True)
        if not filtered.empty:
            team_goals = (
                filtered.groupby("Team", as_index=False)["Goals"].sum().sort_values("Goals", ascending=False)
            )
            
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_plotly_bar_chart(team_goals, "Goals", "Team", "🏆 Goals by Team")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🌟 Top Scorers</div>', unsafe_allow_html=True)
        scorers_pt = (
            filtered.groupby(["Player", "Team"], as_index=False)["Goals"]
            .sum()
            .sort_values(["Goals", "Player"], ascending=[False, True])
        )
        max_rows = int(scorers_pt.shape[0])
        
        if max_rows == 0:
            st.markdown('<div class="info-card">👤 No player data available for current filters.</div>', unsafe_allow_html=True)
        elif max_rows >= 1:
            with st.sidebar:
                if max_rows > 1:
                    st.markdown("### 🎯 Display Options")
                    default_top = min(10, max_rows)
                    top_n = st.slider("🔝 Show top N players", 1, max_rows, int(default_top), key=f"topn_{max_rows}")
                else:
                    top_n = 1
            
            top_df = scorers_pt.head(int(top_n)).reset_index(drop=True)
            st.dataframe(
                top_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("👤 Player", width="medium"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Goals": st.column_config.NumberColumn("⚽ Goals", format="%d", width="small")
                },
            )
            
            if max_rows > 1:
                chart_df = top_df.groupby("Player", as_index=False)["Goals"].sum()
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = create_plotly_bar_chart(chart_df, "Goals", "Player", f"🌟 Top {int(top_n)} Scorers")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        if div_sel == "All Divisions" and not full_df.empty:
            st.markdown('<div class="section-header">📊 Division Distribution</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_pie_chart(full_df)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Division summary table
                div_summary = full_df.groupby('Division').agg({
                    'Goals': 'sum',
                    'Player': 'nunique',
                    'Team': 'nunique'
                }).reset_index()
                div_summary.columns = ['Division', 'Total Goals', 'Players', 'Teams']
                div_summary['Avg Goals/Player'] = (div_summary['Total Goals'] / div_summary['Players']).round(1)
                
                st.markdown("**📈 Division Statistics**")
                st.dataframe(
                    div_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Division": st.column_config.TextColumn("🏅 Division"),
                        "Total Goals": st.column_config.NumberColumn("⚽ Goals", format="%d"),
                        "Players": st.column_config.NumberColumn("👥 Players", format="%d"),
                        "Teams": st.column_config.NumberColumn("🏆 Teams", format="%d"),
                        "Avg Goals/Player": st.column_config.NumberColumn("📊 Avg/Player", format="%.1f")
                    }
                )

    # ---------------------- TEAMS TAB ----------------------
    with t_teams:
        st.markdown('<div class="section-header">🏆 Teams Overview</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">🏆 No teams match your current filters.</div>', unsafe_allow_html=True)
        else:
            # Enhanced team summary with more insights
            team_div = (filtered.groupby("Team")["Division"]
                        .agg(lambda s: ", ".join(sorted(s.astype(str).unique()))).reset_index())
            team_summary = (filtered.groupby("Team", as_index=False)
                            .agg(Total_Goals=("Goals","sum"), Players=("Player","nunique")))
            top_by_team = (filtered.groupby(["Team","Player"], as_index=False)["Goals"].sum())
            top_by_team = (top_by_team.sort_values(["Team","Goals"], ascending=[True,False])
                           .groupby("Team").head(1)
                           .rename(columns={"Player":"Top Scorer","Goals":"Top Scorer Goals"}))

            teams_list = (team_div.merge(team_summary, on="Team", how="left")
                                 .merge(top_by_team, on="Team", how="left")
                          .sort_values(["Total_Goals","Team"], ascending=[False,True]))
            
            # Add average goals per player
            teams_list['Avg Goals/Player'] = (teams_list['Total_Goals'] / teams_list['Players']).round(1)
            teams_list = teams_list[["Division","Team","Players","Total_Goals","Avg Goals/Player","Top Scorer","Top Scorer Goals"]]

            st.dataframe(
                teams_list, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Division": st.column_config.TextColumn("🏅 Division", width="small"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Players": st.column_config.NumberColumn("👥 Players", format="%d", width="small"),
                    "Total_Goals": st.column_config.NumberColumn("⚽ Total Goals", format="%d", width="small"),
                    "Avg Goals/Player": st.column_config.NumberColumn("📊 Avg/Player", format="%.1f", width="small"),
                    "Top Scorer": st.column_config.TextColumn("🌟 Top Scorer", width="medium"),
                    "Top Scorer Goals": st.column_config.NumberColumn("🎯 Goals", format="%d", width="small")
                },
                height=500
            )

            # Team performance chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_plotly_bar_chart(teams_list.rename(columns={"Total_Goals":"Goals"}),
                                          "Goals", "Team", "🏆 Team Performance (Total Goals)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- PLAYERS TAB ----------------------
    with t_players:
        st.markdown('<div class="section-header">👥 Players Directory</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">👤 No players match your current filters.</div>', unsafe_allow_html=True)
        else:
            players_list = (filtered.groupby(["Player","Team","Division"], as_index=False)["Goals"].sum()
                            .sort_values(["Goals","Player"], ascending=[False,True]))
            
            # Add rank
            players_list['Rank'] = range(1, len(players_list) + 1)
            players_list = players_list[['Rank', 'Player', 'Team', 'Division', 'Goals']]
            
            st.dataframe(
                players_list, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("🏅 Rank", format="%d", width="small"),
                    "Player": st.column_config.TextColumn("👤 Player", width="large"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Division": st.column_config.TextColumn("🏆 Division", width="small"),
                    "Goals": st.column_config.NumberColumn("⚽ Goals", format="%d", width="small")
                },
                height=600
            )
            
            # Player statistics summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌟 Highest Scorer", f"{players_list.iloc[0]['Player']}", f"{players_list.iloc[0]['Goals']} goals")
            with col2:
                avg_goals = players_list['Goals'].mean()
                st.metric("📊 Average Goals", f"{avg_goals:.1f}")
            with col3:
                median_goals = players_list['Goals'].median()
                st.metric("📈 Median Goals", f"{median_goals}")

    # ---------------------- ANALYTICS TAB ----------------------
    with t_analytics:
        st.markdown('<div class="section-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
        
        if filtered.empty:
            st.markdown('<div class="info-card">📊 No data available for analytics with current filters.</div>', unsafe_allow_html=True)
        else:
            # Create advanced charts
            fig1, fig2 = create_advanced_analytics_charts(filtered)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance insights
            st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                # Top performing teams
                top_teams = filtered.groupby('Team')['Goals'].sum().sort_values(ascending=False).head(3)
                st.markdown("**🏆 Top Performing Teams:**")
                for i, (team, goals) in enumerate(top_teams.items(), 1):
                    st.write(f"{i}. **{team}**: {goals} goals")
                
                # Goal distribution stats
                st.markdown("**📈 Goal Statistics:**")
                st.write(f"• Highest individual score: **{filtered['Goals'].max()}** goals")
                st.write(f"• Most common score: **{filtered['Goals'].mode().iloc[0]}** goals")
                st.write(f"• Players with 5+ goals: **{len(filtered[filtered['Goals'] >= 5])}** players")
            
            with insights_col2:
                # Division comparison
                if div_sel == "All Divisions":
                    div_stats = full_df.groupby('Division').agg({
                        'Goals': ['sum', 'mean', 'count']
                    }).round(2)
                    div_stats.columns = ['Total Goals', 'Avg Goals', 'Records']
                    
                    st.markdown("**🏅 Division Comparison:**")
                    st.dataframe(div_stats, use_container_width=True)
                else:
                    # Team diversity in current division
                    team_player_count = filtered.groupby('Team')['Player'].nunique().sort_values(ascending=False)
                    st.markdown(f"**⚽ Team Sizes in {div_sel}:**")
                    for team, count in team_player_count.head(5).items():
                        st.write(f"• **{team}**: {count} players")

    # ---------------------- DOWNLOADS TAB ----------------------
    with t_downloads:
        st.markdown('<div class="section-header">📁 Download Center</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Individual Reports**")
            st.caption("Download specific data sets in CSV format")
            
            st.download_button(
                "📋 Full Dataset (All Records)",
                data=full_df.to_csv(index=False), 
                file_name="soccer_fest_2k25_full.csv", 
                mime="text/csv",
                use_container_width=True
            )
            
            st.download_button(
                "🔍 Filtered Data (Current View)",
                data=filtered.to_csv(index=False), 
                file_name="soccer_fest_2k25_filtered.csv", 
                mime="text/csv",
                use_container_width=True
            )

            # Team summary download
            if not filtered.empty:
                teams_summary = (filtered.groupby(["Team","Division"], as_index=False)
                                 .agg(Players=("Player","nunique"), Total_Goals=("Goals","sum"))
                                 .sort_values(["Total_Goals","Team"], ascending=[False,True]))
                st.download_button(
                    "🏆 Teams Summary (Current View)",
                    data=teams_summary.to_csv(index=False), 
                    file_name="teams_summary.csv", 
                    mime="text/csv",
                    use_container_width=True
                )

                # Players summary download
                players_summary = (filtered.groupby(["Player","Team","Division"], as_index=False)["Goals"].sum()
                                   .sort_values(["Goals","Player"], ascending=[False,True]))
                st.download_button(
                    "👥 Players Summary (Current View)",
                    data=players_summary.to_csv(index=False), 
                    file_name="players_summary.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("**📦 Complete Package**")
            st.caption("Get everything in one convenient ZIP file")
            
            # ZIP bundle with all reports
            zip_bytes = make_reports_zip(full_df, filtered)
            st.download_button(
                "📦 Complete Report Package (ZIP)",
                data=zip_bytes, 
                file_name=f"soccer_fest_2k25_reports_{datetime.now().strftime('%Y%m%d')}.zip", 
                mime="application/zip",
                use_container_width=True
            )
            
            st.markdown("**📋 Package Contents:**")
            st.caption("• Full dataset (all records)")
            st.caption("• Filtered dataset (current view)")
            st.caption("• Team performance summary")
            st.caption("• Top scorers ranking")
            
            # Export configuration
            st.markdown("**⚙️ Export Options**")
            export_format = st.selectbox("Choose format:", ["CSV", "Excel (XLSX)"], key="export_format")
            if export_format == "Excel (XLSX)":
                st.info("📝 Excel export will be available in future updates!")

if __name__ == "__main__":
    main()
