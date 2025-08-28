# ABEER BLUESTAR SOCCER FEST 2K25 — Complete Streamlit Dashboard
# Enhanced version with real data processing and advanced features
# Author: AI Assistant | Created: 2025-08-28

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ====================== CONFIGURATION =============================
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== STYLING ===================================
def inject_advanced_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        .stApp { 
            font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .block-container { 
            padding-top: 0.5rem; 
            padding-bottom: 2rem; 
            max-width: 98vw; 
            width: 98vw;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            margin: 1rem auto;
        }
        
        /* Main title styling */
        h1 {
            text-align: center; 
            margin: 0.5rem 0 1.5rem 0; 
            letter-spacing: 0.05em; 
            font-weight: 700; 
            line-height: 1.1;
            font-size: clamp(22px, 3.5vw, 36px);
            background: linear-gradient(45deg, #0ea5e9, #1e40af, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Enhanced buttons */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
            color: white !important;
            border: 0 !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { 
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important;
            filter: brightness(1.1) !important;
        }
        
        /* Enhanced dataframes */
        .stDataFrame {
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important;
        }
        .stDataFrame table {
            background: white !important;
        }
        .stDataFrame th {
            background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
            color: #1e293b !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #cbd5e1 !important;
        }
        
        /* Metric cards enhancement */
        .metric-container {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(59, 130, 246, 0.05));
            border-radius: 15px;
            padding: 1.5rem;
            border-left: 4px solid #0ea5e9;
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.1);
            transition: transform 0.3s ease;
        }
        .metric-container:hover {
            transform: translateY(-3px);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0ea5e9, #3b82f6);
            color: white;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }
        
        /* Success/Info boxes */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid #0ea5e9;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #0ea5e9, #3b82f6);
            border-radius: 10px;
        }
        
        /* Charts background */
        .element-container {
            background: white;
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        /* Loading spinner */
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #0ea5e9;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Enhanced selectbox */
        .stSelectbox > div > div {
            border-radius: 10px;
            border: 2px solid #e5e7eb;
            transition: border-color 0.3s ease;
        }
        .stSelectbox > div > div:focus-within {
            border-color: #0ea5e9;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        }
        
        /* Enhanced text input */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e5e7eb;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0ea5e9;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.5rem;
                margin: 0.5rem;
                width: 95vw;
                max-width: 95vw;
            }
            h1 {
                font-size: clamp(18px, 5vw, 24px);
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Configure Altair theme
    alt.themes.register("tournament_theme", lambda: {
        "config": {
            "view": {"stroke": "transparent", "fill": "white"},
            "background": "white",
            "title": {"font": "Poppins", "fontSize": 18, "color": "#1e293b", "fontWeight": 600},
            "axis": {
                "labelColor": "#64748b", 
                "titleColor": "#374151", 
                "gridColor": "#f1f5f9",
                "labelFont": "Poppins",
                "titleFont": "Poppins"
            },
            "legend": {
                "labelColor": "#64748b", 
                "titleColor": "#374151",
                "labelFont": "Poppins",
                "titleFont": "Poppins"
            },
            "range": {
                "category": ["#0ea5e9", "#34d399", "#60a5fa", "#f59e0b", "#f87171", "#a78bfa", "#fb7185", "#4ade80"]
            }
        }
    })
    alt.themes.enable("tournament_theme")

# ====================== DATA PROCESSING ===========================
def parse_xlsx_without_dependencies(file_bytes: bytes) -> pd.DataFrame:
    """Parse XLSX file without openpyxl dependency using zipfile and XML parsing."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        # Read shared strings
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                root = ET.parse(f).getroot()
                for si in root.findall(".//main:si", ns):
                    text = "".join(t.text or "" for t in si.findall(".//main:t", ns))
                    shared_strings.append(text)
        
        # Read worksheet data
        with z.open("xl/worksheets/sheet1.xml") as f:
            root = ET.parse(f).getroot()
            sheet_data = root.find("main:sheetData", ns)
            
            if sheet_data is None:
                return pd.DataFrame()
            
            rows_data = []
            max_col_idx = 0
            
            for row in sheet_data.findall("main:row", ns):
                row_dict = {}
                for cell in row.findall("main:c", ns):
                    # Get cell reference (like A1, B2, etc.)
                    ref = cell.attrib.get("r", "A1")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    
                    # Convert column letters to index
                    col_idx = 0
                    for ch in col_letters:
                        col_idx = col_idx * 26 + (ord(ch) - 64)
                    col_idx -= 1
                    
                    # Get cell type and value
                    cell_type = cell.attrib.get("t")
                    value_elem = cell.find("main:v", ns)
                    value = value_elem.text if value_elem is not None else None
                    
                    # Handle shared strings
                    if cell_type == "s" and value is not None:
                        idx = int(value)
                        if 0 <= idx < len(shared_strings):
                            value = shared_strings[idx]
                    
                    row_dict[col_idx] = value
                    max_col_idx = max(max_col_idx, col_idx)
                
                rows_data.append(row_dict)
    
    if not rows_data:
        return pd.DataFrame()
    
    # Convert to matrix format
    data_matrix = []
    for row_dict in rows_data:
        row_list = [row_dict.get(i) for i in range(max_col_idx + 1)]
        data_matrix.append(row_list)
    
    return pd.DataFrame(data_matrix)

def safe_read_excel(file_source) -> pd.DataFrame:
    """Safely read Excel file with fallback methods."""
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        file_bytes = file_source.read()
    
    try:
        # Try pandas first
        return pd.read_excel(BytesIO(file_bytes), header=None)
    except ImportError:
        # Fallback to custom parser
        return parse_xlsx_without_dependencies(file_bytes)

def find_division_columns(raw_df: pd.DataFrame) -> tuple[int, int]:
    """Find B Division and A Division column start positions."""
    b_col, a_col = None, None
    
    # Check first two rows for division headers
    for row_idx in range(min(2, len(raw_df))):
        row_data = raw_df.iloc[row_idx].astype(str).str.strip().str.lower()
        
        for col_idx, cell_value in row_data.items():
            if "b division" in cell_value and b_col is None:
                b_col = col_idx
            elif "a division" in cell_value and a_col is None:
                a_col = col_idx
    
    # Fallback to default positions if not found
    if b_col is None and a_col is None:
        b_col = 0
        a_col = 5 if raw_df.shape[1] >= 8 else 4 if raw_df.shape[1] >= 7 else None
    
    return b_col, a_col

def process_tournament_data(xlsx_bytes: bytes) -> pd.DataFrame:
    """Process the tournament Excel data into structured format."""
    raw_df = safe_read_excel(xlsx_bytes)
    
    if raw_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    
    # Find division column positions
    b_start, a_start = find_division_columns(raw_df)
    
    # Determine header and data rows
    header_row = 1 if len(raw_df) > 1 else 0
    data_start_row = header_row + 1
    
    processed_data = []
    
    def extract_division_data(start_col: int, division_name: str):
        """Extract data for a specific division."""
        if start_col is None or start_col + 2 >= raw_df.shape[1]:
            return
        
        end_col = start_col + 3
        division_data = raw_df.iloc[data_start_row:, start_col:end_col].copy()
        
        # Get column headers
        if header_row < len(raw_df):
            headers = raw_df.iloc[header_row, start_col:end_col].tolist()
            headers = [str(h).strip() if h is not None else f"Col_{i}" for i, h in enumerate(headers)]
        else:
            headers = ["Team", "Player", "Goals"]
        
        if len(headers) >= 3:
            division_data.columns = headers[:3]
            division_data = division_data.rename(columns={
                division_data.columns[0]: "Team",
                division_data.columns[1]: "Player", 
                division_data.columns[2]: "Goals"
            })
            
            # Clean and filter data
            division_data = division_data.dropna(subset=["Team", "Player", "Goals"])
            division_data["Goals"] = pd.to_numeric(division_data["Goals"], errors="coerce")
            division_data = division_data.dropna(subset=["Goals"])
            division_data["Goals"] = division_data["Goals"].astype(int)
            division_data["Division"] = division_name
            
            processed_data.extend(division_data.to_dict("records"))
    
    # Extract data for both divisions
    if b_start is not None:
        extract_division_data(b_start, "B Division")
    if a_start is not None:
        extract_division_data(a_start, "A Division")
    
    if not processed_data:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    
    result_df = pd.DataFrame(processed_data)
    return result_df[["Division", "Team", "Player", "Goals"]]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_data(url: str) -> pd.DataFrame:
    """Fetch and process data from Google Sheets with caching."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if not response.content:
            raise ValueError("Downloaded file is empty")
        
        return process_tournament_data(response.content)
    
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {str(e)}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

# ====================== ANALYTICS FUNCTIONS =======================
def calculate_tournament_stats(df: pd.DataFrame) -> dict:
    """Calculate comprehensive tournament statistics."""
    if df.empty:
        return {
            "total_goals": 0,
            "total_players": 0,
            "total_teams": 0,
            "divisions": 0,
            "avg_goals_per_player": 0,
            "avg_goals_per_team": 0,
            "top_scorer_goals": 0,
            "competitive_balance": 0
        }
    
    # Aggregate player data (handle duplicate player entries)
    player_totals = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
    team_totals = df.groupby(["Team", "Division"])["Goals"].sum().reset_index()
    
    stats = {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(player_totals),
        "total_teams": len(team_totals),
        "divisions": df["Division"].nunique(),
        "avg_goals_per_player": round(df["Goals"].sum() / len(player_totals), 2) if len(player_totals) > 0 else 0,
        "avg_goals_per_team": round(df["Goals"].sum() / len(team_totals), 2) if len(team_totals) > 0 else 0,
        "top_scorer_goals": int(player_totals["Goals"].max()) if not player_totals.empty else 0,
        "competitive_balance": round(team_totals["Goals"].std(), 2) if len(team_totals) > 1 else 0
    }
    
    return stats

def get_top_performers(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Get top performing players and teams."""
    if df.empty:
        return {"players": pd.DataFrame(), "teams": pd.DataFrame()}
    
    # Top players (aggregate goals by player-team combination)
    top_players = (df.groupby(["Player", "Team", "Division"])["Goals"]
                   .sum()
                   .reset_index()
                   .sort_values(["Goals", "Player"], ascending=[False, True])
                   .head(top_n))
    
    # Top teams
    top_teams = (df.groupby(["Team", "Division"])["Goals"]
                 .sum()
                 .reset_index()
                 .sort_values("Goals", ascending=False)
                 .head(top_n))
    
    return {"players": top_players, "teams": top_teams}

def create_division_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Create division comparison statistics."""
    if df.empty:
        return pd.DataFrame()
    
    division_stats = df.groupby("Division").agg({
        "Goals": ["sum", "mean", "count"],
        "Team": "nunique",
        "Player": "nunique"
    }).round(2)
    
    division_stats.columns = ["Total_Goals", "Avg_Goals", "Total_Records", "Teams", "Players"]
    division_stats = division_stats.reset_index()
    
    # Calculate percentages
    total_goals = division_stats["Total_Goals"].sum()
    division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"] / total_goals * 100).round(1)
    
    return division_stats

# ====================== VISUALIZATION FUNCTIONS ===================
def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                               title: str, color_scheme: str = "blues") -> alt.Chart:
    """Create enhanced horizontal bar chart with integer ticks."""
    max_val = int(df[x_col].max()) if not df.empty else 1
    tick_values = list(range(0, max_val + 1)) if max_val <= 50 else None
    
    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopRight=6,
            cornerRadiusBottomRight=6,
            opacity=0.8,
            stroke="white",
            strokeWidth=1
        )
        .encode(
            x=alt.X(
                f"{x_col}:Q",
                title="Goals",
                axis=alt.Axis(format="d", tickMinStep=1, values=tick_values, gridOpacity=0.3),
                scale=alt.Scale(domainMin=0, nice=False)
            ),
            y=alt.Y(
                f"{y_col}:N", 
                sort="-x", 
                title=None,
                axis=alt.Axis(labelLimit=200)
            ),
            color=alt.Color(
                f"{x_col}:Q",
                scale=alt.Scale(scheme=color_scheme),
                legend=None
            ),
            tooltip=[
                alt.Tooltip(f"{y_col}:N", title=y_col),
                alt.Tooltip(f"{x_col}:Q", title="Goals", format="d")
            ]
        )
        .properties(
            height=max(300, min(600, len(df) * 25)),
            title=alt.TitleParams(text=title, fontSize=16, anchor="start", fontWeight=600)
        )
        .resolve_scale(color="independent")
    )
    
    return chart

def create_division_donut_chart(df: pd.DataFrame) -> alt.Chart:
    """Create enhanced donut chart for division distribution."""
    if df.empty:
        return alt.Chart().mark_text(text="No data available")
    
    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    
    base = alt.Chart(division_data).add_selection(
        alt.selection_single()
    ).properties(
        width=300,
        height=300,
        title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600)
    )
    
    # Outer ring
    outer = base.mark_arc(
        innerRadius=60,
        outerRadius=120,
        stroke="white",
        strokeWidth=2
    ).encode(
        theta=alt.Theta("Goals:Q", title="Goals"),
        color=alt.Color(
            "Division:N",
            scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]),
            title="Division"
        ),
        opacity=alt.condition(alt.selection_single(), alt.value(1.0), alt.value(0.8)),
        tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")]
    )
    
    # Center text
    center_text = base.mark_text(
        align="center",
        baseline="middle",
        fontSize=18,
        fontWeight="bold",
        color="#1e293b"
    ).encode(
        text=alt.value(f"Total\n{division_data['Goals'].sum()}")
    )
    
    return outer + center_text

def create_advanced_scatter_plot(df: pd.DataFrame) -> go.Figure:
    """Create advanced scatter plot using Plotly."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    # Prepare team statistics
    team_stats = df.groupby(["Team", "Division"]).agg({
        "Player": "nunique",
        "Goals": "sum"
    }).reset_index()
    team_stats.columns = ["Team", "Division", "Players", "Goals"]
    
    fig = px.scatter(
        team_stats,
        x="Players",
        y="Goals", 
        color="Division",
        size="Goals",
        hover_name="Team",
        hover_data={"Players": True, "Goals": True},
        title="Team Performance: Players vs Total Goals",
        color_discrete_map={"A Division": "#0ea5e9", "B Division": "#f59e0b"}
    )
    
    fig.update_traces(
        marker=dict(
            sizemode="diameter",
            sizemin=8,
            sizemax=30,
            line=dict(width=2, color="white"),
            opacity=0.8
        )
    )
    
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Poppins", size=12),
        title=dict(font=dict(size=16, color="#1e293b")),
        xaxis=dict(
            title="Number of Players in Team",
            gridcolor="#f1f5f9",
            zeroline=False
        ),
        yaxis=dict(
            title="Total Goals",
            gridcolor="#f1f5f9",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    return fig

def create_goals_distribution_histogram(df: pd.DataFrame) -> go.Figure:
    """Create goals distribution histogram."""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    # Calculate player totals
    player_goals = df.groupby(["Player", "Team"])["Goals"].sum().values
    
    fig = go.Figure(data=[
        go.Histogram(
            x=player_goals,
            nbinsx=max(1, len(set(player_goals))),
            marker_color="#a78bfa",
            marker_line_color="white",
            marker_line_width=1.5,
            opacity=0.8,
            hovertemplate="<b>%{x} Goals</b><br>%{y} Players<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Distribution of Goals per Player",
        xaxis_title="Goals per Player",
        yaxis_title="Number of Players",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Poppins", size=12),
        title_font=dict(size=16, color="#1e293b"),
        xaxis=dict(gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(gridcolor="#f1f5f9", zeroline=False),
        height=400
    )
    
    return fig

# ====================== UI COMPONENTS =============================
def display_metric_cards(stats: dict):
    """Display enhanced metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-size: 2.5rem; font-weight: 700; color: #0ea5e9; margin-bottom: 0.5rem;">
                    {stats['total_goals']}
                </div>
                <div style="color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
                    TOTAL GOALS
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-size: 2.5rem; font-weight: 700; color: #0ea5e9; margin-bottom: 0.5rem;">
                    {stats['total_players']}
                </div>
                <div style="color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
                    PLAYERS
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-size: 2.5rem; font-weight: 700; color: #0ea5e9; margin-bottom: 0.5rem;">
                    {stats['total_teams']}
                </div>
                <div style="color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
                    TEAMS
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="font-size: 2.5rem; font-weight: 700; color: #0ea5e9; margin-bottom: 0.5rem;">
                    {stats['avg_goals_per_player']}
                </div>
                <div style="color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
                    AVG GOALS/PLAYER
