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
import time

# Optional imports with fallbacks
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    make_subplots = None
    # Only show warning if user is not on a setup page
    pass

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

def find_division_columns(raw_df: pd.DataFrame) -> tuple:
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
    if total_goals > 0:
        division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"] / total_goals * 100).round(1)
    else:
        division_stats["Goal_Share_Pct"] = 0
    
    return division_stats

# ====================== VISUALIZATION FUNCTIONS ===================
def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                               title: str, color_scheme: str = "blues") -> alt.Chart:
    """Create enhanced horizontal bar chart with integer ticks."""
    if df.empty:
        return alt.Chart().mark_text(text="No data available")
    
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

def create_advanced_scatter_plot(df: pd.DataFrame):
    """Create advanced scatter plot using Plotly or Altair fallback."""
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        else:
            return alt.Chart().mark_text(text="No data available")
    
    # Prepare team statistics
    team_stats = df.groupby(["Team", "Division"]).agg({
        "Player": "nunique",
        "Goals": "sum"
    }).reset_index()
    team_stats.columns = ["Team", "Division", "Players", "Goals"]
    
    if PLOTLY_AVAILABLE:
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
    else:
        # Altair fallback
        chart = (
            alt.Chart(team_stats)
            .mark_circle(size=100, opacity=0.8)
            .encode(
                x=alt.X("Players:Q", title="Number of Players in Team"),
                y=alt.Y("Goals:Q", title="Total Goals"),
                color=alt.Color(
                    "Division:N",
                    scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]),
                    title="Division"
                ),
                size=alt.Size("Goals:Q", legend=None),
                tooltip=["Team:N", "Division:N", "Players:Q", "Goals:Q"]
            )
            .properties(
                title="Team Performance: Players vs Total Goals",
                height=400
            )
        )
        return chart

def create_goals_distribution_histogram(df: pd.DataFrame):
    """Create goals distribution histogram using Plotly or Altair fallback."""
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        else:
            return alt.Chart().mark_text(text="No data available")
    
    # Calculate player totals
    player_goals = df.groupby(["Player", "Team"])["Goals"].sum().values
    
    if PLOTLY_AVAILABLE:
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
    else:
        # Altair fallback
        hist_data = pd.DataFrame({"Goals": player_goals})
        chart = (
            alt.Chart(hist_data)
            .mark_bar(color="#a78bfa", opacity=0.8)
            .encode(
                x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
                y=alt.Y("count():Q", title="Number of Players"),
                tooltip=[
                    alt.Tooltip("Goals:Q", bin=True, title="Goals"),
                    alt.Tooltip("count():Q", title="Players")
                ]
            )
            .properties(
                title="Distribution of Goals per Player",
                height=400
            )
        )
        return chart

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
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

def display_insights_cards(df: pd.DataFrame, scope: str = "Tournament"):
    """Display quick insights in card format."""
    if df.empty:
        st.info("📊 No data available for insights.")
        return
    
    # Calculate insights
    stats = calculate_tournament_stats(df)
    top_performers = get_top_performers(df, 5)
    division_comparison = create_division_comparison(df)
    
    # Top team and player
    if not top_performers["teams"].empty:
        top_team = top_performers["teams"].iloc[0]
        top_team_name = top_team["Team"]
        top_team_goals = int(top_team["Goals"])
    else:
        top_team_name, top_team_goals = "—", 0
    
    if not top_performers["players"].empty:
        top_player = top_performers["players"].iloc[0]
        top_player_name = top_player["Player"]
        top_player_team = top_player["Team"]
        top_player_goals = int(top_player["Goals"])
    else:
        top_player_name, top_player_team, top_player_goals = "—", "—", 0
    
    # Display insight cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border-left: 4px solid #0ea5e9; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: #0ea5e9; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    🏆 Top Performing Team
                </div>
                <div style="color: #374151; line-height: 1.5;">
                    <strong>{top_team_name}</strong> leads with <strong>{top_team_goals} goals</strong>, showcasing exceptional team coordination and offensive capability.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if not division_comparison.empty and len(division_comparison) > 1:
            b_goals = division_comparison[division_comparison["Division"] == "B Division"]["Total_Goals"].iloc[0] if "B Division" in division_comparison["Division"].values else 0
            a_goals = division_comparison[division_comparison["Division"] == "A Division"]["Total_Goals"].iloc[0] if "A Division" in division_comparison["Division"].values else 0
            b_pct = division_comparison[division_comparison["Division"] == "B Division"]["Goal_Share_Pct"].iloc[0] if "B Division" in division_comparison["Division"].values else 0
            a_pct = division_comparison[division_comparison["Division"] == "A Division"]["Goal_Share_Pct"].iloc[0] if "A Division" in division_comparison["Division"].values else 0
            
            st.markdown(
                f"""
                <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border-left: 4px solid #f59e0b; margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #f59e0b; margin-bottom: 0.5rem; font-size: 1.1rem;">
                        🎯 Division Performance
                    </div>
                    <div style="color: #374151; line-height: 1.5;">
                        <strong>B Division:</strong> {int(b_goals)} goals ({b_pct}%)<br>
                        <strong>A Division:</strong> {int(a_goals)} goals ({a_pct}%)<br>
                        {"B Division shows higher scoring activity" if b_goals > a_goals else "A Division leads in goal production" if a_goals > b_goals else "Both divisions are evenly matched"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown(
            f"""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border-left: 4px solid #34d399; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: #34d399; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    ⚽ Leading Scorer
                </div>
                <div style="color: #374151; line-height: 1.5;">
                    <strong>{top_player_name}</strong> ({top_player_team}) with <strong>{top_player_goals} goals</strong> - demonstrating consistent performance and clinical finishing.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Competition balance insight
        player_goals = df.groupby(["Player", "Team"])["Goals"].sum()
        goals_1 = len(player_goals[player_goals == 1])
        goals_2_plus = len(player_goals[player_goals >= 2])
        
        st.markdown(
            f"""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border-left: 4px solid #a78bfa; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: #a78bfa; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    📊 Competition Balance
                </div>
                <div style="color: #374151; line-height: 1.5;">
                    <strong>{goals_1} players</strong> scored 1 goal, <strong>{goals_2_plus} players</strong> scored 2+ goals<br>
                    Average: <strong>{stats['avg_goals_per_player']} goals per player</strong><br>
                    {"Well-balanced competition" if stats['competitive_balance'] < 2 else "Some teams dominating" if stats['competitive_balance'] > 4 else "Moderate competition spread"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_enhanced_data_table(df: pd.DataFrame, table_type: str = "records"):
    """Create enhanced data tables with proper formatting."""
    if df.empty:
        st.info(f"📋 No {table_type} data available with current filters.")
        return
    
    if table_type == "records":
        # Main records table
        display_df = df.sort_values("Goals", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small")
            }
        )
    
    elif table_type == "teams":
        # Teams summary table
        teams_summary = df.groupby(["Team", "Division"]).agg({
            "Player": "nunique",
            "Goals": "sum"
        }).reset_index()
        teams_summary.columns = ["Team", "Division", "Players", "Total_Goals"]
        
        # Add top scorer for each team
        top_scorers_list = []
        for team in teams_summary["Team"]:
            team_data = df[df["Team"] == team]
            if not team_data.empty:
                team_player_goals = team_data.groupby("Player")["Goals"].sum()
                top_scorer = team_player_goals.idxmax()
                top_goals = team_player_goals.max()
                top_scorers_list.append({"Team": team, "Top_Scorer": top_scorer, "Top_Scorer_Goals": top_goals})
        
        if top_scorers_list:
            top_scorers_df = pd.DataFrame(top_scorers_list)
            teams_display = teams_summary.merge(top_scorers_df, on="Team", how="left")
        else:
            teams_display = teams_summary.copy()
            teams_display["Top_Scorer"] = "N/A"
            teams_display["Top_Scorer_Goals"] = 0
            
        teams_display = teams_display.sort_values("Total_Goals", ascending=False)
        
        st.dataframe(
            teams_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small"),
                "Top_Scorer": st.column_config.TextColumn("Top Scorer", width="large"),
                "Top_Scorer_Goals": st.column_config.NumberColumn("Goals", format="%d", width="small")
            }
        )
    
    elif table_type == "players":
        # Players summary table
        players_summary = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
        players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
        players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
        
        st.dataframe(
            players_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small")
            }
        )

def create_download_section(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Create comprehensive download section."""
    st.subheader("📥 Download Reports")
    st.caption("**Full** = all data ignoring filters, **Filtered** = current view with active filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Data Exports")
        
        # Full dataset download
        if not full_df.empty:
            st.download_button(
                label="⬇️ Download FULL Dataset (CSV)",
                data=full_df.to_csv(index=False),
                file_name=f"tournament_full_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download complete tournament data"
            )
        
        # Filtered dataset download
        if not filtered_df.empty:
            st.download_button(
                label="⬇️ Download FILTERED Dataset (CSV)",
                data=filtered_df.to_csv(index=False),
                file_name=f"tournament_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download data with current filters applied"
            )
        
    with col2:
        st.subheader("🏆 Summary Reports")
        
        # Teams summary
        if not filtered_df.empty:
            teams_summary = filtered_df.groupby(["Team", "Division"]).agg({
                "Player": "nunique",
                "Goals": "sum"
            }).reset_index()
            teams_summary.columns = ["Team", "Division", "Players_Count", "Total_Goals"]
            teams_summary = teams_summary.sort_values("Total_Goals", ascending=False)
            
            st.download_button(
                label="⬇️ Download TEAMS Summary (CSV)",
                data=teams_summary.to_csv(index=False),
                file_name=f"teams_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download team performance summary"
            )
        
        # Players summary
        if not filtered_df.empty:
            players_summary = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
            players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
            
            st.download_button(
                label="⬇️ Download PLAYERS Summary (CSV)",
                data=players_summary.to_csv(index=False),
                file_name=f"players_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download top scorers summary"
            )
    
    with col3:
        st.subheader("📦 Complete Package")
        
        # Create comprehensive ZIP download
        if st.button("📦 Generate Complete Report Package", help="Generate ZIP with all reports and analytics"):
            with st.spinner("🔄 Generating comprehensive report package..."):
                zip_buffer = create_comprehensive_zip_report(full_df, filtered_df)
                
                st.download_button(
                    label="⬇️ Download Complete Package (ZIP)",
                    data=zip_buffer,
                    file_name=f"tournament_complete_package_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    help="Download ZIP containing all data, summaries, and analytics"
                )

def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    """Create comprehensive ZIP report with multiple files."""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Full dataset
        if not full_df.empty:
            zip_file.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        
        # Filtered dataset
        if not filtered_df.empty:
            zip_file.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
        
        # Teams analysis
        if not filtered_df.empty:
            teams_data = filtered_df.groupby(["Team", "Division"]).agg({
                "Player": ["nunique", "count"],
                "Goals": ["sum", "mean", "max"]
            }).round(2)
            teams_data.columns = ["Unique_Players", "Total_Records", "Total_Goals", "Avg_Goals", "Max_Goals"]
            teams_data = teams_data.reset_index()
            zip_file.writestr("03_teams_detailed_analysis.csv", teams_data.to_csv(index=False))
        
        # Players analysis
        if not filtered_df.empty:
            players_data = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players_data = players_data.sort_values(["Goals", "Player"], ascending=[False, True])
            players_data.insert(0, "Rank", range(1, len(players_data) + 1))
            zip_file.writestr("04_players_ranking.csv", players_data.to_csv(index=False))
        
        # Division comparison
        if not filtered_df.empty:
            division_stats = create_division_comparison(filtered_df)
            if not division_stats.empty:
                zip_file.writestr("05_division_comparison.csv", division_stats.to_csv(index=False))
        
        # Tournament statistics summary
        stats = calculate_tournament_stats(filtered_df)
        stats_df = pd.DataFrame([stats])
        zip_file.writestr("06_tournament_statistics.csv", stats_df.to_csv(index=False))
        
        # README file
        readme_content = f"""
ABEER BLUESTAR SOCCER FEST 2K25 - Data Package
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FILES INCLUDED:
1. 01_full_tournament_data.csv - Complete tournament dataset
2. 02_filtered_tournament_data.csv - Data with applied filters
3. 03_teams_detailed_analysis.csv - Comprehensive team statistics
4. 04_players_ranking.csv - Player rankings and performance
5. 05_division_comparison.csv - Division-wise comparison
6. 06_tournament_statistics.csv - Overall tournament metrics
7. README.txt - This file

TOURNAMENT OVERVIEW:
- Total Goals: {stats['total_goals']}
- Total Players: {stats['total_players']}
- Total Teams: {stats['total_teams']}
- Divisions: {stats['divisions']}
- Average Goals per Player: {stats['avg_goals_per_player']}

For questions or support, please contact the tournament organizers.
        """
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ====================== MAIN APPLICATION ==========================
def main():
    """Main application function."""
    # Check for optional dependencies and show warnings
    if not PLOTLY_AVAILABLE:
        st.sidebar.warning("⚠️ Plotly not installed. Some advanced charts will use Altair instead.")
        with st.sidebar.expander("💡 Install Advanced Charts"):
            st.code("pip install plotly", language="bash")
    
    # Apply styling
    inject_advanced_css()
    
    # Header
    st.markdown("<h1>⚽ ABEER BLUESTAR SOCCER FEST 2K25</h1>", unsafe_allow_html=True)
    
    # Configuration
    GOOGLE_SHEETS_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Refresh button with loading state
        if st.button("🔄 Refresh Tournament Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ Data refreshed successfully!")
            st.rerun()
        
        # Display last refresh time
        last_refresh = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.caption(f"🕒 Last refreshed: {last_refresh}")
        
        st.divider()
        
        # Load data
        with st.spinner("📡 Loading tournament data..."):
            tournament_data = fetch_tournament_data(GOOGLE_SHEETS_URL)
        
        if tournament_data.empty:
            st.error("❌ No tournament data available. Please check the data source.")
            st.stop()
        
        # Keep original data for downloads
        full_tournament_data = tournament_data.copy()
        
        # Filters section
        st.header("🔍 Data Filters")
        
        # Division filter
        division_options = ["All Divisions"] + sorted(tournament_data["Division"].unique().tolist())
        selected_division = st.selectbox("📊 Division", division_options, key="division_filter")
        
        # Apply division filter
        if selected_division != "All Divisions":
            tournament_data = tournament_data[tournament_data["Division"] == selected_division]
        
        # Team filter
        available_teams = sorted(tournament_data["Team"].unique().tolist())
        selected_teams = st.multiselect(
            "🏆 Teams (optional)", 
            available_teams, 
            key="teams_filter",
            help="Select specific teams to focus on"
        )
        
        if selected_teams:
            tournament_data = tournament_data[tournament_data["Team"].isin(selected_teams)]
        
        # Player search
        st.subheader("👤 Player Search")
        player_query = st.text_input(
            "Search players (comma-separated, partial matches OK)", 
            value="",
            key="player_search",
            help="Example: Ahmed, Mohammed, Al-"
        )
        
        if player_query.strip():
            search_terms = [term.strip().lower() for term in player_query.split(",") if term.strip()]
            if search_terms:
                mask = pd.Series(False, index=tournament_data.index)
                player_col = tournament_data["Player"].astype(str).str.lower()
                for term in search_terms:
                    mask = mask | player_col.str.contains(term, na=False, regex=False)
                tournament_data = tournament_data[mask]
        
        # Quick filter for top players
        st.subheader("🥇 Quick Filters")
        min_goals = st.slider(
            "Minimum goals per player", 
            min_value=1, 
            max_value=int(full_tournament_data["Goals"].max()) if not full_tournament_data.empty else 5,
            value=1,
            key="min_goals_filter"
        )
        
        # Apply minimum goals filter
        player_totals = tournament_data.groupby(["Player", "Team"])["Goals"].sum()
        qualifying_players = player_totals[player_totals >= min_goals].index
        if not qualifying_players.empty:
            tournament_data = tournament_data[
                tournament_data.set_index(["Player", "Team"]).index.isin(qualifying_players)
            ].reset_index(drop=True)
        
        # Display filter summary
        st.divider()
        st.caption(f"""
        **Active Filters Summary:**
        - Division: {selected_division}
        - Teams: {len(selected_teams) if selected_teams else 'All'}
        - Player Search: {'Yes' if player_query.strip() else 'No'}
        - Min Goals: {min_goals}
        - Showing: {len(tournament_data)} records
        """)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 OVERVIEW", 
        "⚡ QUICK INSIGHTS", 
        "🏆 TEAMS", 
        "👤 PLAYERS", 
        "📈 ANALYTICS", 
        "📥 DOWNLOADS"
    ])
    
    # Calculate statistics for current data
    current_stats = calculate_tournament_stats(tournament_data)
    top_performers = get_top_performers(tournament_data, 10)
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("📊 Tournament Overview")
        
        # Display metrics
        display_metric_cards(current_stats)
        
        st.divider()
        
        # Main data table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Goal Scoring Records")
            create_enhanced_data_table(tournament_data, "records")
        
        with col2:
            if not tournament_data.empty:
                # Division distribution
                st.subheader("🏁 Division Distribution")
                division_chart = create_division_donut_chart(tournament_data)
                st.altair_chart(division_chart, use_container_width=True)
        
        # Team and player charts
        if not tournament_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Goals by Team")
                team_goals = tournament_data.groupby("Team")["Goals"].sum().reset_index()
                team_goals = team_goals.sort_values("Goals", ascending=False).head(10)
                if not team_goals.empty:
                    team_chart = create_horizontal_bar_chart(
                        team_goals, "Goals", "Team", "Top 10 Teams by Goals", "blues"
                    )
                    st.altair_chart(team_chart, use_container_width=True)
            
            with col2:
                st.subheader("⚽ Top Scorers")
                if not top_performers["players"].empty:
                    top_scorers = top_performers["players"].head(10).copy()
                    top_scorers["Display_Name"] = top_scorers["Player"] + " (" + top_scorers["Team"] + ")"
                    scorer_chart = create_horizontal_bar_chart(
                        top_scorers, "Goals", "Display_Name", "Top 10 Players by Goals", "greens"
                    )
                    st.altair_chart(scorer_chart, use_container_width=True)
    
    # TAB 2: QUICK INSIGHTS
    with tab2:
        st.header("⚡ Quick Tournament Insights")
        
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Total Goals", current_stats["total_goals"])
        with col2:
            st.metric("👥 Active Players", current_stats["total_players"])
        with col3:
            st.metric("🏆 Teams", current_stats["total_teams"])
        with col4:
            st.metric("📊 Divisions", current_stats["divisions"])
        
        st.divider()
        
        # Insights cards
        display_insights_cards(tournament_data, "Current View" if len(tournament_data) < len(full_tournament_data) else "Tournament")
        
        # Division comparison if multiple divisions
        if tournament_data["Division"].nunique() > 1:
            st.subheader("🔄 Division Comparison")
            division_comparison = create_division_comparison(tournament_data)
            if not division_comparison.empty:
                st.dataframe(
                    division_comparison,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d"),
                        "Avg_Goals": st.column_config.NumberColumn("Avg Goals", format="%.2f"),
                        "Total_Records": st.column_config.NumberColumn("Records", format="%d"),
                        "Teams": st.column_config.NumberColumn("Teams", format="%d"),
                        "Players": st.column_config.NumberColumn("Players", format="%d"),
                        "Goal_Share_Pct": st.column_config.NumberColumn("Share %", format="%.1f%%")
                    }
                )
    
    # TAB 3: TEAMS
    with tab3:
        st.header("🏆 Teams Analysis")
        
        if tournament_data.empty:
            st.info("🔍 No teams match your current filters.")
        else:
            # Teams summary table
            st.subheader("📋 Teams Summary")
            create_enhanced_data_table(tournament_data, "teams")
            
            st.divider()
            
            # Team performance chart
            st.subheader("📊 Team Performance Analysis")
            team_analysis = tournament_data.groupby(["Team", "Division"]).agg({
                "Player": "nunique",
                "Goals": "sum"
            }).reset_index()
            team_analysis.columns = ["Team", "Division", "Players", "Goals"]
            team_analysis = team_analysis.sort_values("Goals", ascending=False)
            
            if not team_analysis.empty:
                # Team goals chart
                team_chart = create_horizontal_bar_chart(
                    team_analysis.head(15), "Goals", "Team", "Team Goals Distribution", "viridis"
                )
                st.altair_chart(team_chart, use_container_width=True)
    
    # TAB 4: PLAYERS
    with tab4:
        st.header("👤 Players Analysis")
        
        if tournament_data.empty:
            st.info("🔍 No players match your current filters.")
        else:
            # Players summary table
            st.subheader("📋 Players Ranking")
            create_enhanced_data_table(tournament_data, "players")
            
            st.divider()
            
            # Player performance insights
            col1, col2 = st.columns(2)
            
            with col1:
                # Top scorers by division
                st.subheader("🥇 Top Scorers by Division")
                for division in tournament_data["Division"].unique():
                    div_data = tournament_data[tournament_data["Division"] == division]
                    div_top = div_data.groupby(["Player", "Team"])["Goals"].sum().reset_index()
                    div_top = div_top.sort_values("Goals", ascending=False).head(5)
                    
                    st.write(f"**{division}**")
                    if not div_top.empty:
                        for idx, row in div_top.iterrows():
                            st.write(f"• {row['Player']} ({row['Team']}) - {row['Goals']} goals")
                    else:
                        st.write("• No players found")
                    st.write("")
            
            with col2:
                # Player statistics
                st.subheader("📊 Player Statistics")
                player_goals = tournament_data.groupby(["Player", "Team"])["Goals"].sum()
                
                st.metric("🎯 Highest Individual Score", int(player_goals.max()) if not player_goals.empty else 0)
                st.metric("📈 Average Goals per Player", f"{player_goals.mean():.2f}" if not player_goals.empty else "0.00")
                st.metric("👥 Players with 2+ Goals", len(player_goals[player_goals >= 2]))
                st.metric("⚽ Single Goal Scorers", len(player_goals[player_goals == 1]))
    
    # TAB 5: ANALYTICS
    with tab5:
        st.header("📈 Advanced Analytics")
        
        if tournament_data.empty:
            st.info("🔍 No data available for analytics with current filters.")
        else:
            # Goals distribution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Goals Distribution")
                goals_dist_chart = create_goals_distribution_histogram(tournament_data)
                if PLOTLY_AVAILABLE:
                    st.plotly_chart(goals_dist_chart, use_container_width=True)
                else:
                    st.altair_chart(goals_dist_chart, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Team Performance Matrix")
                scatter_chart = create_advanced_scatter_plot(tournament_data)
                if PLOTLY_AVAILABLE:
                    st.plotly_chart(scatter_chart, use_container_width=True)
                else:
                    st.altair_chart(scatter_chart, use_container_width=True)
            
            st.divider()
            
            # Advanced metrics
            st.subheader("🔍 Detailed Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏆 Team Efficiency**")
                team_efficiency = tournament_data.groupby("Team").agg({
                    "Player": "nunique",
                    "Goals": "sum"
                })
                team_efficiency["Goals_per_Player"] = (team_efficiency["Goals"] / team_efficiency["Player"]).round(2)
                team_efficiency = team_efficiency.sort_values("Goals_per_Player", ascending=False)
                
                for team, data in team_efficiency.head(5).iterrows():
                    st.write(f"• **{team}**: {data['Goals_per_Player']} goals/player")
            
            with col2:
                st.markdown("**⚽ Scoring Patterns**")
                goals_patterns = tournament_data["Goals"].value_counts().sort_index()
                for goals, count in goals_patterns.items():
                    percentage = (count / len(tournament_data) * 100)
                    st.write(f"• **{goals} goal{'s' if goals != 1 else ''}**: {count} records ({percentage:.1f}%)")
            
            with col3:
                st.markdown("**🎯 Division Insights**")
                for division in tournament_data["Division"].unique():
                    div_data = tournament_data[tournament_data["Division"] == division]
                    total_goals = div_data["Goals"].sum()
                    unique_players = div_data["Player"].nunique()
                    avg_goals = total_goals / unique_players if unique_players > 0 else 0
                    st.write(f"• **{division}**:")
                    st.write(f"  - {total_goals} total goals")
                    st.write(f"  - {avg_goals:.2f} avg/player")
    
    # TAB 6: DOWNLOADS
    with tab6:
        create_download_section(full_tournament_data, tournament_data)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; color: #64748b;">
            <p>⚽ <strong>ABEER BLUESTAR SOCCER FEST 2K25</strong> Dashboard</p>
            <p>Built with Streamlit • Real-time data from Google Sheets • Updated automatically</p>
            <p>🏆 Celebrating football excellence and community spirit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ====================== APPLICATION ENTRY POINT ==================
if __name__ == "__main__":
    # Set up session state for refresh tracking
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Run the main application
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("💡 Try refreshing the page or contact support if the issue persists.")
        
        # Display error details in expander for debugging
        with st.expander("🔧 Technical Details"):
            st.exception(e)
