# ABEER BLUESTAR SOCCER FEST 2K25 — Complete Streamlit Dashboard
# Author: AI Assistant | Updated: 2025-08-29
# What's new in this build:
# - Player cards view instead of table in Players tab
# - Yellow and Red cards functionality added
# - Enhanced player profile cards with full statistics
# - Improved data processing to handle card statistics

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Optional imports with fallbacks
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None

# ====================== CONFIGURATION =============================
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== STYLING ===================================
def inject_advanced_css():
    st.markdown(
        """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root{
          /* Adjust this if the sticky tabs should sit a bit lower/higher */
          --sticky-tabs-top: 52px;
        }

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
            position: relative; /* for z layering */
            z-index: 1;
        }

        /* Keep header/toolbar visible so sidebar toggle shows */
        #MainMenu, footer, .stDeployButton,
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            display: none !important;
        }

        /* ----- STICKY TABS (freeze the pane just below the tabs) ----- */
        .block-container [data-testid="stTabs"]:first-of-type{
            position: sticky;
            top: var(--sticky-tabs-top);
            z-index: 6;                           /* above content, below dialogs */
            background: rgba(255,255,255,0.96);   /* frosted background */
            backdrop-filter: blur(8px);
            border-bottom: 1px solid #e2e8f0;
            padding-top: .25rem;
            padding-bottom: .25rem;
            margin-top: .25rem;
        }

        /* App title */
        .app-title{
            display:flex; align-items:center; justify-content:center; gap:12px;
            margin: .75rem 0 1.0rem;
        }
        .app-title .ball{
            font-size: 32px; line-height:1;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,.15));
        }
        .app-title .title{
            font-weight:700; letter-spacing:.05em;
            font-size: clamp(22px, 3.5vw, 36px);
            background: linear-gradient(45deg, #0ea5e9, #1e40af, #7c3aed);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Buttons */
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
            filter: brightness(1.05) !important;
        }

        /* Dataframes */
        .stDataFrame {
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important;
        }

        /* Metric cards */
        .metric-container {
            background: linear-gradient(135deg, rgba(14,165,233,.1), rgba(59,130,246,.05));
            border-radius: 15px;
            padding: 1.5rem;
            border-left: 4px solid #0ea5e9;
            box-shadow: 0 4px 20px rgba(14,165,233,.1);
            transition: transform .2s ease;
        }
        .metric-container:hover { transform: translateY(-3px); }

        /* Player cards hover effect */
        .player-card {
            transition: all 0.3s ease;
        }
        .player-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
        }

        /* Sidebar status pill */
        .status-pill { padding:.5rem .75rem; border-radius:.6rem; font-size:.85rem; margin-top:.5rem; }
        .status-ok  { background:#ecfeff; border-left:4px solid #06b6d4; color:#155e75; }
        .status-warn{ background:#fef9c3; border-left:4px solid #f59e0b; color:#713f12; }
        .status-err { background:#fee2e2; border-left:4px solid #ef4444; color:#7f1d1d; }

        @media (max-width: 768px) {
            .block-container { padding: 1rem .5rem; margin: .5rem; width: 95vw; max-width: 95vw; }
            .app-title .ball{font-size:24px;}
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Altair theme
    alt.themes.register(
        "tournament_theme",
        lambda: {
            "config": {
                "view": {"stroke": "transparent", "fill": "white"},
                "background": "white",
                "title": {"font": "Poppins", "fontSize": 18, "color": "#1e293b", "fontWeight": 600},
                "axis": {
                    "labelColor": "#64748b",
                    "titleColor": "#374151",
                    "gridColor": "#f1f5f9",
                    "labelFont": "Poppins",
                    "titleFont": "Poppins",
                },
                "legend": {
                    "labelColor": "#64748b",
                    "titleColor": "#374151",
                    "labelFont": "Poppins",
                    "titleFont": "Poppins",
                },
                "range": {
                    "category": [
                        "#0ea5e9","#34d399","#60a5fa","#f59e0b",
                        "#f87171","#a78bfa","#fb7185","#4ade80",
                    ]
                },
            }
        },
    )
    alt.themes.enable("tournament_theme")

def notify(msg: str, kind: str = "ok"):
    cls = {"ok": "status-ok", "warn": "status-warn", "err": "status-err"}.get(kind, "status-ok")
    st.markdown(f'<div class="status-pill {cls}">{msg}</div>', unsafe_allow_html=True)

# ---------- Robust Trophy watermark (DOM element, not :before) ----------
def add_world_cup_watermark(*, image_path: str | None = None,
                            image_url: str | None = None,
                            opacity: float = 0.08,
                            size: str = "68vmin",
                            y_offset: str = "6vh"):
    """Shows a big, faint trophy behind the whole app."""
    if image_path:
        ext = "svg+xml" if image_path.lower().endswith(".svg") else "png"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        bg = f"url('data:image/{ext};base64,{b64}')"
    elif image_url:
        bg = f"url('{image_url}')"
    else:
        bg = "url('https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg')"

    st.markdown(
        f"""
    <style>
      #wc-trophy {{
        position: fixed;
        inset: 0;
        background-image: {bg};
        background-repeat: no-repeat;
        background-position: center {y_offset};
        background-size: {size};
        opacity: {opacity};
        pointer-events: none;
        z-index: 0; /* below content; .block-container has z-index:1 */
      }}
    </style>
    <div id="wc-trophy"></div>
    """,
        unsafe_allow_html=True,
    )

# ====================== DATA PROCESSING ===========================
def parse_xlsx_without_dependencies(file_bytes: bytes) -> pd.DataFrame:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                root = ET.parse(f).getroot()
                for si in root.findall(".//main:si", ns):
                    text = "".join(t.text or "" for t in si.findall(".//main:t", ns))
                    shared.append(text)

        if "xl/worksheets/sheet1.xml" not in z.namelist():
            return pd.DataFrame()
        with z.open("xl/worksheets/sheet1.xml") as f:
            root = ET.parse(f).getroot()
            sheet = root.find("main:sheetData", ns)
            if sheet is None:
                return pd.DataFrame()
            rows, max_col = [], 0
            for row in sheet.findall("main:row", ns):
                rd = {}
                for cell in row.findall("main:c", ns):
                    ref = cell.attrib.get("r", "A1")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    col_idx = 0
                    for ch in col_letters:
                        col_idx = col_idx * 26 + (ord(ch) - 64)
                    col_idx -= 1
                    ctype = cell.attrib.get("t")
                    v = cell.find("main:v", ns)
                    val = v.text if v is not None else None
                    if ctype == "s" and val is not None:
                        i = int(val)
                        if 0 <= i < len(shared):
                            val = shared[i]
                    rd[col_idx] = val
                    max_col = max(max_col, col_idx)
                rows.append(rd)

    if not rows:
        return pd.DataFrame()
    matrix = [[r.get(i) for i in range(max_col + 1)] for r in rows]
    return pd.DataFrame(matrix)

def safe_read_excel(file_source) -> pd.DataFrame:
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        file_bytes = file_source.read()

    try:
        return pd.read_excel(BytesIO(file_bytes), header=None)
    except Exception:
        return parse_xlsx_without_dependencies(file_bytes)

def find_division_columns(raw_df: pd.DataFrame):
    b_col, a_col = None, None
    for row_idx in range(min(3, len(raw_df))):
        row = raw_df.iloc[row_idx].astype(str).str.strip().str.lower()
        for col_idx, cell in row.items():
            if "b division" in cell and b_col is None:
                b_col = col_idx
            elif "a division" in cell and a_col is None:
                a_col = col_idx
    if b_col is None and a_col is None:
        b_col = 0
        a_col = 6 if raw_df.shape[1] >= 12 else (5 if raw_df.shape[1] >= 10 else None)
    return b_col, a_col

def process_tournament_data(xlsx_bytes: bytes) -> pd.DataFrame:
    raw_df = safe_read_excel(xlsx_bytes)
    if raw_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals", "Yellow_Cards", "Red_Cards", "Appearances"])

    b_start, a_start = find_division_columns(raw_df)
    header_row = 1 if len(raw_df) > 1 else 0
    data_start_row = header_row + 1

    processed = []

    def extract_div(start_col: int | None, name: str):
        if start_col is None:
            return
        
        # Determine how many columns we have for this division
        # Expected format: Team | Player | Goals | Yellow Cards | Red Cards | Appearances
        cols_needed = 6
        if start_col + cols_needed > raw_df.shape[1]:
            # Fallback to minimal columns if not enough data
            cols_needed = min(3, raw_df.shape[1] - start_col)
        
        df = raw_df.iloc[data_start_row:, start_col:start_col + cols_needed].copy()
        
        # Set column names based on available columns
        if cols_needed >= 6:
            df.columns = ["Team", "Player", "Goals", "Yellow_Cards", "Red_Cards", "Appearances"]
        elif cols_needed >= 3:
            df.columns = ["Team", "Player", "Goals"] + [f"Extra_{i}" for i in range(cols_needed - 3)]
            # Fill missing columns with defaults
            if "Yellow_Cards" not in df.columns:
                df["Yellow_Cards"] = 0
            if "Red_Cards" not in df.columns:
                df["Red_Cards"] = 0
            if "Appearances" not in df.columns:
                df["Appearances"] = 1  # Default to 1 appearance if they have goals
        else:
            return
        
        # Clean the data
        df = df.dropna(subset=["Team", "Player", "Goals"])
        df["Goals"] = pd.to_numeric(df["Goals"], errors="coerce")
        df = df.dropna(subset=["Goals"])
        df["Goals"] = df["Goals"].astype(int)
        
        # Process card data
        for card_col in ["Yellow_Cards", "Red_Cards", "Appearances"]:
            if card_col in df.columns:
                df[card_col] = pd.to_numeric(df[card_col], errors="coerce").fillna(0).astype(int)
        
        df["Division"] = name
        processed.extend(df.to_dict("records"))

    extract_div(b_start, "B Division")
    extract_div(a_start, "A Division")

    if not processed:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals", "Yellow_Cards", "Red_Cards", "Appearances"])
    
    out = pd.DataFrame(processed)
    return out[["Division", "Team", "Player", "Goals", "Yellow_Cards", "Red_Cards", "Appearances"]]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.content:
            raise ValueError("Downloaded file is empty")
        return process_tournament_data(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals", "Yellow_Cards", "Red_Cards", "Appearances"])

# ====================== ANALYTICS FUNCTIONS =======================
def calculate_tournament_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_goals": 0,
            "total_players": 0,
            "total_teams": 0,
            "divisions": 0,
            "avg_goals_per_team": 0,
            "top_scorer_goals": 0,
            "competitive_balance": 0,
            "total_yellow_cards": 0,
            "total_red_cards": 0,
            "total_appearances": 0,
        }

    player_totals = df.groupby(["Player", "Team", "Division"]).agg({
        "Goals": "sum",
        "Yellow_Cards": "sum",
        "Red_Cards": "sum",
        "Appearances": "sum"
    }).reset_index()
    
    team_totals = df.groupby(["Team", "Division"])["Goals"].sum().reset_index()

    return {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(player_totals),
        "total_teams": len(team_totals),
        "divisions": df["Division"].nunique(),
        "avg_goals_per_team": round(df["Goals"].sum() / max(1, len(team_totals)), 2),
        "top_scorer_goals": int(player_totals["Goals"].max()) if not player_totals.empty else 0,
        "competitive_balance": round(team_totals["Goals"].std(), 2) if len(team_totals) > 1 else 0,
        "total_yellow_cards": int(df["Yellow_Cards"].sum()) if "Yellow_Cards" in df.columns else 0,
        "total_red_cards": int(df["Red_Cards"].sum()) if "Red_Cards" in df.columns else 0,
        "total_appearances": int(df["Appearances"].sum()) if "Appearances" in df.columns else 0,
    }

def get_top_performers(df: pd.DataFrame, top_n: int = 10) -> dict:
    if df.empty:
        return {"players": pd.DataFrame(), "teams": pd.DataFrame()}
    
    # Aggregate player stats
    top_players = (
        df.groupby(["Player", "Team", "Division"]).agg({
            "Goals": "sum",
            "Yellow_Cards": "sum",
            "Red_Cards": "sum", 
            "Appearances": "sum"
        }).reset_index()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        .head(top_n)
    )
    
    top_teams = (
        df.groupby(["Team", "Division"])["Goals"]
        .sum()
        .reset_index()
        .sort_values("Goals", ascending=False)
        .head(top_n)
    )
    return {"players": top_players, "teams": top_teams}

def create_division_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    agg_dict = {
        "Goals": "sum",
        "Goals_mean": ("Goals", "mean"),
        "Records": ("Goals", "count"),
        "Teams": ("Team", "nunique"),
        "Players": ("Player", "nunique"),
    }
    
    # Add card columns if they exist
    if "Yellow_Cards" in df.columns:
        agg_dict["Yellow_Cards"] = ("Yellow_Cards", "sum")
    if "Red_Cards" in df.columns:
        agg_dict["Red_Cards"] = ("Red_Cards", "sum")
    if "Appearances" in df.columns:
        agg_dict["Appearances"] = ("Appearances", "sum")
    
    division_stats = (
        df.groupby("Division")
        .agg(agg_dict)
        .round(2)
        .reset_index()
    )
    
    # Rename columns
    base_cols = ["Division", "Total_Goals", "Avg_Goals", "Total_Records", "Teams", "Players"]
    extra_cols = [f"Total_{col}" for col in ["Yellow_Cards", "Red_Cards", "Appearances"] if col in df.columns]
    division_stats.columns = base_cols + extra_cols
    
    total_goals = division_stats["Total_Goals"].sum()
    division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"] / total_goals * 100).round(1) if total_goals else 0
    return division_stats

# ====================== PLAYER CARD FUNCTIONS =====================
def create_player_profile_cards(df: pd.DataFrame):
    """Create individual player profile cards similar to the screenshot"""
    if df.empty:
        st.info("🔍 No players match your current filters.")
        return
    
    # Get player summary data
    player_stats = df.groupby(["Player", "Team", "Division"]).agg({
        "Goals": "sum",
        "Yellow_Cards": "sum",
        "Red_Cards": "sum",
        "Appearances": "sum"
    }).reset_index()
    
    player_stats = player_stats.sort_values(["Goals", "Player"], ascending=[False, True])
    
    # Create cards in a responsive grid layout
    cols_per_row = 4
    
    # Add some spacing and a container
    st.markdown('<div style="margin: 1rem 0;">', unsafe_allow_html=True)
    
    # Process players in batches for grid layout
    for i in range(0, len(player_stats), cols_per_row):
        cols = st.columns(cols_per_row)
        batch = player_stats.iloc[i:i + cols_per_row]
        
        for j, (_, player) in enumerate(batch.iterrows()):
            if j < len(cols):
                with cols[j]:
                    create_single_player_card(player)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_single_player_card(player):
    """Create a single player profile card"""
    # Determine division color
    division_color = "#0ea5e9" if player["Division"] == "B Division" else "#f59e0b"
    division_bg = "14, 165, 233" if player["Division"] == "B Division" else "245, 158, 11"
    
    # Get stats
    goals = int(player["Goals"])
    yellow_cards = int(player["Yellow_Cards"])
    red_cards = int(player["Red_Cards"])
    appearances = int(player["Appearances"])
    
    # Simple award logic
    awards = []
    if goals >= 5:
        awards.append("🥇 Top Scorer")
    elif goals >= 3:
        awards.append("⚽ Goal Machine")
    elif goals >= 1:
        awards.append("🎯 Scorer")
    
    if red_cards == 0 and yellow_cards <= 1:
        awards.append("🏅 Fair Play")
    
    if red_cards == 0 and yellow_cards == 0:
        awards.append("✨ Clean Record")
    
    award_text = " • ".join(awards) if awards else "No awards"
    
    # Create the card
    st.markdown(
        f"""
        <div class="player-card" style="
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-top: 4px solid {division_color};
            margin-bottom: 1rem;
            height: 320px;
            display: flex;
            flex-direction: column;
        ">
            <!-- Player Name -->
            <div style="
                font-size: 1.1rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 0.5rem;
                text-align: center;
                line-height: 1.3;
                min-height: 2.6rem;
                display: flex;
                align-items: center;
                justify-content: center;
            ">{player["Player"]}</div>
            
            <!-- Team and Division -->
            <div style="
                color: white;
                font-size: 0.85rem;
                text-align: center;
                margin-bottom: 1.2rem;
                padding: 0.4rem 0.75rem;
                background: rgba({division_bg}, 0.9);
                border-radius: 20px;
                font-weight: 500;
            ">{player["Team"]} • {player["Division"]}</div>
            
            <!-- Main Goals Display -->
            <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="text-align: center; margin: 1rem 0;">
                    <div style="font-size: 2.2rem; font-weight: 700; color: {division_color}; margin-bottom: 0.25rem;">
                        ⚽ {goals}
                    </div>
                    <div style="color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.8rem;">
                        Goal{"s" if goals != 1 else ""}
                    </div>
                </div>
                
                <!-- Statistics Grid -->
                <div style="
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 0.75rem;
                    margin-top: 1rem;
                    padding-top: 1rem;
                    border-top: 1px solid #e2e8f0;
                ">
                    <div style="text-align: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 0.25rem;">📈</div>
                        <div style="font-size: 0.7rem; color: #64748b; margin-bottom: 0.25rem;">Appearances</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #374151;">{appearances}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 0.25rem;">🟨</div>
                        <div style="font-size: 0.7rem; color: #64748b; margin-bottom: 0.25rem;">Yellow Cards</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #374151;">{yellow_cards}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 0.25rem;">🟥</div>
                        <div style="font-size: 0.7rem; color: #64748b; margin-bottom: 0.25rem;">Red Cards</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #374151;">{red_cards}</div>
                    </div>
                </div>
            </div>
            
            <!-- Awards Section -->
            <div style="
                text-align: center;
                margin-top: 1rem;
                padding-top: 0.75rem;
                border-top: 1px solid #e2e8f0;
                color: #64748b;
                font-size: 0.75rem;
                line-height: 1.3;
                min-height: 2rem;
                display: flex;
                align-items: center;
                justify-content: center;
            ">{award_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ====================== VISUALIZATION FUNCTIONS ===================
def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_scheme: str = "blues") -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")
    max_val = int(df[x_col].max()) if not df.empty else 1
    tick_values = list(range(0, max_val + 1)) if max_val <= 50 else None
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.85, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(f"{x_col}:Q", title="Goals", axis=alt.Axis(format="d", tickMinStep=1, values=tick_values, gridOpacity=0.3), scale=alt.Scale(domainMin=0, nice=False)),
            y=alt.Y(f"{y_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            color=alt.Color(f"{x_col}:Q", scale=alt.Scale(scheme=color_scheme), legend=None),
            tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col), alt.Tooltip(f"{x_col}:Q", title="Goals", format="d")],
        )
        .properties(height=max(300, min(600, len(df) * 25)), title=alt.TitleParams(text=title, fontSize=16, anchor="start", fontWeight=600))
        .resolve_scale(color="independent")
    )

def create_division_donut_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")

    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    sel = alt.selection_single(fields=["Division"], empty="none")

    base = (
        alt.Chart(division_data)
        .add_selection(sel)
        .properties(
            width=300,
            height=300,
            title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600),
        )
    )

    outer = (
        base.mark_arc(innerRadius=60, outerRadius=120, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("Goals:Q", title="Goals"),
            color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]), title="Division"),
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.8)),
            tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")],
        )
    )

    center_text = (
        base.mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#1e293b")
        .encode(text=alt.value(f"Total\n{int(division_data['Goals'].sum())}"))
    )

    return outer + center_text

def create_advanced_scatter_plot(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")

    team_stats = df.groupby(["Team", "Division"]).agg(Players=("Player", "nunique"), Goals=("Goals", "sum")).reset_index()

    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            team_stats, x="Players", y="Goals", color="Division", size="Goals",
            hover_name="Team", hover_data={"Players": True, "Goals": True},
            title="Team Performance: Players vs Total Goals",
        )
        fig.update_traces(marker=dict(sizemode="diameter", sizemin=8, sizemax=30, line=dict(width=2, color="white"), opacity=0.85))
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Poppins", size=12),
            title=dict(font=dict(size=16, color="#1e293b")),
            xaxis=dict(title="Number of Players in Team", gridcolor="#f1f5f9", zeroline=False),
            yaxis=dict(title="Total Goals", gridcolor="#f1f5f9", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=400,
        )
        return fig

    return (
        alt.Chart(team_stats)
        .mark_circle(size=100, opacity=0.85)
        .encode(
            x=alt.X("Players:Q", title="Number of Players in Team"),
            y=alt.Y("Goals:Q", title="Total Goals"),
            color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]), title="Division"),
            size=alt.Size("Goals:Q", legend=None),
            tooltip=["Team:N", "Division:N", "Players:Q", "Goals:Q"],
        )
        .properties(title="Team Performance: Players vs Total Goals", height=400)
    )

def create_goals_distribution_histogram(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")

    player_goals = df.groupby(["Player", "Team"])["Goals"].sum().values
    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            data=[go.Histogram(x=player_goals, nbinsx=max(1, len(set(player_goals))), marker_line_color="white", marker_line_width=1.5, opacity=0.85, hovertemplate="<b>%{x} Goals</b><br>%{y} Players<extra></extra>")]
        )
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
            height=400,
        )
        return fig

    hist_df = pd.DataFrame({"Goals": player_goals})
    return (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
            y=alt.Y("count():Q", title="Number of Players"),
            tooltip=[alt.Tooltip("Goals:Q", bin=True, title="Goals"), alt.Tooltip("count():Q", title="Players")],
        )
        .properties(title="Distribution of Goals per Player", height=400)
    )

def create_cards_analysis_chart(df: pd.DataFrame):
    """Create visualization for yellow and red cards analysis"""
    if df.empty or ("Yellow_Cards" not in df.columns and "Red_Cards" not in df.columns):
        return alt.Chart(pd.DataFrame({"note": ["No card data available"]})).mark_text().encode(text="note:N")
    
    # Aggregate cards by team
    team_cards = df.groupby(["Team", "Division"]).agg({
        "Yellow_Cards": "sum",
        "Red_Cards": "sum"
    }).reset_index()
    
    # Reshape for visualization
    card_data = []
    for _, row in team_cards.iterrows():
        card_data.extend([
            {"Team": row["Team"], "Division": row["Division"], "Card_Type": "Yellow", "Count": row["Yellow_Cards"]},
            {"Team": row["Team"], "Division": row["Division"], "Card_Type": "Red", "Count": row["Red_Cards"]}
        ])
    
    card_df = pd.DataFrame(card_data)
    
    return (
        alt.Chart(card_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("Count:Q", title="Number of Cards"),
            y=alt.Y("Team:N", sort="-x", title="Team"),
            color=alt.Color("Card_Type:N", 
                          scale=alt.Scale(domain=["Yellow", "Red"], range=["#fbbf24", "#ef4444"]),
                          title="Card Type"),
            tooltip=["Team:N", "Division:N", "Card_Type:N", "Count:Q"]
        )
        .properties(
            title="Disciplinary Records by Team",
            height=max(300, min(500, len(team_cards) * 20))
        )
    )

# ====================== UI COMPONENTS =============================
def display_metric_cards(stats: dict):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)

def display_enhanced_metric_cards(stats: dict):
    """Enhanced metrics including cards data"""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#fbbf24;margin-bottom:.5rem;">{stats['total_yellow_cards']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">YELLOW CARDS</div></div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#ef4444;margin-bottom:.5rem;">{stats['total_red_cards']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">RED CARDS</div></div>""", unsafe_allow_html=True)

def display_insights_cards(df: pd.DataFrame, scope: str = "Tournament"):
    if df.empty:
        st.info("No data available for insights.")
        return

    stats = calculate_tournament_stats(df)
    top_perf = get_top_performers(df, 5)
    division_cmp = create_division_comparison(df)

    if not top_perf["teams"].empty:
        tt = top_perf["teams"].iloc[0]
        top_team_name, top_team_goals = tt["Team"], int(tt["Goals"])
    else:
        top_team_name, top_team_goals = "—", 0

    if not top_perf["players"].empty:
        tp = top_perf["players"].iloc[0]
        top_player_name, top_player_team, top_player_goals = tp["Player"], tp["Team"], int(tp["Goals"])
    else:
        top_player_name, top_player_team, top_player_goals = "—", "—", 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #0ea5e9;margin-bottom:1rem;">
            <div style="font-weight:600;color:#0ea5e9;margin-bottom:.5rem;font-size:1.1rem;">🏆 Top Performing Team</div>
            <div style="color:#374151;line-height:1.5;"><strong>{top_team_name}</strong> leads with <strong>{top_team_goals} goals</strong>.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if not division_cmp.empty and len(division_cmp) > 1:
            def get_vals(div):
                s = division_cmp[division_cmp["Division"] == div]
                return (int(s["Total_Goals"].iloc[0]), s["Goal_Share_Pct"].iloc[0]) if not s.empty else (0, 0)
            b_goals, b_pct = get_vals("B Division")
            a_goals, a_pct = get_vals("A Division")

            st.markdown(
                f"""
            <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #f59e0b;margin-bottom:1rem;">
                <div style="font-weight:600;color:#f59e0b;margin-bottom:.5rem;font-size:1.1rem;">🎯 Division Performance</div>
                <div style="color:#374151;line-height:1.5;">
                    <strong>B Division:</strong> {b_goals} goals ({b_pct}%)<br>
                    <strong>A Division:</strong> {a_goals} goals ({a_pct}%)<br>
                    {"B Division shows higher scoring activity" if b_goals > a_goals else ("A Division leads in goal production" if a_goals > b_goals else "Both divisions are evenly matched")}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            f"""
        <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #34d399;margin-bottom:1rem;">
            <div style="font-weight:600;color:#34d399;margin-bottom:.5rem;font-size:1.1rem;">⚽ Leading Scorer</div>
            <div style="color:#374151;line-height:1.5;"><strong>{top_player_name}</strong> ({top_player_team}) with <strong>{top_player_goals} goals</strong>.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        player_goals = df.groupby(["Player", "Team"])["Goals"].sum()
        goals_1 = int((player_goals == 1).sum())
        goals_2_plus = int((player_goals >= 2).sum())

        # Add disciplinary insight
        total_yellow = stats.get("total_yellow_cards", 0)
        total_red = stats.get("total_red_cards", 0)
        discipline_text = f"{total_yellow} yellow, {total_red} red cards issued"

        st.markdown(
            f"""
        <div style="background:white;padding:1.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.05);border-left:4px solid #a78bfa;margin-bottom:1rem;">
            <div style="font-weight:600;color:#a78bfa;margin-bottom:.5rem;font-size:1.1rem;">📊 Tournament Discipline</div>
            <div style="color:#374151;line-height:1.5;">
                <strong>{goals_1} players</strong> scored 1 goal, <strong>{goals_2_plus} players</strong> scored 2+ goals<br>
                <strong>Fair play:</strong> {discipline_text}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

def create_enhanced_data_table(df: pd.DataFrame, table_type: str = "records"):
    if df.empty:
        st.info(f"No {table_type} data available with current filters.")
        return

    if table_type == "records":
        display_df = df.sort_values("Goals", ascending=False).reset_index(drop=True)
        column_config = {
            "Division": st.column_config.TextColumn("Division", width="small"),
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "Player": st.column_config.TextColumn("Player", width="large"),
            "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
        }
        
        # Add card columns if they exist
        if "Yellow_Cards" in display_df.columns:
            column_config["Yellow_Cards"] = st.column_config.NumberColumn("🟨", format="%d", width="small")
        if "Red_Cards" in display_df.columns:
            column_config["Red_Cards"] = st.column_config.NumberColumn("🟥", format="%d", width="small")
        if "Appearances" in display_df.columns:
            column_config["Appearances"] = st.column_config.NumberColumn("📈", format="%d", width="small")
            
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

    elif table_type == "teams":
        teams_summary = df.groupby(["Team", "Division"]).agg({
            "Players": ("Player", "nunique"),
            "Total_Goals": ("Goals", "sum"),
            "Yellow_Cards": ("Yellow_Cards", "sum"),
            "Red_Cards": ("Red_Cards", "sum")
        }).reset_index()
        
        top_rows = []
        for team in teams_summary["Team"].unique():
            team_data = df[df["Team"] == team]
            if team_data.empty:
                continue
            s = team_data.groupby("Player")["Goals"].sum()
            top_rows.append({"Team": team, "Top_Scorer": s.idxmax(), "Top_Scorer_Goals": int(s.max())})
        top_df = pd.DataFrame(top_rows)
        teams_display = teams_summary.merge(top_df, on="Team", how="left").sort_values("Total_Goals", ascending=False)

        st.dataframe(
            teams_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small"),
                "Yellow_Cards": st.column_config.NumberColumn("🟨 Cards", format="%d", width="small"),
                "Red_Cards": st.column_config.NumberColumn("🟥 Cards", format="%d", width="small"),
                "Top_Scorer": st.column_config.TextColumn("Top Scorer", width="large"),
                "Top_Scorer_Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
            },
        )

def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
            
            # Enhanced team analysis with card data
            team_agg = {
                "Unique_Players": ("Player", "nunique"),
                "Total_Records": ("Goals", "count"),
                "Total_Goals": ("Goals", "sum"),
                "Avg_Goals": ("Goals", "mean"),
                "Max_Goals": ("Goals", "max")
            }
            if "Yellow_Cards" in filtered_df.columns:
                team_agg["Total_Yellow_Cards"] = ("Yellow_Cards", "sum")
            if "Red_Cards" in filtered_df.columns:
                team_agg["Total_Red_Cards"] = ("Red_Cards", "sum")
            if "Appearances" in filtered_df.columns:
                team_agg["Total_Appearances"] = ("Appearances", "sum")
                
            teams = (
                filtered_df.groupby(["Team", "Division"])
                .agg(team_agg)
                .round(2)
                .reset_index()
            )
            z.writestr("03_teams_detailed_analysis.csv", teams.to_csv(index=False))
            
            # Enhanced player ranking
            players = filtered_df.groupby(["Player", "Team", "Division"]).agg({
                "Goals": "sum",
                "Yellow_Cards": "sum",
                "Red_Cards": "sum",
                "Appearances": "sum"
            }).reset_index()
            players = players.sort_values(["Goals", "Player"], ascending=[False, True])
            players.insert(0, "Rank", range(1, len(players) + 1))
            z.writestr("04_players_ranking.csv", players.to_csv(index=False))
            
            div_cmp = create_division_comparison(filtered_df)
            if not div_cmp.empty:
                z.writestr("05_division_comparison.csv", div_cmp.to_csv(index=False))
            stats = calculate_tournament_stats(filtered_df)
            z.writestr("06_tournament_statistics.csv", pd.DataFrame([stats]).to_csv(index=False))
        z.writestr("README.txt", f"ABEER BLUESTAR SOCCER FEST 2K25 - Data Package\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_download_section(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.subheader("📥 Download Reports")
    st.caption("**Full** = all data ignoring filters, **Filtered** = current view with active filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Data Exports")
        if not full_df.empty:
            st.download_button(
                label="⬇️ Download FULL Dataset (CSV)",
                data=full_df.to_csv(index=False),
                file_name=f"tournament_full_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download complete tournament data",
            )
        if not filtered_df.empty:
            st.download_button(
                label="⬇️ Download FILTERED Dataset (CSV)",
                data=filtered_df.to_csv(index=False),
                file_name=f"tournament_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download data with current filters applied",
            )

    with col2:
        st.subheader("🏆 Summary Reports")
        if not filtered_df.empty:
            # Enhanced teams summary with card data
            team_agg = {"Players_Count": ("Player", "nunique"), "Total_Goals": ("Goals", "sum")}
            if "Yellow_Cards" in filtered_df.columns:
                team_agg["Yellow_Cards"] = ("Yellow_Cards", "sum")
            if "Red_Cards" in filtered_df.columns:
                team_agg["Red_Cards"] = ("Red_Cards", "sum")
                
            teams_summary = (
                filtered_df.groupby(["Team", "Division"]).agg(team_agg).reset_index()
            ).sort_values("Total_Goals", ascending=False)
            
            st.download_button(
                label="⬇️ Download TEAMS Summary (CSV)",
                data=teams_summary.to_csv(index=False),
                file_name=f"teams_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download team performance summary",
            )

            # Enhanced players summary
            players_summary = filtered_df.groupby(["Player", "Team", "Division"]).agg({
                "Goals": "sum",
                "Yellow_Cards": "sum",
                "Red_Cards": "sum",
                "Appearances": "sum"
            }).reset_index()
            players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
            players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
            st.download_button(
                label="⬇️ Download PLAYERS Summary (CSV)",
                data=players_summary.to_csv(index=False),
                file_name=f"players_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download top scorers summary",
            )

    with col3:
        st.subheader("📦 Complete Package")
        if st.button("📦 Generate Complete Report Package", help="Generate ZIP with all reports and analytics"):
            z = create_comprehensive_zip_report(full_df, filtered_df)
            st.download_button(
                label="⬇️ Download Complete Package (ZIP)",
                data=z,
                file_name=f"tournament_complete_package_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                help="Download ZIP containing all data, summaries, and analytics",
            )

# ====================== MAIN APPLICATION ==========================
def main():
    inject_advanced_css()

    # Title
    st.markdown("""
<div class="app-title">
  <span class="ball">⚽</span>
  <span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span>
</div>
""", unsafe_allow_html=True)

    # Trophy background (URL or local file)
    add_world_cup_watermark(
        image_url="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg",
        opacity=0.10,
        size="68vmin",
        y_offset="6vh"
    )

    GOOGLE_SHEETS_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")

        if st.button("🔄 Refresh Tournament Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notify("Data cache cleared. Reloading…", "ok")
            st.rerun()

        last_refresh = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.caption(f"🕒 Last refreshed: {last_refresh}")
        st.divider()

        # Load data
        with st.spinner("📡 Loading tournament data…"):
            tournament_data = fetch_tournament_data(GOOGLE_SHEETS_URL)

        if tournament_data.empty:
            notify("No tournament data available. Check the published sheet link/permissions.", "err")
            st.stop()

        full_tournament_data = tournament_data.copy()

        st.header("🔍 Data Filters")

        # Division filter
        division_options = ["All Divisions"] + sorted(tournament_data["Division"].unique().tolist())
        selected_division = st.selectbox("📊 Division", division_options, key="division_filter")
        if selected_division != "All Divisions":
            tournament_data = tournament_data[tournament_data["Division"] == selected_division]

        # Team filter
        available_teams = sorted(tournament_data["Team"].unique().tolist())
        selected_teams = st.multiselect(
            "🏆 Teams (optional)",
            available_teams,
            key="teams_filter",
            help="Select specific teams to focus on",
            placeholder="Type to search teams…",
        )
        if selected_teams:
            tournament_data = tournament_data[tournament_data["Team"].isin(selected_teams)]

        # Player search (type-to-search list)
        st.subheader("👤 Player Search")
        player_names = sorted(tournament_data["Player"].dropna().astype(str).unique().tolist())
        selected_players = st.multiselect(
            "Type to search and select players",
            options=player_names,
            default=[],
            key="players_filter",
            placeholder="Start typing a player name…",
            help="Type to search. You can select one or multiple players.",
        )
        if selected_players:
            tournament_data = tournament_data[tournament_data["Player"].isin(selected_players)]
            
        # Card filters
        st.subheader("📋 Disciplinary Filters")
        yellow_card_filter = st.checkbox("Show only players with yellow cards", key="yellow_filter")
        red_card_filter = st.checkbox("Show only players with red cards", key="red_filter")
        
        if yellow_card_filter and "Yellow_Cards" in tournament_data.columns:
            tournament_data = tournament_data[tournament_data["Yellow_Cards"] > 0]
        if red_card_filter and "Red_Cards" in tournament_data.columns:
            tournament
