# ABEER BLUESTAR SOCCER FEST 2K25 — Complete Streamlit Dashboard
# Author: AI Assistant | Updated: 2025-08-28
# - Sticky tabs, frosted UI
# - Player search and filters
# - ⚠️ No Pass_Completion anywhere
# - NEW: Player Profiles as cards (no tables needed)
# - No scatter charts used

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

# Optional imports with fallbacks (used for histogram only)
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None

# ====================== CONFIGURATION =============================
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data sources
GOOGLE_SHEETS_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
)
# Optional: separate player master registry (CSV/XLSX). Leave empty "" if not used.
PLAYER_MASTER_URL = ""  # e.g. "https://docs.google.com/.../pub?output=csv"

# ====================== STYLING ===================================
def inject_advanced_css():
    st.markdown(
        """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root{ --sticky-tabs-top: 52px; }
        .stApp {
            font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .block-container {
            padding-top: 0.5rem; padding-bottom: 2rem;
            max-width: 98vw; width: 98vw;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            margin: 1rem auto;
            position: relative; z-index: 1;
        }
        #MainMenu, footer, .stDeployButton,
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] { display: none !important; }

        /* Sticky first tabs block */
        .block-container [data-testid="stTabs"]:first-of-type{
            position: sticky; top: var(--sticky-tabs-top);
            z-index: 6; background: rgba(255,255,255,0.96);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid #e2e8f0;
            padding-top: .25rem; padding-bottom: .25rem; margin-top: .25rem;
        }

        .app-title{ display:flex; align-items:center; justify-content:center; gap:12px; margin:.75rem 0 1.0rem; }
        .app-title .ball{ font-size:32px; line-height:1; filter: drop-shadow(0 2px 4px rgba(0,0,0,.15)); }
        .app-title .title{
            font-weight:700; letter-spacing:.05em; font-size: clamp(22px, 3.5vw, 36px);
            background: linear-gradient(45deg, #0ea5e9, #1e40af, #7c3aed);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
            color: white !important; border: 0 !important; border-radius: 12px !important;
            padding: 0.6rem 1.2rem !important; font-weight: 600 !important; font-size: 0.9rem !important;
            transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important; filter: brightness(1.05) !important;
        }
        .stDataFrame { border-radius: 15px !important; overflow: hidden !important; box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important; }
        .metric-container {
            background: linear-gradient(135deg, rgba(14,165,233,.1), rgba(59,130,246,.05));
            border-radius: 15px; padding: 1.5rem; border-left: 4px solid #0ea5e9;
            box-shadow: 0 4px 20px rgba(14,165,233,.1); transition: transform .2s ease;
        }
        .metric-container:hover { transform: translateY(-3px); }
        .status-pill { padding:.5rem .75rem; border-radius:.6rem; font-size:.85rem; margin-top:.5rem; }
        .status-ok  { background:#ecfeff; border-left:4px solid #06b6d4; color:#155e75; }
        .status-warn{ background:#fef9c3; border-left:4px solid #f59e0b; color:#713f12; }
        .status-err { background:#fee2e2; border-left:4px solid #ef4444; color:#7f1d1d; }

        /* ===== Player profile cards ===== */
        .player-grid{
          display:grid; gap:16px;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          margin-top: .5rem;
        }
        .player-card{
          background:#0f172a; color:#e2e8f0;
          border-radius:16px; overflow:hidden;
          box-shadow:0 10px 24px rgba(2,6,23,.35);
          border:1px solid rgba(148,163,184,.15);
        }
        .pc-top{ display:grid; grid-template-columns:120px 1fr; gap:16px; padding:16px; align-items:center;}
        .pc-fig{
          background:#d9e6a5; height:120px; border-radius:12px;
          display:flex; align-items:center; justify-content:center; color:#0f172a; font-weight:700;
        }
        .pc-name{ font-size:1.1rem; font-weight:700; letter-spacing:.03em; }
        .pc-sub{ opacity:.85; font-size:.85rem; }
        .pc-body{ padding:16px; border-top:1px solid rgba(148,163,184,.15);}
        .pc-row{ display:flex; align-items:center; gap:10px; margin:.4rem 0;}
        .pc-label{ width:140px; font-size:.82rem; opacity:.85;}
        .pc-bar{
          flex:1; height:16px; background:#1f2937; border-radius:10px; overflow:hidden; position:relative;
          border:1px solid rgba(148,163,184,.12);
        }
        .pc-bar > span{
          display:block; height:100%; background:linear-gradient(90deg,#f97316,#ef4444);
          width:var(--pct,0%); transition:width .4s ease;
        }
        .pc-num{ width:54px; text-align:right; font-variant-numeric:tabular-nums; }
        .pc-footer{ display:flex; gap:8px; padding:12px 16px 16px; flex-wrap:wrap; }
        .pc-pill{
          background:#111827; color:#fbbf24; border:1px dashed rgba(251,191,36,.35);
          padding:.25rem .5rem; border-radius:999px; font-size:.75rem;
        }

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

# ---------- Watermark ----------
def add_world_cup_watermark(*, image_path: str | None = None,
                            image_url: str | None = None,
                            opacity: float = 0.08,
                            size: str = "68vmin",
                            y_offset: str = "6vh"):
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
        position: fixed; inset: 0;
        background-image: {bg};
        background-repeat: no-repeat;
        background-position: center {y_offset};
        background-size: {size};
        opacity: {opacity};
        pointer-events: none; z-index: 0;
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
    for row_idx in range(min(2, len(raw_df))):
        row = raw_df.iloc[row_idx].astype(str).str.strip().str.lower()
        for col_idx, cell in row.items():
            if "b division" in cell and b_col is None:
                b_col = col_idx
            elif "a division" in cell and a_col is None:
                a_col = col_idx
    if b_col is None and a_col is None:
        b_col = 0
        a_col = 5 if raw_df.shape[1] >= 8 else (4 if raw_df.shape[1] >= 7 else None)
    return b_col, a_col

def process_tournament_data(xlsx_bytes: bytes) -> pd.DataFrame:
    raw_df = safe_read_excel(xlsx_bytes)
    if raw_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    b_start, a_start = find_division_columns(raw_df)
    header_row = 1 if len(raw_df) > 1 else 0
    data_start_row = header_row + 1

    processed = []

    def extract_div(start_col: int | None, name: str):
        if start_col is None or start_col + 2 >= raw_df.shape[1]:
            return
        df = raw_df.iloc[data_start_row:, start_col:start_col + 3].copy()
        df.columns = ["Team", "Player", "Goals"]
        df = df.dropna(subset=["Team", "Player", "Goals"])
        df["Goals"] = pd.to_numeric(df["Goals"], errors="coerce")
        df = df.dropna(subset=["Goals"])
        df["Goals"] = df["Goals"].astype(int)
        df["Division"] = name
        processed.extend(df.to_dict("records"))

    extract_div(b_start, "B Division")
    extract_div(a_start, "A Division")

    if not processed:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    out = pd.DataFrame(processed)
    return out[["Division", "Team", "Player", "Goals"]]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.content:
            raise ValueError("Downloaded file is empty")
        return process_tournament_data(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

# ========== OPTIONAL PLAYER MASTER LOADER & PROFILES ==========
def _read_any_table_from_url(url: str) -> pd.DataFrame:
    if not url:
        return pd.DataFrame()
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        b = r.content
        try:
            return pd.read_csv(BytesIO(b))
        except Exception:
            try:
                return pd.read_excel(BytesIO(b))
            except Exception:
                return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def load_player_master(url: str) -> pd.DataFrame:
    """
    Expected columns (all optional except Player):
      Player, Team, Division, Shirt_No, Age, Position, Appearances,
      Yellow_Cards, Red_Cards, Awards
    """
    df = _read_any_table_from_url(url)
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "Shirt No": "Shirt_No",
        "Yellow": "Yellow_Cards",
        "Red": "Red_Cards",
        "Appearence": "Appearances",
    }
    df = df.rename(columns=rename_map)
    return df

def build_player_profiles(base_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    core = (
        base_df.groupby(["Player", "Team", "Division"])["Goals"]
        .sum()
        .reset_index()
        .rename(columns={"Goals": "Total_Goals"})
    )

    if not master_df.empty:
        key_cols = [c for c in ["Player", "Team", "Division"] if c in master_df.columns]
        if "Player" not in key_cols:
            key_cols = ["Player"]
        core = core.merge(master_df, on=key_cols, how="left")

    for col, default in [
        ("Shirt_No", ""), ("Age", ""), ("Position", ""),
        ("Appearances", 0), ("Yellow_Cards", 0), ("Red_Cards", 0),
        ("Awards", "")
    ]:
        if col not in core.columns:
            core[col] = default

    for ncol in ["Appearances", "Yellow_Cards", "Red_Cards", "Total_Goals"]:
        core[ncol] = pd.to_numeric(core[ncol], errors="coerce").fillna(0).astype(int)

    cols = [
        "Player", "Team", "Division", "Shirt_No", "Age", "Position",
        "Appearances", "Total_Goals", "Yellow_Cards", "Red_Cards", "Awards"
    ]
    core = core[[c for c in cols if c in core.columns]].sort_values(
        ["Total_Goals", "Appearances", "Player"], ascending=[False, False, True]
    )
    return core

# ====================== ANALYTICS & VISUALS =======================
def calculate_tournament_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_goals": 0, "total_players": 0, "total_teams": 0, "divisions": 0,
            "avg_goals_per_team": 0, "top_scorer_goals": 0, "competitive_balance": 0,
        }

    player_totals = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
    team_totals = df.groupby(["Team", "Division"])["Goals"].sum().reset_index()

    return {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(player_totals),
        "total_teams": len(team_totals),
        "divisions": df["Division"].nunique(),
        "avg_goals_per_team": round(df["Goals"].sum() / max(1, len(team_totals)), 2),
        "top_scorer_goals": int(player_totals["Goals"].max()) if not player_totals.empty else 0,
        "competitive_balance": round(team_totals["Goals"].std(), 2) if len(team_totals) > 1 else 0,
    }

def create_division_donut_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")
    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    sel = alt.selection_single(fields=["Division"], empty="none")
    base = (
        alt.Chart(division_data)
        .add_selection(sel)
        .properties(width=300, height=300, title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600))
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
    center_text = base.mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#1e293b").encode(
        text=alt.value(f"Total\n{int(division_data['Goals'].sum())}")
    )
    return outer + center_text

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
            title="Distribution of Goals per Player", xaxis_title="Goals per Player", yaxis_title="Number of Players",
            plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Poppins", size=12),
            title_font=dict(size=16, color="#1e293b"),
            xaxis=dict(gridcolor="#f1f5f9", zeroline=False), yaxis=dict(gridcolor="#f1f5f9", zeroline=False),
            height=400,
        )
        return fig
    hist_df = pd.DataFrame({"Goals": player_goals})
    return (
        alt.Chart(hist_df).mark_bar(opacity=0.85)
        .encode(
            x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
            y=alt.Y("count():Q", title="Number of Players"),
            tooltip=[alt.Tooltip("Goals:Q", bin=True, title="Goals"), alt.Tooltip("count():Q", title="Players")],
        ).properties(title="Distribution of Goals per Player", height=400)
    )

# ====================== PLAYER CARDS RENDERER =====================
def render_player_cards(profiles: pd.DataFrame):
    """Show player cards instead of a table. Uses Total_Goals, Appearances, Yellow_Cards, Red_Cards."""
    if profiles.empty:
        st.info("No player records available with the current filters.")
        return

    # scale bars vs current view
    max_g = max(1, int(profiles["Total_Goals"].max()))
    max_a = max(1, int(profiles["Appearances"].max()))
    max_y = max(1, int(profiles["Yellow_Cards"].max()))
    max_r = max(1, int(profiles["Red_Cards"].max()))

    # optional quick filter/search just for this tab
    left, right = st.columns([1, 2])
    with left:
        team_pick = st.multiselect("Filter teams (optional)", sorted(profiles["Team"].unique().tolist()))
    with right:
        name_q = st.text_input("Search player name (optional)", "")

    df = profiles.copy()
    if team_pick:
        df = df[df["Team"].isin(team_pick)]
    if name_q:
        q = name_q.strip().lower()
        df = df[df["Player"].str.lower().str.contains(q)]

    if df.empty:
        st.warning("No players after applying filters.")
        return

    # build cards grid
    cards_html = ['<div class="player-grid">']
    for _, row in df.iterrows():
        goals = int(row["Total_Goals"])
        apps  = int(row["Appearances"])
        yc    = int(row["Yellow_Cards"])
        rc    = int(row["Red_Cards"])
        name  = str(row["Player"])
        team  = str(row["Team"])
        divn  = str(row["Division"])
        pos   = (str(row.get("Position","")) or "—")
        shirt = (str(row.get("Shirt_No","")) or "—")
        age   = (str(row.get("Age","")) or "—")
        awards = str(row.get("Awards","")).strip()

        cards_html.append(f"""
        <div class="player-card">
          <div class="pc-top">
            <div class="pc-fig">#{shirt}</div>
            <div>
              <div class="pc-name">{name}</div>
              <div class="pc-sub">{team} • {divn} • {pos} • Age {age}</div>
            </div>
          </div>
          <div class="pc-body">
            <div class="pc-row">
              <div class="pc-label">Goals</div>
              <div class="pc-bar"><span style="--pct:{min(100, round(goals/max_g*100))}%"></span></div>
              <div class="pc-num">{goals}</div>
            </div>
            <div class="pc-row">
              <div class="pc-label">Appearances</div>
              <div class="pc-bar"><span style="--pct:{min(100, round(apps/max_a*100))}%"></span></div>
              <div class="pc-num">{apps}</div>
            </div>
            <div class="pc-row">
              <div class="pc-label">Yellow Cards</div>
              <div class="pc-bar"><span style="--pct:{min(100, round(yc/max_y*100))}%"></span></div>
              <div class="pc-num">{yc}</div>
            </div>
            <div class="pc-row">
              <div class="pc-label">Red Cards</div>
              <div class="pc-bar"><span style="--pct:{min(100, round(rc/max_r*100))}%"></span></div>
              <div class="pc-num">{rc}</div>
            </div>
          </div>
          <div class="pc-footer">
            {"".join([f'<span class="pc-pill">{a.strip()}</span>' for a in awards.split(",") if a.strip()]) or "<span class='pc-pill' style='opacity:.7;'>No awards</span>"}
          </div>
        </div>
        """)
    cards_html.append("</div>")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

# ====================== UI HELPERS ================================
def display_metric_cards(stats: dict):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)

def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
            teams = (
                filtered_df.groupby(["Team", "Division"])
                .agg(Unique_Players=("Player", "nunique"), Total_Records=("Goals", "count"), Total_Goals=("Goals", "sum"),
                     Avg_Goals=("Goals", "mean"), Max_Goals=("Goals", "max"))
                .round(2).reset_index()
            )
            z.writestr("03_teams_detailed_analysis.csv", teams.to_csv(index=False))
            players = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players = players.sort_values(["Goals", "Player"], ascending=[False, True])
            players.insert(0, "Rank", range(1, len(players) + 1))
            z.writestr("04_players_ranking.csv", players.to_csv(index=False))
            stats = calculate_tournament_stats(filtered_df)
            z.writestr("05_tournament_statistics.csv", pd.DataFrame([stats]).to_csv(index=False))
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
            )
        if not filtered_df.empty:
            st.download_button(
                label="⬇️ Download FILTERED Dataset (CSV)",
                data=filtered_df.to_csv(index=False),
                file_name=f"tournament_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
    with col2:
        st.subheader("🏆 Players / Teams")
        if not filtered_df.empty:
            teams_summary = (
                filtered_df.groupby(["Team", "Division"]).agg(Players_Count=("Player", "nunique"), Total_Goals=("Goals", "sum")).reset_index()
            ).sort_values("Total_Goals", ascending=False)
            st.download_button(
                label="⬇️ Download TEAMS Summary (CSV)",
                data=teams_summary.to_csv(index=False),
                file_name=f"teams_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
            players_summary = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
            players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
            st.download_button(
                label="⬇️ Download PLAYERS Summary (CSV)",
                data=players_summary.to_csv(index=False),
                file_name=f"players_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
    with col3:
        st.subheader("📦 Complete Package")
        if st.button("📦 Generate Complete Report Package"):
            z = create_comprehensive_zip_report(full_df, filtered_df)
            st.download_button(
                label="⬇️ Download Complete Package (ZIP)",
                data=z,
                file_name=f"tournament_complete_package_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
            )

# ====================== MAIN APPLICATION ==========================
def main():
    inject_advanced_css()

    # Title
    st.markdown(
        """<div class="app-title"><span class="ball">⚽</span>
        <span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span></div>""",
        unsafe_allow_html=True,
    )

    # Watermark
    add_world_cup_watermark(
        image_url="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg",
        opacity=0.10, size="68vmin", y_offset="6vh"
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
        with st.spinner("📡 Loading tournament data…"):
            tournament_data = fetch_tournament_data(GOOGLE_SHEETS_URL)
        if tournament_data.empty:
            notify("No tournament data available. Check the published sheet link/permissions.", "err")
            st.stop()
        full_tournament_data = tournament_data.copy()

        st.header("🔍 Data Filters")
        division_options = ["All Divisions"] + sorted(tournament_data["Division"].unique().tolist())
        selected_division = st.selectbox("📊 Division", division_options, key="division_filter")
        if selected_division != "All Divisions":
            tournament_data = tournament_data[tournament_data["Division"] == selected_division]
        available_teams = sorted(tournament_data["Team"].unique().tolist())
        selected_teams = st.multiselect(
            "🏆 Teams (optional)", available_teams, key="teams_filter",
            help="Select specific teams to focus on", placeholder="Type to search teams…",
        )
        if selected_teams:
            tournament_data = tournament_data[tournament_data["Team"].isin(selected_teams)]
        st.subheader("👤 Player Search")
        player_names = sorted(tournament_data["Player"].dropna().astype(str).unique().tolist())
        selected_players = st.multiselect(
            "Type to search and select players", options=player_names, default=[],
            key="players_filter", placeholder="Start typing a player name…",
        )
        if selected_players:
            tournament_data = tournament_data[tournament_data["Player"].isin(selected_players)]

    # Tabs (first block is sticky via CSS)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📊 OVERVIEW", "⚡ QUICK INSIGHTS", "🏆 TEAMS", "👤 PLAYERS", "📈 ANALYTICS", "📥 DOWNLOADS", "📇 PLAYER PROFILES"]
    )

    current_stats = calculate_tournament_stats(tournament_data)

    # TAB 1
    with tab1:
        st.header("📊 Tournament Overview")
        display_metric_cards(current_stats)
        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 Records (raw)")
            if tournament_data.empty:
                st.info("No records for current filters.")
            else:
                show_df = tournament_data.sort_values("Goals", ascending=False).reset_index(drop=True)
                st.dataframe(
                    show_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Division": st.column_config.TextColumn("Division", width="small"),
                        "Team": st.column_config.TextColumn("Team", width="medium"),
                        "Player": st.column_config.TextColumn("Player", width="large"),
                        "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
                    },
                )
        with col2:
            if not tournament_data.empty:
                st.subheader("🏁 Division Distribution")
                st.altair_chart(create_division_donut_chart(tournament_data), use_container_width=True)

    # TAB 2
    with tab2:
        st.header("⚡ Quick Tournament Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Total Goals", current_stats["total_goals"])
        c2.metric("👥 Active Players", current_stats["total_players"])
        c3.metric("🏆 Teams", current_stats["total_teams"])
        c4.metric("📊 Divisions", current_stats["divisions"])
        st.divider()
        st.subheader("📊 Goals Distribution")
        dist = create_goals_distribution_histogram(tournament_data)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(dist, use_container_width=True)
        else:
            st.altair_chart(dist, use_container_width=True)

    # TAB 3 — Teams (simple)
    with tab3:
        st.header("🏆 Teams Summary")
        if tournament_data.empty:
            st.info("No teams for current filters.")
        else:
            teams_summary = (
                tournament_data.groupby(["Team", "Division"])
                .agg(Players=("Player", "nunique"), Total_Goals=("Goals", "sum"))
                .reset_index().sort_values("Total_Goals", ascending=False)
            )
            st.dataframe(
                teams_summary, use_container_width=True, hide_index=True,
                column_config={
                    "Division": st.column_config.TextColumn("Division", width="small"),
                    "Team": st.column_config.TextColumn("Team", width="medium"),
                    "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                    "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small"),
                },
            )

    # TAB 4 — Players ranking quick list (optional)
    with tab4:
        st.header("👤 Players Ranking")
        if tournament_data.empty:
            st.info("No players for current filters.")
        else:
            players_summary = tournament_data.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
            players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
            st.dataframe(
                players_summary, use_container_width=True, hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                    "Player": st.column_config.TextColumn("Player", width="large"),
                    "Team": st.column_config.TextColumn("Team", width="medium"),
                    "Division": st.column_config.TextColumn("Division", width="small"),
                    "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
                },
            )

    # TAB 5 — Analytics (no scatter, keep histogram already in Quick Insights, so keep text insights)
    with tab5:
        st.header("📈 Analytics (Summary)")
        if tournament_data.empty:
            st.info("No analytics for current filters.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🏆 Top Teams (Total Goals)**")
                tg = tournament_data.groupby("Team")["Goals"].sum().sort_values(ascending=False).head(5)
                for team, g in tg.items():
                    st.write(f"• **{team}**: {int(g)} goals")
            with c2:
                st.markdown("**⚽ Scoring Patterns**")
                counts = tournament_data["Goals"].value_counts().sort_index()
                for g, c in counts.items():
                    pct = (c / len(tournament_data) * 100)
                    st.write(f"• **{int(g)} goal{'s' if g != 1 else ''}**: {int(c)} records ({pct:.1f}%)")
            with c3:
                st.markdown("**🎯 Division Insights**")
                for division in tournament_data["Division"].unique():
                    div_data = tournament_data[tournament_data["Division"] == division]
                    total_goals = int(div_data["Goals"].sum())
                    unique_players = int(div_data["Player"].nunique())
                    st.write(f"• **{division}**: {total_goals} goals, {unique_players} players")

    # TAB 6 — Downloads
    with tab6:
        create_download_section(full_tournament_data, tournament_data)

    # TAB 7 — PLAYER PROFILES (card view)
    with tab7:
        st.header("📇 Complete Player Profiles")
        player_master = load_player_master(PLAYER_MASTER_URL)
        profiles = build_player_profiles(tournament_data, player_master)
        render_player_cards(profiles)
        if not profiles.empty:
            st.download_button(
                "⬇️ Download Player Profiles (CSV)",
                data=profiles.to_csv(index=False),
                file_name=f"player_profiles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ====================== ENTRY POINT ===============================
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.exception(e)
