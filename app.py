# app.py
# ABEER BLUESTAR SOCCER FEST 2K25 — Streamlit Dashboard (with Player Card View + CARDS sheet merge)
# Author: AI Assistant
# Last updated: 2025-08-29

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import altair as alt
import pandas as pd
import requests
import streamlit as st

# =========================================================
# --------------------- CONFIG ----------------------------
# =========================================================
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- DATA SOURCES ----
# 1) GOALS workbook (published XLSX that contains the goals grid like before)
GOALS_SHEET_XLSX_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
)

# 2) CARDS sheet (your new link). This can be a normal “edit?gid=...” link. We’ll convert to CSV export.
CARDS_SHEET_LINK = "https://docs.google.com/spreadsheets/d/1Bbx7nCS_j7g1wsK3gHpQQnWkpTqlwkHu/edit?gid=954169595#gid=954169595"

# =========================================================
# --------------------- STYLES ----------------------------
# =========================================================
def inject_advanced_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{ --sticky-tabs-top: 52px; }

.stApp{
  font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
}
.block-container{
  padding-top:.5rem; padding-bottom:2rem;
  max-width:98vw; width:98vw;
  background:rgba(255,255,255,.95);
  backdrop-filter:blur(15px);
  border-radius:20px; box-shadow:0 20px 40px rgba(0,0,0,.15);
  margin:1rem auto; position:relative; z-index:1;
}
#MainMenu, footer, .stDeployButton, div[data-testid="stDecoration"], div[data-testid="stStatusWidget"]{ display:none !important; }

/* sticky first tabs bar */
.block-container [data-testid="stTabs"]:first-of-type{
  position:sticky; top:var(--sticky-tabs-top); z-index:6;
  background:rgba(255,255,255,.96); backdrop-filter:blur(8px);
  border-bottom:1px solid #e2e8f0; padding:.25rem 0; margin-top:.25rem;
}

/* title */
.app-title{ display:flex; align-items:center; justify-content:center; gap:12px; margin:.75rem 0 1rem; }
.app-title .ball{ font-size:32px; line-height:1; filter:drop-shadow(0 2px 4px rgba(0,0,0,.15)); }
.app-title .title{
  font-weight:700; letter-spacing:.05em; font-size:clamp(22px,3.5vw,36px);
  background:linear-gradient(45deg,#0ea5e9,#1e40af,#7c3aed);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 2px 4px rgba(0,0,0,.1);
}

/* buttons */
.stButton > button, .stDownloadButton > button{
  background:linear-gradient(135deg,#0ea5e9,#3b82f6)!important;
  color:#fff!important; border:0!important; border-radius:12px!important;
  padding:.6rem 1.2rem!important; font-weight:600!important; font-size:.9rem!important;
  transition:.3s; box-shadow:0 4px 15px rgba(14,165,233,.3)!important;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform:translateY(-2px)!important; box-shadow:0 8px 25px rgba(14,165,233,.4)!important; filter:brightness(1.05)!important;
}

/* metric card */
.metric-container{
  background:linear-gradient(135deg, rgba(14,165,233,.1), rgba(59,130,246,.05));
  border-radius:15px; padding:1.5rem; border-left:4px solid #0ea5e9;
  box-shadow:0 4px 20px rgba(14,165,233,.1); transition:transform .2s ease;
}
.metric-container:hover{ transform:translateY(-3px); }

/* status pill */
.status-pill{ padding:.5rem .75rem; border-radius:.6rem; font-size:.85rem; margin-top:.5rem; }
.status-ok{ background:#ecfeff; border-left:4px solid #06b6d4; color:#155e75; }
.status-warn{ background:#fef9c3; border-left:4px solid #f59e0b; color:#713f12; }
.status-err{ background:#fee2e2; border-left:4px solid #ef4444; color:#7f1d1d; }

/* ----------- PLAYER CARDS ----------- */
.pgrid{
  display:grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap:18px;
}
@media(min-width:700px){ .pgrid{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media(min-width:1100px){ .pgrid{ grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media(min-width:1450px){ .pgrid{ grid-template-columns: repeat(4, minmax(0, 1fr)); } }

.pcard{
  background:#fff; border-radius:16px; padding:18px;
  box-shadow:0 6px 24px rgba(0,0,0,.06); border:1px solid #eef2f7;
}
.pcard h3{ font-size:1.05rem; margin:.2rem 0 .4rem; color:#0f172a; }
.pcard .sub{ color:#0ea5e9; font-weight:600; }
.pcard .muted{ color:#64748b; font-size:.9rem; margin-bottom:.75rem; }

.pill{
  font-size:.75rem; background:#eef2ff; color:#4f46e5; padding:.25rem .5rem; border-radius:999px;
}
.row{ display:grid; grid-template-columns:auto 1fr auto; gap:.5rem; align-items:center; margin:.35rem 0; }
.label{ color:#475569; white-space:nowrap; }
.dotbar{
  position:relative; height:8px; background:#f1f5f9; border-radius:999px; overflow:hidden;
  border:1px solid #e2e8f0;
}
.dotbar span{
  position:absolute; left:0; top:0; bottom:0; width:var(--pct,0%); background:linear-gradient(90deg,#10b981,#22d3ee);
}
.num{ font-weight:600; color:#1f2937; }

.empty-pill{
  display:inline-block; font-size:.8rem; background:#f8fafc; border:1px solid #e5e7eb; color:#6b7280;
  padding:.25rem .5rem; border-radius:10px;
}
</style>
        """,
        unsafe_allow_html=True,
    )

def add_world_cup_watermark(image_url="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg",
                            opacity=0.10, size="68vmin", y_offset="6vh"):
    st.markdown(
        f"""
<style>
  #wc-trophy {{
    position:fixed; inset:0; background-image:url('{image_url}');
    background-repeat:no-repeat; background-position:center {y_offset};
    background-size:{size}; opacity:{opacity}; pointer-events:none; z-index:0;
  }}
</style>
<div id="wc-trophy"></div>
        """,
        unsafe_allow_html=True,
    )

def notify(msg, kind="ok"):
    cls = {"ok":"status-ok", "warn":"status-warn", "err":"status-err"}.get(kind,"status-ok")
    st.markdown(f'<div class="status-pill {cls}">{msg}</div>', unsafe_allow_html=True)

# =========================================================
# --------------------- DATA UTILS ------------------------
# =========================================================

def _norm_text(s: pd.Series) -> pd.Series:
    """normalize text for safer joins (trim + collapse spaces + lowercase)"""
    return (
        s.astype(str)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
         .str.lower()
    )

def _download(url: str, timeout=30) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def _gsheet_to_csv_url(url: str) -> str:
    """
    Accepts typical Google Sheets links and returns a direct CSV export URL for the specific sheet (using gid if present).
    Examples input:
      - https://docs.google.com/spreadsheets/d/<ID>/edit?gid=<GID>#gid=<GID>
      - https://docs.google.com/spreadsheets/d/<ID>/view#gid=<GID>
    """
    m = re.search(r"/spreadsheets/d/([^/]+)/", url)
    gid = None
    mg = re.search(r"[?#&]gid=(\d+)", url)
    if mg:
        gid = mg.group(1)
    if not m or not gid:
        return url  # give back original; caller will try anyway
    sheet_id = m.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# ---------- XLSX parsing without openpyxl ----------
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

        # Only sheet1; works for the GOALS workbook which is published as a single "sheet1"
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

def safe_read_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
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

# ---------- GOALS sheet (already published XLSX) ----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_goals_data(goals_xlsx_url: str) -> pd.DataFrame:
    content = _download(goals_xlsx_url)
    if not content:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    raw_df = safe_read_excel_bytes(content)
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

# ---------- CARDS sheet (separate sheet; we read CSV via gid) ----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_cards_data(cards_link: str) -> pd.DataFrame:
    """
    Reads the CARDS sheet where there are two side-by-side tables:
      - B Division: Team | Player Name | CARDS
      - A Division: Team | Player Name | CARDS
    Returns tidy rows: Division, Team, Player, Yellow Cards, Red Cards
    """
    csv_url = _gsheet_to_csv_url(cards_link)
    try:
        raw_df = pd.read_csv(csv_url, header=None)
    except Exception:
        # final fallback: try to download XLSX export of the whole workbook and parse sheet1
        alt_bytes = _download(re.sub(r"/edit.*", "", cards_link) + "/export?format=xlsx")
        if not alt_bytes:
            return pd.DataFrame(columns=["Division", "Team", "Player", "Yellow Cards", "Red Cards"])
        raw_df = safe_read_excel_bytes(alt_bytes)

    if raw_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Yellow Cards", "Red Cards"])

    b_start, a_start = find_division_columns(raw_df)
    header_row = 1 if len(raw_df) > 1 else 0
    data_start_row = header_row + 1

    rows = []

    def extract(start_col: int | None, div_name: str):
        if start_col is None or start_col + 2 >= raw_df.shape[1]:
            return
        df = raw_df.iloc[data_start_row:, start_col:start_col + 3].copy()
        df.columns = ["Team", "Player", "CARDS"]
        # keep rows that at least have a player name; team can be empty sometimes
        df = df.dropna(subset=["Player"])
        # Normalize
        for col in ["Team", "Player", "CARDS"]:
            df[col] = df[col].astype(str).str.strip()
        # Map to counts
        df["Yellow Cards"] = (df["CARDS"].str.upper() == "YELLOW").astype(int)
        df["Red Cards"] = (df["CARDS"].str.upper() == "RED").astype(int)
        df["Division"] = div_name
        rows.append(df[["Division", "Team", "Player", "Yellow Cards", "Red Cards"]])

    extract(b_start, "B Division")
    extract(a_start, "A Division")

    if not rows:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Yellow Cards", "Red Cards"])
    out = pd.concat(rows, ignore_index=True)

    # aggregate in case a player appears multiple times
    out = (
        out.groupby(["Division", "Team", "Player"], dropna=False)[["Yellow Cards", "Red Cards"]]
        .sum()
        .reset_index()
    )
    return out

def merge_goals_and_cards(goals_df: pd.DataFrame, cards_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join cards counts onto the goals grid by (Division, Team, Player) with normalized keys."""
    if goals_df.empty:
        return goals_df.assign(**{"Yellow Cards": 0, "Red Cards": 0})

    g = goals_df.copy()
    for col in ["Division", "Team", "Player"]:
        g[f"{col}__k"] = _norm_text(g[col])

    if cards_df.empty:
        g["Yellow Cards"] = 0
        g["Red Cards"] = 0
        return g.drop(columns=[c for c in g.columns if c.endswith("__k")])

    c = cards_df.copy()
    # when Team is empty in cards, we still want to match by player+division
    for col in ["Division", "Team", "Player"]:
        if col not in c.columns:
            c[col] = ""
        c[f"{col}__k"] = _norm_text(c[col])

    merged = g.merge(
        c[["Division__k", "Team__k", "Player__k", "Yellow Cards", "Red Cards"]],
        how="left",
        left_on=["Division__k", "Team__k", "Player__k"],
        right_on=["Division__k", "Team__k", "Player__k"],
        suffixes=("", "_cards"),
    )
    merged["Yellow Cards"] = merged["Yellow Cards"].fillna(0).astype(int)
    merged["Red Cards"] = merged["Red Cards"].fillna(0).astype(int)
    return merged.drop(columns=[c for c in merged.columns if c.endswith("__k")])

# =========================================================
# --------------------- ANALYTICS -------------------------
# =========================================================
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

def get_top_performers(df: pd.DataFrame, top_n: int = 10) -> dict:
    if df.empty:
        return {"players": pd.DataFrame(), "teams": pd.DataFrame()}
    top_players = (
        df.groupby(["Player", "Team", "Division"])["Goals"]
        .sum().reset_index()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        .head(top_n)
    )
    top_teams = (
        df.groupby(["Team", "Division"])["Goals"]
        .sum().reset_index()
        .sort_values("Goals", ascending=False)
        .head(top_n)
    )
    return {"players": top_players, "teams": top_teams}

def create_division_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    division_stats = (
        df.groupby("Division")
          .agg(Goals_sum=("Goals","sum"), Goals_mean=("Goals","mean"), Records=("Goals","count"),
               Teams=("Team","nunique"), Players=("Player","nunique"))
          .round(2).reset_index()
          .rename(columns={"Goals_sum":"Total_Goals", "Goals_mean":"Avg_Goals", "Records":"Total_Records"})
    )
    total_goals = division_stats["Total_Goals"].sum()
    division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"] / total_goals * 100).round(1) if total_goals else 0
    return division_stats

# =========================================================
# ----------------- CHART BUILDERS ------------------------
# =========================================================
def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_scheme: str = "blues") -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note":["No data available"]})).mark_text().encode(text="note:N")
    max_val = int(df[x_col].max()) if not df.empty else 1
    tick_values = list(range(0, max_val + 1)) if max_val <= 50 else None
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=.85, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(f"{x_col}:Q", title="Goals", axis=alt.Axis(format="d", tickMinStep=1, values=tick_values, gridOpacity=.3),
                    scale=alt.Scale(domainMin=0, nice=False)),
            y=alt.Y(f"{y_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            color=alt.Color(f"{x_col}:Q", scale=alt.Scale(scheme=color_scheme), legend=None),
            tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col), alt.Tooltip(f"{x_col}:Q", title="Goals", format="d")],
        )
        .properties(height=max(300, min(600, len(df) * 25)), title=alt.TitleParams(text=title, fontSize=16, anchor="start", fontWeight=600))
        .resolve_scale(color="independent")
    )

def create_division_donut_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"note":["No data available"]})).mark_text().encode(text="note:N")
    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    base = alt.Chart(division_data).properties(width=300, height=300,
        title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600))
    outer = (
        base.mark_arc(innerRadius=60, outerRadius=120, stroke="white", strokeWidth=2)
            .encode(theta="Goals:Q", color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9","#f59e0b"]), title="Division"),
                    tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")])
    )
    center_text = base.mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#1e293b") \
                     .encode(text=alt.value(f"Total\n{int(division_data['Goals'].sum())}"))
    return outer + center_text

# =========================================================
# ---------------------- UI HELPERS -----------------------
# =========================================================
def display_metric_cards(stats: dict):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)

def _fill_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Replace None/NaN in tables: numerics -> 0, text -> ''."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        else:
            out[col] = out[col].fillna("")
    return out

def create_enhanced_data_table(df: pd.DataFrame, table_type: str = "records"):
    if df.empty:
        st.info(f"📋 No {table_type} data available with current filters.")
        return

    if table_type == "records":
        display_df = df[["Division", "Team", "Player", "Goals", "Yellow Cards", "Red Cards"]].copy()
        display_df = display_df.sort_values("Goals", ascending=False).reset_index(drop=True)
        display_df = _fill_for_display(display_df)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
                "Yellow Cards": st.column_config.NumberColumn("🟨 Yellow", format="%d", width="small"),
                "Red Cards": st.column_config.NumberColumn("🟥 Red", format="%d", width="small"),
            },
        )

    elif table_type == "teams":
        base = df.groupby(["Team", "Division"]).agg(
            Players=("Player","nunique"),
            Total_Goals=("Goals","sum"),
            Yellow=("Yellow Cards","sum") if "Yellow Cards" in df.columns else ("Goals","sum"),
            Red=("Red Cards","sum") if "Red Cards" in df.columns else ("Goals","sum"),
        ).reset_index()
        display_df = base.sort_values("Total_Goals", ascending=False)
        display_df = _fill_for_display(display_df)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small"),
                "Yellow": st.column_config.NumberColumn("🟨 Yellow", format="%d", width="small"),
                "Red": st.column_config.NumberColumn("🟥 Red", format="%d", width="small"),
            },
        )

    elif table_type == "players":
        players_summary = (
            df.groupby(["Player", "Team", "Division"])
              .agg(Goals=("Goals","sum"),
                   Yellow=("Yellow Cards","sum") if "Yellow Cards" in df.columns else ("Goals","sum"),
                   Red=("Red Cards","sum") if "Red Cards" in df.columns else ("Goals","sum"))
              .reset_index()
        )
        players_summary = players_summary.sort_values(["Goals","Player"], ascending=[False, True])
        players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
        players_summary = _fill_for_display(players_summary)
        st.dataframe(
            players_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
                "Yellow": st.column_config.NumberColumn("🟨 Yellow", format="%d", width="small"),
                "Red": st.column_config.NumberColumn("🟥 Red", format="%d", width="small"),
            },
        )

def render_player_cards(df: pd.DataFrame):
    """HTML cards, now including Yellow/Red cards if present. Renders, not as literal text."""
    if df.empty:
        st.info("No players to show.")
        return

    # Aggregate per player
    work = (
        df.groupby(["Player", "Team", "Division"], dropna=False)
          .agg(
              Goals=("Goals","sum"),
              Yellow=("Yellow Cards","sum") if "Yellow Cards" in df.columns else ("Goals","sum"),
              Red=("Red Cards","sum") if "Red Cards" in df.columns else ("Goals","sum"),
          )
          .reset_index()
    )
    # If Yellow/Red were fake fallback above (using Goals), fix them to 0
    if "Yellow Cards" not in df.columns: work["Yellow"] = 0
    if "Red Cards" not in df.columns: work["Red"] = 0

    # Rank by Goals desc then Player asc
    work = work.sort_values(["Goals","Player"], ascending=[False, True]).reset_index(drop=True)
    work.insert(0, "Rank", range(1, len(work) + 1))

    max_goals = max(1, int(work["Goals"].max()))
    cards_html = ['<div class="pgrid">']
    for _, r in work.iterrows():
        g_pct = round(r["Goals"] / max_goals * 100, 1)
        cards_html.append(
            f"""
    <div class="pcard">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.25rem">
        <h3 style="margin:0">{r['Player']}</h3>
        <span class="pill" title="Rank">#{int(r['Rank'])}</span>
      </div>
      <div class="sub">{r['Team']}</div>
      <div class="muted">{r['Division']}</div>

      <div class="row">
        <div class="label">⚽ Goals</div>
        <div class="dotbar"><span style="--pct:{g_pct}%"></span></div>
        <div class="num">{int(r['Goals'])}</div>
      </div>

      <div class="row">
        <div class="label">🟨 Yellow Cards</div>
        <div class="dotbar"><span style="--pct:{0 if (r['Yellow'] is None or max_goals == 0) else min(100, round((r['Yellow'] / max_goals) * 100, 1))}%"></span></div>
        <div class="num">{int(r['Yellow'])}</div>
      </div>

      <div class="row">
        <div class="label">🟥 Red Cards</div>
        <div class="dotbar"><span style="--pct:{0 if (r['Red'] is None or max_goals == 0) else min(100, round((r['Red'] / max_goals) * 100, 1))}%"></span></div>
        <div class="num">{int(r['Red'])}</div>
      </div>

      <div style="margin-top:.5rem"><span class="empty-pill">No awards</span></div>
    </div>
            """
        )
    cards_html.append("</div>")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    zbuf = BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
            players = (
                filtered_df.groupby(["Player","Team","Division"])
                           .agg(Goals=("Goals","sum"),
                                Yellow=("Yellow Cards","sum") if "Yellow Cards" in filtered_df.columns else ("Goals","sum"),
                                Red=("Red Cards","sum") if "Red Cards" in filtered_df.columns else ("Goals","sum"))
                           .reset_index()
                           .sort_values(["Goals","Player"], ascending=[False, True])
            )
            players.insert(0, "Rank", range(1, len(players)+1))
            z.writestr("03_players_ranking.csv", players.to_csv(index=False))
            teams = (
                filtered_df.groupby(["Team","Division"])
                           .agg(Players=("Player","nunique"),
                                Total_Goals=("Goals","sum"),
                                Yellow=("Yellow Cards","sum") if "Yellow Cards" in filtered_df.columns else ("Goals","sum"),
                                Red=("Red Cards","sum") if "Red Cards" in filtered_df.columns else ("Goals","sum"))
                           .reset_index()
            )
            z.writestr("04_teams_summary.csv", teams.to_csv(index=False))
        z.writestr("README.txt", f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    zbuf.seek(0)
    return zbuf.getvalue()

def create_download_section(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.subheader("📥 Download Reports")
    st.caption("**Full** = all data, **Filtered** = current view")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Data Exports")
        if not full_df.empty:
            st.download_button("⬇️ Download FULL Dataset (CSV)", full_df.to_csv(index=False),
                               file_name=f"tournament_full_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        if not filtered_df.empty:
            st.download_button("⬇️ Download FILTERED Dataset (CSV)", filtered_df.to_csv(index=False),
                               file_name=f"tournament_filtered_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with col2:
        st.subheader("🏆 Quick Summaries")
        if not filtered_df.empty:
            teams_summary = (
                filtered_df.groupby(["Team","Division"])
                           .agg(Players=("Player","nunique"), Total_Goals=("Goals","sum"))
                           .reset_index()
                           .sort_values("Total_Goals", ascending=False)
            )
            st.download_button("⬇️ Teams Summary (CSV)", teams_summary.to_csv(index=False),
                               file_name=f"teams_summary_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
            players_summary = (
                filtered_df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
                           .sort_values(["Goals","Player"], ascending=[False, True])
            )
            players_summary.insert(0, "Rank", range(1, len(players_summary)+1))
            st.download_button("⬇️ Players Summary (CSV)", players_summary.to_csv(index=False),
                               file_name=f"players_summary_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with col3:
        st.subheader("📦 Complete Package")
        if st.button("📦 Generate ZIP Package"):
            z = create_comprehensive_zip_report(full_df, filtered_df)
            st.download_button("⬇️ Download (ZIP)", z,
                               file_name=f"tournament_package_{datetime.now():%Y%m%d_%H%M}.zip", mime="application/zip")

# =========================================================
# ------------------------ MAIN ---------------------------
# =========================================================
def main():
    inject_advanced_css()
    st.markdown("""
<div class="app-title">
  <span class="ball">⚽</span>
  <span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span>
</div>
""", unsafe_allow_html=True)
    add_world_cup_watermark()

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notify("Cache cleared. Reloading…", "ok")
            st.rerun()
        last_refresh = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.caption(f"🕒 Last refreshed: {last_refresh}")
        st.divider()

        with st.spinner("📡 Loading GOALS data…"):
            goals_df = fetch_goals_data(GOALS_SHEET_XLSX_URL)
        if goals_df.empty:
            notify("No GOALS data found (check the published link/permissions).", "err")
            st.stop()

        with st.spinner("🟨🟥 Loading CARDS sheet…"):
            cards_df = fetch_cards_data(CARDS_SHEET_LINK)
        if cards_df.empty:
            notify("CARDS sheet not found or not public. Cards will show as 0.", "warn")

        tournament_df = merge_goals_and_cards(goals_df, cards_df)
        full_df = tournament_df.copy()

        # Filters
        st.header("🔍 Data Filters")
        division_options = ["All Divisions"] + sorted(tournament_df["Division"].unique().tolist())
        selected_division = st.selectbox("📊 Division", division_options, key="division_filter")
        if selected_division != "All Divisions":
            tournament_df = tournament_df[tournament_df["Division"] == selected_division]

        available_teams = sorted(tournament_df["Team"].unique().tolist())
        selected_teams = st.multiselect("🏆 Teams (optional)", available_teams, key="teams_filter",
                                        placeholder="Type to search teams…")
        if selected_teams:
            tournament_df = tournament_df[tournament_df["Team"].isin(selected_teams)]

        st.subheader("👤 Player Search")
        player_names = sorted(tournament_df["Player"].dropna().astype(str).unique().tolist())
        selected_players = st.multiselect("Type to search and select players", options=player_names,
                                          key="players_filter", placeholder="Start typing a player name…")
        if selected_players:
            tournament_df = tournament_df[tournament_df["Player"].isin(selected_players)]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 OVERVIEW", "⚡ QUICK INSIGHTS", "🏆 TEAMS", "👤 PLAYERS", "🧱 PLAYERS (Card View)", "📥 DOWNLOADS"]
    )

    stats = calculate_tournament_stats(tournament_df)
    top = get_top_performers(tournament_df, 10)

    # OVERVIEW
    with tab1:
        st.header("📊 Tournament Overview")
        display_metric_cards(stats)
        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🎯 Goal Scoring Records")
            create_enhanced_data_table(tournament_df, "records")
        with c2:
            st.subheader("🏁 Division Distribution")
            st.altair_chart(create_division_donut_chart(tournament_df), use_container_width=True)
        if not tournament_df.empty:
            colA, colB = st.columns(2)
            with colA:
                st.subheader("🏆 Goals by Team")
                team_goals = tournament_df.groupby("Team")["Goals"].sum().reset_index().sort_values("Goals", ascending=False).head(10)
                st.altair_chart(create_horizontal_bar_chart(team_goals, "Goals", "Team", "Top 10 Teams by Goals", "blues"), use_container_width=True)
            with colB:
                st.subheader("⚽ Top Scorers")
                ts = top["players"].head(10).copy()
                if not ts.empty:
                    ts["Display_Name"] = ts["Player"] + " (" + ts["Team"] + ")"
                    st.altair_chart(create_horizontal_bar_chart(ts, "Goals", "Display_Name", "Top 10 Players by Goals", "greens"), use_container_width=True)

    # QUICK INSIGHTS
    with tab2:
        st.header("⚡ Quick Tournament Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Total Goals", stats["total_goals"])
        c2.metric("👥 Active Players", stats["total_players"])
        c3.metric("🏆 Teams", stats["total_teams"])
        c4.metric("📊 Divisions", stats["divisions"])
        st.divider()
        # simple blocks
        if not tournament_df.empty:
            division_cmp = create_division_comparison(tournament_df)
            st.subheader("🔄 Division Comparison")
            st.dataframe(_fill_for_display(division_cmp), use_container_width=True, hide_index=True)

    # TEAMS
    with tab3:
        st.header("🏆 Teams Analysis")
        if tournament_df.empty:
            st.info("🔍 No teams match your current filters.")
        else:
            st.subheader("📋 Teams Summary")
            create_enhanced_data_table(tournament_df, "teams")

    # PLAYERS
    with tab4:
        st.header("👤 Players Analysis")
        if tournament_df.empty:
            st.info("🔍 No players match your current filters.")
        else:
            st.subheader("📋 Players Ranking")
            create_enhanced_data_table(tournament_df, "players")

    # PLAYERS (Card View)
    with tab5:
        st.header("🧱 Players (Card View)")
        render_player_cards(tournament_df)

    # DOWNLOADS
    with tab6:
        create_download_section(full_df, tournament_df)

# ---------------- Entry ----------------
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        main()
    except Exception as e:
        st.error(f'🚨 Application Error: "{e}"')
        st.exception(e)
