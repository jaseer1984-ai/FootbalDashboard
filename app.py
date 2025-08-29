# ABEER BLUESTAR SOCCER FEST 2K25 — Streamlit Dashboard
# Players Card View + multi-sheet parsing (GOALS + CARDS with Card Type/Match#/Action)
# Author: AI Assistant | Last updated: 2025-08-29

from __future__ import annotations

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

# Optional (Plotly for Analytics)
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None

# ------------------------- CONFIG -------------------------
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- STYLES -------------------------
def inject_advanced_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{ --sticky-tabs-top: 52px; }
.stApp { font-family:'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:linear-gradient(135deg,#667eea 0%, #764ba2 100%); }
.block-container{
  padding-top:.5rem; padding-bottom:2rem;
  max-width:98vw; width:98vw; margin:1rem auto;
  background:rgba(255,255,255,.95); backdrop-filter:blur(15px);
  border-radius:20px; box-shadow:0 20px 40px rgba(0,0,0,.15);
  position:relative; z-index:1;
}
#MainMenu, footer, .stDeployButton,
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"]{display:none!important}

/* Sticky tabs */
.block-container [data-testid="stTabs"]:first-of-type{
  position:sticky; top:var(--sticky-tabs-top); z-index:6;
  background:rgba(255,255,255,.96); backdrop-filter:blur(8px);
  border-bottom:1px solid #e2e8f0; padding:.25rem 0; margin-top:.25rem;
}

/* Title */
.app-title{display:flex; align-items:center; justify-content:center; gap:12px; margin:.75rem 0 1rem;}
.app-title .ball{font-size:32px; line-height:1; filter:drop-shadow(0 2px 4px rgba(0,0,0,.15));}
.app-title .title{
  font-weight:700; letter-spacing:.05em; font-size:clamp(22px,3.5vw,36px);
  background:linear-gradient(45deg,#0ea5e9,#1e40af,#7c3aed);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 2px 4px rgba(0,0,0,.1);
}

/* Buttons */
.stButton > button, .stDownloadButton > button{
  background:linear-gradient(135deg,#0ea5e9,#3b82f6)!important; color:#fff!important;
  border:0!important; border-radius:12px!important; padding:.6rem 1.2rem!important;
  font-weight:600!important; font-size:.9rem!important; transition:.3s ease!important;
  box-shadow:0 4px 15px rgba(14,165,233,.3)!important;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform:translateY(-2px)!important; box-shadow:0 8px 25px rgba(14,165,233,.4)!important; filter:brightness(1.05)!important;
}

/* Metric card */
.metric-container{ background:linear-gradient(135deg,rgba(14,165,233,.1),rgba(59,130,246,.05));
  border-radius:15px; padding:1.5rem; border-left:4px solid #0ea5e9; box-shadow:0 4px 20px rgba(14,165,233,.1);}

/* Player cards */
.pcard-grid{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); margin-top:.75rem; }
.pcard{ background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:14px; box-shadow:0 8px 18px rgba(0,0,0,.05);
        transition:transform .15s ease, box-shadow .15s ease; }
.pcard:hover{ transform:translateY(-2px); box-shadow:0 12px 24px rgba(0,0,0,.08); }
.pcard h3{ font-size:1rem; line-height:1.25; color:#111827; margin:.25rem 0 .1rem; }
.pcard .sub{ color:#0ea5e9; font-weight:600; }
.pcard .muted{ color:#6b7280; font-size:.85rem; }
.pcard .pill{ display:inline-block; padding:.2rem .55rem; border-radius:999px; background:#f1f5f9; color:#334155; font-size:.75rem; font-weight:600; border:1px solid #e2e8f0; }
.pcard .row{ display:grid; grid-template-columns:1fr auto auto; align-items:center; gap:8px; margin-top:.5rem; }
.pcard .label{ color:#475569; font-size:.9rem; white-space:nowrap; }
.pcard .dotbar{ height:8px; background:#f1f5f9; border-radius:999px; position:relative; overflow:hidden; }
.pcard .dotbar > span{ position:absolute; inset:0; width:var(--pct,0%); background:linear-gradient(90deg,#34d399,#10b981); }
.pcard .action{ margin-top:.5rem; font-size:.8rem; color:#7f1d1d; background:#fee2e2; border:1px solid #fecaca; border-radius:10px; padding:.25rem .5rem; display:inline-block; }

@media (max-width:768px){ .block-container{ padding:1rem .5rem; margin:.5rem; width:95vw; max-width:95vw; } }
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
                "axis": {"labelColor": "#64748b", "titleColor": "#374151", "gridColor": "#f1f5f9",
                         "labelFont": "Poppins", "titleFont": "Poppins"},
                "legend": {"labelColor": "#64748b", "titleColor": "#374151",
                           "labelFont": "Poppins", "titleFont": "Poppins"},
                "range": {"category": ["#0ea5e9","#34d399","#60a5fa","#f59e0b","#f87171","#a78bfa","#fb7185","#4ade80"]},
            }
        },
    )
    alt.themes.enable("tournament_theme")


def add_world_cup_watermark(url: str = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg",
                            opacity: float = 0.10, size: str = "68vmin", y_offset: str = "6vh"):
    st.markdown(
        f"""
<style>
#wc-trophy {{
  position:fixed; inset:0; background:url('{url}') no-repeat center {y_offset} / {size};
  opacity:{opacity}; pointer-events:none; z-index:0;
}}
</style>
<div id="wc-trophy"></div>
""",
        unsafe_allow_html=True,
    )


def notify(msg: str, kind: str = "ok"):
    cls = {"ok": "metric-container", "warn": "status-warn", "err": "status-err"}.get(kind, "metric-container")
    st.markdown(f'<div class="{cls}" style="margin:.5rem 0">{msg}</div>', unsafe_allow_html=True)

# ---------------------- Excel parsing (no xlrd) -------------------
NS_MAIN = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL  = {"rel":  "http://schemas.openxmlformats.org/package/2006/relationships"}
R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

def _parse_shared_strings(z: zipfile.ZipFile) -> list[str]:
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.parse(z.open("xl/sharedStrings.xml")).getroot()
        for si in root.findall(".//main:si", NS_MAIN):
            text = "".join(t.text or "" for t in si.findall(".//main:t", NS_MAIN))
            shared.append(text)
    return shared

def _parse_worksheet(xml_bytes: bytes, shared: list[str]) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    sheet = root.find("main:sheetData", NS_MAIN)
    if sheet is None:
        return pd.DataFrame()
    out_rows, max_col = [], 0
    for row in sheet.findall("main:row", NS_MAIN):
        rd = {}
        for cell in row.findall("main:c", NS_MAIN):
            ref = cell.attrib.get("r", "A1")
            col_letters = "".join(ch for ch in ref if ch.isalpha())
            ci = 0
            for ch in col_letters:
                ci = ci * 26 + (ord(ch) - 64)
            ci -= 1
            v = cell.find("main:v", NS_MAIN)
            val = v.text if v is not None else None
            if cell.attrib.get("t") == "s" and val is not None:
                i = int(val)
                if 0 <= i < len(shared):
                    val = shared[i]
            rd[ci] = val
            max_col = max(max_col, ci)
        out_rows.append(rd)
    if not out_rows:
        return pd.DataFrame()
    matrix = [[r.get(i) for i in range(max_col + 1)] for r in out_rows]
    return pd.DataFrame(matrix)

def parse_xlsx_sheets(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        wb = ET.parse(z.open("xl/workbook.xml")).getroot()
        sheets = wb.findall(".//main:sheets/main:sheet", NS_MAIN)
        id_to_name = {s.attrib.get(R_ID): s.attrib.get("name", f"sheet{s.attrib.get('sheetId','')}") for s in sheets}
        rels = ET.parse(z.open("xl/_rels/workbook.xml.rels")).getroot()
        rid_to_target = {rel.attrib["Id"]: ("xl/" + rel.attrib["Target"]).replace("\\", "/")
                         for rel in rels.findall("rel:Relationship", NS_REL)}
        shared = _parse_shared_strings(z)
        out = {}
        for rid, name in id_to_name.items():
            tgt = rid_to_target.get(rid)
            if tgt and tgt in z.namelist():
                out[name] = _parse_worksheet(z.open(tgt).read(), shared)
        return out

def safe_read_workbook(file_source) -> dict[str, pd.DataFrame]:
    if isinstance(file_source, (str, Path)):
        file_bytes = Path(file_source).read_bytes()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:  # file-like
        file_bytes = file_source.read()

    try:
        xls = pd.ExcelFile(BytesIO(file_bytes))
        return {name: pd.read_excel(xls, sheet_name=name, header=None) for name in xls.sheet_names}
    except Exception:
        return parse_xlsx_sheets(file_bytes)

def find_division_columns(raw_df: pd.DataFrame):
    b_col, a_col = None, None
    for r in range(min(2, len(raw_df))):
        row = raw_df.iloc[r].astype(str).str.strip().str.lower()
        for c, cell in row.items():
            if "b division" in cell and b_col is None: b_col = c
            elif "a division" in cell and a_col is None: a_col = c
    if b_col is None and a_col is None:
        b_col = 0
        a_col = 5 if raw_df.shape[1] >= 8 else (4 if raw_df.shape[1] >= 7 else None)
    return b_col, a_col

def _norm_text(s):
    if pd.isna(s): return None
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

# -------------------------- GOALS sheet --------------------------
def process_goals_sheet(sheet_df: pd.DataFrame) -> pd.DataFrame:
    if sheet_df is None or sheet_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    b_start, a_start = find_division_columns(sheet_df)
    header_row = 1 if len(sheet_df) > 1 else 0
    start_row = header_row + 1

    records = []

    def extract_block(start_col: int | None, division: str):
        if start_col is None: return
        df = sheet_df.iloc[start_row:, start_col:start_col+6].copy()
        df.columns = [f"C{i}" for i in range(df.shape[1])]
        df = df.rename(columns={"C0":"Team","C1":"Player","C2":"Goals","C3":"Appearances","C4":"Yellow Cards","C5":"Red Cards"})
        cols = [c for c in ["Team","Player","Goals","Appearances","Yellow Cards","Red Cards"] if c in df.columns]
        df = df[cols]
        df["Team"] = df["Team"].apply(_norm_text)
        df["Player"] = df["Player"].apply(_norm_text)
        df = df.dropna(subset=["Team","Player"], how="any")
        for c in ["Goals","Appearances","Yellow Cards","Red Cards"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Goals"])
        if df.empty: return
        df["Goals"] = df["Goals"].astype(int)
        df["Division"] = division
        records.extend(df.to_dict("records"))

    extract_block(b_start, "B Division")
    extract_block(a_start, "A Division")

    if not records:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    out = pd.DataFrame(records)

    # Drop optional columns that are fully NaN
    for opt in ["Appearances","Yellow Cards","Red Cards"]:
        if opt in out.columns and out[opt].isna().all(): out = out.drop(columns=[opt])

    for c in ["Team","Player","Division"]: out[c] = out[c].apply(_norm_text)
    return out

# -------------------------- CARDS sheet --------------------------
def _normalize_card_type(x: str | None) -> str | None:
    s = _norm_text(x)
    if not s: return None
    s = s.upper()
    if s.startswith("Y"): return "YELLOW"
    if s.startswith("R"): return "RED"
    return s  # fallback

def process_cards_sheet(sheet_df: pd.DataFrame) -> pd.DataFrame:
    """
    New layout (both divisions side-by-side):
      Team | Player Name | Card Type | Match# | Action
    """
    if sheet_df is None or sheet_df.empty:
        return pd.DataFrame(columns=["Division","Team","Player","Yellow Cards","Red Cards","Card Action"])

    b_start, a_start = find_division_columns(sheet_df)
    header_row = 1 if len(sheet_df) > 1 else 0
    start_row = header_row + 1

    events = []

    def extract_block(start_col: int | None, division: str):
        if start_col is None: return
        df = sheet_df.iloc[start_row:, start_col:start_col+5].copy()
        df.columns = [f"C{i}" for i in range(df.shape[1])]
        df = df.rename(columns={"C0":"Team","C1":"Player","C2":"Card Type","C3":"Match","C4":"Action"})
        for c in ["Team","Player","Action"]:
            if c in df.columns: df[c] = df[c].apply(_norm_text)
        if "Card Type" in df.columns: df["Card Type"] = df["Card Type"].apply(_normalize_card_type)
        if "Match" in df.columns: df["Match"] = pd.to_numeric(df["Match"], errors="coerce").astype("Int64")
        # keep rows that have Player & Team and at least a Card Type (ignore pure blanks)
        df = df.dropna(subset=["Team","Player"], how="any")
        df = df.dropna(subset=["Card Type"], how="all")
        if df.empty: return
        df["Division"] = division
        events.extend(df.to_dict("records"))

    extract_block(b_start, "B Division")
    extract_block(a_start, "A Division")

    if not events:
        return pd.DataFrame(columns=["Division","Team","Player","Yellow Cards","Red Cards","Card Action"])

    cards = pd.DataFrame(events)
    cards["Yellow Cards"] = (cards["Card Type"] == "YELLOW").astype(int)
    cards["Red Cards"]    = (cards["Card Type"] == "RED").astype(int)

    # Collapse per player; keep any action notes (unique, joined)
    def _join_actions(s: pd.Series) -> str | None:
        uniq = [x for x in sorted(set(s.dropna())) if x]
        return " • ".join(uniq) if uniq else None

    grouped = (
        cards.groupby(["Player","Team","Division"], as_index=False)
             .agg(Yellow=("Yellow Cards","sum"),
                  Red=("Red Cards","sum"),
                  CardAction=("Action", _join_actions))
    )
    grouped = grouped.rename(columns={"Yellow":"Yellow Cards","Red":"Red Cards","CardAction":"Card Action"})
    # Ensure ints
    grouped["Yellow Cards"] = grouped["Yellow Cards"].astype(int)
    grouped["Red Cards"]    = grouped["Red Cards"].astype(int)
    return grouped

# --------------------- Merge GOALS + CARDS -----------------------
def merge_goals_cards(goals: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    base = goals.copy() if goals is not None else pd.DataFrame(columns=["Division","Team","Player","Goals"])
    if cards is None or cards.empty:
        merged = base
    else:
        merged = pd.merge(base, cards, on=["Player","Team","Division"], how="outer", suffixes=("", "_cards"))

    for col in ["Goals","Appearances","Yellow Cards","Red Cards"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    # Carry text column if present
    if "Card Action" not in merged.columns and "Card Action_cards" in merged.columns:
        merged["Card Action"] = merged["Card Action_cards"]

    keep = ["Division","Team","Player","Goals"]
    for opt in ["Appearances","Yellow Cards","Red Cards","Card Action"]:
        if opt in merged.columns: keep.append(opt)
    merged = merged[keep].copy()

    for c in ["Division","Team","Player"]:
        merged[c] = merged[c].apply(_norm_text)
    merged["Goals"] = merged["Goals"].fillna(0).astype(int)
    return merged

def build_tournament_dataframe(xlsx_bytes: bytes) -> pd.DataFrame:
    sheets = safe_read_workbook(xlsx_bytes)
    goals_df, cards_df = None, None
    for name, df in sheets.items():
        low = str(name).strip().lower()
        if "goal" in low: goals_df = df
        elif "card" in low: cards_df = df
    if goals_df is None and sheets: goals_df = list(sheets.values())[0]
    if cards_df is None and len(sheets) >= 2: cards_df = list(sheets.values())[1]

    goals = process_goals_sheet(goals_df)
    cards = process_cards_sheet(cards_df)
    return merge_goals_cards(goals, cards)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return build_tournament_dataframe(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division","Team","Player","Goals"])

# --------------------- Analytics helpers -------------------------
def calculate_tournament_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_goals":0,"total_players":0,"total_teams":0,"divisions":0,
                "avg_goals_per_team":0,"top_scorer_goals":0,"competitive_balance":0,
                "total_yellow":0,"total_red":0}

    player_totals = df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
    team_totals   = df.groupby(["Team","Division"])["Goals"].sum().reset_index()

    # Sum cards one row per player to avoid duplicates
    def _sum_cards(col):
        if col not in df.columns: return 0
        dedup = df.drop_duplicates(["Player","Team","Division"])
        return int(dedup[col].fillna(0).astype(int).sum())

    return {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(player_totals),
        "total_teams": len(team_totals),
        "divisions": df["Division"].nunique(),
        "avg_goals_per_team": round(df["Goals"].sum()/max(1,len(team_totals)), 2),
        "top_scorer_goals": int(player_totals["Goals"].max()) if not player_totals.empty else 0,
        "competitive_balance": round(team_totals["Goals"].std(), 2) if len(team_totals)>1 else 0,
        "total_yellow": _sum_cards("Yellow Cards"),
        "total_red":    _sum_cards("Red Cards"),
    }

def get_top_performers(df: pd.DataFrame, top_n: int = 10) -> dict:
    if df.empty: return {"players":pd.DataFrame(), "teams":pd.DataFrame()}
    top_players = (df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
                     .sort_values(["Goals","Player"], ascending=[False,True]).head(top_n))
    top_teams   = (df.groupby(["Team","Division"])["Goals"].sum().reset_index()
                     .sort_values("Goals", ascending=False).head(top_n))
    return {"players": top_players, "teams": top_teams}

def create_division_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    division_stats = (df.groupby("Division")
                        .agg(Goals_sum=("Goals","sum"), Goals_mean=("Goals","mean"),
                             Records=("Goals","count"), Teams=("Team","nunique"),
                             Players=("Player","nunique"))
                        .round(2).reset_index()
                        .rename(columns={"Goals_sum":"Total_Goals","Goals_mean":"Avg_Goals","Records":"Total_Records"}))
    total_goals = division_stats["Total_Goals"].sum()
    division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"]/total_goals*100).round(1) if total_goals else 0
    return division_stats

# ---------------------------- Charts ------------------------------
def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, scheme: str="blues") -> alt.Chart:
    if df.empty: return alt.Chart(pd.DataFrame({"note":["No data"]})).mark_text().encode(text="note:N")
    max_val = int(df[x_col].max()) if not df.empty else 1
    ticks = list(range(0, max_val+1)) if max_val <= 50 else None
    return (alt.Chart(df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=.85, stroke="white", strokeWidth=1)
            .encode(x=alt.X(f"{x_col}:Q", title="Goals", axis=alt.Axis(format="d", tickMinStep=1, values=ticks, gridOpacity=.3),
                            scale=alt.Scale(domainMin=0, nice=False)),
                    y=alt.Y(f"{y_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
                    color=alt.Color(f"{x_col}:Q", scale=alt.Scale(scheme=scheme), legend=None),
                    tooltip=[alt.Tooltip(f"{y_col}:N", title=y_col), alt.Tooltip(f"{x_col}:Q", title="Goals", format="d")])
            .properties(height=max(300, min(600, len(df)*25)), title=alt.TitleParams(text=title, fontSize=16, anchor="start", fontWeight=600))
            .resolve_scale(color="independent"))

def create_division_donut_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty: return alt.Chart(pd.DataFrame({"note":["No data"]})).mark_text().encode(text="note:N")
    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    base = alt.Chart(division_data).properties(width=300, height=300,
                                               title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600))
    outer = base.mark_arc(innerRadius=60, outerRadius=120, stroke="white", strokeWidth=2)\
                .encode(theta=alt.Theta("Goals:Q", title="Goals"),
                        color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9","#f59e0b"]), title="Division"),
                        tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")])
    center = base.mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#1e293b")\
                 .encode(text=alt.value(f"Total\n{int(division_data['Goals'].sum())}"))
    return outer + center

def create_advanced_scatter_plot(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data", x=.5, y=.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note":["No data"]})).mark_text().encode(text="note:N")
    team_stats = df.groupby(["Team","Division"]).agg(Players=("Player","nunique"), Goals=("Goals","sum")).reset_index()
    if PLOTLY_AVAILABLE:
        fig = px.scatter(team_stats, x="Players", y="Goals", color="Division", size="Goals",
                         hover_name="Team", hover_data={"Players":True, "Goals":True},
                         title="Team Performance: Players vs Total Goals")
        fig.update_traces(marker=dict(sizemode="diameter", sizemin=8, sizemax=30, line=dict(width=2, color="white"), opacity=.85))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Poppins", size=12),
                          title=dict(font=dict(size=16, color="#1e293b")),
                          xaxis=dict(title="Players in Team", gridcolor="#f1f5f9", zeroline=False),
                          yaxis=dict(title="Total Goals", gridcolor="#f1f5f9", zeroline=False),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=.5), height=400)
        return fig
    return (alt.Chart(team_stats).mark_circle(size=100, opacity=.85)
            .encode(x=alt.X("Players:Q", title="Players in Team"),
                    y=alt.Y("Goals:Q", title="Total Goals"),
                    color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9","#f59e0b"]), title="Division"),
                    size=alt.Size("Goals:Q", legend=None),
                    tooltip=["Team:N","Division:N","Players:Q","Goals:Q"])
            .properties(title="Team Performance: Players vs Total Goals", height=400))

def create_goals_distribution_histogram(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data", x=.5, y=.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note":["No data"]})).mark_text().encode(text="note:N")
    player_goals = df.groupby(["Player","Team"])["Goals"].sum().values
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Histogram(x=player_goals, nbinsx=max(1,len(set(player_goals))),
                                           marker_line_color="white", marker_line_width=1.5, opacity=.85,
                                           hovertemplate="<b>%{x} Goals</b><br>%{y} Players<extra></extra>")])
        fig.update_layout(title="Distribution of Goals per Player", xaxis_title="Goals per Player",
                          yaxis_title="Number of Players", plot_bgcolor="white", paper_bgcolor="white",
                          font=dict(family="Poppins", size=12), title_font=dict(size=16, color="#1e293b"),
                          xaxis=dict(gridcolor="#f1f5f9", zeroline=False),
                          yaxis=dict(gridcolor="#f1f5f9", zeroline=False), height=400)
        return fig
    hist_df = pd.DataFrame({"Goals": player_goals})
    return (alt.Chart(hist_df).mark_bar(opacity=.85)
            .encode(x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
                    y=alt.Y("count():Q", title="Number of Players"),
                    tooltip=[alt.Tooltip("Goals:Q", bin=True, title="Goals"), alt.Tooltip("count():Q", title="Players")])
            .properties(title="Distribution of Goals per Player", height=400))

# ----------------------------- UI -------------------------------
def display_metric_cards(stats: dict):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)

def create_enhanced_data_table(df: pd.DataFrame, table_type: str="players"):
    if df.empty:
        st.info("📋 No data with current filters."); return
    if table_type == "players":
        summary = df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
        summary = summary.sort_values(["Goals","Player"], ascending=[False,True])
        summary.insert(0, "Rank", range(1, len(summary)+1))
        st.dataframe(summary, use_container_width=True, hide_index=True,
                     column_config={"Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                                    "Player": st.column_config.TextColumn("Player", width="large"),
                                    "Team": st.column_config.TextColumn("Team", width="medium"),
                                    "Division": st.column_config.TextColumn("Division", width="small"),
                                    "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small")})
    else:
        keep = ["Player","Team","Division","Goals"]
        for opt in ["Appearances","Yellow Cards","Red Cards"]: 
            if opt in df.columns: keep.append(opt)
        display_df = df[keep].copy()
        for c in ["Goals","Appearances","Yellow Cards","Red Cards"]:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(display_df[c], errors="coerce").fillna(0).astype(int)
        display_df = display_df.sort_values(["Goals","Player"], ascending=[False,True]).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     column_config={"Player": st.column_config.TextColumn("Player", width="large"),
                                    "Team": st.column_config.TextColumn("Team", width="medium"),
                                    "Division": st.column_config.TextColumn("Division", width="small"),
                                    "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
                                    **({"Appearances": st.column_config.NumberColumn("Appearances", format="%d", width="small")} if "Appearances" in display_df.columns else {}),
                                    **({"Yellow Cards": st.column_config.NumberColumn("Yellow", format="%d", width="small")} if "Yellow Cards" in display_df.columns else {}),
                                    **({"Red Cards": st.column_config.NumberColumn("Red", format="%d", width="small")} if "Red Cards" in display_df.columns else {})})

def render_player_cards(df: pd.DataFrame):
    if df.empty:
        st.info("🔍 No players match your current filters."); return

    work = df.copy()
    for col in ["Appearances","Yellow Cards","Red Cards","Goals"]:
        if col in work.columns: work[col] = pd.to_numeric(work[col], errors="coerce")

    agg = {"Goals": ("Goals","sum")}
    if "Appearances" in work.columns: agg["Appearances"] = ("Appearances","sum")
    if "Yellow Cards" in work.columns: agg["YellowCards"] = ("Yellow Cards","sum")
    if "Red Cards" in work.columns:    agg["RedCards"]   = ("Red Cards","sum")
    if "Card Action" in work.columns:  agg["CardAction"] = ("Card Action", lambda s: " • ".join(sorted({_norm_text(x) for x in s if _norm_text(x)})))

    players = (work.groupby(["Player","Team","Division"]).agg(**agg).reset_index().fillna(0))
    players = players.sort_values(["Goals","Player"], ascending=[False,True]).reset_index(drop=True)
    players.insert(0, "Rank", range(1, len(players)+1))

    max_goals = max(int(players["Goals"].max()), 1)
    max_app = int(players["Appearances"].max()) if "Appearances" in players.columns else 0
    max_yel = int(players["YellowCards"].max()) if "YellowCards" in players.columns else 0
    max_red = int(players["RedCards"].max()) if "RedCards" in players.columns else 0

    html = ['<div class="pcard-grid">']
    for _, row in players.iterrows():
        goals = int(row["Goals"])
        pct_goals = f"{(goals/max_goals)*100:.1f}%"

        html += [
            '<div class="pcard">',
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.25rem">',
            f'<h3 style="margin:0">{row["Player"]}</h3>',
            f'<span class="pill">#{int(row["Rank"])}</span>',
            '</div>',
            f'<div class="sub">{row["Team"]}</div>',
            f'<div class="muted">{row["Division"]}</div>',
            '<div class="row"><div class="label">⚽ Goals</div>',
            f'<div class="dotbar"><span style="--pct:{pct_goals}"></span></div>',
            f'<div class="num">{goals}</div></div>'
        ]

        if "Appearances" in players.columns:
            v = int(row["Appearances"]); pct = f"{(v/max(max_app,1))*100:.1f}%"
            html += ['<div class="row"><div class="label">👕 Appearances</div>',
                     f'<div class="dotbar"><span style="--pct:{pct}"></span></div>',
                     f'<div class="num">{v}</div></div>']
        if "YellowCards" in players.columns:
            v = int(row["YellowCards"]); pct = f"{(v/max(max_yel,1))*100:.1f}%"
            html += ['<div class="row"><div class="label">🟨 Yellow Cards</div>',
                     f'<div class="dotbar"><span style="--pct:{pct}"></span></div>',
                     f'<div class="num">{v}</div></div>']
        if "RedCards" in players.columns:
            v = int(row["RedCards"]); pct = f"{(v/max(max_red,1))*100:.1f}%"
            html += ['<div class="row"><div class="label">🟥 Red Cards</div>',
                     f'<div class="dotbar"><span style="--pct:{pct}"></span></div>',
                     f'<div class="num">{v}</div></div>']

        # Optional action note (from CARDS sheet)
        if "CardAction" in players.columns and _norm_text(row.get("CardAction")):
            html.append(f'<div class="action">🚫 {row["CardAction"]}</div>')
        else:
            html.append('<div style="margin-top:.5rem"><span class="pill">No awards</span></div>')

        html.append('</div>')
    html.append('</div>')
    st.markdown("\n".join(html), unsafe_allow_html=True)

# ------------------------- Downloads ------------------------------
def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
            teams = (filtered_df.groupby(["Team","Division"])
                     .agg(Unique_Players=("Player","nunique"), Total_Records=("Goals","count"),
                          Total_Goals=("Goals","sum"), Avg_Goals=("Goals","mean"), Max_Goals=("Goals","max"))
                     .round(2).reset_index())
            z.writestr("03_teams_detailed_analysis.csv", teams.to_csv(index=False))
            players = filtered_df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
            players = players.sort_values(["Goals","Player"], ascending=[False,True])
            players.insert(0, "Rank", range(1, len(players)+1))
            z.writestr("04_players_ranking.csv", players.to_csv(index=False))
            div_cmp = create_division_comparison(filtered_df)
            if not div_cmp.empty:
                z.writestr("05_division_comparison.csv", div_cmp.to_csv(index=False))
            stats = calculate_tournament_stats(filtered_df)
            z.writestr("06_tournament_statistics.csv", pd.DataFrame([stats]).to_csv(index=False))
        z.writestr("README.txt", f"ABEER BLUESTAR SOCCER FEST 2K25 - Data Package\nGenerated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_download_section(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.subheader("📥 Download Reports")
    st.caption("**Full** = all data ignoring filters, **Filtered** = current view with active filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📊 Data Exports")
        if not full_df.empty:
            st.download_button("⬇️ FULL Dataset (CSV)", full_df.to_csv(index=False),
                               file_name=f"tournament_full_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        if not filtered_df.empty:
            st.download_button("⬇️ FILTERED Dataset (CSV)", filtered_df.to_csv(index=False),
                               file_name=f"tournament_filtered_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with c2:
        st.subheader("🏆 Summary Reports")
        if not filtered_df.empty:
            teams_summary = (filtered_df.groupby(["Team","Division"])
                             .agg(Players_Count=("Player","nunique"), Total_Goals=("Goals","sum")).reset_index()
                             .sort_values("Total_Goals", ascending=False))
            st.download_button("⬇️ TEAMS Summary (CSV)", teams_summary.to_csv(index=False),
                               file_name=f"teams_summary_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
            players_summary = filtered_df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
            players_summary = players_summary.sort_values(["Goals","Player"], ascending=[False,True])
            players_summary.insert(0,"Rank", range(1,len(players_summary)+1))
            st.download_button("⬇️ PLAYERS Summary (CSV)", players_summary.to_csv(index=False),
                               file_name=f"players_summary_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with c3:
        st.subheader("📦 Complete Package")
        if st.button("📦 Generate ZIP Package"):
            z = create_comprehensive_zip_report(full_df, filtered_df)
            st.download_button("⬇️ Download Package (ZIP)", z,
                               file_name=f"tournament_package_{datetime.now():%Y%m%d_%H%M}.zip",
                               mime="application/zip")

# --------------------------- Main --------------------------------
def to_export_xlsx_url(sheet_link_or_id: str) -> str:
    if re.match(r"^[A-Za-z0-9_-]{20,}$", sheet_link_or_id):
        doc_id = sheet_link_or_id
    else:
        m = re.search(r"/d/([A-Za-z0-9_-]+)/", sheet_link_or_id)
        doc_id = m.group(1) if m else sheet_link_or_id
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx"

def main():
    inject_advanced_css()
    st.markdown('<div class="app-title"><span class="ball">⚽</span><span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span></div>', unsafe_allow_html=True)
    add_world_cup_watermark()

    # Your Google Sheet ID (change if needed)
    GOOGLE_SHEET_DOC_ID = "1Bbx7nCS_j7g1wsK3gHpQQnWkpTqlwkHu"
    XLSX_URL = to_export_xlsx_url(GOOGLE_SHEET_DOC_ID)

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
        st.caption(f"🕒 Last refreshed: {st.session_state.get('last_refresh', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        st.divider()

        with st.spinner("📡 Loading tournament data…"):
            tournament_data = fetch_tournament_data(XLSX_URL)
        if tournament_data.empty:
            notify("No tournament data available. Check the sheet link/permissions.", "err")
            st.stop()
        full_data = tournament_data.copy()

        st.header("🔍 Filters")
        divisions = ["All Divisions"] + sorted(tournament_data["Division"].dropna().unique().tolist())
        sel_div = st.selectbox("📊 Division", divisions, key="div")
        if sel_div != "All Divisions":
            tournament_data = tournament_data[tournament_data["Division"] == sel_div]

        teams = sorted(tournament_data["Team"].dropna().unique().tolist())
        sel_teams = st.multiselect("🏆 Teams (optional)", teams, key="teams", placeholder="Type to search teams…")
        if sel_teams: tournament_data = tournament_data[tournament_data["Team"].isin(sel_teams)]

        players = sorted(tournament_data["Player"].dropna().astype(str).unique().tolist())
        sel_players = st.multiselect("👤 Players (optional)", players, key="players", placeholder="Type to search players…")
        if sel_players: tournament_data = tournament_data[tournament_data["Player"].isin(sel_players)]

        # Cards filter (now based on new CARDS sheet)
        for c in ["Yellow Cards","Red Cards"]:
            if c not in tournament_data.columns: tournament_data[c] = 0
        st.subheader("🟨🟥 Cards filter")
        card_filter = st.selectbox("Show players with…", ["All", "Yellow only", "Red only", "Any card"], index=0, key="cards_filter")
        if card_filter == "Yellow only":
            tournament_data = tournament_data[tournament_data["Yellow Cards"] > 0]
        elif card_filter == "Red only":
            tournament_data = tournament_data[tournament_data["Red Cards"] > 0]
        elif card_filter == "Any card":
            tournament_data = tournament_data[(tournament_data["Yellow Cards"] > 0) | (tournament_data["Red Cards"] > 0)]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 OVERVIEW","⚡ QUICK INSIGHTS","🏆 TEAMS","👤 PLAYERS","📈 ANALYTICS","📥 DOWNLOADS"])

    stats = calculate_tournament_stats(tournament_data)
    top = get_top_performers(tournament_data, 10)

    with tab1:
        st.header("📊 Tournament Overview")
        display_metric_cards(stats)
        st.divider()
        c1, c2 = st.columns([2,1])
        with c1:
            st.subheader("🎯 Goal Scoring Records")
            create_enhanced_data_table(tournament_data, "records")
        with c2:
            if not tournament_data.empty:
                st.subheader("🏁 Division Distribution")
                st.altair_chart(create_division_donut_chart(tournament_data), use_container_width=True)
        if not tournament_data.empty:
            a, b = st.columns(2)
            with a:
                st.subheader("🏆 Goals by Team")
                team_goals = (tournament_data.groupby("Team")["Goals"].sum().reset_index()
                              .sort_values("Goals", ascending=False).head(10))
                if not team_goals.empty:
                    st.altair_chart(create_horizontal_bar_chart(team_goals, "Goals", "Team", "Top 10 Teams by Goals", "blues"), use_container_width=True)
            with b:
                st.subheader("⚽ Top Scorers")
                if not top["players"].empty:
                    tdf = top["players"].head(10).copy()
                    tdf["Display"] = tdf["Player"] + " (" + tdf["Team"] + ")"
                    st.altair_chart(create_horizontal_bar_chart(tdf, "Goals", "Display", "Top 10 Players by Goals", "greens"), use_container_width=True)

    with tab2:
        st.header("⚡ Quick Tournament Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Total Goals", stats["total_goals"])
        c2.metric("👥 Active Players", stats["total_players"])
        c3.metric("🏆 Teams", stats["total_teams"])
        c4.metric("📊 Divisions", stats["divisions"])
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("🟨 Yellow Cards", stats["total_yellow"])
        d2.metric("🟥 Red Cards", stats["total_red"])
        d3.metric("⚖️ Goals σ", stats["competitive_balance"])
        d4.metric("🎯 Avg Goals/Team", stats["avg_goals_per_team"])
        st.caption("Cards are summed from the CARDS sheet; each player counted once under current filters.")
        st.divider()

    with tab3:
        st.header("🏆 Teams Analysis")
        if tournament_data.empty:
            st.info("🔍 No teams match your current filters.")
        else:
            st.subheader("📋 Teams Summary")
            ts = (tournament_data.groupby(["Team","Division"])
                  .agg(Players=("Player","nunique"), Total_Goals=("Goals","sum")).reset_index())
            st.dataframe(ts.sort_values("Total_Goals", ascending=False), use_container_width=True, hide_index=True,
                         column_config={"Team": st.column_config.TextColumn("Team", width="medium"),
                                        "Division": st.column_config.TextColumn("Division", width="small"),
                                        "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                                        "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small")})
            st.divider()
            st.subheader("📊 Team Performance Analysis")
            if not ts.empty:
                st.altair_chart(create_horizontal_bar_chart(ts.head(15), "Total_Goals", "Team", "Team Goals Distribution", "viridis"), use_container_width=True)

    with tab4:
        st.header("👤 Players (Card View)")
        render_player_cards(tournament_data)

    with tab5:
        st.header("📈 Advanced Analytics")
        if tournament_data.empty:
            st.info("🔍 No data available for analytics with current filters.")
        else:
            a, b = st.columns(2)
            with a:
                st.subheader("📊 Goals Distribution")
                dist = create_goals_distribution_histogram(tournament_data)
                (st.plotly_chart if PLOTLY_AVAILABLE else st.altair_chart)(dist, use_container_width=True)
            with b:
                st.subheader("🎯 Team Performance Matrix")
                scatter = create_advanced_scatter_plot(tournament_data)
                (st.plotly_chart if PLOTLY_AVAILABLE else st.altair_chart)(scatter, use_container_width=True)

    with tab6:
        create_download_section(full_data, tournament_data)

# ------------------------- Entry point ----------------------------
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.exception(e)
