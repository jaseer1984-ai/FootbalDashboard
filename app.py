# ABEER BLUESTAR SOCCER FEST 2K25 — Streamlit Dashboard (Players Card View + CARDS v2 + POINT TABLE from Google Sheets)
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

# Optional imports (plotly used in Analytics tab if available)
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
        :root{ --sticky-tabs-top: 52px; }

        .stApp { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .block-container {
            padding-top: 0.5rem; padding-bottom: 2rem;
            max-width: 98vw; width: 98vw;
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            margin: 1rem auto;
            position: relative; z-index: 1;
        }

        #MainMenu, footer, .stDeployButton,
        div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { display:none !important; }

        /* Sticky tabs */
        .block-container [data-testid="stTabs"]:first-of-type{
            position: sticky; top: var(--sticky-tabs-top); z-index: 6;
            background: rgba(255,255,255,0.96); backdrop-filter: blur(8px);
            border-bottom: 1px solid #e2e8f0; padding:.25rem 0; margin-top:.25rem;
        }

        /* Title */
        .app-title{ display:flex; align-items:center; justify-content:center; gap:12px; margin:.75rem 0 1rem; }
        .app-title .ball{ font-size:32px; line-height:1; filter: drop-shadow(0 2px 4px rgba(0,0,0,.15)); }
        .app-title .title{
            font-weight:700; letter-spacing:.05em; font-size:clamp(22px,3.5vw,36px);
            background: linear-gradient(45deg,#0ea5e9,#1e40af,#7c3aed);
            -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Buttons */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg,#0ea5e9,#3b82f6)!important; color:#fff!important; border:0!important;
            border-radius:12px!important; padding:.6rem 1.2rem!important; font-weight:600!important; font-size:.9rem!important;
            transition:.3s ease!important; box-shadow:0 4px 15px rgba(14,165,233,.3)!important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px)!important; box-shadow: 0 8px 25px rgba(14,165,233,.4)!important; filter:brightness(1.05)!important;
        }

        /* Status pill (notify) */
        .status-pill{ padding:.5rem .75rem; border-radius:.6rem; font-size:.85rem; margin-top:.5rem; }
        .status-ok{ background:#ecfeff; border-left:4px solid #06b6d4; color:#155e75; }
        .status-warn{ background:#fef9c3; border-left:4px solid #f59e0b; color:#713f12; }
        .status-err{ background:#fee2e2; border-left:4px solid #ef4444; color:#7f1d1d; }

        /* ---------- Player Card CSS ---------- */
        .pcard-grid{
            display:grid; gap:16px;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            margin-top:.75rem;
        }
        .pcard{
            background:#fff; border:1px solid #e5e7eb; border-radius:14px;
            padding:14px; box-shadow:0 8px 18px rgba(0,0,0,.05);
            transition:transform .15s ease, box-shadow .15s ease;
        }
        .pcard:hover{ transform: translateY(-2px); box-shadow:0 12px 24px rgba(0,0,0,.08); }
        .pcard h3{ font-size:1rem; line-height:1.25; color:#111827; margin:.25rem 0 .1rem; }
        .pcard .sub{ color:#0ea5e9; font-weight:600; }
        .pcard .muted{ color:#6b7280; font-size:.85rem; }
        .pcard .pill{
            display:inline-block; padding:.2rem .55rem; border-radius:999px;
            background:#f1f5f9; color:#334155; font-size:.75rem; font-weight:600;
            border:1px solid #e2e8f0;
        }
        .pcard .row{ display:grid; grid-template-columns: 1fr auto auto; align-items:center; gap:8px;
                     margin-top:.5rem; }
        .pcard .label{ color:#475569; font-size:.9rem; white-space:nowrap; }
        .pcard .dotbar{ height:8px; background: #f1f5f9; border-radius: 999px; position:relative; overflow:hidden; }
        .pcard .dotbar > span{
            position:absolute; inset:0; width:var(--pct,0%);
            background: linear-gradient(90deg,#34d399,#10b981);
        }
        .pcard .action{
            margin-top:.5rem; margin-right:.4rem; font-size:.78rem; color:#1f2937;
            background:#f8fafc; border:1px solid #e5e7eb; border-radius:999px; padding:.18rem .55rem;
            display:inline-block;
        }

        /* Hide heading link icons */
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display:none !important; }

        @media (max-width:768px){
            .block-container{ padding: 1rem .5rem; margin:.5rem; width:95vw; max-width:95vw; }
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
                    "category": ["#0ea5e9","#34d399","#60a5fa","#f59e0b","#f87171","#a78bfa","#fb7185","#4ade80"]
                },
            }
        },
    )
    alt.themes.enable("tournament_theme")


def add_world_cup_watermark(image_url: str = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg",
                            opacity: float = 0.10, size: str = "68vmin", y_offset: str = "6vh"):
    st.markdown(
        f"""
    <style>
      #wc-trophy {{
        position: fixed; inset: 0;
        background-image: url('{image_url}');
        background-repeat: no-repeat; background-position: center {y_offset}; background-size: {size};
        opacity: {opacity}; pointer-events: none; z-index: 0;
      }}
    </style>
    <div id="wc-trophy"></div>
    """,
        unsafe_allow_html=True,
    )


def notify(msg: str, kind: str = "ok"):
    cls = {"ok": "status-ok", "warn": "status-warn", "err": "status-err"}.get(kind, "status-ok")
    st.markdown(f'<div class="status-pill {cls}">{msg}</div>', unsafe_allow_html=True)


# ====================== DATA LOADING (multi-sheet, no xlrd) =======
NS_MAIN = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

def _parse_shared_strings(z: zipfile.ZipFile) -> list[str]:
    shared: list[str] = []
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
    rows, max_col = [], 0
    for row in sheet.findall("main:row", NS_MAIN):
        rd = {}
        for cell in row.findall("main:c", NS_MAIN):
            ref = cell.attrib.get("r", "A1")
            col_letters = "".join(ch for ch in ref if ch.isalpha())
            col_idx = 0
            for ch in col_letters:
                col_idx = col_idx * 26 + (ord(ch) - 64)
            col_idx -= 1
            ctype = cell.attrib.get("t")
            v = cell.find("main:v", NS_MAIN)
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

def parse_xlsx_sheets(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: DataFrame} for all sheets in workbook."""
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        wb = ET.parse(z.open("xl/workbook.xml")).getroot()
        sheets = wb.findall(".//main:sheets/main:sheet", NS_MAIN)
        id_to_name = {s.attrib.get(R_ID): s.attrib.get("name", f"sheet{s.attrib.get('sheetId','')}") for s in sheets}
        rels = ET.parse(z.open("xl/_rels/workbook.xml.rels")).getroot()
        rid_to_target = {
            rel.attrib["Id"]: ("xl/" + rel.attrib["Target"]).replace("\\", "/")
            for rel in rels.findall("rel:Relationship", NS_REL)
        }
        shared = _parse_shared_strings(z)
        out: dict[str, pd.DataFrame] = {}
        for rid, name in id_to_name.items():
            target = rid_to_target.get(rid)
            if target and target in z.namelist():
                df = _parse_worksheet(z.open(target).read(), shared)
                out[name] = df
        return out

def safe_read_workbook(file_source) -> dict[str, pd.DataFrame]:
    """
    Read entire workbook into a dict of sheet DataFrames (header=None).
    Uses pandas if available; falls back to our XML parser.
    """
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        file_bytes = file_source.read()

    try:
        xls = pd.ExcelFile(BytesIO(file_bytes))
        return {name: pd.read_excel(xls, sheet_name=name, header=None) for name in xls.sheet_names}
    except Exception:
        return parse_xlsx_sheets(file_bytes)

def find_division_columns(raw_df: pd.DataFrame):
    """Find columns where 'B Division' and 'A Division' headers appear."""
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

def _norm_text(s):
    if pd.isna(s):
        return None
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

# --------- GOALS sheet ----------
def process_goals_sheet(sheet_df: pd.DataFrame) -> pd.DataFrame:
    """Extract normalized records from GOALS sheet blocks."""
    if sheet_df is None or sheet_df.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    b_start, a_start = find_division_columns(sheet_df)
    header_row = 1 if len(sheet_df) > 1 else 0
    data_start_row = header_row + 1

    records = []

    def extract_block(start_col: int | None, division_name: str):
        if start_col is None:
            return
        df = sheet_df.iloc[data_start_row:, start_col : start_col + 6].copy()
        df.columns = [f"C{i}" for i in range(df.shape[1])]
        df = df.rename(columns={"C0": "Team", "C1": "Player", "C2": "Goals",
                                "C3": "Appearances", "C4": "Yellow Cards", "C5": "Red Cards"})
        needed = ["Team", "Player", "Goals", "Appearances", "Yellow Cards", "Red Cards"]
        df = df[[c for c in needed if c in df.columns]]
        df["Team"] = df["Team"].apply(_norm_text)
        df["Player"] = df["Player"].apply(_norm_text)
        df = df.dropna(subset=["Team", "Player"], how="any")
        for c in ["Goals", "Appearances", "Yellow Cards", "Red Cards"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Goals"])
        if df.empty:
            return
        df["Goals"] = df["Goals"].astype(int)
        df["Division"] = division_name
        records.extend(df.to_dict("records"))

    extract_block(b_start, "B Division")
    extract_block(a_start, "A Division")

    if not records:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    out = pd.DataFrame(records)

    # Drop optional columns that are entirely NaN
    for opt in ["Appearances", "Yellow Cards", "Red Cards"]:
        if opt in out.columns and out[opt].isna().all():
            out = out.drop(columns=[opt])

    for c in ["Team", "Player", "Division"]:
        out[c] = out[c].apply(_norm_text)
    return out

# --------- CARDS sheet (Team | Player | Card Type | Match# | Action) ----------
def process_cards_sheet(sheet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract normalized card events from CARDS sheet (two blocks).
    Returns per-player aggregated counts + latest match/action + event summary.
    """
    if sheet_df is None or sheet_df.empty:
        return pd.DataFrame(columns=["Division","Team","Player","Yellow Cards","Red Cards","Last Match","Last Action","Card Events"])

    b_start, a_start = find_division_columns(sheet_df)
    header_row = 1 if len(sheet_df) > 1 else 0
    data_start_row = header_row + 1

    rows = []

    def extract_block(start_col: int | None, division_name: str):
        if start_col is None:
            return
        df = sheet_df.iloc[data_start_row:, start_col : start_col + 5].copy()
        df.columns = [f"C{i}" for i in range(df.shape[1])]
        df = df.rename(columns={"C0": "Team", "C1": "Player", "C2": "Card_Type", "C3": "Match", "C4": "Action"})
        df["Team"] = df["Team"].apply(_norm_text)
        df["Player"] = df["Player"].apply(_norm_text)
        df["Card_Type"] = df["Card_Type"].apply(lambda x: (_norm_text(x) or "").upper())
        df["Action"] = df["Action"].apply(_norm_text)
        df["Match"] = pd.to_numeric(df["Match"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Team", "Player", "Card_Type"])
        if df.empty:
            return
        df["Division"] = division_name

        def make_evt(row):
            m = int(row["Match"]) if pd.notna(row["Match"]) else None
            ct = row["Card_Type"].title() if row["Card_Type"] else "Card"
            act = row["Action"]
            left = f"M{m}: {ct}" if m is not None else ct
            return f"{left}" + (f" — {act}" if act else "")

        df["Event"] = df.apply(make_evt, axis=1)
        rows.append(df)

    extract_block(b_start, "B Division")
    extract_block(a_start, "A Division")

    if not rows:
        return pd.DataFrame(columns=["Division","Team","Player","Yellow Cards","Red Cards","Last Match","Last Action","Card Events"])

    events = pd.concat(rows, ignore_index=True)

    keys = ["Player","Team","Division"]

    counts = (
        events.assign(Yellow=(events["Card_Type"]=="YELLOW").astype(int),
                      Red=(events["Card_Type"]=="RED").astype(int))
              .groupby(keys, as_index=False)[["Yellow","Red"]].sum()
              .rename(columns={"Yellow":"Yellow Cards","Red":"Red Cards"})
    )

    last = (
        events.sort_values(keys+["Match"])
              .groupby(keys, as_index=False)
              .tail(1)[keys+["Match","Action"]]
              .rename(columns={"Match":"Last Match","Action":"Last Action"})
    )

    evsum = (
        events.sort_values(keys+["Match"])
              .groupby(keys)["Event"]
              .apply(lambda s: " • ".join([str(x) for x in s if _norm_text(x)]))
              .reset_index()
              .rename(columns={"Event":"Card Events"})
    )

    out = counts.merge(last, on=keys, how="left").merge(evsum, on=keys, how="left")
    out["Yellow Cards"] = out["Yellow Cards"].astype(int)
    out["Red Cards"] = out["Red Cards"].astype(int)
    out["Last Match"] = pd.to_numeric(out["Last Match"], errors="coerce").astype("Int64")
    return out

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s is not None else ""

def _norm_div_key(s: str) -> str:
    """Normalize any division-like label to 'adivision' or 'bdivision'."""
    s2 = str(s).lower().replace("-", " ")
    if "a" in s2 and "division" in s2:
        return "adivision"
    if "b" in s2 and "division" in s2:
        return "bdivision"
    return re.sub(r"[^a-z0-9]", "", s2)

def canonicalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Unify Team and Player names case-insensitively (use most frequent variant)."""
    out = df.copy()

    def canonize_column(col: str):
        if col not in out.columns:
            return
        tmp = out.dropna(subset=[col]).copy()
        if tmp.empty:
            return
        tmp["k"] = tmp[col].apply(_norm_key)
        # Choose the variant that appears most; tie-break by alphabetical
        pref = (
            tmp.groupby(["k", col]).size().reset_index(name="n")
               .sort_values(["k", "n", col], ascending=[True, False, True])
               .drop_duplicates("k")
        )
        mapping = dict(zip(pref["k"], pref[col]))
        out[col] = out[col].apply(lambda x: mapping.get(_norm_key(x), x) if pd.notna(x) else x)

    canonize_column("Team")
    canonize_column("Player")

    # Add keys for convenience (used in merges)
    if "Team" in out.columns:
        out["TeamKey"] = out["Team"].apply(_norm_key)
    if "Player" in out.columns:
        out["PlayerKey"] = out["Player"].apply(_norm_key)
    if "Division" in out.columns:
        out["DivKey"] = out["Division"].apply(_norm_div_key)

    return out

def merge_goals_cards(goals: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge so card-only players also appear (goals=0); keep last/action/events."""
    base = goals.copy() if goals is not None else pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    merged = (
        base.merge(cards, on=["Player","Team","Division"], how="outer", suffixes=("","_cards"))
        if cards is not None else base
    )

    # Normalize numerics
    for col in ["Goals", "Appearances", "Yellow Cards", "Red Cards"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    keep = ["Division","Team","Player","Goals"]
    for opt in ["Appearances","Yellow Cards","Red Cards","Last Match","Last Action","Card Events"]:
        if opt in merged.columns:
            keep.append(opt)
    merged = merged[keep].copy()

    # Clean text and avoid "None" strings
    for c in ["Division","Team","Player","Last Action","Card Events"]:
        if c in merged.columns:
            merged[c] = merged[c].apply(_norm_text)
    if "Last Match" in merged.columns:
        merged["Last Match"] = pd.to_numeric(merged["Last Match"], errors="coerce").astype("Int64")
    merged["Goals"] = merged["Goals"].fillna(0).astype(int)

    # 👉 unify Team/Player names (case-insensitive)
    merged = canonicalize_names(merged)
    return merged

def build_tournament_dataframe(xlsx_bytes: bytes) -> pd.DataFrame:
    """Read workbook bytes, process GOALS & CARDS, and merge into a clean dataset."""
    sheets = safe_read_workbook(xlsx_bytes)
    goals_df = None
    cards_df = None
    for name, df in sheets.items():
        low = str(name).strip().lower()
        if "goal" in low:
            goals_df = df
        elif "card" in low:
            cards_df = df
    if goals_df is None and sheets:
        goals_df = list(sheets.values())[0]
    if cards_df is None and len(sheets) >= 2:
        cards_df = list(sheets.values())[1]

    goals = process_goals_sheet(goals_df)
    cards = process_cards_sheet(cards_df)
    return merge_goals_cards(goals, cards)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.content:
            raise ValueError("Downloaded file is empty")
        return build_tournament_dataframe(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

# ====================== ANALYTICS HELPERS =========================
def calculate_tournament_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_goals": 0, "total_players": 0, "total_teams": 0, "divisions": 0,
            "avg_goals_per_team": 0, "top_scorer_goals": 0, "competitive_balance": 0,
            "total_yellow": 0, "total_red": 0,
        }
    player_totals = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
    team_totals = df.groupby(["Team", "Division"])["Goals"].sum().reset_index()

    dedup = df.drop_duplicates(subset=["Player","Team","Division"]).copy()
    total_yellow = int(dedup["Yellow Cards"].fillna(0).astype(int).sum()) if "Yellow Cards" in dedup.columns else 0
    total_red    = int(dedup["Red Cards"].fillna(0).astype(int).sum()) if "Red Cards" in dedup.columns else 0

    return {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(player_totals),
        "total_teams": len(team_totals),
        "divisions": df["Division"].nunique(),
        "avg_goals_per_team": round(df["Goals"].sum() / max(1, len(team_totals)), 2),
        "top_scorer_goals": int(player_totals["Goals"].max()) if not player_totals.empty else 0,
        "competitive_balance": round(team_totals["Goals"].std(), 2) if len(team_totals) > 1 else 0,
        "total_yellow": total_yellow,
        "total_red": total_red,
    }

def get_top_performers(df: pd.DataFrame, top_n: int = 10) -> dict:
    if df.empty:
        return {"players": pd.DataFrame(), "teams": pd.DataFrame()}
    top_players = (
        df.groupby(["Player", "Team", "Division"])["Goals"]
        .sum()
        .reset_index()
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
    division_stats = (
        df.groupby("Division")
        .agg(Goals_sum=("Goals", "sum"), Goals_mean=("Goals", "mean"), Records=("Goals", "count"),
             Teams=("Team", "nunique"), Players=("Player", "nunique"))
        .round(2)
        .reset_index()
        .rename(columns={"Goals_sum": "Total_Goals", "Goals_mean": "Avg_Goals", "Records": "Total_Records"})
    )
    total_goals = division_stats["Total_Goals"].sum()
    division_stats["Goal_Share_Pct"] = (division_stats["Total_Goals"] / total_goals * 100).round(1) if total_goals else 0
    return division_stats

# ====================== CHART HELPERS =============================
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
    base = alt.Chart(division_data).properties(width=300, height=300,
                                               title=alt.TitleParams(text="Goals Distribution by Division", fontSize=16, fontWeight=600))
    outer = (
        base.mark_arc(innerRadius=60, outerRadius=120, stroke="white", strokeWidth=2)
        .encode(theta=alt.Theta("Goals:Q", title="Goals"),
                color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]), title="Division"),
                tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")])
    )
    center_text = base.mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#1e293b") \
                      .encode(text=alt.value(f"Total\n{int(division_data['Goals'].sum())}"))
    return outer + center_text

def create_advanced_scatter_plot(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")
    team_stats = df.groupby(["Team", "Division"]).agg(Players=("Player", "nunique"), Goals=("Goals", "sum")).reset_index()
    if PLOTLY_AVAILABLE:
        fig = px.scatter(team_stats, x="Players", y="Goals", color="Division", size="Goals",
                         hover_name="Team", hover_data={"Players": True, "Goals": True},
                         title="Team Performance: Players vs Total Goals")
        fig.update_traces(marker=dict(sizemode="diameter", sizemin=8, sizemax=30, line=dict(width=2, color="white"), opacity=0.85))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Poppins", size=12),
                          title=dict(font=dict(size=16, color="#1e293b")),
                          xaxis=dict(title="Number of Players in Team", gridcolor="#f1f5f9", zeroline=False),
                          yaxis=dict(title="Total Goals", gridcolor="#f1f5f9", zeroline=False),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), height=400)
        return fig
    return (
        alt.Chart(team_stats).mark_circle(size=100, opacity=0.85)
        .encode(x=alt.X("Players:Q", title="Number of Players in Team"),
                y=alt.Y("Goals:Q", title="Total Goals"),
                color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9", "#f59e0b"]), title="Division"),
                size=alt.Size("Goals:Q", legend=None),
                tooltip=["Team:N", "Division:N", "Players:Q", "Goals:Q"])
        .properties(title="Team Performance: Players vs Total Goals", height=400)
    )

def create_goals_distribution_histogram(df: pd.DataFrame):
    if df.empty:
        if PLOTLY_AVAILABLE:
            fig = go.Figure(); fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False); return fig
        return alt.Chart(pd.DataFrame({"note": ["No data available"]})).mark_text().encode(text="note:N")
    player_goals = df.groupby(["Player", "Team"])["Goals"].sum().values
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Histogram(x=player_goals, nbinsx=max(1, len(set(player_goals))),
                                           marker_line_color="white", marker_line_width=1.5, opacity=0.85,
                                           hovertemplate="<b>%{x} Goals</b><br>%{y} Players<extra></extra>")])
        fig.update_layout(title="Distribution of Goals per Player", xaxis_title="Goals per Player", yaxis_title="Number of Players",
                          plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Poppins", size=12),
                          title_font=dict(size=16, color="#1e293b"),
                          xaxis=dict(gridcolor="#f1f5f9", zeroline=False),
                          yaxis=dict(gridcolor="#f1f5f9", zeroline=False), height=400)
        return fig
    hist_df = pd.DataFrame({"Goals": player_goals})
    return (
        alt.Chart(hist_df).mark_bar(opacity=0.85)
        .encode(x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
                y=alt.Y("count():Q", title="Number of Players"),
                tooltip=[alt.Tooltip("Goals:Q", bin=True, title="Goals"), alt.Tooltip("count():Q", title="Players")])
        .properties(title="Distribution of Goals per Player", height=400)
    )

# ====================== UI HELPERS ================================
def display_metric_cards(stats: dict):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_goals']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_players']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.5rem;font-weight:700;color:#0ea5e9;margin-bottom:.5rem;">{stats['total_teams']}</div><div style="color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">TEAMS</div></div>""", unsafe_allow_html=True)

def create_enhanced_data_table(df: pd.DataFrame, table_type: str = "players"):
    if df.empty:
        st.info("📋 No data with current filters.")
        return
    if table_type == "players":
        players_summary = df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
        players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
        players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
        st.dataframe(
            players_summary,
            use_container_width=True, hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
            },
        )
    elif table_type == "records":
        keep = ["Player", "Team", "Division", "Goals"]
        for opt in ["Appearances", "Yellow Cards", "Red Cards", "Last Match", "Last Action", "Card Events"]:
            if opt in df.columns:
                keep.append(opt)
        display_df = df[keep].copy()

        # Last Match as text with '-' for missing
        if "Last Match" in display_df.columns:
            lm = pd.to_numeric(display_df["Last Match"], errors="coerce")
            display_df["Last Match"] = lm.apply(lambda x: "-" if pd.isna(x) else str(int(x)))

        # Coerce numeric columns
        for c in ["Goals", "Appearances", "Yellow Cards", "Red Cards"]:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(display_df[c], errors="coerce").fillna(0).astype(int)

        # Replace None with '-' in object columns
        for c in display_df.select_dtypes(include="object").columns:
            display_df[c] = display_df[c].replace({None: "-"}).fillna("-")

        display_df = display_df.sort_values(["Goals", "Player"], ascending=[False, True]).reset_index(drop=True)

        st.dataframe(
            display_df, use_container_width=True, hide_index=True,
            column_config={
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Division": st.column_config.TextColumn("Division", width="small"),
                "Goals": st.column_config.NumberColumn("Goals", format="%d", width="small"),
                **({"Appearances": st.column_config.NumberColumn("Appearances", format="%d", width="small")} if "Appearances" in display_df.columns else {}),
                **({"Yellow Cards": st.column_config.NumberColumn("Yellow", format="%d", width="small")} if "Yellow Cards" in display_df.columns else {}),
                **({"Red Cards": st.column_config.NumberColumn("Red", format="%d", width="small")} if "Red Cards" in display_df.columns else {}),
                # ⬇️ renamed labels here
                **({"Last Match": st.column_config.TextColumn("Match", width="small")} if "Last Match" in display_df.columns else {}),
                **({"Last Action": st.column_config.TextColumn("Action", width="large")} if "Last Action" in display_df.columns else {}),
                **({"Card Events": st.column_config.TextColumn("Card Events", width="large")} if "Card Events" in display_df.columns else {}),
            },
        )


# ---------- Players Card View renderer ----------
def render_player_cards(df: pd.DataFrame):
    """
    Renders players as responsive cards.
    Shows Appearances/Yellow/Red + Action/Match chips if those columns exist.
    """
    if df.empty:
        st.info("🔍 No players match your current filters.")
        return

    work = df.copy()

    for col in ["Appearances", "Yellow Cards", "Red Cards", "Goals"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg_map = {"Goals": ("Goals", "sum")}
    if "Appearances" in work.columns: agg_map["Appearances"] = ("Appearances", "sum")
    if "Yellow Cards" in work.columns: agg_map["YellowCards"] = ("Yellow Cards", "sum")
    if "Red Cards" in work.columns:    agg_map["RedCards"]   = ("Red Cards", "sum")
    if "Last Match" in work.columns:   agg_map["LastMatch"]  = ("Last Match", "max")
    if "Last Action" in work.columns:  agg_map["LastAction"] = ("Last Action", "first")
    if "Card Events" in work.columns:  agg_map["CardEvents"] = ("Card Events", "first")

    players = (
        work.groupby(["Player", "Team", "Division"])
            .agg(**agg_map)
            .reset_index()
            .fillna(0)
    )

    players = players.sort_values(["Goals", "Player"], ascending=[False, True]).reset_index(drop=True)
    players.insert(0, "Rank", range(1, len(players) + 1))

    max_goals = max(int(players["Goals"].max()), 1)
    max_app = int(players["Appearances"].max()) if "Appearances" in players.columns else 0
    max_yel = int(players["YellowCards"].max()) if "YellowCards" in players.columns else 0
    max_red = int(players["RedCards"].max()) if "RedCards" in players.columns else 0

    html = ['<div class="pcard-grid">']
    for _, row in players.iterrows():
        goals = int(row["Goals"])
        pct_goals = f"{(goals/max_goals)*100:.1f}%" if max_goals else "0%"

        html.append('<div class="pcard">')
        html.append('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.25rem">')
        html.append(f'<h3 style="margin:0">{row["Player"]}</h3>')
        html.append(f'<span class="pill" title="Rank">#{int(row["Rank"])}</span>')
        html.append('</div>')
        html.append(f'<div class="sub">{row["Team"]}</div>')
        html.append(f'<div class="muted">{row["Division"]}</div>')

        html.append('<div class="row">')
        html.append('<div class="label">⚽ Goals</div>')
        html.append(f'<div class="dotbar"><span style="--pct:{pct_goals}"></span></div>')
        html.append(f'<div class="num">{goals}</div>')
        html.append('</div>')

        if "Appearances" in players.columns:
            val = int(row["Appearances"])
            pct = f"{(val/max(max_app,1))*100:.1f}%"
            html.append('<div class="row">')
            html.append('<div class="label">👕 Appearances</div>')
            html.append(f'<div class="dotbar"><span style="--pct:{pct}"></span></div>')
            html.append(f'<div class="num">{val}</div>')
            html.append('</div>')

        if "YellowCards" in players.columns:
            val = int(row["YellowCards"])
            pct = f"{(val/max(max_yel,1))*100:.1f}%"
            html.append('<div class="row">')
            html.append('<div class="label">🟨 Yellow Cards</div>')
            html.append(f'<div class="dotbar"><span style="--pct:{pct}"></span></div>')
            html.append(f'<div class="num">{val}</div>')
            html.append('</div>')

        if "RedCards" in players.columns:
            val = int(row["RedCards"])
            pct = f"{(val/max(max_red,1))*100:.1f}%"
            html.append('<div class="row">')
            html.append('<div class="label">🟥 Red Cards</div>')
            html.append(f'<div class="dotbar"><span style="--pct:{pct}"></span></div>')
            html.append(f'<div class="num">{val}</div>')
            html.append('</div>')

        chips = []
        if "CardEvents" in players.columns and pd.notna(row.get("CardEvents", None)):
            for piece in str(row["CardEvents"]).split(" • "):
                if piece.strip():
                    chips.append(f'<span class="action">{piece.strip()}</span>')
        elif "LastMatch" in players.columns or "LastAction" in players.columns:
            lm = int(row["LastMatch"]) if "LastMatch" in players.columns and pd.notna(row["LastMatch"]) else None
            la = row.get("LastAction", None)
            if lm is not None or la:
                msg = (f"M{lm}: " if lm is not None else "") + (la or "Card recorded")
                chips.append(f'<span class="action">{msg}</span>')

        if chips:
            html.append('<div style="margin-top:.5rem">' + "".join(chips) + "</div>")
        else:
            html.append('<div style="margin-top:.5rem"><span class="pill">No awards</span></div>')

        html.append('</div>')
    html.append('</div>')

    st.markdown("\n".join(html), unsafe_allow_html=True)

# ====================== DOWNLOAD PACKAGE ==========================
def create_comprehensive_zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
            teams = (
                filtered_df.groupby(["Team", "Division"])
                .agg(Unique_Players=("Player", "nunique"), Total_Records=("Goals", "count"),
                     Total_Goals=("Goals", "sum"), Avg_Goals=("Goals", "mean"), Max_Goals=("Goals", "max"))
                .round(2)
                .reset_index()
            )
            z.writestr("03_teams_detailed_analysis.csv", teams.to_csv(index=False))
            players = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players = players.sort_values(["Goals", "Player"], ascending=[False, True])
            players.insert(0, "Rank", range(1, len(players) + 1))
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
    # (Hidden) st.caption explanations removed
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Data Exports")
        if not full_df.empty:
            st.download_button(
                label="⬇️ Download FULL Dataset (CSV)",
                data=full_df.to_csv(index=False),
                file_name=f"tournament_full_data_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
        if not filtered_df.empty:
            st.download_button(
                label="⬇️ Download FILTERED Dataset (CSV)",
                data=filtered_df.to_csv(index=False),
                file_name=f"tournament_filtered_data_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )

    with col2:
        st.subheader("🏆 Summary Reports")
        if not filtered_df.empty:
            teams_summary = (
                filtered_df.groupby(["Team", "Division"]).agg(Players_Count=("Player", "nunique"), Total_Goals=("Goals", "sum")).reset_index()
            ).sort_values("Total_Goals", ascending=False)
            st.download_button(
                label="⬇️ Download TEAMS Summary (CSV)",
                data=teams_summary.to_csv(index=False),
                file_name=f"teams_summary_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )

            players_summary = filtered_df.groupby(["Player", "Team", "Division"])["Goals"].sum().reset_index()
            players_summary = players_summary.sort_values(["Goals", "Player"], ascending=[False, True])
            players_summary.insert(0, "Rank", range(1, len(players_summary) + 1))
            st.download_button(
                label="⬇️ Download PLAYERS Summary (CSV)",
                data=players_summary.to_csv(index=False),
                file_name=f"players_summary_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )

    with col3:
        st.subheader("📦 Complete Package")
        if st.button("📦 Generate Complete Report Package"):
            z = create_comprehensive_zip_report(full_df, filtered_df)
            st.download_button(
                label="⬇️ Download Complete Package (ZIP)",
                data=z,
                file_name=f"tournament_complete_package_{datetime.now():%Y%m%d_%H%M}.zip",
                mime="application/zip",
            )

# ====================== POINT TABLE (Google Sheets) ===============
POINT_TABLE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"

def _coerce_first_row_as_header(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()
    header_idx = 0
    for i in range(min(6, len(df))):
        if df.iloc[i].notna().sum() >= 2:
            header_idx = i
            break
    header = df.iloc[header_idx].astype(str).str.strip().tolist()
    df2 = df.iloc[header_idx + 1 :].copy()
    # ensure unique col names
    cols, seen = [], {}
    for c in header:
        k = c if c not in seen else f"{c}_{seen[c]+1}"
        seen[c] = seen.get(c, 0) + 1
        cols.append(k)
    df2.columns = cols
    return df2.reset_index(drop=True)

def normalize_point_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    rename = {}
    for col in df.columns:
        key = _norm_key(col)
        if key in ("team","teams","club","clubs"): rename[col] = "Team"
        elif key in ("p","mp","m","played","matchesplayed"): rename[col] = "P"
        elif key in ("w","win","wins"): rename[col] = "W"
        elif key in ("d","draw","draws"): rename[col] = "D"
        elif key in ("l","loss","losses"): rename[col] = "L"
        elif key in ("gf","goalsfor","goalsf","for"): rename[col] = "GF"
        elif key in ("ga","goalsagainst","goalsa","against"): rename[col] = "GA"
        elif key in ("gd","goaldifference","diff","difference"): rename[col] = "GD"
        elif key in ("pts","point","points","score"): rename[col] = "Pts"
    df = df.rename(columns=rename)

    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).str.strip()

    for c in ["P","W","D","L","GF","GA","GD","Pts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "GD" not in df.columns and "GF" in df.columns and "GA" in df.columns:
        df["GD"] = df["GF"] - df["GA"]

    sort_cols = [c for c in ["Pts","GD","GF"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] + [False]*(len(sort_cols)-1), kind="mergesort")

    # Final tidy
    df = df.dropna(axis=1, how="all").reset_index(drop=True)
    return df

def _norm_sheetname(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def _pick_sheet_name(all_names: list[str], target: str) -> str | None:
    want = _norm_sheetname(target)
    for nm in all_names:
        if _norm_sheetname(nm) == want:
            return nm
    # fallback: contains
    for nm in all_names:
        if _norm_sheetname(target) in _norm_sheetname(nm):
            return nm
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_point_tables(url: str) -> dict[str, pd.DataFrame]:
    """
    Download the published xlsx and return normalized point tables for:
    'B-Division Group A', 'B-Division Group B', 'A-Division'
    """
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    raw = r.content
    sheets = safe_read_workbook(raw)  # header=None frames

    # Coerce headers and normalize each sheet
    cleaned = {name: normalize_point_table(_coerce_first_row_as_header(df)) for name, df in sheets.items()}

    names = list(cleaned.keys())
    out = {}
    for label in ["B-Division Group A", "B-Division Group B", "A-Division"]:
        sel = _pick_sheet_name(names, label)
        out[label] = cleaned.get(sel, pd.DataFrame())
    return out

def display_point_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data found for this group yet.")
        return
    # Show standard header labels; ensure blanks for object NA
    show = df.copy()
    for c in show.select_dtypes(include="object").columns:
        show[c] = show[c].replace({None: ""}).fillna("")
    ordered = [c for c in ["Team","P","W","D","L","Pts","GF","GA","GD"] if c in show.columns]
    if ordered:
        show = show[ordered]
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            **({"Team": st.column_config.TextColumn("TEAM", width="large")} if "Team" in show.columns else {}),
            **({"P": st.column_config.NumberColumn("M", format="%d", width="small")} if "P" in show.columns else {}),
            **({"W": st.column_config.NumberColumn("W", format="%d", width="small")} if "W" in show.columns else {}),
            **({"D": st.column_config.NumberColumn("D", format="%d", width="small")} if "D" in show.columns else {}),
            **({"L": st.column_config.NumberColumn("L", format="%d", width="small")} if "L" in show.columns else {}),
            **({"Pts": st.column_config.NumberColumn("PTS", format="%d", width="small")} if "Pts" in show.columns else {}),
            **({"GF": st.column_config.NumberColumn("GF", format="%d", width="small")} if "GF" in show.columns else {}),
            **({"GA": st.column_config.NumberColumn("GA", format="%d", width="small")} if "GA" in show.columns else {}),
            **({"GD": st.column_config.NumberColumn("GD", format="%d", width="small")} if "GD" in show.columns else {}),
        },
    )

# ====================== MAIN APP =================================
def to_export_xlsx_url(sheet_link_or_id: str) -> str:
    """
    Accept either a full Google Sheets link or just the doc id and return
    an 'export?format=xlsx' URL that downloads the whole workbook.
    """
    if re.match(r"^[A-Za-z0-9_-]{20,}$", sheet_link_or_id):
        doc_id = sheet_link_or_id
    else:
        m = re.search(r"/d/([A-Za-z0-9_-]+)/", sheet_link_or_id)
        doc_id = m.group(1) if m else sheet_link_or_id
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx"

def main():
    inject_advanced_css()

    # Title
    st.markdown(
        """
<div class="app-title">
  <span class="ball">⚽</span>
  <span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span>
</div>
""",
        unsafe_allow_html=True,
    )
    add_world_cup_watermark()

    # 👉 Set your Google Sheet doc ID (or paste the edit link) for GOALS + CARDS
    GOOGLE_SHEET_DOC_ID = "1Bbx7nCS_j7g1wsK3gHpQQnWkpTqlwkHu"  # your sheet ID
    GOOGLE_SHEETS_URL = to_export_xlsx_url(GOOGLE_SHEET_DOC_ID)

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        if st.button("🔄 Refresh Tournament Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        division_options = ["All Divisions"] + sorted(tournament_data["Division"].dropna().unique().tolist())
        selected_division = st.selectbox("📊 Division", division_options, key="division_filter")
        if selected_division != "All Divisions":
            tournament_data = tournament_data[tournament_data["Division"] == selected_division]

        # Team filter
        available_teams = sorted(tournament_data["Team"].dropna().unique().tolist())
        selected_teams = st.multiselect(
            "🏆 Teams (optional)",
            available_teams,
            key="teams_filter",
            help="Select specific teams to focus on",
            placeholder="Type to search teams…",
        )
        if selected_teams:
            tournament_data = tournament_data[tournament_data["Team"].isin(selected_teams)]

        # Player search
        st.subheader("👤 Player Search")
        player_names = sorted(tournament_data["Player"].dropna().astype(str).unique().tolist())
        selected_players = st.multiselect(
            "Type to search and select players",
            options=player_names,
            default=[],
            key="players_filter",
            placeholder="Start typing a player name…",
            help="You can select one or more players.",
        )
        if selected_players:
            tournament_data = tournament_data[tournament_data["Player"].isin(selected_players)]

        # --- Cards filter (ONLY All / Yellow / Red) ---
        for c in ["Yellow Cards", "Red Cards"]:
            if c not in tournament_data.columns:
                tournament_data[c] = 0

        st.subheader("🟨🟥 Cards filter")
        card_filter = st.selectbox(
            "Show players with…",
            ["All", "Yellow only", "Red only"],  # removed "Any card"
            index=0,
            key="cards_filter",
            help="Filter to players who have cards recorded on the CARDS sheet.",
        )
        if card_filter == "Yellow only":
            tournament_data = tournament_data[tournament_data["Yellow Cards"] > 0]
        elif card_filter == "Red only":
            tournament_data = tournament_data[tournament_data["Red Cards"] > 0]

    # Tabs (sticky) — POINT TABLE added
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📊 OVERVIEW", "⚡ QUICK INSIGHTS", "🏆 TEAMS", "👤 PLAYERS", "📈 ANALYTICS", "📋 POINT TABLE", "📥 DOWNLOADS"]
    )

    current_stats = calculate_tournament_stats(tournament_data)
    top_performers = get_top_performers(tournament_data, 10)

    # TAB 1 — Overview
    with tab1:
        st.header("📊 Tournament Overview")
        display_metric_cards(current_stats)
        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 Goal Scoring Records")
            create_enhanced_data_table(tournament_data, "records")
        with col2:
            if not tournament_data.empty:
                st.subheader("🏁 Division Distribution")
                st.altair_chart(create_division_donut_chart(tournament_data), use_container_width=True)
        if not tournament_data.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🏆 Goals by Team")
                team_goals = (tournament_data.groupby("Team")["Goals"].sum().reset_index()
                              .sort_values("Goals", ascending=False).head(10))
                if not team_goals.empty:
                    st.altair_chart(create_horizontal_bar_chart(team_goals, "Goals", "Team", "Top 10 Teams by Goals", "blues"), use_container_width=True)
            with c2:
                st.subheader("⚽ Top Scorers")
                if not top_performers["players"].empty:
                    ts = top_performers["players"].head(10).copy()
                    ts["Display_Name"] = ts["Player"] + " (" + ts["Team"] + ")"
                    st.altair_chart(create_horizontal_bar_chart(ts, "Goals", "Display_Name", "Top 10 Players by Goals", "greens"), use_container_width=True)

    # TAB 2 — Quick Insights
    with tab2:
        st.header("⚡ Quick Tournament Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Total Goals", current_stats["total_goals"])
        c2.metric("👥 Active Players", current_stats["total_players"])
        c3.metric("🏆 Teams", current_stats["total_teams"])
        c4.metric("📊 Divisions", current_stats["divisions"])
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("🟨 Yellow Cards", current_stats["total_yellow"])
        c6.metric("🟥 Red Cards", current_stats["total_red"])
        c7.metric("⚖️ Team Goals σ", current_stats["competitive_balance"])
        c8.metric("🎯 Avg Goals/Team", current_stats["avg_goals_per_team"])
        # (Hidden) explanatory caption removed
        st.divider()

    # Helpers for PTS merge
    def build_pts_lookup() -> pd.DataFrame:
        """Return DataFrame with TeamKey, DivKey, Pts from all point tables."""
        sections = fetch_point_tables(POINT_TABLE_URL)
        blocks = []
        for label, dfpts in sections.items():
            if dfpts is None or dfpts.empty or "Team" not in dfpts.columns:
                continue
            if "Pts" not in dfpts.columns:
                continue
            tmp = dfpts[["Team", "Pts"]].copy()
            # Normalize numeric
            tmp["Pts"] = pd.to_numeric(tmp["Pts"], errors="coerce").fillna(0).astype(int)
            # Map sheet label -> division text used in tournament_data
            div_txt = "A Division" if label.lower().startswith("a") else "B Division"
            tmp["Division"] = div_txt
            tmp["TeamKey"] = tmp["Team"].apply(_norm_key)
            tmp["DivKey"] = tmp["Division"].apply(_norm_div_key)
            blocks.append(tmp[["TeamKey", "DivKey", "Pts"]])
        if not blocks:
            return pd.DataFrame(columns=["TeamKey","DivKey","Pts"])
        pts_all = pd.concat(blocks, ignore_index=True)
        # If duplicates (e.g., same team across group A/B), keep max Pts
        pts_all = pts_all.groupby(["TeamKey","DivKey"], as_index=False)["Pts"].max()
        return pts_all

    pts_lookup_df = build_pts_lookup()

    # TAB 3 — Teams
    with tab3:
        st.header("🏆 Teams Analysis")
        if tournament_data.empty:
            st.info("🔍 No teams match your current filters.")
        else:
            st.subheader("📋 Teams Summary")
            teams_summary = tournament_data.groupby(["Team", "Division"]).agg(Players=("Player", "nunique"), Total_Goals=("Goals", "sum")).reset_index()

            # 👉 Merge PTS from point tables (robust name/division normalization)
            teams_summary["TeamKey"] = teams_summary["Team"].apply(_norm_key)
            teams_summary["DivKey"] = teams_summary["Division"].apply(_norm_div_key)
            if not pts_lookup_df.empty:
                teams_summary = teams_summary.merge(pts_lookup_df, on=["TeamKey","DivKey"], how="left")
            else:
                teams_summary["Pts"] = 0
            teams_summary["Pts"] = teams_summary["Pts"].fillna(0).astype(int)
            teams_summary = teams_summary.drop(columns=["TeamKey","DivKey"])

            st.dataframe(
                teams_summary.sort_values("Total_Goals", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "Team": st.column_config.TextColumn("Team", width="medium"),
                    "Division": st.column_config.TextColumn("Division", width="small"),
                    "Players": st.column_config.NumberColumn("Players", format="%d", width="small"),
                    "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d", width="small"),
                    "Pts": st.column_config.NumberColumn("PTS", format="%d", width="small"),
                },
            )
            st.divider()
            st.subheader("📊 Team Performance Analysis")
            team_analysis = teams_summary.sort_values("Total_Goals", ascending=False)
            if not team_analysis.empty:
                st.altair_chart(create_horizontal_bar_chart(team_analysis.head(15), "Total_Goals", "Team", "Team Goals Distribution", "viridis"), use_container_width=True)

    # TAB 4 — Players (CARD VIEW)
    with tab4:
        st.header("👤 Players (Card View)")
        render_player_cards(tournament_data)

    # TAB 5 — Analytics
    with tab5:
        st.header("📈 Advanced Analytics")
        if tournament_data.empty:
            st.info("🔍 No data available for analytics with current filters.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Goals Distribution")
                dist = create_goals_distribution_histogram(tournament_data)
                (st.plotly_chart if PLOTLY_AVAILABLE else st.altair_chart)(dist, use_container_width=True)
            with c2:
                st.subheader("🎯 Team Performance Matrix")
                scatter = create_advanced_scatter_plot(tournament_data)
                (st.plotly_chart if PLOTLY_AVAILABLE else st.altair_chart)(scatter, use_container_width=True)

    # TAB 6 — POINT TABLE (from Google Sheets)
    with tab6:
        st.header("📋 Point Table")
        # (Hidden) explanatory caption removed
        try:
            sections = fetch_point_tables(POINT_TABLE_URL)
            sub1, sub2, sub3 = st.tabs(["B-Division Group A", "B-Division Group B", "A-Division"])
            with sub1:
                display_point_table(sections.get("B-Division Group A"))
            with sub2:
                display_point_table(sections.get("B-Division Group B"))
            with sub3:
                display_point_table(sections.get("A-Division"))
        except Exception as e:
            st.error(f"Couldn't load the Point Table workbook. {e}")

    # TAB 7 — Downloads
    with tab7:
        create_download_section(full_tournament_data, tournament_data)

# ====================== ENTRY POINT ===============================
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.exception(e)
