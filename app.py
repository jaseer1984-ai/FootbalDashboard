# ABEER BLUESTAR SOCCER FEST 2K25 — Streamlit Dashboard (White Cards)
# Author: AI Assistant | Rev: 2025-08-28

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

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# CSS / Theme (WHITE BACKGROUND + White player cards)
# ---------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
          :root{ --sticky-tabs-top: 52px; }

          /* App and container -> WHITE */
          .stApp {
            font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
            background:#ffffff;                    /* white background */
          }
          .block-container {
            padding-top:.5rem; padding-bottom:2rem;
            max-width:98vw; width:98vw;
            background:#ffffff;                    /* keep content white */
            border-radius:20px;
            box-shadow:0 0 0 rgba(0,0,0,0);        /* no outer shadow */
            margin:1rem auto; position:relative; z-index:1;
          }

          /* Hide some streamlit chrome */
          #MainMenu, footer, .stDeployButton,
          div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { display:none !important; }

          /* Sticky tabs (first tabs row) */
          .block-container [data-testid="stTabs"]:first-of-type{
            position:sticky; top:var(--sticky-tabs-top); z-index:6;
            background:rgba(255,255,255,.96); backdrop-filter:blur(8px);
            border-bottom:1px solid #e2e8f0; padding:.25rem 0; margin-top:.25rem;
          }

          /* Title */
          .app-title{ display:flex; align-items:center; justify-content:center; gap:12px; margin:.75rem 0 1rem; }
          .app-title .ball{ font-size:32px; line-height:1; }
          .app-title .title{
            font-weight:700; letter-spacing:.05em; font-size:clamp(22px,3.5vw,36px);
            background:linear-gradient(45deg,#0ea5e9,#1e40af,#7c3aed);
            -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
          }

          /* Buttons */
          .stButton > button, .stDownloadButton > button{
            background:linear-gradient(135deg,#0ea5e9,#3b82f6)!important;
            color:#fff!important; border:0!important; border-radius:12px!important;
            padding:.6rem 1.2rem!important; font-weight:600!important; font-size:.9rem!important;
            transition:.3s!important; box-shadow:0 4px 15px rgba(14,165,233,.3)!important;
          }
          .stButton > button:hover, .stDownloadButton > button:hover{
            transform:translateY(-2px)!important; box-shadow:0 8px 25px rgba(14,165,233,.4)!important;
            filter:brightness(1.05)!important;
          }

          /* Metric card look */
          .metric-container{ background:linear-gradient(135deg,rgba(14,165,233,.08),rgba(59,130,246,.05));
            border-radius:15px; padding:1.25rem; border-left:4px solid #0ea5e9; }

          /* ---------- PLAYER CARDS (WHITE) ---------- */
          .players-grid{
            display:grid; gap:16px;
            grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
            margin-top:.5rem;
          }
          .pcard{
            background:#ffffff; color:#1e293b;
            border-radius:16px; padding:16px 16px 12px 16px;
            border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,.08);
          }
          .pcard h3{ margin:0 0 8px 0; font-size:1.15rem; line-height:1.35; font-weight:700; color:#111827; }
          .pcard .sub{ font-size:.9rem; color:#475569; }
          .pcard .muted{ color:#94a3b8; font-size:.9rem; margin:.15rem 0 .25rem; }

          .row{ display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:10px; margin:.5rem 0; }
          .label{ font-size:.9rem; color:#334155; white-space:nowrap; }
          .dotbar{ position:relative; height:10px; border-radius:999px; background:#f1f5f9; border:1px solid #e2e8f0; overflow:hidden; }
          .dotbar>span{ position:absolute; inset:0; width:var(--pct,0%); height:100%;
            background:linear-gradient(90deg,#3b82f6,#06b6d4); transition:width .4s ease; }
          .num{ width:32px; text-align:right; font-variant-numeric:tabular-nums; color:#111827; }

          .pill{ display:inline-block; margin-top:.5rem; padding:.25rem .55rem; border-radius:9999px; font-size:.75rem;
            background:#f8fafc; color:#0f172a; border:1px dashed #94a3b8; }

          @media (max-width:768px){
            .block-container{ padding:1rem .5rem; margin:.5rem; width:95vw; max-width:95vw; }
            .app-title .ball{ font-size:24px; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Altair theme
    alt.themes.register(
        "white_theme",
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
                "legend": {"labelFont": "Poppins", "titleFont": "Poppins"},
            }
        },
    )
    alt.themes.enable("white_theme")


def app_title():
    st.markdown(
        """
        <div class="app-title">
          <span class="ball">⚽</span>
          <span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Robust XLSX loader (works even without openpyxl on some hosts)
# ---------------------------------------------------------------------
def parse_xlsx_robust(file_bytes: bytes) -> pd.DataFrame:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                root = ET.parse(f).getroot()
                for si in root.findall(".//m:si", ns):
                    text = "".join(t.text or "" for t in si.findall(".//m:t", ns))
                    shared.append(text)

        if "xl/worksheets/sheet1.xml" not in z.namelist():
            return pd.DataFrame()
        with z.open("xl/worksheets/sheet1.xml") as f:
            root = ET.parse(f).getroot()
            sheet = root.find("m:sheetData", ns)
            if sheet is None:
                return pd.DataFrame()
            rows, max_col = [], 0
            for row in sheet.findall("m:row", ns):
                rd = {}
                for cell in row.findall("m:c", ns):
                    ref = cell.attrib.get("r", "A1")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    col_idx = 0
                    for ch in col_letters:
                        col_idx = col_idx * 26 + (ord(ch) - 64)
                    col_idx -= 1
                    ctype = cell.attrib.get("t")
                    v = cell.find("m:v", ns)
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


def safe_read_excel(file_or_bytes) -> pd.DataFrame:
    if isinstance(file_or_bytes, (str, Path)):
        data = Path(file_or_bytes).read_bytes()
    elif isinstance(file_or_bytes, bytes):
        data = file_or_bytes
    else:
        data = file_or_bytes.read()

    # Try pandas engine first
    try:
        return pd.read_excel(BytesIO(data), header=None)
    except Exception:
        return parse_xlsx_robust(data)


# ---------------------------------------------------------------------
# Tournament data shaping
# ---------------------------------------------------------------------
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
    raw = safe_read_excel(xlsx_bytes)
    if raw.empty:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    b_start, a_start = find_division_columns(raw)
    header_row = 1 if len(raw) > 1 else 0
    data_start = header_row + 1
    records = []

    def pull(col_start: int | None, name: str):
        if col_start is None or col_start + 2 >= raw.shape[1]:
            return
        df = raw.iloc[data_start:, col_start:col_start + 3].copy()
        df.columns = ["Team", "Player", "Goals"]
        df = df.dropna(subset=["Team", "Player", "Goals"])
        df["Goals"] = pd.to_numeric(df["Goals"], errors="coerce")
        df = df.dropna(subset=["Goals"])
        df["Goals"] = df["Goals"].astype(int)
        df["Division"] = name
        records.extend(df.to_dict("records"))

    pull(b_start, "B Division")
    pull(a_start, "A Division")
    if not records:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
    out = pd.DataFrame(records)
    return out[["Division", "Team", "Player", "Goals"]]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sheet(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.content:
            return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
        return process_tournament_data(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])


# ---------------------------------------------------------------------
# Stats & charts
# ---------------------------------------------------------------------
def calc_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(total_goals=0, players=0, teams=0, divisions=0)
    return dict(
        total_goals=int(df["Goals"].sum()),
        players=int(df.groupby(["Player", "Team", "Division"]).size().shape[0]),
        teams=int(df.groupby(["Team", "Division"]).size().shape[0]),
        divisions=int(df["Division"].nunique()),
    )


def team_bar(df: pd.DataFrame):
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data"]})).mark_text().encode(text="note:N")
    tg = df.groupby("Team")["Goals"].sum().reset_index().sort_values("Goals", ascending=False).head(10)
    return (
        alt.Chart(tg)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("Goals:Q", title="Goals", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y("Team:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("Goals:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Team:N", alt.Tooltip("Goals:Q", format="d")],
        )
        .properties(height=max(300, min(600, len(tg) * 25)), title="Top 10 Teams by Goals")
    )


def players_bar(df: pd.DataFrame):
    if df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data"]})).mark_text().encode(text="note:N")
    pl = (
        df.groupby(["Player", "Team"])["Goals"]
        .sum()
        .reset_index()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        .head(10)
    )
    pl["Who"] = pl["Player"] + " (" + pl["Team"] + ")"
    return (
        alt.Chart(pl)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("Goals:Q", title="Goals", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y("Who:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("Goals:Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=["Player:N", "Team:N", alt.Tooltip("Goals:Q", format="d")],
        )
        .properties(height=max(300, min(600, len(pl) * 25)), title="Top 10 Players by Goals")
    )


# ---------------------------------------------------------------------
# PLAYER CARDS (white)
# ---------------------------------------------------------------------
def build_players_cards_html(df: pd.DataFrame) -> str:
    """
    Creates the white cards HTML. No table; exactly the card layout you asked for.
    Shows: Name, sub (Team • Division • Age —), dummy stats (Appearances/Cards=0), and pill.
    """
    if df.empty:
        return "<p>No players match current filters.</p>"

    # Aggregate goals by player/team/division
    agg = (
        df.groupby(["Player", "Team", "Division"])["Goals"]
        .sum()
        .reset_index()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        .reset_index(drop=True)
    )

    max_goals = max(1, int(agg["Goals"].max()))
    html = ['<div class="players-grid">']

    for _, row in agg.iterrows():
        name = str(row["Player"]).strip()
        team = str(row["Team"]).strip()
        division = str(row["Division"]).strip()
        goals = int(row["Goals"])
        pct = int(round(goals / max_goals * 100))  # for the bar width

        html.append(
            f"""
            <div class="pcard">
              <h3>{name}</h3>
              <div class="sub">{team} • {division} • Age —</div>
              <div class="muted">—</div>

              <div class="row">
                <div class="label">⚽ Goals</div>
                <div class="dotbar"><span style="--pct:{pct}%"></span></div>
                <div class="num">{goals}</div>
              </div>

              <div class="row">
                <div class="label">👕 Appearances</div>
                <div class="dotbar"><span style="--pct:0%"></span></div>
                <div class="num">0</div>
              </div>

              <div class="row">
                <div class="label">🟨 Yellow Cards</div>
                <div class="dotbar"><span style="--pct:0%"></span></div>
                <div class="num">0</div>
              </div>

              <div class="row">
                <div class="label">🟥 Red Cards</div>
                <div class="dotbar"><span style="--pct:0%"></span></div>
                <div class="num">0</div>
              </div>

              <span class="pill">No awards</span>
            </div>
            """
        )

    html.append("</div>")
    return "\n".join(html)


# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------
def zip_package(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty:
            z.writestr("01_full_tournament_data.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered_tournament_data.csv", filtered_df.to_csv(index=False))
        z.writestr("README.txt", f"ABEER BLUESTAR SOCCER FEST 2K25\nGenerated: {datetime.now():%Y-%m-%d %H:%M}")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    inject_css()
    app_title()

    # Source (published) — replace with your sheet if needed
    SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")

        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

        st.caption(f"Last refresh: {st.session_state.get('last_refresh', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        st.divider()

        with st.spinner("Fetching data…"):
            raw_df = fetch_sheet(SHEET_URL)

        if raw_df.empty:
            st.error("No data loaded. Check Google Sheet publish/permissions.")
            st.stop()

        full_df = raw_df.copy()

        # Filters
        st.subheader("Filters")
        div_opt = ["All"] + sorted(full_df["Division"].unique().tolist())
        sel_div = st.selectbox("Division", div_opt)
        work = full_df.copy()
        if sel_div != "All":
            work = work[work["Division"] == sel_div]

        team_opt = sorted(work["Team"].unique().tolist())
        sel_teams = st.multiselect("Teams", team_opt, placeholder="Type to search…")
        if sel_teams:
            work = work[work["Team"].isin(sel_teams)]

        player_opt = sorted(work["Player"].dropna().astype(str).unique().tolist())
        sel_players = st.multiselect("Players", player_opt, placeholder="Type to search…")
        if sel_players:
            work = work[work["Player"].isin(sel_players)]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 OVERVIEW", "⚡ QUICK INSIGHTS", "🏆 TEAMS", "👤 PLAYERS", "📈 ANALYTICS", "📥 DOWNLOADS"]
    )

    # OVERVIEW
    with tab1:
        st.header("📊 Tournament Overview")
        stats = calc_stats(work)
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="metric-container"><div style="font-size:2rem;font-weight:700;color:#0ea5e9;">{stats["total_goals"]}</div>'
            f'<div style="color:#475569;font-weight:600;">Total Goals</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="metric-container"><div style="font-size:2rem;font-weight:700;color:#0ea5e9;">{stats["players"]}</div>'
            f'<div style="color:#475569;font-weight:600;">Players</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="metric-container"><div style="font-size:2rem;font-weight:700;color:#0ea5e9;">{stats["teams"]}</div>'
            f'<div style="color:#475569;font-weight:600;">Teams</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()

        colL, colR = st.columns(2)
        with colL:
            st.subheader("🏆 Goals by Team")
            st.altair_chart(team_bar(work), use_container_width=True)
        with colR:
            st.subheader("⚽ Top Scorers")
            st.altair_chart(players_bar(work), use_container_width=True)

    # QUICK INSIGHTS (simple text)
    with tab2:
        st.header("⚡ Quick Insights")
        if work.empty:
            st.info("No data for current filters.")
        else:
            top_team = work.groupby("Team")["Goals"].sum().sort_values(ascending=False).head(1)
            top_player = (
                work.groupby(["Player", "Team"])["Goals"]
                .sum()
                .sort_values(ascending=False)
                .head(1)
            )
            col1, col2 = st.columns(2)
            with col1:
                if not top_team.empty:
                    st.success(f"🏅 **Top Team**: {top_team.index[0]} — {int(top_team.iloc[0])} goals")
            with col2:
                if not top_player.empty:
                    who = top_player.index[0]
                    st.success(f"🎯 **Top Scorer**: {who[0]} ({who[1]}) — {int(top_player.iloc[0])} goals")

    # TEAMS (keep a simple summary)
    with tab3:
        st.header("🏆 Teams Summary")
        if work.empty:
            st.info("No teams with current filters.")
        else:
            tdf = (
                work.groupby(["Team", "Division"])
                .agg(Players=("Player", "nunique"), Goals=("Goals", "sum"))
                .reset_index()
                .sort_values("Goals", ascending=False)
            )
            st.dataframe(tdf, use_container_width=True, hide_index=True)

    # PLAYERS — WHITE CARDS (NO TABLE)
    with tab4:
        st.header("👤 Player Profiles")
        cards_html = build_players_cards_html(work)
        st.markdown(cards_html, unsafe_allow_html=True)  # IMPORTANT: markdown + unsafe_allow_html

    # ANALYTICS (optional small section)
    with tab5:
        st.header("📈 Analytics")
        if work.empty:
            st.info("No analytics for current filters.")
        else:
            by_div = (
                work.groupby("Division")
                .agg(Total_Goals=("Goals", "sum"), Players=("Player", "nunique"), Teams=("Team", "nunique"))
                .reset_index()
            )
            st.dataframe(by_div, use_container_width=True, hide_index=True)

    # DOWNLOADS
    with tab6:
        st.header("📥 Downloads")
        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "⬇️ Full Dataset (CSV)",
                data=full_df.to_csv(index=False),
                file_name=f"tournament_full_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Filtered Dataset (CSV)",
                data=work.to_csv(index=False),
                file_name=f"tournament_filtered_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
        with colB:
            if st.button("📦 Build ZIP Package"):
                z = zip_package(full_df, work)
                st.download_button(
                    "⬇️ Download Package (ZIP)",
                    data=z,
                    file_name=f"tournament_package_{datetime.now():%Y%m%d_%H%M}.zip",
                    mime="application/zip",
                )


# ---------------------------------------------------------------------
# Run app
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main()
