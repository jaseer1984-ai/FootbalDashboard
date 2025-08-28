# ABEER BLUESTAR SOCCER FEST 2K25 — Streamlit Dashboard (Light Player Cards)
# Author: AI Assistant | Updated: 2025-08-28

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

# ============== CONFIG ==============
st.set_page_config(
    page_title="ABEER BLUESTAR SOCCER FEST 2K25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== STYLES ==============
def inject_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{ --sticky-tabs-top: 52px; }

.stApp{ font-family:'Poppins',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); }
.block-container{
  padding-top:.5rem; padding-bottom:2rem; max-width:98vw; width:98vw; margin:1rem auto;
  background:rgba(255,255,255,.95); backdrop-filter:blur(14px);
  border-radius:20px; box-shadow:0 20px 40px rgba(0,0,0,.15); position:relative; z-index:1;
}

/* keep sidebar toggle, hide other chrome */
#MainMenu, footer, .stDeployButton, div[data-testid="stDecoration"], div[data-testid="stStatusWidget"]{display:none!important}

/* sticky tabs */
.block-container [data-testid="stTabs"]:first-of-type{
  position:sticky; top:var(--sticky-tabs-top); z-index:6;
  background:rgba(255,255,255,.96); backdrop-filter:blur(8px);
  border-bottom:1px solid #e2e8f0; padding:.3rem 0; margin-top:.25rem;
}

/* title */
.app-title{display:flex;align-items:center;justify-content:center;gap:12px;margin:.75rem 0 1rem}
.app-title .ball{font-size:32px;line-height:1;filter:drop-shadow(0 2px 4px rgba(0,0,0,.15))}
.app-title .title{font-weight:700;letter-spacing:.05em;font-size:clamp(22px,3.5vw,36px);
  background:linear-gradient(45deg,#0ea5e9,#1e40af,#7c3aed);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}

/* buttons */
.stButton > button, .stDownloadButton > button{
  background:linear-gradient(135deg,#0ea5e9,#3b82f6)!important;color:#fff!important;border:0!important;
  border-radius:12px!important;padding:.6rem 1.2rem!important;font-weight:600!important;font-size:.9rem!important;
  transition:.25s!important;box-shadow:0 4px 14px rgba(14,165,233,.28)!important}
.stButton > button:hover,.stDownloadButton > button:hover{transform:translateY(-2px)!important;filter:brightness(1.05)!important}

/* metric cards */
.metric-container{background:linear-gradient(135deg,rgba(14,165,233,.10),rgba(59,130,246,.05));
  border-left:4px solid #0ea5e9;border-radius:14px;padding:1.2rem;box-shadow:0 4px 18px rgba(14,165,233,.1)}
.stDataFrame{border-radius:14px!important;overflow:hidden!important;box-shadow:0 8px 30px rgba(0,0,0,.06)!important}

/* status pill */
.status-pill{padding:.5rem .75rem;border-radius:.6rem;font-size:.85rem;margin-top:.5rem}
.status-ok{background:#ecfeff;border-left:4px solid #06b6d4;color:#155e75}
.status-warn{background:#fef9c3;border-left:4px solid #f59e0b;color:#713f12}
.status-err{background:#fee2e2;border-left:4px solid #ef4444;color:#7f1d1d}

/* -------- LIGHT PLAYER CARDS (your screenshot model) -------- */
.player-grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));align-items:stretch}
.player-card{
  background:#ffffff;border:1px solid #eceff5;border-radius:12px;
  box-shadow:0 6px 18px rgba(17,24,39,.06); overflow:hidden;
}
.pc-pad{padding:14px}
.pc-name{font-weight:700;color:#1f2937;line-height:1.2}
.pc-sub{color:#6b7280;font-size:.85rem;margin-top:2px}
.pc-stat{display:grid;grid-template-columns:1.1fr .25fr;align-items:center;gap:8px;margin:.45rem 0}
.pc-stat .label{color:#374151;font-size:.86rem;display:flex;gap:7px;align-items:center}
.pc-stat .num{text-align:right;font-variant-numeric:tabular-nums;color:#111827}
.pc-pill{display:inline-flex;gap:6px;align-items:center;
  border:1px dashed #eab308;color:#92400e;background:#fef3c7;
  border-radius:999px;padding:.20rem .5rem;font-size:.75rem}
.pc-hr{height:1px;background:#f2f4f8;margin:.35rem 0 .55rem 0}
.pc-ico{opacity:.9}

/* mobile */
@media (max-width:768px){
  .block-container{padding:1rem .6rem;margin:.5rem;width:95vw;max-width:95vw}
  .app-title .ball{font-size:24px}
}
</style>
""",
        unsafe_allow_html=True,
    )

    alt.themes.register(
        "tournament_theme",
        lambda: {
            "config": {
                "view": {"stroke": "transparent", "fill": "white"},
                "background": "white",
                "title": {"font": "Poppins", "fontSize": 18, "color": "#1e293b", "fontWeight": 600},
                "axis": {"labelColor": "#64748b", "titleColor": "#374151", "gridColor": "#f1f5f9"},
                "legend": {"labelColor": "#64748b", "titleColor": "#374151"},
                "range": {"category": ["#0ea5e9","#34d399","#60a5fa","#f59e0b","#f87171","#a78bfa","#fb7185","#4ade80"]},
            }
        },
    )
    alt.themes.enable("tournament_theme")

def trophy_watermark():
    st.markdown(
        """
<style>
  #wc-trophy{position:fixed;inset:0;background:url('https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f3c6.svg') no-repeat center 6vh/68vmin;opacity:.10;pointer-events:none;z-index:0}
</style>
<div id="wc-trophy"></div>
""",
        unsafe_allow_html=True,
    )

def notify(msg: str, kind: str = "ok"):
    cls = {"ok": "status-ok", "warn": "status-warn", "err": "status-err"}.get(kind, "status-ok")
    st.markdown(f'<div class="status-pill {cls}">{msg}</div>', unsafe_allow_html=True)

# ============== DATA HELPERS ==============
def parse_xlsx_no_deps(file_bytes: bytes) -> pd.DataFrame:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as f:
                root = ET.parse(f).getroot()
                for si in root.findall(".//m:si", ns):
                    shared.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))

        if "xl/worksheets/sheet1.xml" not in z.namelist():
            return pd.DataFrame()
        with z.open("xl/worksheets/sheet1.xml") as f:
            root = ET.parse(f).getroot()
            sheet = root.find("m:sheetData", ns)
            if sheet is None: return pd.DataFrame()
            rows, max_col = [], 0
            for row in sheet.findall("m:row", ns):
                rd = {}
                for cell in row.findall("m:c", ns):
                    ref = cell.attrib.get("r","A1")
                    col_letters = "".join(ch for ch in ref if ch.isalpha())
                    col_idx = 0
                    for ch in col_letters: col_idx = col_idx*26 + (ord(ch)-64)
                    col_idx -= 1
                    t = cell.attrib.get("t")
                    v = cell.find("m:v", ns)
                    val = v.text if v is not None else None
                    if t == "s" and val is not None:
                        i = int(val)
                        if 0 <= i < len(shared): val = shared[i]
                    rd[col_idx] = val
                    if col_idx > max_col: max_col = col_idx
                rows.append(rd)
    if not rows: return pd.DataFrame()
    matrix = [[r.get(i) for i in range(max_col+1)] for r in rows]
    return pd.DataFrame(matrix)

def safe_read_excel(bytes_or_path) -> pd.DataFrame:
    if isinstance(bytes_or_path, (str, Path)):
        file_bytes = Path(bytes_or_path).read_bytes()
    elif isinstance(bytes_or_path, bytes):
        file_bytes = bytes_or_path
    else:
        file_bytes = bytes_or_path.read()
    try:
        return pd.read_excel(BytesIO(file_bytes), header=None)
    except Exception:
        return parse_xlsx_no_deps(file_bytes)

def find_division_cols(df: pd.DataFrame):
    b_col = a_col = None
    for i in range(min(2, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        for j, cell in row.items():
            if "b division" in cell and b_col is None: b_col = j
            if "a division" in cell and a_col is None: a_col = j
    if b_col is None and a_col is None:
        b_col = 0
        a_col = 5 if df.shape[1] >= 8 else (4 if df.shape[1] >= 7 else None)
    return b_col, a_col

def process_tournament_data(xbytes: bytes) -> pd.DataFrame:
    raw = safe_read_excel(xbytes)
    if raw.empty: return pd.DataFrame(columns=["Division","Team","Player","Goals"])
    b_start, a_start = find_division_cols(raw)
    header_row = 1 if len(raw) > 1 else 0
    start_row = header_row + 1
    out = []

    def take(col_start, name):
        if col_start is None or col_start+2 >= raw.shape[1]: return
        df = raw.iloc[start_row:, col_start:col_start+3].copy()
        df.columns = ["Team","Player","Goals"]
        df = df.dropna(subset=["Team","Player","Goals"])
        df["Goals"] = pd.to_numeric(df["Goals"], errors="coerce")
        df = df.dropna(subset=["Goals"])
        df["Goals"] = df["Goals"].astype(int)
        df["Division"] = name
        out.extend(df.to_dict("records"))

    take(b_start,"B Division")
    take(a_start,"A Division")
    if not out: return pd.DataFrame(columns=["Division","Team","Player","Goals"])
    df = pd.DataFrame(out)[["Division","Team","Player","Goals"]]
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if not r.content: raise ValueError("Empty file")
        return process_tournament_data(r.content)
    except Exception:
        return pd.DataFrame(columns=["Division","Team","Player","Goals"])

# ============== ANALYTICS ==============
def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_goals":0,"total_players":0,"total_teams":0,"divisions":0,"competitive_balance":0}
    pt = df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index()
    tt = df.groupby(["Team","Division"])["Goals"].sum().reset_index()
    return {
        "total_goals": int(df["Goals"].sum()),
        "total_players": len(pt),
        "total_teams": len(tt),
        "divisions": df["Division"].nunique(),
        "competitive_balance": round(tt["Goals"].std(),2) if len(tt)>1 else 0,
    }

def division_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    d = (df.groupby("Division").agg(Total_Goals=("Goals","sum"), Avg_Goals=("Goals","mean"),
         Total_Records=("Goals","count"), Teams=("Team","nunique"), Players=("Player","nunique"))
         .round(2).reset_index())
    s = d["Total_Goals"].sum()
    d["Goal_Share_Pct"] = (d["Total_Goals"]/s*100).round(1) if s else 0
    return d

def bar_chart(df, x, y, title, scheme="blues"):
    if df.empty:
        return alt.Chart(pd.DataFrame({"note":["No data"]})).mark_text().encode(text="note:N")
    max_val = int(df[x].max())
    ticks = list(range(0, max_val+1)) if max_val <= 50 else None
    return (alt.Chart(df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=.9, stroke="white", strokeWidth=1)
            .encode(x=alt.X(f"{x}:Q", title="Goals", axis=alt.Axis(format="d", values=ticks, gridOpacity=.3), scale=alt.Scale(domainMin=0, nice=False)),
                    y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                    color=alt.Color(f"{x}:Q", scale=alt.Scale(scheme=scheme), legend=None),
                    tooltip=[y, alt.Tooltip(x, format="d")])
            .properties(height=max(300, min(600, len(df)*24)), title=alt.TitleParams(text=title, fontWeight=600)))

def donut_division(df: pd.DataFrame):
    if df.empty:
        return alt.Chart(pd.DataFrame({"note":["No data"]})).mark_text().encode(text="note:N")
    division_data = df.groupby("Division")["Goals"].sum().reset_index()
    sel = alt.selection_single(fields=["Division"], empty="none")
    base = alt.Chart(division_data).add_selection(sel).properties(width=280, height=280, title=alt.TitleParams(text="Goals by Division", fontWeight=600))
    arc = base.mark_arc(innerRadius=58, outerRadius=118, stroke="white", strokeWidth=2)\
        .encode(theta="Goals:Q", color=alt.Color("Division:N", scale=alt.Scale(range=["#0ea5e9","#f59e0b"])),
                opacity=alt.condition(sel, alt.value(1.0), alt.value(0.85)),
                tooltip=["Division:N", alt.Tooltip("Goals:Q", format="d")])
    center = base.mark_text(align="center", baseline="middle", fontSize=16, fontWeight="bold", color="#1e293b")\
        .encode(text=alt.value(f"Total\n{int(division_data['Goals'].sum())}"))
    return arc + center

# ============== DOWNLOAD PACKAGE ==============
def zip_report(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if not full_df.empty: z.writestr("01_full.csv", full_df.to_csv(index=False))
        if not filtered_df.empty:
            z.writestr("02_filtered.csv", filtered_df.to_csv(index=False))
            teams = (filtered_df.groupby(["Team","Division"])
                     .agg(Unique_Players=("Player","nunique"), Total_Records=("Goals","count"),
                          Total_Goals=("Goals","sum"), Avg_Goals=("Goals","mean"), Max_Goals=("Goals","max"))
                     .round(2).reset_index())
            players = (filtered_df.groupby(["Player","Team","Division"])["Goals"].sum()
                       .reset_index().sort_values(["Goals","Player"], ascending=[False,True]))
            players.insert(0,"Rank", range(1,len(players)+1))
            z.writestr("03_teams_summary.csv", teams.to_csv(index=False))
            z.writestr("04_players_ranking.csv", players.to_csv(index=False))
            dcmp = division_comparison(filtered_df)
            if not dcmp.empty: z.writestr("05_division_comparison.csv", dcmp.to_csv(index=False))
            z.writestr("06_stats.json", pd.Series(stats(filtered_df)).to_json())
        z.writestr("README.txt", f"ABEER BLUESTAR SOCCER FEST 2K25 package\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    buf.seek(0)
    return buf.getvalue()

def downloads_section(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.subheader("📥 Download Reports")
    st.caption("**Full** = all data; **Filtered** = current view")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("📊 Data")
        if not full_df.empty:
            st.download_button("⬇️ Full Dataset (CSV)", data=full_df.to_csv(index=False), file_name=f"tournament_full_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        if not filtered_df.empty:
            st.download_button("⬇️ Filtered Dataset (CSV)", data=filtered_df.to_csv(index=False), file_name=f"tournament_filtered_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with c2:
        st.subheader("🏆 Summaries")
        if not filtered_df.empty:
            teams = (filtered_df.groupby(["Team","Division"]).agg(Players=("Player","nunique"), Total_Goals=("Goals","sum")).reset_index().sort_values("Total_Goals",ascending=False))
            st.download_button("⬇️ Teams Summary (CSV)", data=teams.to_csv(index=False), file_name=f"teams_summary_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
            players = (filtered_df.groupby(["Player","Team","Division"])["Goals"].sum().reset_index().sort_values(["Goals","Player"],ascending=[False,True]))
            players.insert(0,"Rank", range(1,len(players)+1))
            st.download_button("⬇️ Players Summary (CSV)", data=players.to_csv(index=False), file_name=f"players_summary_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with c3:
        st.subheader("📦 Package")
        if st.button("📦 Build Complete ZIP"):
            z = zip_report(full_df, filtered_df)
            st.download_button("⬇️ Download Package (ZIP)", data=z, file_name=f"soccerfest_package_{datetime.now():%Y%m%d_%H%M}.zip", mime="application/zip")

# ============== UI FRAGMENTS ==============
def metric_row(label: str, value: int) -> str:
    # simple icon decisions matching your screenshot vibe
    ico = {
        "Goals": "⚽",
        "Appearances": "👟",
        "Yellow Cards": "🟨",
        "Red Cards": "🟥",
    }.get(label, "•")
    return f"""
      <div class="pc-stat">
        <div class="label"><span class="pc-ico">{ico}</span><span>{label}</span></div>
        <div class="num">{int(value)}</div>
      </div>
    """

def render_player_cards(df: pd.DataFrame):
    if df.empty:
        st.info("No players match your current filters.")
        return
    # Aggregate to unique player → totals
    players = (df.groupby(["Player","Team","Division"])["Goals"].sum()
                 .reset_index().sort_values(["Goals","Player"], ascending=[False,True]))
    # Build cards HTML
    cards = []
    for _, r in players.iterrows():
        name = str(r["Player"])
        team = str(r["Team"])
        div = str(r["Division"])
        goals = int(r["Goals"])
        # Others not tracked in sheet → 0
        appearances = 0
        yc = 0
        rc = 0
        sub = f"{team} • {div} • Age —"
        card = f"""
        <div class="player-card">
          <div class="pc-pad">
            <div class="pc-name">{name}</div>
            <div class="pc-sub">{sub}</div>
            <div class="pc-hr"></div>
            {metric_row("Goals", goals)}
            {metric_row("Appearances", appearances)}
            {metric_row("Yellow Cards", yc)}
            {metric_row("Red Cards", rc)}
            <div class="pc-hr"></div>
            <span class="pc-pill">No awards</span>
          </div>
        </div>
        """
        cards.append(card)
    html = '<div class="player-grid">' + "".join(cards) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def metric_deck(s: dict):
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-container"><div style="font-size:2.2rem;font-weight:700;color:#0ea5e9;margin-bottom:.25rem">{s['total_goals']}</div><div style="color:#64748b;font-weight:600">TOTAL GOALS</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-container"><div style="font-size:2.2rem;font-weight:700;color:#0ea5e9;margin-bottom:.25rem">{s['total_players']}</div><div style="color:#64748b;font-weight:600">PLAYERS</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-container"><div style="font-size:2.2rem;font-weight:700;color:#0ea5e9;margin-bottom:.25rem">{s['total_teams']}</div><div style="color:#64748b;font-weight:600">TEAMS</div></div>""", unsafe_allow_html=True)

# ============== MAIN ==============
def main():
    inject_css()
    st.markdown('<div class="app-title"><span class="ball">⚽</span><span class="title">ABEER BLUESTAR SOCCER FEST 2K25</span></div>', unsafe_allow_html=True)
    trophy_watermark()

    # SOURCE
    GOOGLE_SHEETS_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    # SIDEBAR
    with st.sidebar:
        st.header("🎛️ Controls")
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notify("Cache cleared. Reloading…", "ok")
            st.rerun()
        st.caption(f"🕒 Last refreshed: {st.session_state.get('last_refresh', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        st.divider()
        with st.spinner("📡 Loading tournament data…"):
            df = fetch_tournament(GOOGLE_SHEETS_URL)
        if df.empty:
            notify("No tournament data. Check the sheet link/permissions.", "err")
            st.stop()

        full_df = df.copy()

        st.header("🔍 Filters")
        division_opts = ["All Divisions"] + sorted(df["Division"].unique().tolist())
        sel_div = st.selectbox("📊 Division", division_opts)
        if sel_div != "All Divisions":
            df = df[df["Division"] == sel_div]

        teams_avail = sorted(df["Team"].unique().tolist())
        sel_teams = st.multiselect("🏆 Teams", teams_avail, placeholder="Type to search teams…")
        if sel_teams:
            df = df[df["Team"].isin(sel_teams)]

        st.subheader("👤 Player Search")
        players_avail = sorted(df["Player"].dropna().astype(str).unique().tolist())
        sel_players = st.multiselect("Type to search players", options=players_avail, placeholder="Start typing a name…")
        if sel_players:
            df = df[df["Player"].isin(sel_players)]

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 OVERVIEW","⚡ QUICK INSIGHTS","🏆 TEAMS","👤 PLAYERS","📈 ANALYTICS","📥 DOWNLOADS"])

    S = stats(df)

    with tab1:
        st.header("📊 Tournament Overview")
        metric_deck(S)
        st.divider()
        c1,c2 = st.columns([2,1])
        with c1:
            st.subheader("🎯 Goal Records (top 20)")
            st.dataframe(df.sort_values("Goals", ascending=False).head(20).reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        with c2:
            st.subheader("🏁 Divisions")
            st.altair_chart(donut_division(df), use_container_width=True)
        if not df.empty:
            c3,c4 = st.columns(2)
            with c3:
                st.subheader("🏆 Goals by Team")
                tg = df.groupby("Team")["Goals"].sum().reset_index().sort_values("Goals",ascending=False).head(10)
                st.altair_chart(bar_chart(tg,"Goals","Team","Top 10 Teams by Goals","blues"), use_container_width=True)
            with c4:
                st.subheader("⚽ Top Scorers")
                tp = (df.groupby(["Player","Team"])["Goals"].sum().reset_index().sort_values("Goals",ascending=False).head(10))
                tp["Display"] = tp["Player"] + " (" + tp["Team"] + ")"
                st.altair_chart(bar_chart(tp,"Goals","Display","Top 10 Players by Goals","greens"), use_container_width=True)

    with tab2:
        st.header("⚡ Quick Insights")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🎯 Total Goals", S["total_goals"])
        c2.metric("👥 Players", S["total_players"])
        c3.metric("🏆 Teams", S["total_teams"])
        c4.metric("📊 Divisions", S["divisions"])
        st.divider()
        dcmp = division_comparison(df)
        if not dcmp.empty:
            st.dataframe(dcmp, use_container_width=True, hide_index=True)

    with tab3:
        st.header("🏆 Teams")
        if df.empty:
            st.info("No teams under current filters.")
        else:
            st.subheader("📋 Summary")
            teams = (df.groupby(["Team","Division"]).agg(Players=("Player","nunique"), Total_Goals=("Goals","sum")).reset_index())
            st.dataframe(teams.sort_values("Total_Goals",ascending=False), use_container_width=True, hide_index=True)
            st.divider()
            st.subheader("📊 Team Goals")
            st.altair_chart(bar_chart(teams.sort_values("Total_Goals",ascending=False).head(15),"Total_Goals","Team","Goals by Team","viridis"), use_container_width=True)

    with tab4:
        st.header("👤 Player Profiles")
        render_player_cards(df)  # <<< light card grid replaces the table

    with tab5:
        st.header("📈 Analytics")
        if df.empty:
            st.info("No data available for analytics.")
        else:
            c1,c2 = st.columns(2)
            with c1:
                st.subheader("📊 Goals Distribution")
                player_goals = df.groupby(["Player","Team"])["Goals"].sum().values
                hist_df = pd.DataFrame({"Goals": player_goals})
                chart = (alt.Chart(hist_df).mark_bar(opacity=.9)
                         .encode(x=alt.X("Goals:Q", bin=alt.Bin(maxbins=10), title="Goals per Player"),
                                 y=alt.Y("count():Q", title="Players"),
                                 tooltip=[alt.Tooltip("Goals:Q", bin=True, title="Goals"), alt.Tooltip("count():Q", title="Players")])
                         .properties(height=380, title="Distribution of Goals per Player"))
                st.altair_chart(chart, use_container_width=True)
            with c2:
                st.subheader("🎯 Division Comparison (table)")
                st.dataframe(division_comparison(df), use_container_width=True, hide_index=True)
            st.divider()
            c3,c4,c5 = st.columns(3)
            with c3:
                st.markdown("**Top Teams (by goals)**")
                tg = df.groupby("Team")["Goals"].sum().sort_values(ascending=False).head(5)
                for t,g in tg.items(): st.write(f"• **{t}** — {int(g)}")
            with c4:
                st.markdown("**Scoring Breakdown**")
                counts = df["Goals"].value_counts().sort_index()
                for g,c in counts.items(): st.write(f"• {int(g)} goal{'s' if g!=1 else ''}: {int(c)} records")
            with c5:
                st.markdown("**Division Notes**")
                for dname in df["Division"].unique():
                    ddf = df[df["Division"]==dname]
                    st.write(f"• **{dname}** — {int(ddf['Goals'].sum())} goals, {int(ddf['Player'].nunique())} players")

    with tab6:
        downloads_section(full_df, df)

# ============== ENTRY ==============
if __name__ == "__main__":
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.exception(e)
