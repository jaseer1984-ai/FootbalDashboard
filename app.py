# ABEER BLUESTAR SOCCER FEST 2K25 — Enhanced Dashboard (Altair + Streamlit)
# - Modern UI (beautiful light theme) via CSS injection (wrapped in <style> ✅)
# - Hidden Google Sheets XLSX URL (no input box shown)
# - Refresh button + "last refreshed" timestamp
# - Robust XLSX parsing without openpyxl (XML fallback)
# - Filters: Division, Team, Player search + refine by multiselect
# - Tabs: Overview, Teams, Players, Analytics, Downloads
# - Integer tick marks on charts (no floating labels)
# - Tables hide index/serial numbers
# - Downloads: Full (ALL rows), Filtered (current view), summaries, ZIP

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests
from datetime import datetime


# ---------------------- THEME / CSS -------------------------------------------
def inject_beautiful_css():
    """Injects all custom CSS correctly (wrapped in <style> so it won't render as text)."""
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
      .stApp{
        font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
        background:linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%);
        color:#1e293b;
      }
      .block-container{padding-top:1rem;padding-bottom:2rem;max-width:95vw;width:95vw;}

      .main-title{
        background:linear-gradient(135deg,#0ea5e9,#3b82f6,#8b5cf6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
        text-align:center;font-size:clamp(28px,4vw,48px);font-weight:700;margin:.5rem 0 1.5rem 0;letter-spacing:-.02em;
      }

      /* Filter panel */
      .filter-panel{
        background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
        border:1px solid #e2e8f0;border-radius:16px;padding:1.5rem;margin-bottom:2rem;
        box-shadow:0 4px 6px -1px rgba(0,0,0,.1);
      }
      .filter-title{font-size:1.125rem;font-weight:600;color:#1e293b;margin-bottom:1rem;display:flex;gap:.5rem;align-items:center;}

      /* Sidebar look */
      .css-1d391kg{background:linear-gradient(180deg,#f8fafc 0%,#ffffff 100%) !important;border-right:1px solid #e2e8f0;}
      .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3{color:#1e293b !important;font-weight:600;}

      /* Buttons */
      .stButton>button,.stDownloadButton>button{
        background:linear-gradient(135deg,#0ea5e9,#0284c7) !important;color:#fff !important;border:0 !important;
        border-radius:12px !important;padding:.75rem 1.5rem !important;font-weight:600 !important;font-size:.875rem !important;
        letter-spacing:.05em;text-transform:uppercase;box-shadow:0 4px 12px rgba(14,165,233,.2);transition:.3s ease !important;
      }
      .stButton>button:hover,.stDownloadButton>button:hover{
        background:linear-gradient(135deg,#0284c7,#0369a1) !important;transform:translateY(-2px);
        box-shadow:0 8px 20px rgba(14,165,233,.3);
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"]{
        background:#fff;border-radius:12px;padding:.25rem;gap:.25rem;box-shadow:0 2px 4px rgba(0,0,0,.05);border:1px solid #e2e8f0;
      }
      .stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#64748b;font-weight:500;border:none;transition:.3s;}
      .stTabs [data-baseweb="tab"]:hover{background:#f1f5f9;color:#1e293b;}
      .stTabs [aria-selected="true"]{
        background:linear-gradient(135deg,#0ea5e9,#0284c7) !important;color:#fff !important;
        box-shadow:0 2px 8px rgba(14,165,233,.3);
      }

      /* Tables */
      .stDataFrame{background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;box-shadow:0 2px 4px rgba(0,0,0,.05);}

      /* Section headers */
      .section-header{font-size:1.5rem;font-weight:600;color:#1e293b;margin:2rem 0 1rem;padding-bottom:.5rem;border-bottom:2px solid #e2e8f0;}

      /* Info card */
      .info-card{
        background:linear-gradient(135deg,#fef3c7,#fbbf24);border:1px solid #f59e0b;border-radius:12px;padding:1rem;color:#92400e;
        text-align:center;font-weight:500;box-shadow:0 2px 4px rgba(245,158,11,.1);
      }

      /* Chart container */
      .chart-container{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:1.5rem;margin:1rem 0;box-shadow:0 4px 6px -1px rgba(0,0,0,.05);}

      /* Inputs */
      .stSelectbox label,.stMultiSelect label,.stTextInput label{color:#1e293b !important;font-weight:500 !important;}
      .stSelectbox>div>div,.stMultiSelect>div>div,.stTextInput input{background:#fff !important;border:1px solid #e2e8f0 !important;border-radius:8px !important;}

      /* Metrics (st.metric) */
      [data-testid="metric-container"]{
        background:linear-gradient(135deg,#fff 0%,#f8fafc 100%);border:1px solid #e2e8f0;padding:1.5rem;border-radius:16px;
        box-shadow:0 4px 6px -1px rgba(0,0,0,.05);position:relative;overflow:hidden;
      }
      [data-testid="metric-container"]::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#0ea5e9,#3b82f6,#8b5cf6);}
      [data-testid="metric-container"] [data-testid="stMetricValue"]{
        font-size:2rem !important;font-weight:700 !important;background:linear-gradient(135deg,#0ea5e9,#3b82f6) !important;
        -webkit-background-clip:text !important;-webkit-text-fill-color:transparent !important;background-clip:text !important;
      }
      [data-testid="metric-container"] [data-testid="stMetricLabel"]{color:#64748b !important;font-weight:500 !important;text-transform:uppercase;font-size:.75rem;letter-spacing:.05em;}

      /* Slider accent */
      .stSlider>div>div>div>div{background:#0ea5e9 !important;}

      /* Fade-in */
      @keyframes fadeIn{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
      .stDataFrame,.chart-container{animation:fadeIn .5s ease-out;}
    </style>
    """, unsafe_allow_html=True)


# ---------------------- XLSX fallback (no openpyxl) ---------------------------
def _parse_xlsx_without_openpyxl(file_bytes: bytes) -> pd.DataFrame:
    """Return first worksheet as raw grid (header=None), parsing the XLSX XML."""
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
    """Read Excel as raw grid (header=None). Try pandas; fall back to XML parser if engine missing."""
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


# ---------------------- Parser for the dual-table layout ----------------------
def _find_block_start_indices(raw: pd.DataFrame) -> tuple[int | None, int | None]:
    """Find the start columns for B Division and A Division within the first two rows."""
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
    """Parse the sheet into tidy [Division, Team, Player, Goals]."""
    raw = _read_excel_raw(xlsx_bytes)

    b_start, a_start = _find_block_start_indices(raw)
    if b_start is None and a_start is None:  # fallback guess
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

    if b_start is not None:
        extract_block(b_start, "B Division")
    if a_start is not None:
        extract_block(a_start, "A Division")

    if not frames:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Team", "Player", "Goals"])
    combined["Goals"] = pd.to_numeric(combined["Goals"], errors="coerce")
    combined = combined.dropna(subset=["Goals"])
    combined["Goals"] = combined["Goals"].astype(int)
    return combined[["Division", "Team", "Player", "Goals"]]


# ---------------------- Fetch helpers -----------------------------------------
def fetch_xlsx_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content:
        raise ValueError("Downloaded file is empty.")
    return r.content


# ---------------------- UI helpers --------------------------------------------
def display_beautiful_metrics(df: pd.DataFrame) -> None:
    total = int(df["Goals"].sum()) if not df.empty else 0
    players = df["Player"].nunique() if not df.empty else 0
    teams = df["Team"].nunique() if not df.empty else 0
    avg_goals = round(total / players, 1) if players > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚽ TOTAL GOALS", f"{total:,}")
    c2.metric("👥 PLAYERS", f"{players:,}")
    c3.metric("🏆 TEAMS", f"{teams:,}")
    c4.metric("📊 AVG GOALS/PLAYER", f"{avg_goals}")


def create_filter_panel(full_df: pd.DataFrame):
    """Render the top filter panel and return (div_sel, team_sel, player_query, data_div)."""
    st.markdown('<div class="filter-panel"><div class="filter-title">🔍 Smart Filters & Search</div></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        div_opts = ["🌟 All Divisions"] + [f"🏅 {d}" for d in sorted(full_df["Division"].unique().tolist())]
        div_sel = st.selectbox("Division", div_opts, key="div_filter")
        if div_sel == "🌟 All Divisions":
            div_sel_clean = "All Divisions"
        else:
            div_sel_clean = div_sel[2:] if div_sel.startswith("🏅 ") else div_sel

    with col2:
        data_div = full_df if div_sel_clean == "All Divisions" else full_df[full_df["Division"] == div_sel_clean]
        team_opts = [f"⚽ {t}" for t in sorted(data_div["Team"].unique().tolist())]
        team_sel = st.multiselect("Teams (Optional)", team_opts, key="team_filter")
        team_sel_clean = [t[2:] if t.startswith("⚽ ") else t for t in team_sel]

    col3, col4 = st.columns([3, 1])
    with col3:
        player_query = st.text_input(
            "🔍 Search Players",
            value="",
            placeholder="Type player names separated by commas (e.g., Ahmed, Mohammed, Ali)",
            key="player_search",
        )
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All", key="clear_filters"):
            st.session_state.div_filter = "🌟 All Divisions"
            st.session_state.team_filter = []
            st.session_state.player_search = ""
            st.rerun()

    return div_sel_clean, team_sel_clean, player_query, (full_df if div_sel_clean == "All Divisions" else full_df[full_df["Division"] == div_sel_clean])


def create_altair_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col=None) -> alt.Chart:
    """Horizontal bar chart with integer ticks (no decimals)."""
    max_val = int(df[x_col].max()) if not df.empty else 1
    axis = alt.Axis(format='d', tickMinStep=1, values=list(range(0, max_val + 1)) if max_val <= 50 else None)
    if color_col:
        mark = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke='white', strokeWidth=1)
        enc = {
            "x": alt.X(f"{x_col}:Q", axis=axis, title=x_col.replace('_', ' ').title(), scale=alt.Scale(domainMin=0, nice=False)),
            "y": alt.Y(f"{y_col}:N", sort="-x", title=y_col.replace('_', ' ').title()),
            "color": alt.Color(f"{color_col}:N", scale=alt.Scale(range=['#0ea5e9','#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444'])),
            "tooltip": [y_col, x_col, color_col],
        }
    else:
        mark = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#0ea5e9', stroke='white', strokeWidth=1)
        enc = {
            "x": alt.X(f"{x_col}:Q", axis=axis, title=x_col.replace('_', ' ').title(), scale=alt.Scale(domainMin=0, nice=False)),
            "y": alt.Y(f"{y_col}:N", sort="-x", title=y_col.replace('_', ' ').title()),
            "tooltip": [y_col, x_col],
        }
    return mark.encode(**enc).properties(title=alt.TitleParams(text=title, fontSize=16, anchor='start', color='#1e293b'), height=400)


def create_pie_chart(df: pd.DataFrame) -> alt.Chart:
    agg = df.groupby("Division")["Goals"].sum().reset_index()
    pie = alt.Chart(agg).mark_arc(innerRadius=30, stroke='white', strokeWidth=2).encode(
        theta=alt.Theta('Goals:Q', title='Goals'),
        color=alt.Color('Division:N', scale=alt.Scale(range=['#0ea5e9','#3b82f6','#8b5cf6','#10b981']), title='Division'),
        tooltip=['Division:N', 'Goals:Q'],
    )
    text = alt.Chart(agg).mark_text(align='center', baseline='middle', fontSize=12, fontWeight='bold', color='white').encode(
        theta='Goals:Q', text=alt.Text('Goals:Q', format='d')
    )
    return (pie + text).properties(title=alt.TitleParams(text='🏆 Goals Distribution by Division', fontSize=16, anchor='start', color='#1e293b'), width=400, height=400)


def create_advanced_analytics_charts(df: pd.DataFrame):
    if df.empty:
        return None, None
    hist = alt.Chart(df).mark_bar(color='#0ea5e9', stroke='white', strokeWidth=1, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        alt.X('Goals:Q', bin=alt.Bin(maxbins=20), title='Goals'),
        alt.Y('count():Q', title='Number of Players'),
        tooltip=['Goals:Q', 'count():Q'],
    ).properties(title=alt.TitleParams(text='📊 Goals Distribution', fontSize=16, anchor='start', color='#1e293b'), width=400, height=300)

    team_stats = df.groupby('Team').agg({'Goals': 'sum', 'Player': 'nunique'}).reset_index()
    team_stats.columns = ['Team', 'Total_Goals', 'Player_Count']
    scatter = alt.Chart(team_stats).mark_circle(size=100, stroke='white', strokeWidth=2).encode(
        x=alt.X('Player_Count:Q', title='Number of Players'),
        y=alt.Y('Total_Goals:Q', title='Total Goals'),
        size=alt.Size('Total_Goals:Q', scale=alt.Scale(range=[100, 500]), title='Total Goals'),
        color=alt.Color('Total_Goals:Q', scale=alt.Scale(scheme='blues'), title='Total Goals'),
        tooltip=['Team:N', 'Player_Count:Q', 'Total_Goals:Q'],
    ).properties(title=alt.TitleParams(text='🎯 Team Performance: Players vs Goals', fontSize=16, anchor='start', color='#1e293b'), width=400, height=300)
    return hist, scatter


def make_reports_zip(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    team_goals = filtered_df.groupby("Team", as_index=False)["Goals"].sum().sort_values("Goals", ascending=False)
    top_scorers = filtered_df.groupby(["Player", "Team"], as_index=False)["Goals"].sum().sort_values(["Goals", "Player"], ascending=[False, True])[["Player", "Team", "Goals"]]
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("records_full.csv", full_df.to_csv(index=False))
        z.writestr("records_filtered.csv", filtered_df.to_csv(index=False))
        z.writestr("team_goals_filtered.csv", team_goals.to_csv(index=False))
        z.writestr("top_scorers_filtered.csv", top_scorers.to_csv(index=False))
    buf.seek(0)
    return buf.read()


# ---------------------- App ---------------------------------------------------
def main():
    st.set_page_config(page_title="ABEER BLUESTAR SOCCER FEST 2K25", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")
    inject_beautiful_css()
    st.markdown('<h1 class="main-title">⚽ ABEER BLUESTAR SOCCER FEST 2K25</h1>', unsafe_allow_html=True)

    # Hidden, hard-coded Google Sheets XLSX URL
    XLSX_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    # Sidebar controls
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

    full_df = data.copy()  # keep an unfiltered copy for FULL downloads

    with st.sidebar:
        last_ref = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.caption(f"📅 Last refreshed: {last_ref}")
        st.markdown("---")
        st.markdown("### 📋 Data Summary")
        st.caption(f"📊 Total Records: {len(full_df)}")
        st.caption(f"🏆 Divisions: {full_df['Division'].nunique()}")
        st.caption(f"⚽ Teams: {full_df['Team'].nunique()}")
        st.caption(f"👥 Players: {full_df['Player'].nunique()}")

    # Filters panel
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
    with st.sidebar:
        if current_players:
            st.markdown("### 👥 Refine Player Selection")
            players_pick = st.multiselect("Select specific players:", options=current_players)
            if players_pick:
                filtered = filtered[filtered["Player"].isin(players_pick)]

    # Show active filters summary
    if (div_sel != "All Divisions") or team_sel or player_query.strip() or (current_players and 'players_pick' in locals() and players_pick):
        with st.expander("🔍 Active Filters", expanded=False):
            st.write(f"**Division:** {div_sel}")
            if team_sel:
                st.write(f"**Teams:** {', '.join(team_sel)}")
            if player_query.strip():
                st.write(f"**Search:** {player_query}")
            if 'players_pick' in locals() and players_pick:
                st.write(f"**Selected Players:** {', '.join(players_pick[:3])}" + (f" (+{len(players_pick)-3} more)" if len(players_pick) > 3 else ""))
            st.caption(f"Showing {len(filtered)} of {len(full_df)} records")

    # Tabs
    t_overview, t_teams, t_players, t_analytics, t_downloads = st.tabs(["🏠 OVERVIEW", "🏆 TEAMS", "👥 PLAYERS", "📊 ANALYTICS", "⬇️ DOWNLOADS"])

    # OVERVIEW
    with t_overview:
        display_beautiful_metrics(filtered)

        st.markdown('<div class="section-header">🏅 Goal Scoring Records</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">📭 No records match your current filters. Adjust the filters above to see data.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                filtered.sort_values("Goals", ascending=False).reset_index(drop=True),
                use_container_width=True, hide_index=True,
                column_config={
                    "Division": st.column_config.TextColumn("🏅 Division", width="small"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Player": st.column_config.TextColumn("👤 Player", width="medium"),
                    "Goals": st.column_config.NumberColumn("⚽ Goals", format="%d", width="small"),
                },
                height=420,
            )

        st.markdown('<div class="section-header">🏆 Team Performance</div>', unsafe_allow_html=True)
        if not filtered.empty:
            team_goals = filtered.groupby("Team", as_index=False)["Goals"].sum().sort_values("Goals", ascending=False)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.altair_chart(create_altair_bar_chart(team_goals, "Goals", "Team", "🏆 Goals by Team"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🌟 Top Scorers</div>', unsafe_allow_html=True)
        scorers_pt = filtered.groupby(["Player", "Team"], as_index=False)["Goals"].sum().sort_values(["Goals", "Player"], ascending=[False, True])
        max_rows = int(scorers_pt.shape[0])
        if max_rows == 0:
            st.markdown('<div class="info-card">👤 No player data available for current filters.</div>', unsafe_allow_html=True)
        else:
            with st.sidebar:
                if max_rows > 1:
                    st.markdown("### 🎯 Display Options")
                    default_top = min(10, max_rows)
                    top_n = st.slider("🔝 Show top N players", 1, max_rows, int(default_top), key=f"topn_{max_rows}")
                else:
                    top_n = 1
            top_df = scorers_pt.head(int(top_n)).reset_index(drop=True)
            st.dataframe(
                top_df, use_container_width=True, hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("👤 Player", width="large"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Goals": st.column_config.NumberColumn("⚽ Goals", format="%d", width="small"),
                },
            )
            if max_rows > 1:
                chart_df = top_df.groupby("Player", as_index=False)["Goals"].sum()
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.altair_chart(create_altair_bar_chart(chart_df, "Goals", "Player", f"🌟 Top {int(top_n)} Scorers"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if div_sel == "All Divisions" and not full_df.empty:
            st.markdown('<div class="section-header">📊 Division Distribution</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.altair_chart(create_pie_chart(full_df), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                div_summary = full_df.groupby('Division').agg({'Goals': 'sum', 'Player': 'nunique', 'Team': 'nunique'}).reset_index()
                div_summary.columns = ['Division', 'Total Goals', 'Players', 'Teams']
                div_summary['Avg Goals/Player'] = (div_summary['Total Goals'] / div_summary['Players']).round(1)
                st.markdown("**📈 Division Statistics**")
                st.dataframe(
                    div_summary, use_container_width=True, hide_index=True,
                    column_config={
                        "Division": st.column_config.TextColumn("🏅 Division"),
                        "Total Goals": st.column_config.NumberColumn("⚽ Goals", format="%d"),
                        "Players": st.column_config.NumberColumn("👥 Players", format="%d"),
                        "Teams": st.column_config.NumberColumn("🏆 Teams", format="%d"),
                        "Avg Goals/Player": st.column_config.NumberColumn("📊 Avg/Player", format="%.1f"),
                    },
                )

    # TEAMS
    with t_teams:
        st.markdown('<div class="section-header">🏆 Teams Overview</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">🏆 No teams match your current filters.</div>', unsafe_allow_html=True)
        else:
            team_div = filtered.groupby("Team")["Division"].agg(lambda s: ", ".join(sorted(s.astype(str).unique()))).reset_index()
            team_summary = filtered.groupby("Team", as_index=False).agg(Total_Goals=("Goals", "sum"), Players=("Player", "nunique"))
            top_by_team = filtered.groupby(["Team", "Player"], as_index=False)["Goals"].sum()
            top_by_team = top_by_team.sort_values(["Team", "Goals"], ascending=[True, False]).groupby("Team").head(1)
            top_by_team = top_by_team.rename(columns={"Player": "Top Scorer", "Goals": "Top Scorer Goals"})
            teams_list = (
                team_div.merge(team_summary, on="Team", how="left")
                        .merge(top_by_team, on="Team", how="left")
                        .sort_values(["Total_Goals", "Team"], ascending=[False, True])
            )
            teams_list["Avg Goals/Player"] = (teams_list["Total_Goals"] / teams_list["Players"]).round(1)
            teams_list = teams_list[["Division", "Team", "Players", "Total_Goals", "Avg Goals/Player", "Top Scorer", "Top Scorer Goals"]]
            st.dataframe(
                teams_list, use_container_width=True, hide_index=True,
                column_config={
                    "Division": st.column_config.TextColumn("🏅 Division", width="small"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Players": st.column_config.NumberColumn("👥 Players", format="%d", width="small"),
                    "Total_Goals": st.column_config.NumberColumn("⚽ Total Goals", format="%d", width="small"),
                    "Avg Goals/Player": st.column_config.NumberColumn("📊 Avg/Player", format="%.1f", width="small"),
                    "Top Scorer": st.column_config.TextColumn("🌟 Top Scorer", width="medium"),
                    "Top Scorer Goals": st.column_config.NumberColumn("🎯 Goals", format="%d", width="small"),
                },
                height=520,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.altair_chart(create_altair_bar_chart(teams_list.rename(columns={"Total_Goals": "Goals"}), "Goals", "Team", "🏆 Team Performance (Total Goals)"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # PLAYERS
    with t_players:
        st.markdown('<div class="section-header">👥 Players Directory</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">👤 No players match your current filters.</div>', unsafe_allow_html=True)
        else:
            players_list = filtered.groupby(["Player", "Team", "Division"], as_index=False)["Goals"].sum().sort_values(["Goals", "Player"], ascending=[False, True])
            players_list["Rank"] = range(1, len(players_list) + 1)
            players_list = players_list[["Rank", "Player", "Team", "Division", "Goals"]]
            st.dataframe(
                players_list, use_container_width=True, hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("🏅 Rank", format="%d", width="small"),
                    "Player": st.column_config.TextColumn("👤 Player", width="large"),
                    "Team": st.column_config.TextColumn("⚽ Team", width="medium"),
                    "Division": st.column_config.TextColumn("🏆 Division", width="small"),
                    "Goals": st.column_config.NumberColumn("⚽ Goals", format="%d", width="small"),
                },
                height=620,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌟 Highest Scorer", f"{players_list.iloc[0]['Player']}", f"{players_list.iloc[0]['Goals']} goals")
            with col2:
                st.metric("📊 Average Goals", f"{players_list['Goals'].mean():.1f}")
            with col3:
                st.metric("📈 Median Goals", f"{players_list['Goals'].median()}")

    # ANALYTICS
    with t_analytics:
        st.markdown('<div class="section-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="info-card">📊 No data available for analytics with current filters.</div>', unsafe_allow_html=True)
        else:
            fig1, fig2 = create_advanced_analytics_charts(filtered)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.altair_chart(fig1, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.altair_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
            ic1, ic2 = st.columns(2)
            with ic1:
                top_teams = filtered.groupby('Team')['Goals'].sum().sort_values(ascending=False).head(3)
                st.markdown("**🏆 Top Performing Teams:**")
                for i, (team, goals) in enumerate(top_teams.items(), 1):
                    st.write(f"{i}. **{team}**: {goals} goals")
                st.markdown("**📈 Goal Statistics:**")
                st.write(f"• Highest individual score: **{filtered['Goals'].max()}** goals")
                st.write(f"• Most common score: **{filtered['Goals'].mode().iloc[0]}** goals")
                st.write(f"• Players with 5+ goals: **{len(filtered[filtered['Goals'] >= 5])}** players")
            with ic2:
                if div_sel == "All Divisions":
                    div_stats = full_df.groupby('Division').agg({'Goals': ['sum', 'mean', 'count']}).round(2)
                    div_stats.columns = ['Total Goals', 'Avg Goals', 'Records']
                    st.markdown("**🏅 Division Comparison:**")
                    st.dataframe(div_stats, use_container_width=True)
                else:
                    team_player_cnt = filtered.groupby('Team')['Player'].nunique().sort_values(ascending=False)
                    st.markdown(f"**⚽ Team Sizes in {div_sel}:**")
                    for team, count in team_player_cnt.head(5).items():
                        st.write(f"• **{team}**: {count} players")

    # DOWNLOADS
    with t_downloads:
        st.markdown('<div class="section-header">📁 Download Center</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Individual Reports**")
            st.caption("Download specific datasets in CSV format.")
            st.download_button("📋 Full Dataset (All Records)", data=full_df.to_csv(index=False), file_name="soccer_fest_2k25_full.csv", mime="text/csv", use_container_width=True)
            st.download_button("🔍 Filtered Data (Current View)", data=filtered.to_csv(index=False), file_name="soccer_fest_2k25_filtered.csv", mime="text/csv", use_container_width=True)
            if not filtered.empty:
                teams_summary = filtered.groupby(["Team", "Division"], as_index=False).agg(Players=("Player", "nunique"), Total_Goals=("Goals", "sum")).sort_values(["Total_Goals", "Team"], ascending=[False, True])
                st.download_button("🏆 Teams Summary (Current View)", data=teams_summary.to_csv(index=False), file_name="teams_summary.csv", mime="text/csv", use_container_width=True)
                players_summary = filtered.groupby(["Player", "Team", "Division"], as_index=False)["Goals"].sum().sort_values(["Goals", "Player"], ascending=[False, True])
                st.download_button("👥 Players Summary (Current View)", data=players_summary.to_csv(index=False), file_name="players_summary.csv", mime="text/csv", use_container_width=True)

        with c2:
            st.markdown("**📦 Complete Package**")
            st.caption("Get everything in one convenient ZIP file.")
            zip_bytes = make_reports_zip(full_df, filtered)
            st.download_button("📦 Complete Report Package (ZIP)", data=zip_bytes, file_name=f"soccer_fest_2k25_reports_{datetime.now().strftime('%Y%m%d')}.zip", mime="application/zip", use_container_width=True)
            st.markdown("**📋 Package Contents:**")
            st.caption("• Full dataset (all records)")
            st.caption("• Filtered dataset (current view)")
            st.caption("• Team performance summary")
            st.caption("• Top scorers ranking")

            st.markdown("**⚙️ Export Options**")
            export_format = st.selectbox("Choose format:", ["CSV", "Excel (XLSX)"], key="export_format")
            if export_format == "Excel (XLSX)":
                st.info("📝 Excel export will be available in a future update.")

if __name__ == "__main__":
    main()
