# app.py — ABEER BLUESTAR SOCCER FEST 2K25 Dashboard - DARK GAMING THEME
# Complete dashboard with futuristic dark gaming theme

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests
from datetime import datetime

# ---------------------- Dark Gaming Theme CSS --------------------------------
def inject_dark_gaming_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
      /* Dark Gaming Theme */
      .stApp { 
        font-family: 'Rajdhani', monospace; 
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%) !important;
        color: #00ff88 !important;
      }
      
      .block-container { 
        padding: 1rem 2rem; 
        max-width: 1400px; 
        background: rgba(0,0,0,0.3);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,255,136,0.2);
      }
      
      /* Neon Headers */
      h1 { 
        font-family: 'Orbitron', monospace !important;
        font-weight: 900 !important;
        text-align: center !important;
        color: #00ff88 !important;
        text-shadow: 0 0 20px #00ff88, 0 0 40px #00ff88, 0 0 60px #00ff88;
        margin-bottom: 2rem !important;
        font-size: 2.5rem !important;
        letter-spacing: 3px !important;
      }
      
      h2, h3 { 
        color: #00ccff !important; 
        text-shadow: 0 0 10px #00ccff;
        font-family: 'Orbitron', monospace !important;
        border-bottom: 2px solid #00ccff;
        padding-bottom: 0.5rem;
      }
      
      /* Cyber Buttons */
      .stButton > button, .stDownloadButton > button {
        background: linear-gradient(45deg, #ff0080, #ff8c00) !important;
        color: white !important;
        border: 2px solid #ff0080 !important;
        border-radius: 25px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 0 20px rgba(255,0,128,0.5) !important;
        transition: all 0.3s ease !important;
      }
      
      .stButton > button:hover, .stDownloadButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 30px rgba(255,0,128,0.8) !important;
        background: linear-gradient(45deg, #ff8c00, #ff0080) !important;
      }
      
      /* Glowing Cards */
      [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(0,204,255,0.1) 100%) !important;
        border: 2px solid #00ff88 !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        box-shadow: 0 0 25px rgba(0,255,136,0.3) !important;
        backdrop-filter: blur(5px) !important;
      }
      
      [data-testid="metric-container"] > div {
        color: #00ff88 !important;
      }
      
      /* Sidebar Styling */
      .css-1d391kg, .css-1v3fvcr {
        background: rgba(0,0,0,0.8) !important;
        border-right: 2px solid #00ff88 !important;
      }
      
      .stSelectbox > div > div {
        background: rgba(0,0,0,0.7) !important;
        color: #00ff88 !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px !important;
      }
      
      .stMultiSelect > div > div {
        background: rgba(0,0,0,0.7) !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px !important;
      }
      
      .stTextInput > div > div > input {
        background: rgba(0,0,0,0.7) !important;
        color: #00ff88 !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px !important;
      }
      
      /* DataFrames */
      .stDataFrame {
        border: 2px solid #00ccff !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 0 25px rgba(0,204,255,0.3) !important;
      }
      
      .stDataFrame table {
        background: rgba(0,0,0,0.8) !important;
        color: #00ff88 !important;
      }
      
      .stDataFrame th {
        background: rgba(0,204,255,0.2) !important;
        color: #00ccff !important;
        text-transform: uppercase !important;
        font-weight: bold !important;
      }
      
      /* Charts */
      .vega-embed {
        border: 2px solid #00ccff !important;
        border-radius: 15px !important;
        background: rgba(0,0,0,0.5) !important;
        box-shadow: 0 0 25px rgba(0,204,255,0.3) !important;
      }
      
      /* Scrollbars */
      ::-webkit-scrollbar {
        width: 12px;
      }
      
      ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.3);
      }
      
      ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00ff88, #00ccff);
        border-radius: 6px;
      }
      
      ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #00ccff, #ff0080);
      }
      
      /* Sidebar Labels */
      .css-1d391kg label, .css-1v3fvcr label {
        color: #00ff88 !important;
        font-family: 'Orbitron', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
      }
      
      /* Slider */
      .stSlider > div > div > div {
        color: #00ccff !important;
      }
      
      /* Caption text */
      .stCaption {
        color: #00ccff !important;
      }
      
      /* Info messages */
      .stInfo {
        background: rgba(0,255,136,0.1) !important;
        color: #00ff88 !important;
        border: 1px solid #00ff88 !important;
        border-radius: 10px !important;
      }
      
      /* Animations */
      @keyframes glow {
        0% { box-shadow: 0 0 20px rgba(0,255,136,0.3); }
        50% { box-shadow: 0 0 40px rgba(0,255,136,0.6); }
        100% { box-shadow: 0 0 20px rgba(0,255,136,0.3); }
      }
      
      h1 {
        animation: glow 2s ease-in-out infinite;
      }
    </style>
    """, unsafe_allow_html=True)

    # Dark Altair theme
    alt.themes.register("cyberpunk", lambda: {
        "config": {
            "background": "rgba(0,0,0,0.8)",
            "title": {"font": "Orbitron", "fontSize": 16, "color": "#00ccff"},
            "axis": {
                "labelColor": "#00ff88", 
                "titleColor": "#00ccff", 
                "gridColor": "rgba(0,255,136,0.2)",
                "domainColor": "#00ff88"
            },
            "legend": {"labelColor": "#00ff88", "titleColor": "#00ccff"},
            "range": {"category": ["#00ff88","#00ccff","#ff0080","#ff8c00","#ff4444","#8800ff"]},
            "view": {"stroke": "transparent"}
        }
    })
    alt.themes.enable("cyberpunk")

# ========================= XLSX fallback (no openpyxl) ========================
def _parse_xlsx_without_openpyxl(file_bytes: bytes) -> pd.DataFrame:
    """Return first worksheet as raw grid (header=None)."""
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
            rows = []
            max_col_idx = 0
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
    """Read Excel as raw grid (header=None). Tries pandas; falls back if engine missing."""
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

# ========================= Robust block parser ================================
def _find_block_start_indices(raw: pd.DataFrame) -> tuple[int | None, int | None]:
    """Scan first two rows for 'B Division' and 'A Division' labels."""
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
    """Parse the dual-table sheet into tidy [Division, Team, Player, Goals]."""
    raw = _read_excel_raw(xlsx_bytes)

    b_start, a_start = _find_block_start_indices(raw)
    if b_start is None and a_start is None:
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

# ========================= Fetch helpers =====================================
def fetch_xlsx_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content:
        raise ValueError("Downloaded file is empty.")
    return r.content

# ========================= UI helpers ========================================
def display_metrics(df: pd.DataFrame) -> None:
    total = int(df["Goals"].sum()) if not df.empty else 0
    players = df["Player"].nunique() if not df.empty else 0
    teams = df["Team"].nunique() if not df.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("TOTAL GOALS", f"{total}")
    c2.metric("NUMBER OF PLAYERS", f"{players}")
    c3.metric("NUMBER OF TEAMS", f"{teams}")

def bar_chart(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    """Horizontal bar chart with INTEGER ticks (no decimals)."""
    max_val = int(df[value].max()) if not df.empty else 1
    # Limit explicit ticks to 50 to avoid clutter when values are large
    tick_vals = list(range(0, max_val + 1)) if max_val <= 50 else None
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X(
                f"{value}:Q",
                title="Number of Goals",
                axis=alt.Axis(format="d", tickMinStep=1, values=tick_vals),
                scale=alt.Scale(domainMin=0, nice=False),
            ),
            y=alt.Y(f"{category}:N", sort="-x", title=category),
            tooltip=[category, value],
        )
        .properties(height=400, title=title)
    )

def pie_chart(df: pd.DataFrame) -> alt.Chart:
    agg = df.groupby("Division")["Goals"].sum().reset_index()
    return (
        alt.Chart(agg)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Goals:Q", title="Goals"),
            color=alt.Color("Division:N", title="Division"),
            tooltip=["Division", "Goals"],
        )
        .properties(title="Goals by Division")
    )

def make_reports_zip(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    """Create a ZIP with CSVs for full and filtered views and summaries (Top Scorers includes Team)."""
    team_goals = (
        filtered_df.groupby("Team", as_index=False)["Goals"]
        .sum()
        .sort_values("Goals", ascending=False)
    )
    top_scorers = (
        filtered_df.groupby(["Player", "Team"], as_index=False)["Goals"]
        .sum()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        [["Player", "Team", "Goals"]]
    )
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("records_full.csv", full_df.to_csv(index=False))
        z.writestr("records_filtered.csv", filtered_df.to_csv(index=False))
        z.writestr("team_goals_filtered.csv", team_goals.to_csv(index=False))
        z.writestr("top_scorers_filtered.csv", top_scorers.to_csv(index=False))
    buf.seek(0)
    return buf.read()

# ========================= App ===============================================
def main():
    st.set_page_config(page_title="ABEER BLUESTAR SOCCER FEST 2K25", page_icon="⚽", layout="wide")
    inject_dark_gaming_css()

    # Heading
    st.markdown(
        "<h1 style='text-align:center; margin-top:0; margin-bottom:0.25rem;'>"
        "ABEER BLUESTAR SOCCER FEST 2K25"
        "</h1>",
        unsafe_allow_html=True,
    )

    # Data source (fixed Google Sheets XLSX)
    XLSX_URL = (
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    )

    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @st.cache_data(ttl=300)
    def load_data():
        data_bytes = fetch_xlsx_bytes_from_url(XLSX_URL)
        return load_and_prepare_data_from_bytes(data_bytes)

    try:
        data = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    last_ref = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.sidebar.caption(f"Last refreshed: {last_ref}")

    # Filters
    st.sidebar.header("Filters")
    div_opts = ["All"] + sorted(data["Division"].unique().tolist())
    div_sel = st.sidebar.selectbox("Division", div_opts)
    data_div = data if div_sel == "All" else data[data["Division"] == div_sel]

    team_opts = sorted(data_div["Team"].unique().tolist())
    team_sel = st.sidebar.multiselect("Team (optional)", team_opts)
    filtered = data_div if not team_sel else data_div[data_div["Team"].isin(team_sel)]

    st.sidebar.subheader("Player search")
    player_query = st.sidebar.text_input("Type player names (comma-separated, partial OK)", value="")
    if player_query.strip():
        tokens = [t.strip().lower() for t in player_query.split(",") if t.strip()]
        if tokens:
            mask = False
            for t in tokens:
                mask = mask | filtered["Player"].str.lower().str.contains(t, na=False)
            filtered = filtered[mask]

    current_players = sorted(filtered["Player"].unique().tolist())
    players_pick = st.sidebar.multiselect("Players (refine optional)", options=current_players)
    if players_pick:
        filtered = filtered[filtered["Player"].isin(players_pick)]

    # Metrics
    display_metrics(filtered)

    # Records table
    st.subheader("Goal Scoring Records")
    if filtered.empty:
        st.info("No records under current filters.")
    else:
        table_df = filtered.sort_values("Goals", ascending=False).reset_index(drop=True)
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
        )

    # Goals by Team (integer ticks)
    st.subheader("Goals by Team")
    team_goals = (
        filtered.groupby("Team", as_index=False)["Goals"]
        .sum()
        .sort_values("Goals", ascending=False)
    )
    if team_goals.empty:
        st.info("No team data to display for the current filters.")
    else:
        st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

    # Top Scorers (safe for 0/1/2+ players) with integer ticks
    st.subheader("Top Scorers")
    player_goals_total = (
        filtered.groupby("Player", as_index=False)["Goals"]
        .sum()
        .sort_values("Goals", ascending=False)
    )
    max_players = int(player_goals_total.shape[0])

    if max_players == 0:
        st.info("No player data to display for the current filters.")
    elif max_players == 1:
        st.caption("Only one player found — showing that player.")
        single_df = player_goals_total.head(1).reset_index(drop=True)
        st.dataframe(
            single_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
        )
        st.altair_chart(bar_chart(single_df, "Player", "Goals", "Top 1 Player by Goals"), use_container_width=True)
    else:
        default_top = min(10, max_players)
        top_n = st.sidebar.slider("Top N players", min_value=1, max_value=max_players, value=int(default_top),
                                  key=f"topn_{max_players}")
        top_df = player_goals_total.head(int(top_n)).reset_index(drop=True)
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
        )
        st.altair_chart(bar_chart(top_df, "Player", "Goals", f"Top {int(top_n)} Players by Goals"),
                        use_container_width=True)

    # Downloads
    st.subheader("Download Reports")
    st.download_button(
        "⬇️ Download FULL records (CSV)",
        data=data.to_csv(index=False),
        file_name="records_full.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download FILTERED records (CSV)",
        data=filtered.to_csv(index=False),
        file_name="records_filtered.csv",
        mime="text/csv",
    )
    top_scorers_with_team = (
        filtered.groupby(["Player", "Team"], as_index=False)["Goals"]
        .sum()
        .sort_values(["Goals", "Player"], ascending=[False, True])
        [["Player", "Team", "Goals"]]
    )
    st.download_button(
        "⬇️ Download TOP SCORERS (with Team) CSV",
        data=top_scorers_with_team.to_csv(index=False),
        file_name="top_scorers_filtered.csv",
        mime="text/csv",
    )
    zip_bytes = make_reports_zip(data, filtered)
    st.download_button(
        "📦 Download ALL reports (ZIP)",
        data=zip_bytes,
        file_name="football_reports.zip",
        mime="application/zip",
    )

    # Division distribution (pie)
    if div_sel == "All" and not data.empty:
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(data), use_container_width=True)

if __name__ == "__main__":
    main()
