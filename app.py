# ABEER BLUESTAR SOCCER FEST 2K25 — Streamlit Dashboard (with tabs)
# Tabs: OVERVIEW, TEAMS (Teams List), PLAYERS (Players List), DOWNLOADS
# - Integer ticks on charts / safe slider
# - Full vs Filtered downloads (+ Teams/Players CSV)
# - Robust XLSX parsing without openpyxl

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests
from datetime import datetime

# ---------------------- FLUID LOOK & FEEL ------------------------------------
def inject_fluid_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
      .stApp { font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
      .block-container { padding-top:.6rem; padding-bottom:2rem; max-width:96vw; width:96vw; }
      @media (min-width: 1600px) { .block-container { max-width:1500px; width:1500px; } }
      /* Smaller, wrapping title */
      h1 {
        text-align:center; margin:.2rem 0 .7rem 0; letter-spacing:.06em; font-weight:700; line-height:1.15;
        font-size: clamp(22px, 3vw, 36px); white-space:normal; overflow-wrap:anywhere; word-break:break-word;
      }
      .stButton > button, .stDownloadButton > button {
        background:#0ea5e9 !important; color:#fff !important; border:0 !important;
        border-radius:10px !important; padding:.45rem .9rem !important; font-weight:600 !important;
      }
      .stButton > button:hover, .stDownloadButton > button:hover { filter:brightness(1.06); }
      .stDataFrame table { border-radius:10px; overflow:hidden; }
    </style>
    """, unsafe_allow_html=True)

    alt.themes.register("fluid", lambda: {
        "config": {
            "view": {"stroke": "transparent"},
            "background": "transparent",
            "title": {"font": "Poppins", "fontSize": 18, "color": "#0b1e35"},
            "axis":  {"labelColor": "#345", "titleColor": "#234", "gridColor": "#e6eef8"},
            "legend":{"labelColor": "#345", "titleColor": "#234"},
            "range": {"category": ["#0ea5e9","#34d399","#60a5fa","#f59e0b","#f87171","#a78bfa"]}
        }
    })
    alt.themes.enable("fluid")

# ========================= XLSX fallback (no openpyxl) ========================
def _parse_xlsx_without_openpyxl(file_bytes: bytes) -> pd.DataFrame:
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

    if b_start is not None: extract_block(b_start, "B Division")
    if a_start is not None: extract_block(a_start, "A Division")

    if not frames:
        return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Team", "Player", "Goals"])
    combined["Goals"] = pd.to_numeric(combined["Goals"], errors="coerce")
    combined = combined.dropna(subset=["Goals"])
    combined["Goals"] = combined["Goals"].astype(int)
    return combined[["Division", "Team", "Player", "Goals"]]

# ========================= Fetch helpers ======================================
def fetch_xlsx_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content:
        raise ValueError("Downloaded file is empty.")
    return r.content

# ========================= UI helpers =========================================
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
    inject_fluid_css()

    st.markdown("<h1>ABEER BLUESTAR SOCCER FEST 2K25</h1>", unsafe_allow_html=True)

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

    # Keep a permanent full copy for downloads (ALWAYS all rows)
    full_df = data.copy()

    last_ref = st.session_state.get("last_refresh", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.sidebar.caption(f"Last refreshed: {last_ref}")

    # ===== Filters =====
    st.sidebar.header("Filters")
    div_opts = ["All"] + sorted(full_df["Division"].unique().tolist())
    div_sel = st.sidebar.selectbox("Division", div_opts)
    data_div = full_df if div_sel == "All" else full_df[full_df["Division"] == div_sel]

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

    # ===== Tabs =====
    t_overview, t_teams, t_players, t_downloads = st.tabs(["OVERVIEW", "TEAMS", "PLAYERS", "DOWNLOADS"])

    # ---------------------- OVERVIEW ----------------------
    with t_overview:
        display_metrics(filtered)

        st.subheader("Goal Scoring Records")
        if filtered.empty:
            st.info("No records under current filters.")
        else:
            st.dataframe(
                filtered.sort_values("Goals", ascending=False).reset_index(drop=True),
                use_container_width=True, hide_index=True,
                column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
            )

        st.subheader("Goals by Team")
        team_goals = (
            filtered.groupby("Team", as_index=False)["Goals"].sum().sort_values("Goals", ascending=False)
        )
        if team_goals.empty:
            st.info("No team data to display for the current filters.")
        else:
            st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

        # Top Scorers table with Team + safe slider
        st.subheader("Top Scorers")
        scorers_pt = (
            filtered.groupby(["Player", "Team"], as_index=False)["Goals"]
            .sum()
            .sort_values(["Goals", "Player"], ascending=[False, True])
        )
        max_rows = int(scorers_pt.shape[0])
        if max_rows == 0:
            st.info("No player data to display for the current filters.")
        elif max_rows == 1:
            st.caption("Only one scorer found — showing that row.")
            single_df = scorers_pt.head(1).reset_index(drop=True)
            st.dataframe(
                single_df, use_container_width=True, hide_index=True,
                column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
            )
            st.altair_chart(
                bar_chart(single_df.groupby("Player", as_index=False)["Goals"].sum(),
                          "Player", "Goals", "Top 1 Scorer by Goals"),
                use_container_width=True,
            )
        else:
            default_top = min(10, max_rows)
            top_n = st.sidebar.slider("Top N players", 1, max_rows, int(default_top), key=f"topn_{max_rows}")
            top_df = scorers_pt.head(int(top_n)).reset_index(drop=True)
            st.dataframe(
                top_df, use_container_width=True, hide_index=True,
                column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
            )
            chart_df = top_df.groupby("Player", as_index=False)["Goals"].sum()
            st.altair_chart(bar_chart(chart_df, "Player", "Goals", f"Top {int(top_n)} Scorers by Goals"),
                            use_container_width=True)

        if div_sel == "All" and not full_df.empty:
            st.subheader("Goals Distribution by Division")
            st.altair_chart(pie_chart(full_df), use_container_width=True)

    # ---------------------- TEAMS (Teams List) ----------------------
    with t_teams:
        st.subheader("Teams List")
        if filtered.empty:
            st.info("No teams under current filters.")
        else:
            # Summary: Division(s), Team, Players, Total Goals, Top Scorer, Top Scorer Goals
            team_div = (filtered.groupby("Team")["Division"]
                        .agg(lambda s: ", ".join(sorted(s.astype(str).unique()))).reset_index())
            team_summary = (filtered.groupby("Team", as_index=False)
                            .agg(Total_Goals=("Goals","sum"), Players=("Player","nunique")))
            top_by_team = (filtered.groupby(["Team","Player"], as_index=False)["Goals"].sum())
            top_by_team = (top_by_team.sort_values(["Team","Goals"], ascending=[True,False])
                           .groupby("Team").head(1)
                           .rename(columns={"Player":"Top Scorer","Goals":"Top Scorer Goals"}))

            teams_list = (team_div.merge(team_summary, on="Team", how="left")
                                 .merge(top_by_team, on="Team", how="left")
                          .sort_values(["Total_Goals","Team"], ascending=[False,True]))
            teams_list = teams_list[["Division","Team","Players","Total_Goals","Top Scorer","Top Scorer Goals"]]

            st.dataframe(
                teams_list, use_container_width=True, hide_index=True,
                column_config={
                    "Total_Goals": st.column_config.NumberColumn("Total Goals", format="%d"),
                    "Players": st.column_config.NumberColumn("Players", format="%d"),
                    "Top Scorer Goals": st.column_config.NumberColumn("Top Scorer Goals", format="%d"),
                },
            )

            st.altair_chart(bar_chart(teams_list.rename(columns={"Total_Goals":"Goals"}),
                                      "Team", "Goals", "Team Totals (Goals)"),
                            use_container_width=True)

    # ---------------------- PLAYERS (Players List) ----------------------
    with t_players:
        st.subheader("Players List")
        if filtered.empty:
            st.info("No players under current filters.")
        else:
            players_list = (filtered.groupby(["Player","Team","Division"], as_index=False)["Goals"].sum()
                            .sort_values(["Goals","Player"], ascending=[False,True]))
            st.dataframe(
                players_list, use_container_width=True, hide_index=True,
                column_config={"Goals": st.column_config.NumberColumn("Goals", format="%d")},
            )

    # ---------------------- DOWNLOADS ----------------------
    with t_downloads:
        st.subheader("Download Reports")
        st.caption("**Full** = all divisions, ignores filters. **Filtered** = current view.")

        st.download_button("⬇️ Download FULL (ALL rows) CSV",
            data=full_df.to_csv(index=False), file_name="records_full.csv", mime="text/csv")
        st.download_button("⬇️ Download FILTERED (current view) CSV",
            data=filtered.to_csv(index=False), file_name="records_filtered.csv", mime="text/csv")

        # Extra downloads for new tabs
        teams_list_dl = (filtered.groupby(["Team","Division"], as_index=False)
                         .agg(Players=("Player","nunique"), Total_Goals=("Goals","sum"))
                         .sort_values(["Total_Goals","Team"], ascending=[False,True]))
        st.download_button("⬇️ Download TEAMS LIST (current view) CSV",
            data=teams_list_dl.to_csv(index=False), file_name="teams_list_filtered.csv", mime="text/csv")

        players_list_dl = (filtered.groupby(["Player","Team","Division"], as_index=False)["Goals"].sum()
                           .sort_values(["Goals","Player"], ascending=[False,True]))
        st.download_button("⬇️ Download PLAYERS LIST (current view) CSV",
            data=players_list_dl.to_csv(index=False), file_name="players_list_filtered.csv", mime="text/csv")

        # ZIP bundle (includes full/filtered + summaries)
        zip_bytes = make_reports_zip(full_df, filtered)
        st.download_button("📦 Download ALL reports (ZIP)",
            data=zip_bytes, file_name="football_reports.zip", mime="application/zip")

if __name__ == "__main__":
    main()
