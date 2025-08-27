# app.py — Football Goals Dashboard (with Refresh button, direct URL input, player search)
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import requests

# ========== XLSX fallback (no openpyxl) ==========
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

# ========== Robust block parser ==========
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
        temp = raw.iloc[data_start:, start_col:end_col].copy()
        if len(labels) != 3:
            return
        temp.columns = labels
        temp = temp.rename(columns={labels[0]: "Team", labels[1]: "Player", labels[2]: "Goals"})
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

# ========== Fetch helpers ==========
def fetch_xlsx_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content:
        raise ValueError("Downloaded file is empty.")
    return r.content

# ========== UI helpers ==========
def display_metrics(df: pd.DataFrame) -> None:
    total = int(df["Goals"].sum()) if not df.empty else 0
    players = df["Player"].nunique() if not df.empty else 0
    teams = df["Team"].nunique() if not df.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Goals", f"{total}")
    c2.metric("Number of Players", f"{players}")
    c3.metric("Number of Teams", f"{teams}")

def bar_chart(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title="Number of Goals"),
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

# ========== App ==========
def main():
    st.set_page_config(page_title="Football Goals Dashboard", page_icon="⚽", layout="wide")
    st.title("Football Goals Dashboard")

    st.sidebar.header("Data source (URL)")
    default_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRpCD-Wh_NnGQjJ1Mh3tuU5Mdcl8TK41JopMUcSnfqww8wkPXKKgRyg7v4sC_vuUw/pub?output=xlsx"
    xlsx_url = st.sidebar.text_input("Paste XLSX URL", value=default_url)

    # --- Refresh button ---
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()

    @st.cache_data(ttl=300)
    def load_data(url):
        data_bytes = fetch_xlsx_bytes_from_url(url)
        return load_and_prepare_data_from_bytes(data_bytes)

    try:
        data = load_data(xlsx_url.strip())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # -------- Filters --------
    st.sidebar.header("Filters")
    div_opts = ["All"] + sorted(data["Division"].unique().tolist())
    div_sel = st.sidebar.selectbox("Division", div_opts)
    data_div = data if div_sel == "All" else data[data["Division"] == div_sel]

    team_opts = sorted(data_div["Team"].unique().tolist())
    team_sel = st.sidebar.multiselect("Team (optional)", team_opts)
    filtered = data_div if not team_sel else data_div[data_div["Team"].isin(team_sel)]

    # Player search
    st.sidebar.subheader("Player search")
    player_query = st.sidebar.text_input("Type player names (comma-separated, partial OK)")
    if player_query.strip():
        tokens = [t.strip().lower() for t in player_query.split(",") if t.strip()]
        if tokens:
            mask = False
            for t in tokens:
                mask = mask | filtered["Player"].str.lower().str.contains(t, na=False)
            filtered = filtered[mask]

    players_pick = st.sidebar.multiselect("Players (refine optional)", options=sorted(filtered["Player"].unique().tolist()))
    if players_pick:
        filtered = filtered[filtered["Player"].isin(players_pick)]

    # -------- Metrics --------
    display_metrics(filtered)

    # -------- Table --------
    st.subheader("Goal Scoring Records")
    if filtered.empty:
        st.info("No records under current filters.")
    else:
        st.dataframe(filtered.sort_values("Goals", ascending=False), use_container_width=True)

    # -------- Goals by Team --------
    st.subheader("Goals by Team")
    team_goals = (
        filtered.groupby("Team")["Goals"].sum().reset_index().sort_values("Goals", ascending=False)
    )
    if team_goals.empty:
        st.info("No team data to display.")
    else:
        st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

    # -------- Top Scorers --------
    st.subheader("Top Scorers")
    player_goals = (
        filtered.groupby("Player")["Goals"].sum().reset_index().sort_values("Goals", ascending=False)
    )
    max_players = int(player_goals.shape[0])
    if max_players == 0:
        st.info("No player data.")
    elif max_players == 1:
        st.caption("Only one player found.")
        st.altair_chart(bar_chart(player_goals.head(1), "Player", "Goals", "Top 1 Player"), use_container_width=True)
    else:
        default_top = min(10, max_players)
        top_n = st.sidebar.slider("Top N players", 1, max_players, int(default_top))
        st.altair_chart(bar_chart(player_goals.head(int(top_n)), "Player", "Goals", f"Top {int(top_n)} Players"), use_container_width=True)

    # -------- Division Distribution --------
    if div_sel == "All" and not data.empty:
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(data), use_container_width=True)

if __name__ == "__main__":
    main()
