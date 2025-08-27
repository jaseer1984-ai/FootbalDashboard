# app.py — Football Goals Dashboard (safe for 0/1 players, team filter, XLSX fallback)
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

# =========================
# XLSX fallback (no openpyxl)
# =========================
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
            for row in rows_xml.findall("main:row", ns):
                rdict = {}
                for c in row.findall("main:c", ns):
                    ref = c.attrib.get("r", "A1")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    t = c.attrib.get("t")
                    v = c.find("main:v", ns)
                    val = v.text if v is not None else None
                    if t == "s" and val is not None:
                        idx = int(val)
                        if 0 <= idx < len(shared):
                            val = shared[idx]
                    rdict[col] = val
                rows.append(rdict)
    if not rows:
        return pd.DataFrame()
    cols = sorted({k for r in rows for k in r.keys()}, key=lambda s: [ord(c) for c in s])
    data = [[r.get(c) for c in cols] for r in rows]
    return pd.DataFrame(data, columns=cols)

def _read_excel_safely(file_like_or_path) -> pd.DataFrame:
    # capture bytes for fallback
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        with open(p, "rb") as fh:
            b = fh.read()
        src = p
    else:
        b = file_like_or_path.read()
        file_like_or_path.seek(0)
        src = file_like_or_path
    try:
        return pd.read_excel(src)
    except ImportError:
        return _parse_xlsx_without_openpyxl(b)

# =========================
# Data shaping
# =========================
def load_and_prepare_data(file_like_or_path) -> pd.DataFrame:
    df = _read_excel_safely(file_like_or_path)

    # Case 1: pandas gave named columns
    if "B Division" in df.columns and "A Division" in df.columns:
        b = df[["B Division", "Unnamed: 1", "Unnamed: 2"]].copy()
        b.columns = ["Team", "Player", "Goals"]
        b["Division"] = "B Division"

        a = df[["A Division", "Unnamed: 6", "Unnamed: 7"]].copy()
        a.columns = ["Team", "Player", "Goals"]
        a["Division"] = "A Division"

    # Case 2: fallback sheet (letters; row1=header, row2+=data)
    else:
        if len(df) < 3:
            return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
        header = df.iloc[1].tolist()
        body = df.iloc[2:].reset_index(drop=True)
        body.columns = header

        b = body.iloc[:, :3].copy()
        b.columns = ["Team", "Player", "Goals"]
        b["Division"] = "B Division"

        a = body.iloc[:, 3:].copy()
        a.columns = ["Team", "Player", "Goals"]
        a["Division"] = "A Division"

    out = pd.concat([b, a], ignore_index=True)
    out = out.dropna(subset=["Team", "Player", "Goals"])
    out["Goals"] = pd.to_numeric(out["Goals"], errors="coerce")
    out = out.dropna(subset=["Goals"])
    out["Goals"] = out["Goals"].astype(int)
    return out

# =========================
# UI helpers
# =========================
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

def _find_local_excel() -> Path | None:
    for p in [Path("Goal Score.xlsx"), Path(__file__).parent / "Goal Score.xlsx"]:
        if p.exists():
            return p
    return None

# =========================
# App
# =========================
def main():
    st.set_page_config(page_title="Football Goals Dashboard", page_icon="⚽", layout="wide")
    st.title("Football Goals Dashboard")
    st.write("Explore goal-scoring statistics for A and B divisions.")

    # Data source
    st.sidebar.header("Data source")
    up = st.sidebar.file_uploader("Upload Goal Score.xlsx", type=["xlsx"])
    if up is not None:
        data = load_and_prepare_data(up)
    else:
        local = _find_local_excel()
        if local is None:
            st.warning("No Excel file found. Upload **Goal Score.xlsx** from the sidebar.")
            st.stop()
        data = load_and_prepare_data(local)

    # -------- Filters --------
    st.sidebar.header("Filters")
    div_opts = ["All"] + sorted(data["Division"].unique().tolist())
    div_sel = st.sidebar.selectbox("Division", div_opts)
    data_div = data if div_sel == "All" else data[data["Division"] == div_sel]

    team_opts = sorted(data_div["Team"].unique().tolist())
    team_sel = st.sidebar.multiselect("Team (optional)", team_opts)
    filtered = data_div if not team_sel else data_div[data_div["Team"].isin(team_sel)]

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
        filtered.groupby("Team")["Goals"]
        .sum()
        .reset_index()
        .sort_values("Goals", ascending=False)
    )
    if team_goals.empty:
        st.info("No team data to display for the current filters.")
    else:
        st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

    # -------- Top Scorers (handles 0 / 1 / many players) --------
    st.subheader("Top Scorers")
    player_goals = (
        filtered.groupby("Player")["Goals"]
        .sum()
        .reset_index()
        .sort_values("Goals", ascending=False)
    )
    max_players = int(player_goals.shape[0])

    if max_players == 0:
        st.info("No player data to display for the current filters (try clearing the Team filter).")
    elif max_players == 1:
        st.caption("Only one player found — showing that player.")
        st.altair_chart(
            bar_chart(player_goals.head(1), "Player", "Goals", "Top 1 Player by Goals"),
            use_container_width=True,
        )
    else:
        default_top = min(10, max_players)
        top_n = st.sidebar.slider(
            "Top N players",
            min_value=2,                # avoid min==max crash when 1 player (we're in else so >=2)
            max_value=max_players,
            value=int(default_top),
        )
        st.altair_chart(
            bar_chart(player_goals.head(int(top_n)), "Player", "Goals", f"Top {int(top_n)} Players by Goals"),
            use_container_width=True,
        )

    # -------- Division distribution --------
    if div_sel == "All" and not data.empty:
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(data), use_container_width=True)

if __name__ == "__main__":
    main()
