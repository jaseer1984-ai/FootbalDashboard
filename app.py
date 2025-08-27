# app.py — Football Goals Dashboard (Division + Team filters, XLSX fallback)
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

# -----------------------------
# XLSX fallback parser (no openpyxl)
# -----------------------------
def _parse_xlsx_without_openpyxl(file_bytes: bytes) -> pd.DataFrame:
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                root = ET.parse(f).getroot()
                for si in root.findall('.//main:si', ns):
                    text = ''.join(t.text or '' for t in si.findall('.//main:t', ns))
                    shared_strings.append(text)
        with z.open('xl/worksheets/sheet1.xml') as f:
            root = ET.parse(f).getroot()
            sheet_data = root.find('main:sheetData', ns)
            rows = []
            for row in sheet_data.findall('main:row', ns):
                row_data = {}
                for c in row.findall('main:c', ns):
                    ref = c.attrib.get('r', 'A1')
                    col = ''.join(ch for ch in ref if ch.isalpha())
                    t = c.attrib.get('t')
                    v = c.find('main:v', ns)
                    val = v.text if v is not None else None
                    if t == 's' and val is not None:
                        idx = int(val)
                        if 0 <= idx < len(shared_strings):
                            val = shared_strings[idx]
                    row_data[col] = val
                rows.append(row_data)
        if not rows:
            return pd.DataFrame()
        all_cols = sorted({c for r in rows for c in r.keys()},
                          key=lambda s: [ord(ch) for ch in s])
        data = [[r.get(c) for c in all_cols] for r in rows]
        return pd.DataFrame(data, columns=all_cols)

def _read_excel_safely(file_like_or_path) -> pd.DataFrame:
    file_bytes = None
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        with open(p, 'rb') as fh:
            file_bytes = fh.read()
        src = p
    else:
        file_bytes = file_like_or_path.read()
        file_like_or_path.seek(0)
        src = file_like_or_path
    try:
        return pd.read_excel(src)
    except ImportError:
        return _parse_xlsx_without_openpyxl(file_bytes)

def load_and_prepare_data(file_like_or_path) -> pd.DataFrame:
    raw_df = _read_excel_safely(file_like_or_path)
    if "B Division" in raw_df.columns and "A Division" in raw_df.columns:
        b_df = raw_df[["B Division", "Unnamed: 1", "Unnamed: 2"]].copy()
        b_df.columns = ["Team", "Player", "Goals"]
        b_df["Division"] = "B Division"
        a_df = raw_df[["A Division", "Unnamed: 6", "Unnamed: 7"]].copy()
        a_df.columns = ["Team", "Player", "Goals"]
        a_df["Division"] = "A Division"
    else:
        if len(raw_df) < 3:
            return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
        header = raw_df.iloc[1].tolist()
        data_rows = raw_df.iloc[2:].reset_index(drop=True)
        data_rows.columns = header
        b_df = data_rows.iloc[:, :3].copy()
        b_df.columns = ["Team", "Player", "Goals"]
        b_df["Division"] = "B Division"
        a_df = data_rows.iloc[:, 3:].copy()
        a_df.columns = ["Team", "Player", "Goals"]
        a_df["Division"] = "A Division"

    combined = pd.concat([b_df, a_df], ignore_index=True)
    combined = combined.dropna(subset=["Team", "Player", "Goals"])
    combined["Goals"] = pd.to_numeric(combined["Goals"], errors="coerce")
    combined = combined.dropna(subset=["Goals"])
    combined["Goals"] = combined["Goals"].astype(int)
    return combined

def display_metrics(df: pd.DataFrame) -> None:
    total_goals = int(df["Goals"].sum()) if not df.empty else 0
    num_players = df["Player"].nunique() if not df.empty else 0
    num_teams = df["Team"].nunique() if not df.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Goals", f"{total_goals}")
    c2.metric("Number of Players", f"{num_players}")
    c3.metric("Number of Teams", f"{num_teams}")

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
    division_goals = df.groupby("Division")["Goals"].sum().reset_index()
    return (
        alt.Chart(division_goals)
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
        if p.exists(): return p
    return None

def main() -> None:
    st.set_page_config(
        page_title="Football Goals Dashboard",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Football Goals Dashboard")
    st.write("Explore goal-scoring statistics for A and B divisions.")

    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload Goal Score.xlsx", type=["xlsx"])
    if uploaded is not None:
        df = load_and_prepare_data(uploaded)
    else:
        local = _find_local_excel()
        if local is None:
            st.warning("No Excel file found. Please upload **Goal Score.xlsx** using the sidebar.")
            st.stop()
        df = load_and_prepare_data(local)

    # -------- Filters: Division then Team --------
    st.sidebar.header("Filters")
    div_options = ["All"] + sorted(df["Division"].unique().tolist())
    chosen_div = st.sidebar.selectbox("Division", div_options)

    # Apply division filter first
    df_div = df if chosen_div == "All" else df[df["Division"] == chosen_div]

    # Team multiselect (context-aware to current division)
    team_options = sorted(df_div["Team"].unique().tolist())
    chosen_teams = st.sidebar.multiselect("Team (optional)", options=team_options)

    if chosen_teams:
        filtered = df_div[df_div["Team"].isin(chosen_teams)]
    else:
        filtered = df_div

    # -------- Metrics --------
    display_metrics(filtered)

    # -------- Table --------
    st.subheader("Goal Scoring Records")
    st.dataframe(filtered.sort_values(by="Goals", ascending=False), use_container_width=True)

    # -------- Goals by Team --------
    st.subheader("Goals by Team")
    team_goals = (
        filtered.groupby("Team")["Goals"].sum().reset_index().sort_values(by="Goals", ascending=False)
    )
    st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

    # -------- Top Scorers --------
    st.subheader("Top Scorers")
    player_goals = (
        filtered.groupby("Player")["Goals"].sum().reset_index().sort_values(by="Goals", ascending=False)
    )
    max_players = player_goals.shape[0] or 1
    top_n = st.sidebar.slider("Top N players", 1, max_players, min(10, max_players))
    st.altair_chart(
        bar_chart(player_goals.head(top_n), "Player", "Goals", f"Top {top_n} Players by Goals"),
        use_container_width=True,
    )

    # -------- Distribution by Division --------
    if chosen_div == "All":
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(df), use_container_width=True)

if __name__ == "__main__":
    main()
