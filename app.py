"""
Football Goals Dashboard
=======================

This Streamlit application provides an interactive dashboard to explore goal‑
scoring statistics for two football divisions (A and B) using the provided
Excel file ``Goal Score.xlsx``.
"""

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

def load_and_prepare_data(excel_path: Path) -> pd.DataFrame:
    """
    Load the Excel file and reshape it into a tidy DataFrame.

    This function first attempts to read the file with pandas (which uses
    openpyxl under the hood).  If openpyxl isn’t installed, a fallback parser
    based on the standard library is used.  The worksheet contains two
    side‑by‑side tables for B and A divisions; they are split, cleaned and
    concatenated into a single DataFrame with columns Division, Team,
    Player and Goals.
    """
    # Try pandas; fallback if openpyxl is missing
    try:
        raw_df = pd.read_excel(excel_path)
    except ImportError:
        raw_df = _parse_xlsx_without_openpyxl(excel_path)

    # If columns are letters (from fallback), use row 1 as header
    first_col = list(raw_df.columns)[0]
    if first_col in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        if len(raw_df) < 3:
            return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])
        header = raw_df.iloc[1].values
        data_rows = raw_df.iloc[2:].reset_index(drop=True)
        data_rows.columns = header
        b_df = data_rows.iloc[:, :3].copy()
        b_df.columns = ["Team", "Player", "Goals"]
        b_df["Division"] = "B Division"
        a_df = data_rows.iloc[:, 3:].copy()
        a_df.columns = ["Team", "Player", "Goals"]
        a_df["Division"] = "A Division"
        combined_df = pd.concat([b_df, a_df], ignore_index=True)
    else:
        b_df = raw_df[["B Division", "Unnamed: 1", "Unnamed: 2"]].copy()
        b_df.columns = ["Team", "Player", "Goals"]
        b_df["Division"] = "B Division"
        a_df = raw_df[["A Division", "Unnamed: 6", "Unnamed: 7"]].copy()
        a_df.columns = ["Team", "Player", "Goals"]
        a_df["Division"] = "A Division"
        combined_df = pd.concat([b_df, a_df], ignore_index=True)

    # Clean and convert goals
    combined_df = combined_df.dropna(subset=["Team", "Player", "Goals"])
    combined_df["Goals"] = pd.to_numeric(combined_df["Goals"], errors="coerce")
    combined_df = combined_df.dropna(subset=["Goals"])
    combined_df["Goals"] = combined_df["Goals"].astype(int)
    return combined_df

def _parse_xlsx_without_openpyxl(file_obj: Path) -> pd.DataFrame:
    """
    Fallback parser for Excel .xlsx files using only built‑in modules.

    It reads the first worksheet from a .xlsx archive and returns a
    DataFrame with columns labelled by Excel column letters.
    """
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    # Read file into memory
    with open(file_obj, 'rb') as f:
        bytes_data = f.read()

    with zipfile.ZipFile(BytesIO(bytes_data)) as z:
        # Parse shared strings
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                for si in root.findall('.//main:si', ns):
                    text = ''.join([t.text for t in si.findall('.//main:t', ns) if t.text])
                    shared_strings.append(text)

        # Parse the first worksheet
        with z.open('xl/worksheets/sheet1.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            sheet_data = root.find('main:sheetData', ns)
            rows = []
            for row in sheet_data.findall('main:row', ns):
                row_data = {}
                for c in row.findall('main:c', ns):
                    cell_ref = c.attrib['r']
                    col = ''.join(ch for ch in cell_ref if ch.isalpha())
                    cell_type = c.attrib.get('t')
                    v = c.find('main:v', ns)
                    val = v.text if v is not None else None
                    if cell_type == 's' and val is not None:
                        idx = int(val)
                        val = shared_strings[idx] if idx < len(shared_strings) else val
                    row_data[col] = val
                rows.append(row_data)

        # Build DataFrame
        if not rows:
            return pd.DataFrame()
        col_letters = sorted({col for r in rows for col in r.keys()},
                             key=lambda s: [ord(c) for c in s])
        data = []
        for r in rows:
            data.append([r.get(col, None) for col in col_letters])
        return pd.DataFrame(data, columns=col_letters)

def display_metrics(df: pd.DataFrame) -> None:
    """Display high‑level summary metrics."""
    total_goals = int(df["Goals"].sum())
    num_players = df["Player"].nunique()
    num_teams = df["Team"].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Goals", f"{total_goals}")
    col2.metric("Number of Players", f"{num_players}")
    col3.metric("Number of Teams", f"{num_teams}")

def bar_chart(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    """Create a bar chart using Altair."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title="Number of Goals"),
            y=alt.Y(f"{category}:N", sort="-x", title=category),
            tooltip=[category, value],
        )
        .properties(height=400, title=title)
    )
    return chart

def pie_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a donut chart showing goals contribution by division."""
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

def main() -> None:
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Football Goals Dashboard",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Football Goals Dashboard")
    st.write("Explore goal‑scoring statistics for A and B divisions.")

    # Load and prepare data using fallback if necessary
    excel_path = Path("Goal Score.xlsx")
    data = load_and_prepare_data(excel_path)

    # Sidebar filters
    st.sidebar.header("Filters")
    divisions = ["All"] + sorted(data["Division"].unique().tolist())
    selected_division = st.sidebar.selectbox("Select Division", divisions)
    if selected_division == "All":
        filtered_data = data.copy()
    else:
        filtered_data = data[data["Division"] == selected_division]

    # Display metrics
    display_metrics(filtered_data)

    # Data table
    st.subheader("Goal Scoring Records")
    st.dataframe(filtered_data.sort_values(by="Goals", ascending=False), use_container_width=True)

    # Goals by team
    st.subheader("Goals by Team")
    team_goals = (
        filtered_data.groupby("Team")["Goals"].sum().reset_index().sort_values(by="Goals", ascending=False)
    )
    st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

    # Top scorers with adjustable top N
    st.subheader("Top Scorers")
    player_goals = (
        filtered_data.groupby("Player")["Goals"].sum().reset_index().sort_values(by="Goals", ascending=False)
    )
    max_players = player_goals.shape[0]
    default_top = min(10, max_players)
    top_n = st.sidebar.slider("Number of top players to display", 1, max_players, default_top)
    top_players = player_goals.head(top_n)
    st.altair_chart(
        bar_chart(top_players, "Player", "Goals", f"Top {top_n} Players by Goals"),
        use_container_width=True,
    )

    # Division pie chart
    if selected_division == "All":
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(data), use_container_width=True)

if __name__ == "__main__":
    main()
