"""
Football Goals Dashboard
=======================

This Streamlit application provides an interactive dashboard to explore goal‑
scoring statistics for two football divisions (A and B) using the provided
Excel file ``Goal Score.xlsx``.  The app reads the Excel file, restructures
its contents into a tidy format, and then computes metrics such as total
goals, number of players and teams.  Users can filter the data by division
using the sidebar and view summary tables and charts.  Bar charts display
goals by team and top scorers, while a pie chart compares the total
contribution of each division.  The app uses Streamlit for the user
interface and Altair for the charts, as recommended when building
dashboards【22939637440300†L217-L230】.

To run the app locally, make sure you have the dependencies listed in
``requirements.txt`` installed and then execute ``streamlit run app.py``
from a terminal.  When deploying on GitHub/Streamlit Cloud, include this
file and ``Goal Score.xlsx`` in your repository.

"""

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path


def load_and_prepare_data(excel_path: Path) -> pd.DataFrame:
    """Load the Excel file and reshape it into a tidy DataFrame.

    The input file contains two sections side‑by‑side: one for the B
    division and one for the A division.  Each section has columns for
    team, player name and number of goals.  This function extracts each
    section, assigns a division label and concatenates them into a single
    long‑format DataFrame with columns ``Division``, ``Team``, ``Player``
    and ``Goals``.  Rows with missing or non‑numeric goal counts are
    dropped.  Goal counts are converted to integers.

    Parameters
    ----------
    excel_path : Path
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        A tidy DataFrame containing division, team, player and goals.
    """
    # Read the entire Excel sheet
    raw_df = pd.read_excel(excel_path)

    # Extract columns for the B Division
    b_df = raw_df[["B Division", "Unnamed: 1", "Unnamed: 2"]].copy()
    b_df.columns = ["Team", "Player", "Goals"]
    b_df["Division"] = "B Division"
    # Drop rows where critical fields are missing
    b_df = b_df.dropna(subset=["Team", "Player", "Goals"])
    # Convert goals to numeric, coercing invalid values to NaN
    b_df["Goals"] = pd.to_numeric(b_df["Goals"], errors="coerce")
    b_df = b_df.dropna(subset=["Goals"])

    # Extract columns for the A Division
    a_df = raw_df[["A Division", "Unnamed: 6", "Unnamed: 7"]].copy()
    a_df.columns = ["Team", "Player", "Goals"]
    a_df["Division"] = "A Division"
    a_df = a_df.dropna(subset=["Team", "Player", "Goals"])
    a_df["Goals"] = pd.to_numeric(a_df["Goals"], errors="coerce")
    a_df = a_df.dropna(subset=["Goals"])

    # Combine and reset index
    combined_df = pd.concat([b_df, a_df], ignore_index=True)
    # Ensure goals are integers
    combined_df["Goals"] = combined_df["Goals"].astype(int)
    return combined_df


def display_metrics(df: pd.DataFrame) -> None:
    """Display high‑level summary metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame for which to compute metrics.
    """
    total_goals = int(df["Goals"].sum())
    num_players = df["Player"].nunique()
    num_teams = df["Team"].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Goals", f"{total_goals}")
    col2.metric("Number of Players", f"{num_players}")
    col3.metric("Number of Teams", f"{num_teams}")


def bar_chart(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    """Create a bar chart using Altair.

    Parameters
    ----------
    df : pd.DataFrame
        Data used to create the chart.  The DataFrame should have
        columns for the category (categorical variable) and value
        (numeric variable).
    category : str
        Column name for the categorical variable (e.g. team or player).
    value : str
        Column name for the numeric variable (e.g. goals).
    title : str
        Title of the chart.

    Returns
    -------
    alt.Chart
        The generated bar chart.
    """
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title="Number of Goals"),
            y=alt.Y(f"{category}:N", sort="-x", title=category),
            tooltip=[category, value],
        )
        .properties(
            height=400,
            title=title,
        )
    )
    return chart


def pie_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a pie (donut) chart showing goals contribution by division.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns ``Division`` and ``Goals``.

    Returns
    -------
    alt.Chart
        The generated pie chart.
    """
    division_goals = df.groupby("Division")["Goals"].sum().reset_index()
    chart = (
        alt.Chart(division_goals)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Goals:Q", title="Goals"),
            color=alt.Color("Division:N", title="Division"),
            tooltip=["Division", "Goals"],
        )
        .properties(title="Goals by Division")
    )
    return chart


def main() -> None:
    """Main function for the Streamlit app."""
    # Set page configuration: wide layout and custom title【22939637440300†L242-L248】
    st.set_page_config(
        page_title="Football Goals Dashboard",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Football Goals Dashboard")
    st.write("Explore goal‑scoring statistics for A and B divisions.")

    # Load and prepare data
    excel_path = Path("Goal Score.xlsx")
    data = load_and_prepare_data(excel_path)

    # Sidebar filters
    st.sidebar.header("Filters")
    divisions = ["All"] + sorted(data["Division"].unique().tolist())
    selected_division = st.sidebar.selectbox("Select Division", divisions)
    # Filter data by selected division
    if selected_division == "All":
        filtered_data = data.copy()
    else:
        filtered_data = data[data["Division"] == selected_division]

    # Show metrics
    display_metrics(filtered_data)

    # Display raw data table
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
    # Top N slider
    max_players = player_goals.shape[0]
    default_top = min(10, max_players)
    top_n = st.sidebar.slider("Number of top players to display", 1, max_players, default_top)
    top_players = player_goals.head(top_n)
    st.altair_chart(
        bar_chart(top_players, "Player", "Goals", f"Top {top_n} Players by Goals"),
        use_container_width=True,
    )

    # Goals by division pie chart (only show when viewing all divisions)
    if selected_division == "All":
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(data), use_container_width=True)


if __name__ == "__main__":
    main()
