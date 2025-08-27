import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

def load_and_prepare_data(file_obj) -> pd.DataFrame:
    """Load the Excel file and reshape it into a tidy DataFrame."""
    # Accept either a Path or a file‑like object
    raw_df = pd.read_excel(file_obj)

    # Process B division columns
    b_df = raw_df[["B Division", "Unnamed: 1", "Unnamed: 2"]].copy()
    b_df.columns = ["Team", "Player", "Goals"]
    b_df["Division"] = "B Division"
    b_df = b_df.dropna(subset=["Team", "Player", "Goals"])
    b_df["Goals"] = pd.to_numeric(b_df["Goals"], errors="coerce")
    b_df = b_df.dropna(subset=["Goals"])

    # Process A division columns
    a_df = raw_df[["A Division", "Unnamed: 6", "Unnamed: 7"]].copy()
    a_df.columns = ["Team", "Player", "Goals"]
    a_df["Division"] = "A Division"
    a_df = a_df.dropna(subset=["Team", "Player", "Goals"])
    a_df["Goals"] = pd.to_numeric(a_df["Goals"], errors="coerce")
    a_df = a_df.dropna(subset=["Goals"])

    # Combine and convert goals to integer
    combined_df = pd.concat([b_df, a_df], ignore_index=True)
    combined_df["Goals"] = combined_df["Goals"].astype(int)
    return combined_df

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
    """Create a donut chart showing goals by division."""
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

def display_metrics(df: pd.DataFrame) -> None:
    """Display high‑level summary metrics."""
    total_goals = int(df["Goals"].sum())
    num_players = df["Player"].nunique()
    num_teams = df["Team"].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Goals", f"{total_goals}")
    col2.metric("Number of Players", f"{num_players}")
    col3.metric("Number of Teams", f"{num_teams}")

def main():
    # Page configuration:contentReference[oaicite:2]{index=2}
    st.set_page_config(
        page_title="Football Goals Dashboard",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Football Goals Dashboard")
    st.write("Explore goal‑scoring statistics for A and B divisions.")

    # Attempt to load the Excel file from the script directory
    script_dir = Path(__file__).parent
    default_file_path = script_dir / "Goal Score.xlsx"

    # Sidebar uploader in case the file isn’t found or to use custom file
    uploaded_file = st.sidebar.file_uploader("Upload Goal Score.xlsx", type=["xlsx"])
    if uploaded_file:
        data = load_and_prepare_data(uploaded_file)
    else:
        if default_file_path.exists():
            data = load_and_prepare_data(default_file_path)
        else:
            st.error("No data file found. Please upload Goal Score.xlsx.")
            return

    # Sidebar division filter and top‑N slider
    st.sidebar.header("Filters")
    divisions = ["All"] + sorted(data["Division"].unique().tolist())
    selected_division = st.sidebar.selectbox("Select Division", divisions)
    if selected_division == "All":
        filtered_data = data.copy()
    else:
        filtered_data = data[data["Division"] == selected_division]
    max_players = filtered_data["Player"].nunique()
    top_n = st.sidebar.slider(
        "Number of top players to display",
        min_value=1,
        max_value=max_players,
        value=min(10, max_players),
    )

    # Summary metrics
    display_metrics(filtered_data)

    # Data table
    st.subheader("Goal Scoring Records")
    st.dataframe(
        filtered_data.sort_values(by="Goals", ascending=False),
        use_container_width=True,
    )

    # Goals by team
    st.subheader("Goals by Team")
    team_goals = (
        filtered_data.groupby("Team")["Goals"]
        .sum()
        .reset_index()
        .sort_values(by="Goals", ascending=False)
    )
    st.altair_chart(bar_chart(team_goals, "Team", "Goals", "Goals by Team"), use_container_width=True)

    # Top scorers
    st.subheader(f"Top {top_n} Scorers")
    player_goals = (
        filtered_data.groupby("Player")["Goals"]
        .sum()
        .reset_index()
        .sort_values(by="Goals", ascending=False)
    )
    top_players = player_goals.head(top_n)
    st.altair_chart(
        bar_chart(top_players, "Player", "Goals", f"Top {top_n} Players by Goals"),
        use_container_width=True,
    )

    # Goals distribution by division
    if selected_division == "All":
        st.subheader("Goals Distribution by Division")
        st.altair_chart(pie_chart(data), use_container_width=True)

if __name__ == "__main__":
    main()
