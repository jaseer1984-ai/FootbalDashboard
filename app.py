import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Football Goal Scoring Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the Excel data"""
    try:
        # Read the Excel file
        df = pd.read_excel('Goal Score.xlsx', sheet_name='Sheet1', header=None)
        
        # Process B Division data (columns 0-2)
        b_division_data = []
        b_start_row = 2  # Data starts from row 3 (index 2)
        
        for i in range(b_start_row, len(df)):
            if pd.notna(df.iloc[i, 0]) and pd.notna(df.iloc[i, 1]) and pd.notna(df.iloc[i, 2]):
                b_division_data.append({
                    'Division': 'B Division',
                    'Team': df.iloc[i, 0],
                    'Player': df.iloc[i, 1],
                    'Goals': df.iloc[i, 2]
                })
        
        # Process A Division data (columns 5-7)
        a_division_data = []
        
        for i in range(b_start_row, len(df)):
            if pd.notna(df.iloc[i, 5]) and pd.notna(df.iloc[i, 6]) and pd.notna(df.iloc[i, 7]):
                a_division_data.append({
                    'Division': 'A Division',
                    'Team': df.iloc[i, 5],
                    'Player': df.iloc[i, 6],
                    'Goals': df.iloc[i, 7]
                })
        
        # Combine both divisions
        all_data = b_division_data + a_division_data
        processed_df = pd.DataFrame(all_data)
        
        return processed_df
    
    except FileNotFoundError:
        # Sample data for demonstration when file is not available
        sample_data = {
            'Division': ['B Division', 'B Division', 'A Division', 'A Division', 'B Division', 'A Division'] * 5,
            'Team': ['Bluestar B', 'Newcastle FC', 'Blasters FC', 'Real Kerala', 'Bluestar B', 'Blasters FC'] * 5,
            'Player': ['Alan Solaman', 'Muhammed Shuhaib', 'Muhammed Jiyad', 'Muhamed Ashik', 'Sufaid Thazhathe', 'Muhammed Niyas'] * 5,
            'Goals': np.random.randint(1, 5, 30)
        }
        return pd.DataFrame(sample_data)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Football Goal Scoring Dashboard</h1>
        <p>Track player performance across divisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🎯 Filters")
    
    # Division filter
    divisions = ['All'] + list(df['Division'].unique())
    selected_division = st.sidebar.selectbox("Select Division:", divisions)
    
    # Team filter
    if selected_division != 'All':
        teams = ['All'] + list(df[df['Division'] == selected_division]['Team'].unique())
    else:
        teams = ['All'] + list(df['Team'].unique())
    selected_team = st.sidebar.selectbox("Select Team:", teams)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_division != 'All':
        filtered_df = filtered_df[filtered_df['Division'] == selected_division]
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['Team'] == selected_team]
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_goals = filtered_df['Goals'].sum()
        st.metric("🥅 Total Goals", total_goals)
    
    with col2:
        total_players = filtered_df['Player'].nunique()
        st.metric("👥 Total Players", total_players)
    
    with col3:
        total_teams = filtered_df['Team'].nunique()
        st.metric("🏆 Total Teams", total_teams)
    
    with col4:
        avg_goals = round(filtered_df['Goals'].mean(), 2) if len(filtered_df) > 0 else 0
        st.metric("📊 Avg Goals/Player", avg_goals)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏅 Top Goal Scorers")
        top_scorers = filtered_df.groupby('Player')['Goals'].sum().sort_values(ascending=False).head(10)
        
        if not top_scorers.empty:
            # Create a DataFrame for better display
            top_scorers_df = pd.DataFrame({
                'Player': top_scorers.index,
                'Goals': top_scorers.values
            })
            
            # Display as bar chart using Streamlit's built-in chart
            st.bar_chart(top_scorers_df.set_index('Player'))
            
            # Also show as a table
            st.dataframe(top_scorers_df, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")
    
    with col2:
        st.subheader("🏟️ Goals by Team")
        team_goals = filtered_df.groupby('Team')['Goals'].sum().sort_values(ascending=False)
        
        if not team_goals.empty:
            # Create a DataFrame for better display
            team_goals_df = pd.DataFrame({
                'Team': team_goals.index,
                'Goals': team_goals.values
            })
            
            # Display as bar chart
            st.bar_chart(team_goals_df.set_index('Team'))
            
            # Show percentage breakdown
            team_goals_df['Percentage'] = (team_goals_df['Goals'] / team_goals_df['Goals'].sum() * 100).round(1)
            st.dataframe(team_goals_df, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🆚 Division Comparison")
        if len(df['Division'].unique()) > 1:
            division_stats = df.groupby('Division').agg({
                'Goals': ['sum', 'mean', 'count'],
                'Player': 'nunique',
                'Team': 'nunique'
            }).round(2)
            
            division_stats.columns = ['Total Goals', 'Avg Goals', 'Total Records', 'Unique Players', 'Unique Teams']
            
            # Reset index to make Division a column
            division_stats_display = division_stats.reset_index()
            
            # Display as table
            st.dataframe(division_stats_display, use_container_width=True)
            
            # Show total goals by division as chart
            total_goals_by_div = df.groupby('Division')['Goals'].sum()
            st.bar_chart(total_goals_by_div)
        else:
            st.info("Need data from multiple divisions for comparison.")
    
    with col2:
        st.subheader("📈 Goal Distribution")
        
        if not filtered_df.empty:
            # Create histogram data
            goal_counts = filtered_df['Goals'].value_counts().sort_index()
            
            # Display as bar chart
            st.bar_chart(goal_counts)
            
            # Show statistics
            st.write("**Distribution Statistics:**")
            st.write(f"• Most common score: {goal_counts.index[0]} goals ({goal_counts.iloc[0]} players)")
            st.write(f"• Highest score: {filtered_df['Goals'].max()} goals")
            st.write(f"• Lowest score: {filtered_df['Goals'].min()} goals")
        else:
            st.info("No data available for the selected filters.")
    
    # Data Table
    st.subheader("📋 Detailed Data")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search players or teams:", "")
    if search_term:
        mask = (
            filtered_df['Player'].str.contains(search_term, case=False, na=False) |
            filtered_df['Team'].str.contains(search_term, case=False, na=False)
        )
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df
    
    # Sort options
    sort_by = st.selectbox("Sort by:", ['Goals (Descending)', 'Goals (Ascending)', 'Player Name', 'Team Name'])
    
    if sort_by == 'Goals (Descending)':
        display_df = display_df.sort_values('Goals', ascending=False)
    elif sort_by == 'Goals (Ascending)':
        display_df = display_df.sort_values('Goals', ascending=True)
    elif sort_by == 'Player Name':
        display_df = display_df.sort_values('Player')
    else:
        display_df = display_df.sort_values('Team')
    
    # Display the dataframe with highlighting for top performers
    if not display_df.empty:
        # Add rank column
        display_df_with_rank = display_df.copy()
        display_df_with_rank['Rank'] = display_df_with_rank['Goals'].rank(method='dense', ascending=False).astype(int)
        
        # Reorder columns
        display_df_with_rank = display_df_with_rank[['Rank', 'Division', 'Team', 'Player', 'Goals']]
        
        st.dataframe(
            display_df_with_rank.reset_index(drop=True),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df_with_rank.to_csv(index=False)
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name="football_goals_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No data matches your search criteria.")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Statistics:**")
        if not df.empty:
            st.write(f"• Total goals scored: {df['Goals'].sum()}")
            st.write(f"• Average goals per player: {df['Goals'].mean():.2f}")
            st.write(f"• Highest individual score: {df['Goals'].max()}")
            st.write(f"• Total unique players: {df['Player'].nunique()}")
            st.write(f"• Total unique teams: {df['Team'].nunique()}")
            
            # Top performer
            top_player = df.groupby('Player')['Goals'].sum().idxmax()
            top_goals = df.groupby('Player')['Goals'].sum().max()
            st.write(f"• Top scorer: {top_player} ({top_goals} goals)")
    
    with col2:
        st.write("**Division Breakdown:**")
        if not df.empty:
            for division in df['Division'].unique():
                div_data = df[df['Division'] == division]
                st.write(f"**{division}:**")
                st.write(f"  - Total goals: {div_data['Goals'].sum()}")
                st.write(f"  - Players: {div_data['Player'].nunique()}")
                st.write(f"  - Teams: {div_data['Team'].nunique()}")
                if len(div_data) > 0:
                    st.write(f"  - Avg goals/player: {div_data['Goals'].mean():.2f}")
                    
                    # Top scorer in division
                    div_top_player = div_data.groupby('Player')['Goals'].sum().idxmax()
                    div_top_goals = div_data.groupby('Player')['Goals'].sum().max()
                    st.write(f"  - Top scorer: {div_top_player} ({div_top_goals} goals)")

if __name__ == "__main__":
    main()
