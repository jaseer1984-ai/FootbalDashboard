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
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the Excel data"""
    try:
        # Read the Excel file - no headers since we have custom structure
        df = pd.read_excel('Goal Score.xlsx', sheet_name='Sheet1', header=None)
        
        all_data = []
        
        # Process each row starting from row 2 (index 2, after headers)
        for i in range(2, len(df)):
            # Process B Division data (columns A, B, C = indexes 0, 1, 2)
            if (pd.notna(df.iloc[i, 0]) and 
                pd.notna(df.iloc[i, 1]) and 
                pd.notna(df.iloc[i, 2])):
                all_data.append({
                    'Division': 'B Division',
                    'Team': str(df.iloc[i, 0]).strip(),
                    'Player': str(df.iloc[i, 1]).strip(),
                    'Goals': int(df.iloc[i, 2])
                })
            
            # Process A Division data (columns F, G, H = indexes 5, 6, 7)
            if (pd.notna(df.iloc[i, 5]) and 
                pd.notna(df.iloc[i, 6]) and 
                pd.notna(df.iloc[i, 7])):
                all_data.append({
                    'Division': 'A Division',
                    'Team': str(df.iloc[i, 5]).strip(),
                    'Player': str(df.iloc[i, 6]).strip(),
                    'Goals': int(df.iloc[i, 7])
                })
        
        processed_df = pd.DataFrame(all_data)
        return processed_df
    
    except FileNotFoundError:
        st.error("❌ Goal Score.xlsx file not found. Please upload your Excel file.")
        # Return empty DataFrame
        return pd.DataFrame(columns=['Division', 'Team', 'Player', 'Goals'])
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {str(e)}")
        return pd.DataFrame(columns=['Division', 'Team', 'Player', 'Goals'])

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
    
    if df.empty:
        st.warning("⚠️ No data loaded. Please check your Excel file format.")
        st.info("Expected format: B Division (columns A,B,C) and A Division (columns F,G,H)")
        return
    
    # Show data loading success
    st.success(f"✅ Successfully loaded {len(df)} player records!")
    
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
    
    # Goal range filter
    min_goals = st.sidebar.slider("Minimum Goals:", 
                                  min_value=int(df['Goals'].min()), 
                                  max_value=int(df['Goals'].max()), 
                                  value=int(df['Goals'].min()))
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_division != 'All':
        filtered_df = filtered_df[filtered_df['Division'] == selected_division]
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['Team'] == selected_team]
    filtered_df = filtered_df[filtered_df['Goals'] >= min_goals]
    
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
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_scorer_goals = filtered_df.groupby('Player')['Goals'].sum().max() if not filtered_df.empty else 0
        st.metric("🏅 Highest Score", top_scorer_goals)
    
    with col2:
        teams_with_goals = filtered_df.groupby('Team')['Goals'].sum()
        top_team_goals = teams_with_goals.max() if not teams_with_goals.empty else 0
        st.metric("🏆 Top Team Goals", top_team_goals)
    
    with col3:
        multi_goal_players = len(filtered_df[filtered_df['Goals'] > 1]) if not filtered_df.empty else 0
        st.metric("⭐ Multi-Goal Players", multi_goal_players)
    
    with col4:
        total_records = len(filtered_df)
        st.metric("📋 Total Records", total_records)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏅 Top Goal Scorers")
        if not filtered_df.empty:
            top_scorers = filtered_df.groupby('Player')['Goals'].sum().sort_values(ascending=False).head(10)
            
            if not top_scorers.empty:
                # Create a DataFrame for better display
                top_scorers_df = pd.DataFrame({
                    'Player': top_scorers.index,
                    'Goals': top_scorers.values
                })
                
                # Display as bar chart using Streamlit's built-in chart
                st.bar_chart(top_scorers_df.set_index('Player'))
                
                # Show top 5 in a nice format
                st.write("**Top 5 Scorers:**")
                for idx, (player, goals) in enumerate(top_scorers.head().items(), 1):
                    st.write(f"{idx}. **{player}**: {goals} goals")
        else:
            st.info("No players match the selected filters.")
    
    with col2:
        st.subheader("🏟️ Goals by Team")
        if not filtered_df.empty:
            team_goals = filtered_df.groupby('Team')['Goals'].sum().sort_values(ascending=False)
            
            if not team_goals.empty:
                # Create a DataFrame for better display
                team_goals_df = pd.DataFrame({
                    'Team': team_goals.index,
                    'Goals': team_goals.values
                })
                
                # Display as bar chart
                st.bar_chart(team_goals_df.set_index('Team'))
                
                # Show breakdown with percentages
                team_goals_df['Percentage'] = (team_goals_df['Goals'] / team_goals_df['Goals'].sum() * 100).round(1)
                st.write("**Team Breakdown:**")
                for _, row in team_goals_df.head().iterrows():
                    st.write(f"• **{row['Team']}**: {row['Goals']} goals ({row['Percentage']}%)")
        else:
            st.info("No teams match the selected filters.")
    
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
            
            # Display as styled table
            st.dataframe(
                division_stats_display.style.format({
                    'Total Goals': '{:.0f}',
                    'Avg Goals': '{:.2f}',
                    'Total Records': '{:.0f}',
                    'Unique Players': '{:.0f}',
                    'Unique Teams': '{:.0f}'
                }),
                use_container_width=True
            )
            
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
            if not goal_counts.empty:
                most_common_goals = goal_counts.index[0]
                most_common_count = goal_counts.iloc[0]
                st.write(f"• Most common score: **{most_common_goals}** goals ({most_common_count} players)")
                st.write(f"• Highest score: **{filtered_df['Goals'].max()}** goals")
                st.write(f"• Lowest score: **{filtered_df['Goals'].min()}** goals")
                st.write(f"• Score range: **{filtered_df['Goals'].max() - filtered_df['Goals'].min()}** goals")
        else:
            st.info("No data available for the selected filters.")
    
    # Data Table
    st.subheader("📋 Detailed Player Data")
    
    # Search and sort controls
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search players or teams:", "")
    with col2:
        sort_by = st.selectbox("Sort by:", [
            'Goals (Descending)', 
            'Goals (Ascending)', 
            'Player Name', 
            'Team Name',
            'Division'
        ])
    
    # Apply search filter
    if search_term:
        mask = (
            filtered_df['Player'].str.contains(search_term, case=False, na=False) |
            filtered_df['Team'].str.contains(search_term, case=False, na=False)
        )
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df.copy()
    
    # Apply sorting
    if sort_by == 'Goals (Descending)':
        display_df = display_df.sort_values('Goals', ascending=False)
    elif sort_by == 'Goals (Ascending)':
        display_df = display_df.sort_values('Goals', ascending=True)
    elif sort_by == 'Player Name':
        display_df = display_df.sort_values('Player')
    elif sort_by == 'Team Name':
        display_df = display_df.sort_values('Team')
    else:  # Division
        display_df = display_df.sort_values(['Division', 'Goals'], ascending=[True, False])
    
    # Display the dataframe with enhanced formatting
    if not display_df.empty:
        # Add rank column for goals
        display_df_with_rank = display_df.copy()
        display_df_with_rank['Goal Rank'] = display_df_with_rank['Goals'].rank(method='dense', ascending=False).astype(int)
        
        # Reorder columns
        display_df_with_rank = display_df_with_rank[['Goal Rank', 'Division', 'Team', 'Player', 'Goals']]
        
        # Color coding for top performers
        def highlight_top_scorers(val):
            if isinstance(val, (int, float)) and val >= 2:
                return 'background-color: #d4edda; color: #155724; font-weight: bold'
            return ''
        
        styled_df = display_df_with_rank.style.applymap(highlight_top_scorers, subset=['Goals'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Download button and summary
        col1, col2 = st.columns([1, 1])
        with col1:
            csv = display_df_with_rank.to_csv(index=False)
            st.download_button(
                label="📥 Download data as CSV",
                data=csv,
                file_name="football_goals_data.csv",
                mime="text/csv"
            )
        
        with col2:
            st.info(f"Showing {len(display_df_with_rank)} of {len(df)} total records")
            
    else:
        st.warning("🔍 No data matches your search criteria. Try adjusting your filters.")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Statistics:**")
        if not df.empty:
            st.write(f"• **Total goals scored**: {df['Goals'].sum()}")
            st.write(f"• **Average goals per player**: {df['Goals'].mean():.2f}")
            st.write(f"• **Highest individual score**: {df['Goals'].max()}")
            st.write(f"• **Total unique players**: {df['Player'].nunique()}")
            st.write(f"• **Total unique teams**: {df['Team'].nunique()}")
            
            # Top performer
            top_player_data = df.groupby('Player')['Goals'].sum()
            top_player = top_player_data.idxmax()
            top_goals = top_player_data.max()
            st.write(f"• **Top scorer**: {top_player} ({top_goals} goals)")
            
            # Most goals in single game
            st.write(f"• **Highest single-game score**: {df['Goals'].max()}")
    
    with col2:
        st.write("**Division Breakdown:**")
        if not df.empty:
            for division in sorted(df['Division'].unique()):
                div_data = df[df['Division'] == division]
                st.write(f"**{division}:**")
                st.write(f"  - Total goals: **{div_data['Goals'].sum()}**")
                st.write(f"  - Players: **{div_data['Player'].nunique()}**")
                st.write(f"  - Teams: **{div_data['Team'].nunique()}**")
                if len(div_data) > 0:
                    st.write(f"  - Avg goals/player: **{div_data['Goals'].mean():.2f}**")
                    
                    # Top scorer in division
                    div_top_data = div_data.groupby('Player')['Goals'].sum()
                    div_top_player = div_top_data.idxmax()
                    div_top_goals = div_top_data.max()
                    st.write(f"  - Top scorer: **{div_top_player}** ({div_top_goals} goals)")

if __name__ == "__main__":
    main()
