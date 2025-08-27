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
def load_data(uploaded_file=None):
    """Load and process the CSV data"""
    try:
        # Try to read from uploaded file first, then from local file
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, header=None)
        else:
            try:
                df = pd.read_csv('goal_score_data.csv', header=None)
            except FileNotFoundError:
                # Fallback: try to read Excel if CSV doesn't exist
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
    
    except Exception as e:
        st.error(f"❌ Error reading data file: {str(e)}")
        return pd.DataFrame(columns=['Division', 'Team', 'Player', 'Goals'])

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Football Goal Scoring Dashboard</h1>
        <p>Track player performance across divisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.subheader("📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your goal scoring data file", 
        type=['csv', 'xlsx'], 
        help="Upload a CSV or Excel file with B Division (columns A,B,C) and A Division (columns F,G,H)"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df.empty:
        if uploaded_file is None:
            st.info("📤 Please upload your data file above, or ensure it's in the GitHub repository.")
            st.write("**Expected file format:**")
            st.write("- **CSV**: goal_score_data.csv")
            st.write("- **Excel**: Goal Score.xlsx")
            st.write("- Row 1: 'B Division' in column B, 'A Division' in column G")
            st.write("- Row 2: Headers (Team, Player Name, Nos. of Goals)")
            st.write("- Row 3+: Data")
            
            # Conversion instructions
            with st.expander("💡 How to Convert Excel to CSV"):
                st.write("If you're having issues with Excel files, convert to CSV:")
                st.code("""
# Run this locally with your Excel file:
import pandas as pd
df = pd.read_excel('Goal Score.xlsx', sheet_name='Sheet1', header=None)
df.to_csv('goal_score_data.csv', index=False, header=False)
                """)
                
            return
        else:
            st.error("❌ Could not process the uploaded file. Please check the format.")
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
    if not df.empty:
        min_goals = st.sidebar.slider("Minimum Goals:", 
                                      min_value=int(df['Goals'].min()), 
                                      max_value=int(df['Goals'].max()), 
                                      value=int(df['Goals'].min()))
    else:
        min_goals = 1
    
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
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏅 Top Goal Scorers")
        if not filtered_df.empty:
            top_scorers = filtered_df.groupby('Player')['Goals'].sum().sort_values(ascending=False).head(10)
            
            if not top_scorers.empty:
                top_scorers_df = pd.DataFrame({
                    'Player': top_scorers.index,
                    'Goals': top_scorers.values
                })
                
                st.bar_chart(top_scorers_df.set_index('Player'))
                
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
                team_goals_df = pd.DataFrame({
                    'Team': team_goals.index,
                    'Goals': team_goals.values
                })
                
                st.bar_chart(team_goals_df.set_index('Team'))
                
                team_goals_df['Percentage'] = (team_goals_df['Goals'] / team_goals_df['Goals'].sum() * 100).round(1)
                st.write("**Team Breakdown:**")
                for _, row in team_goals_df.head().iterrows():
                    st.write(f"• **{row['Team']}**: {row['Goals']} goals ({row['Percentage']}%)")
        else:
            st.info("No teams match the selected filters.")
    
    # Data Table
    st.subheader("📋 Detailed Player Data")
    
    # Search functionality
    search_term = st.text_input("🔍 Search players or teams:", "")
    
    # Apply search filter
    if search_term:
        mask = (
            filtered_df['Player'].str.contains(search_term, case=False, na=False) |
            filtered_df['Team'].str.contains(search_term, case=False, na=False)
        )
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df.copy()
    
    # Display the dataframe
    if not display_df.empty:
        display_df_sorted = display_df.sort_values('Goals', ascending=False)
        display_df_sorted['Rank'] = display_df_sorted['Goals'].rank(method='dense', ascending=False).astype(int)
        display_df_sorted = display_df_sorted[['Rank', 'Division', 'Team', 'Player', 'Goals']]
        
        st.dataframe(display_df_sorted.reset_index(drop=True), use_container_width=True)
        
        # Download button
        csv = display_df_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="football_goals_filtered.csv",
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
            st.write(f"• **Total goals scored**: {df['Goals'].sum()}")
            st.write(f"• **Average goals per player**: {df['Goals'].mean():.2f}")
            st.write(f"• **Highest individual score**: {df['Goals'].max()}")
            st.write(f"• **Total unique players**: {df['Player'].nunique()}")
            st.write(f"• **Total unique teams**: {df['Team'].nunique()}")
    
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

if __name__ == "__main__":
    main()
