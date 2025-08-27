import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
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
        
        fig_top_scorers = px.bar(
            x=top_scorers.values,
            y=top_scorers.index,
            orientation='h',
            title="Top 10 Goal Scorers",
            labels={'x': 'Goals', 'y': 'Player'},
            color=top_scorers.values,
            color_continuous_scale='viridis'
        )
        fig_top_scorers.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_top_scorers, use_container_width=True)
    
    with col2:
        st.subheader("🏟️ Goals by Team")
        team_goals = filtered_df.groupby('Team')['Goals'].sum().sort_values(ascending=False)
        
        fig_team_goals = px.pie(
            values=team_goals.values,
            names=team_goals.index,
            title="Goals Distribution by Team"
        )
        fig_team_goals.update_traces(textposition='inside', textinfo='percent+label')
        fig_team_goals.update_layout(height=400)
        st.plotly_chart(fig_team_goals, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🆚 Division Comparison")
        division_stats = df.groupby('Division').agg({
            'Goals': ['sum', 'mean', 'count'],
            'Player': 'nunique',
            'Team': 'nunique'
        }).round(2)
        
        division_stats.columns = ['Total Goals', 'Avg Goals', 'Total Records', 'Unique Players', 'Unique Teams']
        
        fig_division = go.Figure()
        
        fig_division.add_trace(go.Bar(
            name='Total Goals',
            x=division_stats.index,
            y=division_stats['Total Goals'],
            yaxis='y',
            offsetgroup=1
        ))
        
        fig_division.add_trace(go.Scatter(
            name='Avg Goals',
            x=division_stats.index,
            y=division_stats['Avg Goals'],
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig_division.update_layout(
            title='Division Performance Overview',
            xaxis=dict(title='Division'),
            yaxis=dict(title='Total Goals', side='left'),
            yaxis2=dict(title='Average Goals', side='right', overlaying='y'),
            height=400
        )
        
        st.plotly_chart(fig_division, use_container_width=True)
    
    with col2:
        st.subheader("📈 Goal Distribution")
        
        fig_hist = px.histogram(
            filtered_df, 
            x='Goals',
            nbins=max(1, filtered_df['Goals'].max() - filtered_df['Goals'].min() + 1),
            title="Distribution of Goals Scored",
            labels={'Goals': 'Number of Goals', 'count': 'Number of Players'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
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
    
    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Statistics:**")
        st.write(f"• Total goals scored: {df['Goals'].sum()}")
        st.write(f"• Average goals per player: {df['Goals'].mean():.2f}")
        st.write(f"• Highest individual score: {df['Goals'].max()}")
        st.write(f"• Total unique players: {df['Player'].nunique()}")
        st.write(f"• Total unique teams: {df['Team'].nunique()}")
    
    with col2:
        st.write("**Division Breakdown:**")
        for division in df['Division'].unique():
            div_data = df[df['Division'] == division]
            st.write(f"**{division}:**")
            st.write(f"  - Total goals: {div_data['Goals'].sum()}")
            st.write(f"  - Players: {div_data['Player'].nunique()}")
            st.write(f"  - Teams: {div_data['Team'].nunique()}")
            st.write(f"  - Avg goals/player: {div_data['Goals'].mean():.2f}")

if __name__ == "__main__":
    main()
