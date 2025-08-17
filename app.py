import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os

# Try to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Charts will be displayed as tables.")

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    st.error("❌ Google API libraries not available. Using sample data.")

# Page configuration
st.set_page_config(
    page_title="ZABIN Premier League Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .team-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Google Drive Authentication (with fallback)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_google_credentials():
    """Get Google credentials from Streamlit secrets"""
    if not GOOGLE_API_AVAILABLE:
        return None
    
    try:
        creds_info = st.secrets["google_credentials"]
        creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
        return creds
    except Exception:
        return None

def authenticate_google_drive():
    """Authenticate and return Google Drive service"""
    if not GOOGLE_API_AVAILABLE:
        return None
    
    creds = get_google_credentials()
    if creds and creds.valid:
        return build('drive', 'v3', credentials=creds)
    return None

@st.cache_data(ttl=300)
def load_sample_data():
    """Load sample tournament data for demo purposes"""
    
    # Sample standings data
    standings_data = {
        'Team': ['STRIKERS', 'TITANS', 'BLUE STAR B', 'NEW CASTLE FC', 'WARRIORS', 'FALCONS', 'ACC B', 'YAS FC'],
        'Division': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        'Matches': [1, 1, 1, 1, 1, 1, 0, 1],
        'Wins': [1, 1, 1, 0, 0, 0, 0, 0],
        'Draws': [0, 0, 0, 0, 0, 0, 0, 0],
        'Losses': [0, 0, 0, 1, 1, 1, 0, 1],
        'Points': [3, 3, 3, 0, 0, 0, 0, 0],
        'Goals_For': [1, 1, 3, 2, 0, 0, 0, 2],
        'Goals_Against': [0, 0, 2, 3, 1, 1, 0, 3],
        'Goal_Difference': [1, 1, 1, -1, -1, -1, 0, -1],
        'Yellow_Cards': [0, 1, 0, 0, 0, 1, 0, 0],
        'Red_Cards': [0, 0, 0, 0, 0, 0, 0, 0]
    }
    
    # Sample matches data
    matches_data = {
        'Date': ['2025-08-15', '2025-08-15', '2025-08-16'],
        'Home_Team': ['BLUE STAR B', 'STRIKERS', 'TITANS'],
        'Away_Team': ['YAS FC', 'WARRIORS', 'FALCONS'],
        'Home_Score': [3, 1, 1],
        'Away_Score': [2, 0, 0],
        'Division': ['B', 'A', 'A'],
        'Venue': ['Bluestar Stadium', 'Main Stadium', 'Main Stadium'],
        'Attendance': [150, 200, 180]
    }
    
    # Sample players data
    players_data = {
        'Player': ['Jamseer Thelkkath', 'Muhamed Shahid', 'Shajahaan Manjapulli', 'Alan Solaman', 'Arslan Variyamkundil'],
        'Team': ['YAS FC', 'YAS FC', 'YAS FC', 'BLUE STAR B', 'BLUE STAR B'],
        'Goals': [2, 1, 1, 0, 0],
        'Assists': [0, 1, 0, 2, 1],
        'Yellow_Cards': [0, 0, 1, 0, 0],
        'Red_Cards': [0, 0, 0, 0, 0],
        'Minutes_Played': [90, 90, 85, 90, 75]
    }
    
    return pd.DataFrame(standings_data), pd.DataFrame(matches_data), pd.DataFrame(players_data)

@st.cache_data(ttl=300)
def load_data_from_google_drive():
    """Load data from Google Drive or fallback to sample data"""
    
    service = authenticate_google_drive()
    
    if service is None:
        st.info("📄 Using sample data for demo. Configure Google Drive for real-time data.")
        return load_sample_data()
    
    try:
        # Get folder ID from secrets
        folder_id = st.secrets.get("app_settings", {}).get("google_drive_folder_id", "15-KmaOB3ealj2ZuP_-xeK272lHMfQhDf")
        
        # Try to access the folder
        folder = service.files().get(fileId=folder_id).execute()
        st.success(f"📁 Connected to: {folder.get('name', 'Tournament Data')}")
        
        # List files in folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get('files', [])
        
        if not files:
            st.warning("📂 No files found in Google Drive folder. Using sample data.")
            return load_sample_data()
        
        st.info(f"📄 Found {len(files)} files in Google Drive")
        
        # For now, return sample data but indicate real connection
        # In production, you would process the actual Excel files here
        return load_sample_data()
        
    except Exception as e:
        st.error(f"❌ Error accessing Google Drive: {str(e)}")
        st.info("📄 Falling back to sample data")
        return load_sample_data()

def create_chart_fallback(data, chart_type, title):
    """Create a simple table when Plotly is not available"""
    st.subheader(title)
    if chart_type == "bar":
        st.bar_chart(data)
    elif chart_type == "line":
        st.line_chart(data)
    else:
        st.dataframe(data)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ ZABIN PREMIER LEAGUE DASHBOARD</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">In-House Football Tournament - Season 1</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard")
    
    # Status indicators
    google_status = "🟢 Connected" if GOOGLE_API_AVAILABLE and get_google_credentials() else "🔴 Offline"
    plotly_status = "🟢 Available" if PLOTLY_AVAILABLE else "🟡 Limited"
    
    st.sidebar.markdown(f"""
    **System Status:**
    - Google Drive: {google_status}
    - Charts: {plotly_status}
    """)
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Navigation
    page = st.sidebar.selectbox("Choose a section:", 
                               ["🏠 Overview", "📋 Standings", "⚽ Matches", "👤 Players", "📈 Analytics"])
    
    # Load data
    with st.spinner("📥 Loading tournament data..."):
        standings_df, matches_df, players_df = load_data_from_google_drive()
    
    # Display pages
    if page == "🏠 Overview":
        display_overview(standings_df, matches_df, players_df)
    elif page == "📋 Standings":
        display_standings(standings_df)
    elif page == "⚽ Matches":
        display_matches(matches_df)
    elif page == "👤 Players":
        display_players(players_df)
    elif page == "📈 Analytics":
        display_analytics(standings_df, matches_df, players_df)

def display_overview(standings_df, matches_df, players_df):
    """Display overview page"""
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Teams", len(standings_df))
    with col2:
        st.metric("Matches Played", len(matches_df))
    with col3:
        total_goals = int(standings_df['Goals_For'].sum())
        st.metric("Total Goals", total_goals)
    with col4:
        if len(matches_df) > 0:
            avg_goals = (matches_df['Home_Score'].sum() + matches_df['Away_Score'].sum()) / len(matches_df)
        else:
            avg_goals = 0
        st.metric("Avg Goals/Match", f"{avg_goals:.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Current Leaders")
        leaders = standings_df.nlargest(5, 'Points')
        for _, row in leaders.iterrows():
            division_color = "#ff6b6b" if row['Division'] == 'A' else "#4ecdc4"
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {division_color}22 0%, transparent 100%); 
                        padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {division_color};">
                <strong>{row['Team']}</strong> (Div {row['Division']}) - {row['Points']} pts (GD: {row['Goal_Difference']:+.0f})
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚽ Recent Matches")
        for _, match in matches_df.iterrows():
            result_color = "#28a745" if match['Home_Score'] > match['Away_Score'] else "#dc3545" if match['Home_Score'] < match['Away_Score'] else "#ffc107"
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid {result_color};">
                <div style="font-weight: bold; margin-bottom: 5px;">{match['Home_Team']} vs {match['Away_Team']}</div>
                <div style="font-size: 1.5em; color: {result_color}; margin: 5px 0;">{match['Home_Score']} - {match['Away_Score']}</div>
                <div style="color: #666; font-size: 0.9em;">{match['Date']} • {match['Venue']} • Division {match['Division']}</div>
            </div>
            """, unsafe_allow_html=True)

def display_standings(standings_df):
    """Display standings page"""
    st.header("📋 League Standings")
    
    # Division filter
    divisions = ['All'] + list(standings_df['Division'].unique())
    division_filter = st.selectbox("Select Division:", divisions)
    
    if division_filter != "All":
        filtered_standings = standings_df[standings_df['Division'] == division_filter]
    else:
        filtered_standings = standings_df
    
    # Sort standings
    filtered_standings = filtered_standings.sort_values(['Points', 'Goal_Difference'], ascending=[False, False])
    
    # Display table
    st.dataframe(
        filtered_standings[['Team', 'Division', 'Matches', 'Wins', 'Draws', 'Losses', 'Points', 'Goals_For', 'Goals_Against', 'Goal_Difference']],
        use_container_width=True
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if PLOTLY_AVAILABLE:
            fig = px.bar(filtered_standings, x='Team', y='Points', color='Division',
                         title="Points by Team", color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Points by Team")
            chart_data = filtered_standings.set_index('Team')['Points']
            st.bar_chart(chart_data)
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig = px.scatter(filtered_standings, x='Goals_For', y='Points', 
                           hover_data=['Team'], title="Goals vs Points",
                           color='Division', color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Goals vs Points")
            chart_data = filtered_standings[['Goals_For', 'Points']]
            st.scatter_chart(chart_data)

def display_matches(matches_df):
    """Display matches page"""
    st.header("⚽ Match Results")
    
    for _, match in matches_df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 3])
        
        with col1:
            st.markdown(f"**{match['Home_Team']}**")
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>{match['Home_Score']}</h3>", unsafe_allow_html=True)
        with col3:
            st.markdown("<h3 style='text-align: center;'>-</h3>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<h3 style='text-align: center;'>{match['Away_Score']}</h3>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"**{match['Away_Team']}**")
        
        st.caption(f"{match['Date']} • {match['Venue']} • Division {match['Division']} • Attendance: {match['Attendance']}")
        st.markdown("---")
    
    # Goals analysis
    if PLOTLY_AVAILABLE:
        matches_df['Total_Goals'] = matches_df['Home_Score'] + matches_df['Away_Score']
        matches_df['Match'] = matches_df['Home_Team'] + ' vs ' + matches_df['Away_Team']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=matches_df['Match'], y=matches_df['Home_Score'], name='Home Goals'))
        fig.add_trace(go.Bar(x=matches_df['Match'], y=matches_df['Away_Score'], name='Away Goals'))
        fig.update_layout(title="Goals by Match", barmode='stack', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📊 Goals by Match")
        chart_data = matches_df[['Home_Score', 'Away_Score']]
        st.bar_chart(chart_data)

def display_players(players_df):
    """Display players page"""
    st.header("👤 Player Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥅 Top Scorers")
        top_scorers = players_df.nlargest(5, 'Goals')
        for _, player in top_scorers.iterrows():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #1f77b4;">
                <strong>{player['Player']}</strong> ({player['Team']}) - {player['Goals']} goals
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 Top Assists")
        top_assists = players_df.nlargest(5, 'Assists')
        for _, player in top_assists.iterrows():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #ff7f0e;">
                <strong>{player['Player']}</strong> ({player['Team']}) - {player['Assists']} assists
            </div>
            """, unsafe_allow_html=True)
    
    # Full statistics
    st.subheader("📊 All Player Statistics")
    st.dataframe(players_df, use_container_width=True)

def display_analytics(standings_df, matches_df, players_df):
    """Display analytics page"""
    st.header("📈 Tournament Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if PLOTLY_AVAILABLE:
            division_goals = standings_df.groupby('Division')['Goals_For'].sum().reset_index()
            fig = px.pie(division_goals, values='Goals_For', names='Division', 
                         title="Goals by Division", color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Goals by Division")
            division_data = standings_df.groupby('Division')['Goals_For'].sum()
            st.bar_chart(division_data)
    
    with col2:
        st.subheader("📊 Tournament Statistics")
        total_teams = len(standings_df)
        total_matches = len(matches_df)
        total_goals = standings_df['Goals_For'].sum()
        avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 0
        
        st.metric("Participating Teams", total_teams)
        st.metric("Completed Matches", total_matches)
        st.metric("Goals Scored", int(total_goals))
        st.metric("Average Goals per Match", f"{avg_goals_per_match:.1f}")

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>⚽ ZABIN Premier League Dashboard • Built with Streamlit</p>
    <p>Real-time data sync with Google Drive</p>
</div>
""", unsafe_allow_html=True)
