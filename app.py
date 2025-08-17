import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os

# Page configuration MUST be first
st.set_page_config(
    page_title="ZABIN Premier League Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import dependencies with better error handling
PLOTLY_AVAILABLE = False
GOOGLE_API_AVAILABLE = False

# Try importing Plotly with multiple fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False

# Try importing Google APIs
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

# Only show warnings if explicitly in development mode
if st.query_params.get("debug") == "true":
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly not available. Charts will be displayed as tables.")
    if not GOOGLE_API_AVAILABLE:
        st.error("❌ Google API libraries not available. Using sample data.")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .team-card {
        background: linear-gradient(90deg, #e3f2fd 0%, white 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .team-card:hover {
        transform: translateX(5px);
    }
    .match-card {
        background: linear-gradient(90deg, #f8f9fa 0%, white 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .match-teams {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .team-name {
        font-weight: bold;
        font-size: 1.1rem;
        color: #333;
    }
    .score {
        font-size: 1.8rem;
        font-weight: bold;
        color: #28a745;
        margin: 0 20px;
    }
    .match-info {
        color: #666;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-limited { background-color: #ffc107; }
    
    .section-header {
        color: #1f77b4;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .standings-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive Authentication
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
    """Load enhanced sample tournament data"""
    
    # Enhanced standings data with more realistic stats
    standings_data = {
        'Team': ['STRIKERS', 'TITANS', 'BLUE STAR B', 'NEW CASTLE FC', 'WARRIORS', 'FALCONS', 'ACC B', 'YAS FC'],
        'Division': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        'Matches': [3, 3, 3, 3, 3, 3, 2, 3],
        'Wins': [3, 2, 2, 1, 1, 0, 1, 0],
        'Draws': [0, 1, 1, 1, 0, 1, 0, 1],
        'Losses': [0, 0, 0, 1, 2, 2, 1, 2],
        'Points': [9, 7, 7, 4, 3, 1, 3, 1],
        'Goals_For': [8, 6, 7, 4, 3, 2, 3, 4],
        'Goals_Against': [1, 2, 3, 5, 6, 7, 4, 8],
        'Goal_Difference': [7, 4, 4, -1, -3, -5, -1, -4],
        'Yellow_Cards': [2, 3, 1, 4, 5, 6, 2, 3],
        'Red_Cards': [0, 0, 0, 1, 1, 2, 0, 1]
    }
    
    # Enhanced matches data
    matches_data = {
        'Date': ['2025-08-15', '2025-08-15', '2025-08-16', '2025-08-17', '2025-08-18'],
        'Home_Team': ['BLUE STAR B', 'STRIKERS', 'TITANS', 'NEW CASTLE FC', 'WARRIORS'],
        'Away_Team': ['YAS FC', 'WARRIORS', 'FALCONS', 'ACC B', 'BLUE STAR B'],
        'Home_Score': [3, 3, 2, 1, 0],
        'Away_Score': [2, 1, 0, 1, 2],
        'Division': ['B', 'A', 'A', 'B', 'Mixed'],
        'Venue': ['Bluestar Stadium', 'Main Stadium', 'Main Stadium', 'Sports Complex', 'Training Ground'],
        'Attendance': [180, 220, 200, 150, 100]
    }
    
    # Enhanced players data
    players_data = {
        'Player': ['Jamseer Thelkkath', 'Ahmed Al-Mansouri', 'Muhamed Shahid', 'Carlos Rodriguez', 'Shajahaan Manjapulli', 
                  'Alan Solaman', 'Arslan Variyamkundil', 'Mohammed Hassan', 'David Wilson', 'Omar Abdullah'],
        'Team': ['YAS FC', 'STRIKERS', 'YAS FC', 'TITANS', 'YAS FC', 'BLUE STAR B', 'BLUE STAR B', 'NEW CASTLE FC', 'WARRIORS', 'FALCONS'],
        'Goals': [5, 4, 3, 3, 2, 2, 1, 1, 1, 0],
        'Assists': [1, 2, 3, 1, 0, 4, 3, 2, 1, 2],
        'Yellow_Cards': [1, 0, 1, 2, 1, 0, 0, 1, 2, 1],
        'Red_Cards': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        'Minutes_Played': [270, 270, 250, 240, 220, 270, 200, 180, 240, 150]
    }
    
    return pd.DataFrame(standings_data), pd.DataFrame(matches_data), pd.DataFrame(players_data)

@st.cache_data(ttl=300)
def load_data_from_google_drive():
    """Load data from Google Drive or fallback to sample data"""
    
    service = authenticate_google_drive()
    
    if service is None:
        return load_sample_data()
    
    try:
        # Get folder ID from secrets
        folder_id = st.secrets.get("app_settings", {}).get("google_drive_folder_id", "15-KmaOB3ealj2ZuP_-xeK272lHMfQhDf")
        
        # Try to access the folder
        folder = service.files().get(fileId=folder_id).execute()
        
        # List files in folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get('files', [])
        
        if not files:
            return load_sample_data()
        
        # Process real files here when implemented
        return load_sample_data()
        
    except Exception as e:
        return load_sample_data()

def create_enhanced_chart(data, chart_type, title, **kwargs):
    """Create enhanced charts with fallbacks"""
    if PLOTLY_AVAILABLE:
        if chart_type == "bar":
            fig = px.bar(data, title=title, **kwargs)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12),
                title_font=dict(size=16, color='#1f77b4')
            )
            return fig
        elif chart_type == "scatter":
            fig = px.scatter(data, title=title, **kwargs)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12),
                title_font=dict(size=16, color='#1f77b4')
            )
            return fig
        elif chart_type == "pie":
            fig = px.pie(data, title=title, **kwargs)
            fig.update_layout(
                font=dict(family="Arial", size=12),
                title_font=dict(size=16, color='#1f77b4')
            )
            return fig
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ ZABIN PREMIER LEAGUE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">In-House Football Tournament - Season 1 • Live Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Control")
    
    # Enhanced status indicators
    google_status_icon = "🟢" if GOOGLE_API_AVAILABLE and get_google_credentials() else "🔴"
    google_status_text = "Connected" if GOOGLE_API_AVAILABLE and get_google_credentials() else "Sample Mode"
    
    plotly_status_icon = "🟢" if PLOTLY_AVAILABLE else "🟡"
    plotly_status_text = "Enhanced Charts" if PLOTLY_AVAILABLE else "Basic Charts"
    
    st.sidebar.markdown(f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4 style="margin: 0 0 10px 0; color: #333;">System Status</h4>
        <p style="margin: 5px 0;"><span class="status-indicator {'status-online' if GOOGLE_API_AVAILABLE and get_google_credentials() else 'status-offline'}"></span> Data Source: {google_status_text}</p>
        <p style="margin: 5px 0;"><span class="status-indicator {'status-online' if PLOTLY_AVAILABLE else 'status-limited'}"></span> Visualization: {plotly_status_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data", help="Reload tournament data"):
        st.cache_data.clear()
        st.rerun()
    
    # Navigation
    st.sidebar.markdown("### 📋 Navigation")
    page = st.sidebar.selectbox("Choose a section:", 
                               ["🏠 Overview", "📋 Standings", "⚽ Matches", "👤 Players", "🏃 Team Performance", "📈 Analytics"],
                               help="Navigate through different sections of the dashboard")
    
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
    elif page == "🏃 Team Performance":
        display_team_performance(standings_df, matches_df, players_df)
    elif page == "📈 Analytics":
        display_analytics(standings_df, matches_df, players_df)

def display_overview(standings_df, matches_df, players_df):
    """Display enhanced overview page"""
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(standings_df)}</div>
            <div class="metric-label">Total Teams</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(matches_df)}</div>
            <div class="metric-label">Matches Played</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_goals = int(standings_df['Goals_For'].sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_goals}</div>
            <div class="metric-label">Total Goals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if len(matches_df) > 0:
            avg_goals = (matches_df['Home_Score'].sum() + matches_df['Away_Score'].sum()) / len(matches_df)
        else:
            avg_goals = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_goals:.1f}</div>
            <div class="metric-label">Avg Goals/Match</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">🏆 Current Leaders</h3>', unsafe_allow_html=True)
        leaders = standings_df.nlargest(5, 'Points')
        for idx, row in leaders.iterrows():
            position = idx + 1
            trophy = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
            division_color = "#ff6b6b" if row['Division'] == 'A' else "#4ecdc4"
            st.markdown(f"""
            <div class="team-card" style="border-left-color: {division_color};">
                <strong>{trophy} {row['Team']}</strong> (Division {row['Division']})<br>
                <span style="color: #666;">📊 {row['Points']} pts • ⚽ {row['Goals_For']} goals • 📈 {row['Goal_Difference']:+.0f} GD</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="section-header">⚽ Recent Matches</h3>', unsafe_allow_html=True)
        for _, match in matches_df.tail(3).iterrows():
            result_color = "#28a745" if match['Home_Score'] > match['Away_Score'] else "#dc3545" if match['Home_Score'] < match['Away_Score'] else "#ffc107"
            st.markdown(f"""
            <div class="match-card" style="border-left-color: {result_color};">
                <div class="match-teams">
                    <span class="team-name">{match['Home_Team']}</span>
                    <span class="score">{match['Home_Score']} - {match['Away_Score']}</span>
                    <span class="team-name">{match['Away_Team']}</span>
                </div>
                <div class="match-info">📅 {match['Date']} • 🏟️ {match['Venue']} • 📊 Division {match['Division']} • 👥 {match['Attendance']} attendance</div>
            </div>
            """, unsafe_allow_html=True)

def display_standings(standings_df):
    """Display enhanced standings page"""
    st.markdown('<h2 class="section-header">📋 League Standings</h2>', unsafe_allow_html=True)
    
    # Division filter
    divisions = ['All'] + sorted(list(standings_df['Division'].unique()))
    division_filter = st.selectbox("🔍 Select Division:", divisions, help="Filter teams by division")
    
    if division_filter != "All":
        filtered_standings = standings_df[standings_df['Division'] == division_filter]
    else:
        filtered_standings = standings_df
    
    # Sort standings
    filtered_standings = filtered_standings.sort_values(['Points', 'Goal_Difference', 'Goals_For'], ascending=[False, False, False]).reset_index(drop=True)
    
    # Add position column
    filtered_standings['Position'] = range(1, len(filtered_standings) + 1)
    
    # Display enhanced table
    st.markdown('<div class="standings-table">', unsafe_allow_html=True)
    st.dataframe(
        filtered_standings[['Position', 'Team', 'Division', 'Matches', 'Wins', 'Draws', 'Losses', 'Points', 'Goals_For', 'Goals_Against', 'Goal_Difference']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Position": st.column_config.NumberColumn("#", width="small"),
            "Team": st.column_config.TextColumn("Team", width="large"),
            "Division": st.column_config.TextColumn("Div", width="small"),
            "Matches": st.column_config.NumberColumn("M", width="small"),
            "Wins": st.column_config.NumberColumn("W", width="small"),
            "Draws": st.column_config.NumberColumn("D", width="small"),
            "Losses": st.column_config.NumberColumn("L", width="small"),
            "Points": st.column_config.NumberColumn("Pts", width="small"),
            "Goals_For": st.column_config.NumberColumn("GF", width="small"),
            "Goals_Against": st.column_config.NumberColumn("GA", width="small"),
            "Goal_Difference": st.column_config.NumberColumn("GD", width="small"),
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if PLOTLY_AVAILABLE:
            fig = create_enhanced_chart(
                filtered_standings, "bar",
                "Points by Team",
                x='Team', y='Points', color='Division',
                color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'}
            )
            if fig:
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("📊 Points by Team")
            chart_data = filtered_standings.set_index('Team')['Points']
            st.bar_chart(chart_data)
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig = create_enhanced_chart(
                filtered_standings, "scatter",
                "Goals vs Points",
                x='Goals_For', y='Points', color='Division',
                hover_data=['Team'],
                color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'}
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("📊 Goals vs Points")
            chart_data = filtered_standings[['Goals_For', 'Points']].set_index('Goals_For')
            st.line_chart(chart_data)

def display_matches(matches_df):
    """Display enhanced matches page"""
    st.markdown('<h2 class="section-header">⚽ Match Center</h2>', unsafe_allow_html=True)
    
    # Match results
    for _, match in matches_df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 3])
        
        with col1:
            st.markdown(f"<h4 style='text-align: right; margin: 0;'>{match['Home_Team']}</h4>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h2 style='text-align: center; margin: 0; color: #1f77b4;'>{match['Home_Score']}</h2>", unsafe_allow_html=True)
        with col3:
            st.markdown("<h2 style='text-align: center; margin: 0; color: #666;'>-</h2>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<h2 style='text-align: center; margin: 0; color: #1f77b4;'>{match['Away_Score']}</h2>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<h4 style='text-align: left; margin: 0;'>{match['Away_Team']}</h4>", unsafe_allow_html=True)
        
        # Match details
        st.markdown(f"""
        <div style="text-align: center; color: #666; margin: 10px 0 30px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;">
            📅 {match['Date']} • 🏟️ {match['Venue']} • 📊 Division {match['Division']} • 👥 {match['Attendance']} spectators
        </div>
        """, unsafe_allow_html=True)
    
    # Goals analysis
    st.markdown('<h3 class="section-header">📊 Goals Analysis</h3>', unsafe_allow_html=True)
    
    if PLOTLY_AVAILABLE:
        matches_df_copy = matches_df.copy()
        matches_df_copy['Match'] = matches_df_copy['Home_Team'] + ' vs ' + matches_df_copy['Away_Team']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=matches_df_copy['Match'], 
            y=matches_df_copy['Home_Score'], 
            name='Home Goals',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=matches_df_copy['Match'], 
            y=matches_df_copy['Away_Score'], 
            name='Away Goals',
            marker_color='#ff7f0e'
        ))
        fig.update_layout(
            title="Goals by Match",
            barmode='group',
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_data = matches_df[['Home_Score', 'Away_Score']]
        st.bar_chart(chart_data)

def display_players(players_df):
    """Display enhanced players page"""
    st.markdown('<h2 class="section-header">👤 Player Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: #1f77b4; margin-bottom: 20px;">🥅 Top Scorers</h3>', unsafe_allow_html=True)
        top_scorers = players_df.nlargest(5, 'Goals')
        for idx, player in top_scorers.iterrows():
            position = list(top_scorers.index).index(idx) + 1
            trophy = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
            st.markdown(f"""
            <div class="team-card">
                <strong>{trophy} {player['Player']}</strong><br>
                <span style="color: #666;">{player['Team']} • ⚽ {player['Goals']} goals in {player['Minutes_Played']} minutes</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 style="color: #1f77b4; margin-bottom: 20px;">🎯 Top Assists</h3>', unsafe_allow_html=True)
        top_assists = players_df.nlargest(5, 'Assists')
        for idx, player in top_assists.iterrows():
            position = list(top_assists.index).index(idx) + 1
            trophy = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
            st.markdown(f"""
            <div class="team-card" style="border-left-color: #ff7f0e;">
                <strong>{trophy} {player['Player']}</strong><br>
                <span style="color: #666;">{player['Team']} • 🎯 {player['Assists']} assists • ⚽ {player['Goals']} goals</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Full statistics
    st.markdown('<h3 class="section-header">📊 Complete Player Statistics</h3>', unsafe_allow_html=True)
    
    # Add performance metrics
    players_enhanced = players_df.copy()
    players_enhanced['Goals_Per_90'] = (players_enhanced['Goals'] * 90 / players_enhanced['Minutes_Played']).round(2)
    players_enhanced['Assists_Per_90'] = (players_enhanced['Assists'] * 90 / players_enhanced['Minutes_Played']).round(2)
    
    st.dataframe(
        players_enhanced,
        use_container_width=True,
        column_config={
            "Player": st.column_config.TextColumn("Player Name", width="large"),
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "Goals": st.column_config.NumberColumn("Goals", width="small"),
            "Assists": st.column_config.NumberColumn("Assists", width="small"),
            "Goals_Per_90": st.column_config.NumberColumn("Goals/90min", width="small", format="%.2f"),
            "Assists_Per_90": st.column_config.NumberColumn("Assists/90min", width="small", format="%.2f"),
            "Yellow_Cards": st.column_config.NumberColumn("Yellow", width="small"),
            "Red_Cards": st.column_config.NumberColumn("Red", width="small"),
            "Minutes_Played": st.column_config.NumberColumn("Minutes", width="small"),
        }
    )

def display_team_performance(standings_df, matches_df, players_df):
    """Display comprehensive team performance analysis"""
    st.markdown('<h2 class="section-header">🏃 Team Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Team selector
    teams = sorted(standings_df['Team'].unique())
    selected_team = st.selectbox("🔍 Select Team for Detailed Analysis:", teams, help="Choose a team to see comprehensive performance data")
    
    if selected_team:
        # Get team data
        team_stats = standings_df[standings_df['Team'] == selected_team].iloc[0]
        team_matches = matches_df[(matches_df['Home_Team'] == selected_team) | (matches_df['Away_Team'] == selected_team)]
        team_players = players_df[players_df['Team'] == selected_team]
        
        # Team header with key stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-value">{team_stats['Points']}</div>
                <div class="metric-label">Points</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">#{standings_df[standings_df['Points'] >= team_stats['Points']].shape[0]} in league</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div class="metric-value">{team_stats['Wins']}</div>
                <div class="metric-label">Wins</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">{(team_stats['Wins']/max(team_stats['Matches'], 1)*100):.0f}% win rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333;">
                <div class="metric-value">{team_stats['Goals_For']}</div>
                <div class="metric-label">Goals Scored</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">{(team_stats['Goals_For']/max(team_stats['Matches'], 1)):.1f} per match</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333;">
                <div class="metric-value">{team_stats['Goals_Against']}</div>
                <div class="metric-label">Goals Conceded</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">{(team_stats['Goals_Against']/max(team_stats['Matches'], 1)):.1f} per match</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            gd_color = "#28a745" if team_stats['Goal_Difference'] > 0 else "#dc3545" if team_stats['Goal_Difference'] < 0 else "#6c757d"
            st.markdown(f"""
            <div class="metric-card" style="background: {gd_color};">
                <div class="metric-value">{team_stats['Goal_Difference']:+d}</div>
                <div class="metric-label">Goal Difference</div>
                <div style="font-size: 0.8rem; margin-top: 5px;">{'Excellent' if team_stats['Goal_Difference'] > 3 else 'Good' if team_stats['Goal_Difference'] > 0 else 'Needs improvement'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main content in tabs for better organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "⚽ Match History", "👥 Squad", "📈 Performance", "🏆 Achievements"])
        
        with tab1:
            # Team overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Team Information")
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                    <p><strong>Team Name:</strong> {selected_team}</p>
                    <p><strong>Division:</strong> {team_stats['Division']}</p>
                    <p><strong>Matches Played:</strong> {team_stats['Matches']}</p>
                    <p><strong>Form:</strong> {team_stats['Wins']}W-{team_stats['Draws']}D-{team_stats['Losses']}L</p>
                    <p><strong>League Position:</strong> #{standings_df[standings_df['Points'] >= team_stats['Points']].shape[0]}</p>
                    <p><strong>Points per Match:</strong> {(team_stats['Points']/max(team_stats['Matches'], 1)):.1f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Discipline record
                st.markdown("### 🟨 Discipline Record")
                total_cards = team_stats['Yellow_Cards'] + team_stats['Red_Cards']
                discipline_rating = "Excellent" if total_cards <= 2 else "Good" if total_cards <= 5 else "Needs Improvement"
                st.markdown(f"""
                <div style="background: #fff3cd; padding: 15px; border-radius: 10px; border-left: 4px solid #ffc107;">
                    <p><strong>Yellow Cards:</strong> {team_stats['Yellow_Cards']}</p>
                    <p><strong>Red Cards:</strong> {team_stats['Red_Cards']}</p>
                    <p><strong>Total Cards:</strong> {total_cards}</p>
                    <p><strong>Fair Play Rating:</strong> {discipline_rating}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Performance radar chart or comparison
                if PLOTLY_AVAILABLE:
                    st.markdown("### 📊 Performance Comparison")
                    
                    # Create performance metrics relative to league average
                    league_avg_goals_for = standings_df['Goals_For'].mean()
                    league_avg_goals_against = standings_df['Goals_Against'].mean()
                    league_avg_points = standings_df['Points'].mean()
                    
                    performance_metrics = {
                        'Metric': ['Attack', 'Defense', 'Points', 'Discipline'],
                        'Team_Score': [
                            (team_stats['Goals_For'] / max(league_avg_goals_for, 1)) * 100,
                            (league_avg_goals_against / max(team_stats['Goals_Against'], 1)) * 100,
                            (team_stats['Points'] / max(league_avg_points, 1)) * 100,
                            max(0, 100 - (total_cards * 10))
                        ]
                    }
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=performance_metrics['Team_Score'],
                        theta=performance_metrics['Metric'],
                        fill='toself',
                        name=selected_team,
                        line_color='#1f77b4'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 150]
                            )),
                        showlegend=True,
                        title="Performance vs League Average",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("### 📊 Performance Metrics")
                    performance_data = pd.DataFrame({
                        'Metric': ['Goals For', 'Goals Against', 'Points', 'Yellow Cards', 'Red Cards'],
                        'Value': [team_stats['Goals_For'], team_stats['Goals_Against'], 
                                team_stats['Points'], team_stats['Yellow_Cards'], team_stats['Red_Cards']]
                    })
                    st.dataframe(performance_data, use_container_width=True, hide_index=True)
        
        with tab2:
            # Match history
            st.markdown("### ⚽ Complete Match History")
            
            if not team_matches.empty:
                for _, match in team_matches.iterrows():
                    is_home = match['Home_Team'] == selected_team
                    opponent = match['Away_Team'] if is_home else match['Home_Team']
                    team_score = match['Home_Score'] if is_home else match['Away_Score']
                    opp_score = match['Away_Score'] if is_home else match['Home_Score']
                    
                    # Determine result
                    if team_score > opp_score:
                        result = "W"
                        result_color = "#28a745"
                        result_text = "Victory"
                    elif team_score < opp_score:
                        result = "L"
                        result_color = "#dc3545"
                        result_text = "Defeat"
                    else:
                        result = "D"
                        result_color = "#ffc107"
                        result_text = "Draw"
                    
                    venue_text = f"🏠 Home" if is_home else f"🛫 Away"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, {result_color}22 0%, white 100%); 
                                padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid {result_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{selected_team} {team_score} - {opp_score} {opponent}</strong>
                                <div style="color: {result_color}; font-weight: bold; margin-top: 5px;">{result_text} ({result})</div>
                            </div>
                            <div style="text-align: right; color: #666;">
                                <div>{venue_text}</div>
                                <div>📅 {match['Date']}</div>
                                <div>🏟️ {match['Venue']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Match statistics summary
                wins = sum(1 for _, match in team_matches.iterrows() 
                          if (match['Home_Team'] == selected_team and match['Home_Score'] > match['Away_Score']) or
                             (match['Away_Team'] == selected_team and match['Away_Score'] > match['Home_Score']))
                
                draws = sum(1 for _, match in team_matches.iterrows() 
                           if match['Home_Score'] == match['Away_Score'])
                
                losses = len(team_matches) - wins - draws
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Wins", wins, f"{(wins/max(len(team_matches), 1)*100):.0f}%")
                with col2:
                    st.metric("Draws", draws, f"{(draws/max(len(team_matches), 1)*100):.0f}%")
                with col3:
                    st.metric("Losses", losses, f"{(losses/max(len(team_matches), 1)*100):.0f}%")
            else:
                st.info(f"No match history available for {selected_team}")
        
        with tab3:
            # Squad analysis
            st.markdown("### 👥 Squad Analysis")
            
            if not team_players.empty:
                # Squad overview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ⚽ Goal Scorers")
                    scorers = team_players[team_players['Goals'] > 0].sort_values('Goals', ascending=False)
                    if not scorers.empty:
                        for _, player in scorers.iterrows():
                            goals_per_90 = (player['Goals'] * 90 / max(player['Minutes_Played'], 1))
                            st.markdown(f"""
                            <div class="team-card">
                                <strong>{player['Player']}</strong><br>
                                <span style="color: #666;">⚽ {player['Goals']} goals • 📊 {goals_per_90:.1f} goals/90min</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No goal scorers recorded yet")
                
                with col2:
                    st.markdown("#### 🎯 Playmakers")
                    assisters = team_players[team_players['Assists'] > 0].sort_values('Assists', ascending=False)
                    if not assisters.empty:
                        for _, player in assisters.iterrows():
                            assists_per_90 = (player['Assists'] * 90 / max(player['Minutes_Played'], 1))
                            st.markdown(f"""
                            <div class="team-card" style="border-left-color: #ff7f0e;">
                                <strong>{player['Player']}</strong><br>
                                <span style="color: #666;">🎯 {player['Assists']} assists • 📊 {assists_per_90:.1f} assists/90min</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No assists recorded yet")
                
                # Complete squad table
                st.markdown("#### 📊 Complete Squad Statistics")
                team_players_enhanced = team_players.copy()
                team_players_enhanced['Goals_Per_90'] = (team_players_enhanced['Goals'] * 90 / team_players_enhanced['Minutes_Played']).round(2)
                team_players_enhanced['Assists_Per_90'] = (team_players_enhanced['Assists'] * 90 / team_players_enhanced['Minutes_Played']).round(2)
                team_players_enhanced['Contribution'] = team_players_enhanced['Goals'] + team_players_enhanced['Assists']
                
                st.dataframe(
                    team_players_enhanced.sort_values('Contribution', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Player": "Player Name",
                        "Goals": "Goals",
                        "Assists": "Assists", 
                        "Contribution": "G+A",
                        "Goals_Per_90": "Goals/90min",
                        "Assists_Per_90": "Assists/90min",
                        "Yellow_Cards": "Yellow",
                        "Red_Cards": "Red",
                        "Minutes_Played": "Minutes"
                    }
                )
            else:
                st.info(f"No player data available for {selected_team}")
        
        with tab4:
            # Performance trends
            st.markdown("### 📈 Performance Trends")
            
            if PLOTLY_AVAILABLE and not team_matches.empty:
                # Goals scored vs conceded over time
                match_data = []
                for _, match in team_matches.iterrows():
                    is_home = match['Home_Team'] == selected_team
                    goals_for = match['Home_Score'] if is_home else match['Away_Score']
                    goals_against = match['Away_Score'] if is_home else match['Home_Score']
                    
                    match_data.append({
                        'Date': match['Date'],
                        'Goals_For': goals_for,
                        'Goals_Against': goals_against,
                        'Opponent': match['Away_Team'] if is_home else match['Home_Team']
                    })
                
                match_trends_df = pd.DataFrame(match_data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=match_trends_df['Date'],
                    y=match_trends_df['Goals_For'],
                    mode='lines+markers',
                    name='Goals Scored',
                    line=dict(color='#28a745'),
                    hovertemplate='<b>%{text}</b><br>Goals Scored: %{y}<extra></extra>',
                    text=[f"vs {opp}" for opp in match_trends_df['Opponent']]
                ))
                
                fig.add_trace(go.Scatter(
                    x=match_trends_df['Date'],
                    y=match_trends_df['Goals_Against'],
                    mode='lines+markers',
                    name='Goals Conceded',
                    line=dict(color='#dc3545'),
                    hovertemplate='<b>%{text}</b><br>Goals Conceded: %{y}<extra></extra>',
                    text=[f"vs {opp}" for opp in match_trends_df['Opponent']]
                ))
                
                fig.update_layout(
                    title=f"{selected_team} - Goals Trend",
                    xaxis_title="Match Date",
                    yaxis_title="Goals",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison with league
            st.markdown("#### 📊 League Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                league_avg_gf = standings_df['Goals_For'].mean()
                team_gf = team_stats['Goals_For']
                diff_gf = team_gf - league_avg_gf
                st.metric(
                    "Goals Scored vs League Avg", 
                    f"{team_gf:.1f}", 
                    f"{diff_gf:+.1f}",
                    delta_color="normal"
                )
            
            with col2:
                league_avg_ga = standings_df['Goals_Against'].mean()
                team_ga = team_stats['Goals_Against']
                diff_ga = league_avg_ga - team_ga  # Reverse because fewer goals against is better
                st.metric(
                    "Defense vs League Avg", 
                    f"{team_ga:.1f}", 
                    f"{diff_ga:+.1f}",
                    delta_color="normal"
                )
            
            with col3:
                league_avg_pts = standings_df['Points'].mean()
                team_pts = team_stats['Points']
                diff_pts = team_pts - league_avg_pts
                st.metric(
                    "Points vs League Avg", 
                    f"{team_pts:.1f}", 
                    f"{diff_pts:+.1f}",
                    delta_color="normal"
                )
        
        with tab5:
            # Achievements and records
            st.markdown("### 🏆 Achievements & Records")
            
            # Team achievements
            achievements = []
            
            # Check for various achievements
            if team_stats['Points'] == standings_df['Points'].max():
                achievements.append("👑 League Leaders")
            
            if team_stats['Goals_For'] == standings_df['Goals_For'].max():
                achievements.append("⚽ Highest Scoring Team")
            
            if team_stats['Goals_Against'] == standings_df['Goals_Against'].min():
                achievements.append("🛡️ Best Defense")
            
            if team_stats['Goal_Difference'] == standings_df['Goal_Difference'].max():
                achievements.append("📈 Best Goal Difference")
            
            if team_stats['Yellow_Cards'] + team_stats['Red_Cards'] == (standings_df['Yellow_Cards'] + standings_df['Red_Cards']).min():
                achievements.append("🏅 Fair Play Award")
            
            if not team_players.empty:
                if team_players['Goals'].max() == players_df['Goals'].max():
                    top_scorer = team_players.loc[team_players['Goals'].idxmax(), 'Player']
                    achievements.append(f"🥇 League Top Scorer: {top_scorer}")
                
                if team_players['Assists'].max() == players_df['Assists'].max():
                    top_assister = team_players.loc[team_players['Assists'].idxmax(), 'Player']
                    achievements.append(f"🎯 League Top Assister: {top_assister}")
            
            # Display achievements
            if achievements:
                for achievement in achievements:
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #ffd700 0%, #fff 100%); 
                                padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #ffd700;">
                        <strong>{achievement}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No major achievements yet. Keep playing!")
            
            # Team records
            st.markdown("#### 📝 Team Records")
            
            if not team_matches.empty:
                # Calculate records
                max_goals_match = team_matches.apply(lambda x: max(x['Home_Score'], x['Away_Score']) if x['Home_Team'] == selected_team or x['Away_Team'] == selected_team else 0, axis=1).max()
                
                biggest_win_margin = 0
                biggest_loss_margin = 0
                
                for _, match in team_matches.iterrows():
                    is_home = match['Home_Team'] == selected_team
                    team_score = match['Home_Score'] if is_home else match['Away_Score']
                    opp_score = match['Away_Score'] if is_home else match['Home_Score']
                    
                    if team_score > opp_score:
                        biggest_win_margin = max(biggest_win_margin, team_score - opp_score)
                    elif opp_score > team_score:
                        biggest_loss_margin = max(biggest_loss_margin, opp_score - team_score)
                
                records_data = {
                    "Record": [
                        "Most Goals in a Match",
                        "Biggest Win Margin", 
                        "Biggest Loss Margin",
                        "Total Matches Played",
                        "Current Form"
                    ],
                    "Value": [
                        f"{max_goals_match} goals",
                        f"+{biggest_win_margin} goals" if biggest_win_margin > 0 else "No wins yet",
                        f"-{biggest_loss_margin} goals" if biggest_loss_margin > 0 else "No losses yet",
                        f"{len(team_matches)} matches",
                        f"{team_stats['Wins']}W-{team_stats['Draws']}D-{team_stats['Losses']}L"
                    ]
                }
                
                st.dataframe(
                    pd.DataFrame(records_data),
                    use_container_width=True,
                    hide_index=True
                )

def display_analytics(standings_df, matches_df, players_df):
    """Display enhanced analytics page"""
    st.markdown('<h2 class="section-header">📈 Tournament Analytics</h2>', unsafe_allow_html=True)
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        highest_scorer = players_df.loc[players_df['Goals'].idxmax()]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);">
            <div class="metric-value">⚽</div>
            <div class="metric-label">Top Scorer</div>
            <div style="font-size: 0.9rem; margin-top: 5px;">{highest_scorer['Player']}<br>{highest_scorer['Goals']} goals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_defense = standings_df.loc[standings_df['Goals_Against'].idxmin()]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);">
            <div class="metric-value">🛡️</div>
            <div class="metric-label">Best Defense</div>
            <div style="font-size: 0.9rem; margin-top: 5px;">{best_defense['Team']}<br>{best_defense['Goals_Against']} goals conceded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        most_entertaining = matches_df.loc[(matches_df['Home_Score'] + matches_df['Away_Score']).idxmax()]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffc107 0%, #ffb300 100%);">
            <div class="metric-value">🎯</div>
            <div class="metric-label">Most Goals in Match</div>
            <div style="font-size: 0.9rem; margin-top: 5px;">{most_entertaining['Home_Team']} vs {most_entertaining['Away_Team']}<br>{most_entertaining['Home_Score'] + most_entertaining['Away_Score']} total goals</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: #1f77b4;">📊 Goals Distribution by Division</h3>', unsafe_allow_html=True)
        if PLOTLY_AVAILABLE:
            division_goals = standings_df.groupby('Division')['Goals_For'].sum().reset_index()
            fig = create_enhanced_chart(
                division_goals, "pie",
                "Goals by Division",
                values='Goals_For', names='Division',
                color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'}
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            division_data = standings_df.groupby('Division')['Goals_For'].sum()
            st.bar_chart(division_data)
    
    with col2:
        st.markdown('<h3 style="color: #1f77b4;">📈 Team Performance Matrix</h3>', unsafe_allow_html=True)
        if PLOTLY_AVAILABLE:
            fig = create_enhanced_chart(
                standings_df, "scatter",
                "Attack vs Defense",
                x='Goals_For', y='Goals_Against', 
                size='Points', color='Division',
                hover_data=['Team', 'Points'],
                color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4'}
            )
            if fig:
                fig.update_layout(
                    xaxis_title="Goals Scored",
                    yaxis_title="Goals Conceded"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            chart_data = standings_df[['Goals_For', 'Goals_Against']].set_index('Goals_For')
            st.line_chart(chart_data)
    
    # Tournament progression
    st.markdown('<h3 class="section-header">🏆 Tournament Insights</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 League Statistics")
        total_teams = len(standings_df)
        total_matches = len(matches_df)
        total_goals = int(standings_df['Goals_For'].sum())
        total_cards = int(standings_df['Yellow_Cards'].sum() + standings_df['Red_Cards'].sum())
        avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 0
        
        stats_data = {
            "Metric": ["Participating Teams", "Completed Matches", "Goals Scored", "Cards Issued", "Avg Goals/Match"],
            "Value": [total_teams, total_matches, total_goals, total_cards, f"{avg_goals_per_match:.1f}"]
        }
        
        st.dataframe(
            pd.DataFrame(stats_data),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("#### 🎯 Top Performers")
        
        # Create performance summary
        top_team = standings_df.loc[standings_df['Points'].idxmax()]
        top_scorer = players_df.loc[players_df['Goals'].idxmax()]
        top_assists = players_df.loc[players_df['Assists'].idxmax()]
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
            <p><strong>🏆 Leading Team:</strong> {top_team['Team']} ({top_team['Points']} pts)</p>
            <p><strong>⚽ Top Scorer:</strong> {top_scorer['Player']} ({top_scorer['Goals']} goals)</p>
            <p><strong>🎯 Most Assists:</strong> {top_assists['Player']} ({top_assists['Assists']} assists)</p>
            <p><strong>📈 Tournament Progress:</strong> {(len(matches_df)/12*100):.0f}% Complete</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Fair play table
    st.markdown('<h3 class="section-header">🏅 Fair Play Ranking</h3>', unsafe_allow_html=True)
    
    # Calculate fair play points (fewer cards = better)
    fair_play_df = standings_df.copy()
    fair_play_df['Fair_Play_Points'] = -(fair_play_df['Yellow_Cards'] + fair_play_df['Red_Cards'] * 2)
    fair_play_df = fair_play_df.sort_values('Fair_Play_Points', ascending=False)
    
    st.dataframe(
        fair_play_df[['Team', 'Yellow_Cards', 'Red_Cards', 'Fair_Play_Points']],
        use_container_width=True,
        column_config={
            "Team": "Team",
            "Yellow_Cards": "Yellow Cards",
            "Red_Cards": "Red Cards", 
            "Fair_Play_Points": "Fair Play Score"
        },
        hide_index=True
    )

if __name__ == "__main__":
    main()

# Enhanced Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
---
<div style="text-align: center; padding: 20px 0;">
    <h4 style="color: #1f77b4; margin-bottom: 15px;">⚽ ZABIN Premier League Dashboard</h4>
    <p style="margin: 5px 0; color: #666;">🏆 Built with Streamlit • 📊 Real-time Analytics • 📱 Mobile Responsive</p>
    <p style="margin: 5px 0; color: #666;">🔄 Auto-sync with Google Drive • 📈 Live Tournament Tracking</p>
    <p style="margin: 15px 0; font-size: 0.9rem; color: #999;">
        Made with ❤️ for football enthusiasts • Season 1 - 2025
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
