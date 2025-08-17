import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
import requests
import json
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="ZABIN Premier League Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .auth-section {
        background-color: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 20px 0;
    }
    .setup-info {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive Authentication using Streamlit Secrets
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_google_credentials():
    """Get Google credentials from Streamlit secrets"""
    try:
        # Get credentials from Streamlit secrets
        creds_info = st.secrets["google_credentials"]
        
        # Create credentials object
        creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
        
        return creds
    except Exception as e:
        return None

def authenticate_google_drive():
    """Authenticate and return Google Drive service"""
    # First try to get credentials from secrets
    creds = get_google_credentials()
    
    if creds and creds.valid:
        return build('drive', 'v3', credentials=creds)
    
    # If no valid credentials from secrets, show setup instructions
    st.error("❌ Google Drive authentication not configured")
    show_deployment_setup()
    st.stop()

def show_deployment_setup():
    """Show setup instructions for deployment"""
    st.markdown("""
    <div class="setup-info">
        <h3>🚀 Deployment Setup Required</h3>
        <p>To deploy this dashboard, you need to configure Google Drive authentication in Streamlit Cloud:</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 Step-by-Step Setup Instructions", expanded=True):
        st.markdown("""
        ### 1. Google Cloud Console Setup
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing one
        3. Enable **Google Drive API**
        4. Create **OAuth 2.0 Client ID** credentials
        5. Add your Streamlit app URL to authorized redirect URIs
        6. Download the credentials JSON file
        
        ### 2. Get Authorized User Credentials
        Run this Python script locally to get your credentials:
        ```python
        from google_auth_oauthlib.flow import Flow
        
        # Use your downloaded credentials.json
        flow = Flow.from_client_secrets_file(
            'credentials.json',
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
        
        auth_url, _ = flow.authorization_url(prompt='consent')
        print(f'Go to: {auth_url}')
        
        code = input('Enter authorization code: ')
        flow.fetch_token(code=code)
        
        # Print credentials for Streamlit secrets
        creds = flow.credentials
        print("Add this to your Streamlit secrets:")
        print({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes
        })
        ```
        
        ### 3. Configure Streamlit Secrets
        In your Streamlit Cloud app settings, add these secrets:
        ```toml
        [google_credentials]
        token = "your_access_token"
        refresh_token = "your_refresh_token"
        token_uri = "https://oauth2.googleapis.com/token"
        client_id = "your_client_id"
        client_secret = "your_client_secret"
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        
        [app_settings]
        google_drive_folder_id = "15-KmaOB3ealj2ZuP_-xeK272lHMfQhDf"
        ```
        
        ### 4. Required Files for GitHub Repository
        Create these files in your repository:
        
        **requirements.txt:**
        ```
        streamlit==1.28.1
        pandas==2.0.3
        plotly==5.15.0
        openpyxl==3.1.2
        google-auth==2.23.0
        google-auth-oauthlib==1.0.0
        google-auth-httplib2==0.1.0
        google-api-python-client==2.97.0
        numpy==1.24.3
        ```
        
        **README.md:**
        ```markdown
        # ZABIN Premier League Dashboard
        
        A Streamlit dashboard for football tournament management.
        
        ## Deployment
        1. Fork this repository
        2. Connect to Streamlit Cloud
        3. Configure secrets as described above
        4. Deploy!
        ```
        """)

def find_folder_by_id(service, folder_id):
    """Get folder info by ID"""
    try:
        folder = service.files().get(fileId=folder_id).execute()
        return folder
    except Exception as e:
        st.error(f"❌ Cannot access folder with ID {folder_id}: {str(e)}")
        return None

def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder"""
    try:
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, modifiedTime)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"❌ Error listing files: {str(e)}")
        return []

def download_file_from_drive(service, file_id):
    """Download file from Google Drive and return as bytes"""
    try:
        request = service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file_io.seek(0)
        return file_io
    except Exception as e:
        st.error(f"❌ Error downloading file: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_google_drive():
    """Load tournament data from Google Drive folder"""
    try:
        # Get folder ID from secrets or use default
        folder_id = st.secrets.get("app_settings", {}).get("google_drive_folder_id", "15-KmaOB3ealj2ZuP_-xeK272lHMfQhDf")
        
        service = authenticate_google_drive()
        
        # Verify folder access
        folder = find_folder_by_id(service, folder_id)
        if not folder:
            return None, None, None
        
        st.success(f"📁 Connected to folder: {folder.get('name', 'Unknown')}")
        
        # List files in folder
        files = list_files_in_folder(service, folder_id)
        
        if not files:
            st.warning("📂 No files found in the folder")
            return None, None, None
        
        standings_df = None
        matches_df = None
        players_df = None
        
        # Process each file
        progress_bar = st.progress(0)
        for i, file in enumerate(files):
            file_name = file['name']
            file_id = file['id']
            
            progress_bar.progress((i + 1) / len(files))
            
            if file_name.endswith('.xlsx'):
                st.write(f"📄 Processing: {file_name}")
                
                file_content = download_file_from_drive(service, file_id)
                if file_content:
                    if 'point' in file_name.lower() or 'table' in file_name.lower():
                        standings_df = process_point_table(file_content, file_name)
                    elif 'match' in file_name.lower():
                        matches_df, players_df = process_match_report(file_content, file_name)
        
        progress_bar.empty()
        return standings_df, matches_df, players_df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None

def process_point_table(file_content, file_name):
    """Process the Point Table Excel file"""
    try:
        # Read all sheets to find the right one
        xl_file = pd.ExcelFile(file_content)
        
        # Try different sheet names that might contain standings
        possible_sheets = ['Point Table', 'Standings', 'Table', 'Points']
        target_sheet = None
        
        for sheet in possible_sheets:
            if sheet in xl_file.sheet_names:
                target_sheet = sheet
                break
        
        if not target_sheet:
            target_sheet = xl_file.sheet_names[0]
        
        # Read the sheet
        df = pd.read_excel(file_content, sheet_name=target_sheet)
        
        # Look for the table headers
        header_row_idx = None
        
        for idx, row in df.iterrows():
            row_str = str(row.values).upper()
            if 'TEAM' in row_str and ('PTS' in row_str or 'POINTS' in row_str):
                header_row_idx = idx
                break
        
        if header_row_idx is not None:
            # Re-read with proper header
            df = pd.read_excel(file_content, sheet_name=target_sheet, skiprows=header_row_idx)
            df.columns = df.columns.astype(str).str.strip()
            
            # Find team column
            team_col = None
            for col in df.columns:
                if 'TEAM' in str(col).upper():
                    team_col = col
                    break
            
            if team_col:
                # Clean data
                df = df.dropna(subset=[team_col])
                df = df[~df[team_col].astype(str).str.contains('TEAM', case=False, na=False)]
                
                # Rename columns
                column_mapping = {}
                for col in df.columns:
                    col_upper = str(col).upper().strip()
                    if 'TEAM' in col_upper:
                        column_mapping[col] = 'Team'
                    elif col_upper in ['M', 'MATCHES', 'PLAYED']:
                        column_mapping[col] = 'Matches'
                    elif col_upper in ['W', 'WINS', 'WON']:
                        column_mapping[col] = 'Wins'
                    elif col_upper in ['D', 'DRAWS', 'DRAW']:
                        column_mapping[col] = 'Draws'
                    elif col_upper in ['L', 'LOSSES', 'LOST']:
                        column_mapping[col] = 'Losses'
                    elif col_upper in ['PTS', 'POINTS']:
                        column_mapping[col] = 'Points'
                    elif col_upper in ['GF', 'GOALS FOR', 'FOR']:
                        column_mapping[col] = 'Goals_For'
                    elif col_upper in ['GA', 'GOALS AGAINST', 'AGAINST']:
                        column_mapping[col] = 'Goals_Against'
                    elif col_upper in ['GD', 'GOAL DIFFERENCE', 'DIFF']:
                        column_mapping[col] = 'Goal_Difference'
                    elif col_upper in ['Y', 'YELLOW']:
                        column_mapping[col] = 'Yellow_Cards'
                    elif col_upper in ['R', 'RED']:
                        column_mapping[col] = 'Red_Cards'
                
                df = df.rename(columns=column_mapping)
                
                # Add division if missing
                if 'Division' not in df.columns:
                    df['Division'] = 'A'
                
                # Convert to numeric
                numeric_columns = ['Matches', 'Wins', 'Draws', 'Losses', 'Points', 'Goals_For', 'Goals_Against', 'Goal_Difference', 'Yellow_Cards', 'Red_Cards']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Remove empty rows
                df = df[df['Team'].astype(str).str.strip() != '']
                
                st.success(f"✅ Processed {len(df)} teams from standings")
                return df
        
        return None
        
    except Exception as e:
        st.error(f"❌ Error processing standings: {str(e)}")
        return None

def process_match_report(file_content, file_name):
    """Process the Match Report Excel file"""
    try:
        xl_file = pd.ExcelFile(file_content)
        matches_data = []
        players_data = []
        
        for sheet_name in xl_file.sheet_names:
            if 'match' in sheet_name.lower():
                try:
                    df = pd.read_excel(file_content, sheet_name=sheet_name)
                    
                    # Extract match information
                    team1, team2 = None, None
                    score1, score2 = 0, 0
                    match_date = '2025-08-15'
                    venue = 'Stadium'
                    division = 'A'
                    
                    # Look for team names and match info
                    for idx, row in df.iterrows():
                        if idx > 15:
                            break
                        
                        row_str = ' '.join([str(cell) for cell in row.values if pd.notna(cell)])
                        row_upper = row_str.upper()
                        
                        # Team detection
                        team_names = ['BLUSTAR', 'YAS FC', 'STRIKERS', 'TITANS', 'WARRIORS', 'FALCONS', 'NEW CASTLE', 'ACC']
                        found_teams = [team for team in team_names if team in row_upper]
                        
                        if len(found_teams) >= 2:
                            team1, team2 = found_teams[0], found_teams[1]
                        elif len(found_teams) == 1:
                            if not team1:
                                team1 = found_teams[0]
                            elif not team2:
                                team2 = found_teams[0]
                        
                        # Venue detection
                        if 'VENUE' in row_upper or 'STADIUM' in row_upper:
                            venue_parts = row_str.split(':')
                            if len(venue_parts) > 1:
                                venue = venue_parts[-1].strip()
                        
                        # Division detection
                        if 'DIVISION' in row_upper and 'B' in row_upper:
                            division = 'B'
                        
                        # Date detection
                        for cell in row.values:
                            if pd.notna(cell) and isinstance(cell, (int, float)) and cell > 40000:
                                try:
                                    date_obj = pd.to_datetime('1900-01-01') + pd.Timedelta(days=cell-2)
                                    match_date = date_obj.strftime('%Y-%m-%d')
                                except:
                                    pass
                    
                    # Score detection (simplified)
                    # Look for goal tallies
                    for idx, row in df.iterrows():
                        for cell in row.values:
                            if pd.notna(cell) and isinstance(cell, (int, float)) and 0 < cell <= 10:
                                # This could be a goal count
                                col_idx = list(row.values).index(cell)
                                if col_idx < len(df.columns) / 2:
                                    score1 = max(score1, int(cell))
                                else:
                                    score2 = max(score2, int(cell))
                    
                    # Create match record
                    if team1 and team2:
                        match_info = {
                            'Date': match_date,
                            'Home_Team': team1,
                            'Away_Team': team2,
                            'Home_Score': min(score1, 10),
                            'Away_Score': min(score2, 10),
                            'Division': division,
                            'Venue': venue,
                            'Attendance': 150
                        }
                        matches_data.append(match_info)
                
                except Exception as e:
                    continue
        
        matches_df = pd.DataFrame(matches_data) if matches_data else None
        players_df = create_sample_players_df()  # Simplified for now
        
        if matches_df is not None:
            st.success(f"✅ Processed {len(matches_df)} matches")
        
        return matches_df, players_df
        
    except Exception as e:
        st.error(f"❌ Error processing matches: {str(e)}")
        return None, None

def create_sample_players_df():
    """Create sample players dataframe"""
    return pd.DataFrame({
        'Player': ['Jamseer Thelkkath', 'Muhamed Shahid', 'Alan Solaman'],
        'Team': ['YAS FC', 'YAS FC', 'BLUE STAR B'],
        'Goals': [2, 1, 0],
        'Assists': [0, 1, 2],
        'Yellow_Cards': [0, 1, 0],
        'Red_Cards': [0, 0, 0],
        'Minutes_Played': [90, 85, 90]
    })

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ ZABIN PREMIER LEAGUE DASHBOARD</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">In-House Football Tournament - Season 1</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard")
    
    # Check if properly configured
    if "google_credentials" not in st.secrets:
        st.sidebar.error("❌ Not configured for deployment")
        show_deployment_setup()
        return
    
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
    
    if standings_df is None and matches_df is None:
        st.warning("⚠️ No data available. Please check your Google Drive connection.")
        return
    
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        teams_count = len(standings_df) if standings_df is not None else 0
        st.metric("Total Teams", teams_count)
    with col2:
        matches_count = len(matches_df) if matches_df is not None else 0
        st.metric("Matches Played", matches_count)
    with col3:
        if standings_df is not None:
            total_goals = int(standings_df['Goals_For'].sum())
        else:
            total_goals = 0
        st.metric("Total Goals", total_goals)
    with col4:
        if matches_df is not None and len(matches_df) > 0:
            avg_goals = (matches_df['Home_Score'].sum() + matches_df['Away_Score'].sum()) / len(matches_df)
        else:
            avg_goals = 0
        st.metric("Avg Goals/Match", f"{avg_goals:.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Current Leaders")
        if standings_df is not None and not standings_df.empty:
            leaders = standings_df.nlargest(5, 'Points')[['Team', 'Points', 'Goal_Difference']]
            for _, row in leaders.iterrows():
                st.markdown(f"**{row['Team']}** - {row['Points']} pts (GD: {row['Goal_Difference']:+.0f})")
    
    with col2:
        st.subheader("⚽ Recent Matches")
        if matches_df is not None and not matches_df.empty:
            for _, match in matches_df.iterrows():
                st.markdown(f"**{match['Home_Team']}** {match['Home_Score']}-{match['Away_Score']} **{match['Away_Team']}**")
                st.caption(f"{match['Date']} • {match['Venue']}")

def display_standings(standings_df):
    """Display standings page"""
    st.header("📋 League Standings")
    
    if standings_df is not None and not standings_df.empty:
        # Sort standings
        standings_sorted = standings_df.sort_values(['Points', 'Goal_Difference'], ascending=[False, False])
        
        st.dataframe(
            standings_sorted[['Team', 'Matches', 'Wins', 'Draws', 'Losses', 'Points', 'Goals_For', 'Goals_Against', 'Goal_Difference']],
            use_container_width=True
        )
        
        # Points chart
        fig = px.bar(standings_sorted, x='Team', y='Points', title="Points by Team")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No standings data available")

def display_matches(matches_df):
    """Display matches page"""
    st.header("⚽ Match Results")
    
    if matches_df is not None and not matches_df.empty:
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
            
            st.caption(f"{match['Date']} • {match['Venue']} • Division {match['Division']}")
            st.markdown("---")
    else:
        st.warning("No match data available")

def display_players(players_df):
    """Display players page"""
    st.header("👤 Player Statistics")
    
    if players_df is not None and not players_df.empty:
        st.dataframe(players_df, use_container_width=True)
    else:
        st.warning("No player data available")

def display_analytics(standings_df, matches_df, players_df):
    """Display analytics page"""
    st.header("📈 Tournament Analytics")
    
    if standings_df is not None and not standings_df.empty:
        fig = px.scatter(standings_df, x='Goals_For', y='Points', 
                        hover_data=['Team'], title="Goals vs Points")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for analytics")

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>⚽ ZABIN Premier League Dashboard • Deployed via GitHub & Streamlit Cloud</p>
    <p>Real-time data from Google Drive</p>
</div>
""", unsafe_allow_html=True)