# app.py — Enhanced Visual Football Goals Dashboard
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import numpy as np

# Page config with custom styling
st.set_page_config(
    page_title="⚽ Football Goals Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced visual appeal
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    /* Main styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom metrics styling */
    [data-testid="metric-container"] {
        background-color: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# XLSX fallback parser (keeping your original robust implementation)
# -----------------------------
def _parse_xlsx_without_openpyxl(file_bytes: bytes) -> pd.DataFrame:
    """Read the first worksheet from an .xlsx file using only the standard library."""
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

    with zipfile.ZipFile(BytesIO(file_bytes)) as z:
        # shared strings
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                root = ET.parse(f).getroot()
                for si in root.findall('.//main:si', ns):
                    text = ''.join(t.text or '' for t in si.findall('.//main:t', ns))
                    shared_strings.append(text)

        # first worksheet (sheet1.xml)
        with z.open('xl/worksheets/sheet1.xml') as f:
            root = ET.parse(f).getroot()
            sheet_data = root.find('main:sheetData', ns)
            rows = []
            for row in sheet_data.findall('main:row', ns):
                row_data = {}
                for c in row.findall('main:c', ns):
                    ref = c.attrib.get('r', 'A1')
                    col_letters = ''.join(ch for ch in ref if ch.isalpha())
                    t = c.attrib.get('t')
                    v = c.find('main:v', ns)
                    val = v.text if v is not None else None
                    if t == 's' and val is not None:
                        idx = int(val)
                        if 0 <= idx < len(shared_strings):
                            val = shared_strings[idx]
                    row_data[col_letters] = val
                rows.append(row_data)

        if not rows:
            return pd.DataFrame()

        all_cols = sorted({c for r in rows for c in r.keys()},
                          key=lambda s: [ord(ch) for ch in s])
        data = [[r.get(c) for c in all_cols] for r in rows]
        return pd.DataFrame(data, columns=all_cols)

def _read_excel_safely(file_like_or_path) -> pd.DataFrame:
    """Try pandas.read_excel first, fallback to custom parser if needed."""
    file_bytes = None
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        with open(p, 'rb') as fh:
            file_bytes = fh.read()
    else:
        file_bytes = file_like_or_path.read()
        file_like_or_path.seek(0)

    try:
        return pd.read_excel(file_like_or_path)
    except ImportError:
        return _parse_xlsx_without_openpyxl(file_bytes)

@st.cache_data
def load_and_prepare_data(file_like_or_path) -> pd.DataFrame:
    """Load and prepare data with caching for better performance."""
    raw_df = _read_excel_safely(file_like_or_path)

    if "B Division" in raw_df.columns and "A Division" in raw_df.columns:
        b_df = raw_df[["B Division", "Unnamed: 1", "Unnamed: 2"]].copy()
        b_df.columns = ["Team", "Player", "Goals"]
        b_df["Division"] = "B Division"

        a_df = raw_df[["A Division", "Unnamed: 6", "Unnamed: 7"]].copy()
        a_df.columns = ["Team", "Player", "Goals"]
        a_df["Division"] = "A Division"
    else:
        if len(raw_df) < 3:
            return pd.DataFrame(columns=["Division", "Team", "Player", "Goals"])

        header = raw_df.iloc[1].tolist()
        data_rows = raw_df.iloc[2:].reset_index(drop=True)
        data_rows.columns = header

        b_df = data_rows.iloc[:, :3].copy()
        b_df.columns = ["Team", "Player", "Goals"]
        b_df["Division"] = "B Division"

        a_df = data_rows.iloc[:, 3:].copy()
        a_df.columns = ["Team", "Player", "Goals"]
        a_df["Division"] = "A Division"

    combined = pd.concat([b_df, a_df], ignore_index=True)
    combined = combined.dropna(subset=["Team", "Player", "Goals"])
    combined["Goals"] = pd.to_numeric(combined["Goals"], errors="coerce")
    combined = combined.dropna(subset=["Goals"])
    combined["Goals"] = combined["Goals"].astype(int)
    return combined

# -----------------------------
# Enhanced Visual Components
# -----------------------------

def create_hero_metrics(df: pd.DataFrame):
    """Create visually appealing hero metrics."""
    total_goals = int(df["Goals"].sum()) if not df.empty else 0
    num_players = df["Player"].nunique() if not df.empty else 0
    num_teams = df["Team"].nunique() if not df.empty else 0
    avg_goals = round(df["Goals"].mean(), 1) if not df.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0; font-size: 2.5rem;">⚽</h2>
            <h1 style="margin: 0.5rem 0; color: #333;">{total_goals}</h1>
            <p style="color: #666; margin: 0;">Total Goals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #764ba2; margin: 0; font-size: 2.5rem;">👥</h2>
            <h1 style="margin: 0.5rem 0; color: #333;">{num_players}</h1>
            <p style="color: #666; margin: 0;">Players</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0; font-size: 2.5rem;">🏆</h2>
            <h1 style="margin: 0.5rem 0; color: #333;">{num_teams}</h1>
            <p style="color: #666; margin: 0;">Teams</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #764ba2; margin: 0; font-size: 2.5rem;">📊</h2>
            <h1 style="margin: 0.5rem 0; color: #333;">{avg_goals}</h1>
            <p style="color: #666; margin: 0;">Avg Goals/Player</p>
        </div>
        """, unsafe_allow_html=True)

def create_modern_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_sequence=None):
    """Create a modern, animated bar chart using Plotly."""
    if color_sequence is None:
        color_sequence = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    fig = px.bar(
        df, 
        x=y_col, 
        y=x_col, 
        orientation='h',
        title=title,
        color=y_col,
        color_continuous_scale='Viridis',
        text=y_col
    )
    
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker=dict(
            line=dict(width=0.5, color='DarkSlateGrey'),
            cornerradius="30%"
        )
    )
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=12),
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=False)
    )
    
    return fig

def create_donut_chart(df: pd.DataFrame):
    """Create an interactive donut chart for divisions."""
    division_goals = df.groupby("Division")["Goals"].sum().reset_index()
    
    fig = go.Figure(data=[go.Pie(
        labels=division_goals['Division'], 
        values=division_goals['Goals'],
        hole=.3,
        marker_colors=['#667eea', '#764ba2'],
        textinfo='label+percent+value',
        textfont_size=14,
        hovertemplate="<b>%{label}</b><br>Goals: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title="Goals Distribution by Division",
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=12),
        height=400
    )
    
    return fig

def create_scatter_performance(df: pd.DataFrame):
    """Create a scatter plot showing team performance."""
    team_stats = df.groupby(['Team', 'Division']).agg({
        'Goals': 'sum',
        'Player': 'nunique'
    }).reset_index()
    team_stats['Goals_per_Player'] = team_stats['Goals'] / team_stats['Player']
    
    fig = px.scatter(
        team_stats,
        x='Player',
        y='Goals',
        color='Division',
        size='Goals_per_Player',
        hover_data=['Team'],
        title="Team Performance: Goals vs Players",
        color_discrete_sequence=['#667eea', '#764ba2']
    )
    
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=12),
        height=500
    )
    
    return fig

def create_goals_distribution(df: pd.DataFrame):
    """Create a histogram showing goal distribution."""
    fig = px.histogram(
        df,
        x='Goals',
        color='Division',
        title='Distribution of Individual Goal Counts',
        nbins=max(df['Goals'].max(), 10),
        color_discrete_sequence=['#667eea', '#764ba2'],
        opacity=0.8
    )
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=12),
        height=400,
        bargap=0.1
    )
    
    return fig

def create_top_performers_gauge(df: pd.DataFrame):
    """Create gauge charts for top performers."""
    if df.empty:
        return go.Figure()
    
    top_scorer = df.loc[df['Goals'].idxmax()]
    max_possible = df['Goals'].max() * 1.2  # Set gauge max to 120% of top score
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = top_scorer['Goals'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"🏆 {top_scorer['Player']}<br><sub>Top Scorer</sub>"},
        delta = {'reference': df['Goals'].mean()},
        gauge = {
            'axis': {'range': [None, max_possible]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_possible*0.5], 'color': "lightgray"},
                {'range': [max_possible*0.5, max_possible*0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_possible*0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=12)
    )
    
    return fig

def _find_local_excel() -> Path | None:
    """Find local Excel file."""
    candidates = [
        Path("Goal Score.xlsx"),
        Path(__file__).parent / "Goal Score.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def main() -> None:
    # Hero Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin: 0; font-size: 3rem;">⚽ Football Goals Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Explore goal-scoring statistics with beautiful visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        uploaded = st.file_uploader(
            "Upload Goal Score.xlsx", 
            type=["xlsx"],
            help="Upload your Excel file containing goal statistics"
        )
        
        # Load data
        if uploaded is not None:
            df = load_and_prepare_data(uploaded)
            st.success("✅ File uploaded successfully!")
        else:
            local_path = _find_local_excel()
            if local_path is None:
                st.error("⚠️ No Excel file found. Please upload Goal Score.xlsx")
                st.stop()
            df = load_and_prepare_data(local_path)
            st.info("📋 Using local file")

        # Enhanced filters
        st.markdown("### 🎛️ Filters")
        divisions = ["All"] + sorted(df["Division"].unique().tolist())
        chosen_div = st.selectbox(
            "Select Division", 
            divisions,
            help="Filter data by division"
        )
        
        # Additional filters
        if not df.empty:
            min_goals = int(df['Goals'].min())
            max_goals = int(df['Goals'].max())
            goal_range = st.slider(
                "Goal Range",
                min_goals, max_goals,
                (min_goals, max_goals),
                help="Filter players by goal count"
            )
            
            # Team filter
            teams = ["All"] + sorted(df["Team"].unique().tolist())
            selected_team = st.selectbox("Select Team", teams, help="Filter by specific team")

    # Apply filters
    filtered = df.copy()
    if chosen_div != "All":
        filtered = filtered[filtered["Division"] == chosen_div]
    if not df.empty:
        filtered = filtered[(filtered["Goals"] >= goal_range[0]) & (filtered["Goals"] <= goal_range[1])]
    if selected_team != "All":
        filtered = filtered[filtered["Team"] == selected_team]

    # Hero Metrics
    create_hero_metrics(filtered)

    # Main content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Teams", "⭐ Players", "📈 Analytics"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Goal Distribution by Division")
            if not filtered.empty:
                donut_fig = create_donut_chart(filtered)
                st.plotly_chart(donut_fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")
        
        with col2:
            st.subheader("🏅 Top Performer")
            if not filtered.empty:
                gauge_fig = create_top_performers_gauge(filtered)
                st.plotly_chart(gauge_fig, use_container_width=True)

        # Goals distribution histogram
        if not filtered.empty:
            st.subheader("📊 Goals Distribution")
            hist_fig = create_goals_distribution(filtered)
            st.plotly_chart(hist_fig, use_container_width=True)

    with tab2:
        st.subheader("🏆 Team Performance")
        
        if not filtered.empty:
            # Team goals bar chart
            team_goals = filtered.groupby("Team")["Goals"].sum().reset_index().sort_values("Goals", ascending=False)
            team_fig = create_modern_bar_chart(team_goals, "Team", "Goals", "Goals by Team")
            st.plotly_chart(team_fig, use_container_width=True)
            
            # Team performance scatter
            st.subheader("📈 Team Efficiency Analysis")
            scatter_fig = create_scatter_performance(filtered)
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("No team data available for the selected filters.")

    with tab3:
        st.subheader("⭐ Player Performance")
        
        if not filtered.empty:
            # Top players selector
            max_players = len(filtered['Player'].unique())
            top_n = st.slider("Number of top players to display", 1, min(max_players, 20), min(10, max_players))
            
            player_goals = filtered.groupby("Player")["Goals"].sum().reset_index().sort_values("Goals", ascending=False).head(top_n)
            player_fig = create_modern_bar_chart(player_goals, "Player", "Goals", f"Top {top_n} Players by Goals")
            st.plotly_chart(player_fig, use_container_width=True)
            
            # Player details table
            st.subheader("📋 Player Details")
            detailed_table = filtered[['Player', 'Team', 'Division', 'Goals']].sort_values('Goals', ascending=False)
            st.dataframe(
                detailed_table,
                use_container_width=True,
                height=400,
                column_config={
                    "Goals": st.column_config.ProgressColumn(
                        "Goals",
                        help="Number of goals scored",
                        min_value=0,
                        max_value=int(filtered['Goals'].max()) if not filtered.empty else 1
                    )
                }
            )
        else:
            st.info("No player data available for the selected filters.")

    with tab4:
        st.subheader("📈 Advanced Analytics")
        
        if not filtered.empty:
            # Key insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Key Insights")
                total_goals = filtered['Goals'].sum()
                avg_goals = filtered['Goals'].mean()
                top_team = filtered.groupby('Team')['Goals'].sum().idxmax()
                top_team_goals = filtered.groupby('Team')['Goals'].sum().max()
                
                insights = [
                    f"🎯 **{total_goals}** total goals scored across all teams",
                    f"📊 **{avg_goals:.1f}** average goals per player",
                    f"🏆 **{top_team}** leads with **{top_team_goals}** goals",
                    f"👥 **{filtered['Player'].nunique()}** players in competition"
                ]
                
                for insight in insights:
                    st.markdown(f"- {insight}")
            
            with col2:
                st.markdown("#### 📊 Statistical Summary")
                stats_df = pd.DataFrame({
                    'Metric': ['Min Goals', 'Max Goals', 'Median Goals', 'Std Deviation'],
                    'Value': [
                        filtered['Goals'].min(),
                        filtered['Goals'].max(),
                        filtered['Goals'].median(),
                        round(filtered['Goals'].std(), 2)
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for analytics.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Built with ❤️ using Streamlit • Enhanced Visual Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
