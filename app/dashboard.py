"""
Streamlit dashboard for DeepMatch AI predictions.
Provides interactive interface for match outcome predictions.
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="DeepMatch AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ DeepMatch AI - Premier League Match Prediction")
st.markdown("AI-powered prediction system using deep learning with team embeddings")

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"
API_PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
API_TEAMS_ENDPOINT = f"{API_BASE_URL}/teams"

# CSS Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-result {
        font-size: 24px;
        font-weight: bold;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def get_known_teams() -> Optional[list]:
    """Fetch list of known teams from API."""
    try:
        response = requests.get(API_TEAMS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return sorted(response.json()["teams"])
    except Exception as e:
        st.error(f"Cannot connect to API. Make sure the API is running: {e}")
        return None


def make_prediction(
    home_team: str,
    away_team: str,
    B365H: float,
    B365D: float,
    B365A: float,
    home_avg_goals_for: float,
    home_avg_goals_against: float,
    home_avg_points: float,
    away_avg_goals_for: float,
    away_avg_goals_against: float,
    away_avg_points: float,
    form_points_diff: float,
    form_goals_for_diff: float,
    form_goals_against_diff: float,
) -> Optional[Dict]:
    """Call prediction API."""
    try:
        payload = {
            "home_team": home_team,
            "away_team": away_team,
            "B365H": B365H,
            "B365D": B365D,
            "B365A": B365A,
            "home_avg_goals_for": home_avg_goals_for,
            "home_avg_goals_against": home_avg_goals_against,
            "home_avg_points": home_avg_points,
            "away_avg_goals_for": away_avg_goals_for,
            "away_avg_goals_against": away_avg_goals_against,
            "away_avg_points": away_avg_points,
            "form_points_diff": form_points_diff,
            "form_goals_for_diff": form_goals_for_diff,
            "form_goals_against_diff": form_goals_against_diff,
        }
        
        response = requests.post(
            API_PREDICT_ENDPOINT,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json()['detail']}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please start the API server first: `uvicorn app.api:app --reload`")
        return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.markdown("### ⚙️ Configuration")
api_status = st.sidebar.empty()

# Check API status
try:
    response = requests.get(f"{API_BASE_URL}/", timeout=2)
    if response.status_code == 200:
        api_status.success("✅ API Connected")
    else:
        api_status.error("❌ API Unavailable")
except:
    api_status.error("❌ API Offline")

# ============================================================================
# Main Interface
# ============================================================================

# Get known teams
teams = get_known_teams()

if teams is None:
    st.error("Cannot fetch teams. Please ensure the API is running.")
    st.stop()

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏠 Home Team")
    home_team = st.selectbox(
        "Select home team",
        teams,
        key="home_team"
    )

with col2:
    st.markdown("### ⭐ Away Team")
    away_team = st.selectbox(
        "Select away team",
        teams,
        key="away_team",
        index=1 if len(teams) > 1 else 0
    )

# Separator
st.divider()

# Betting Odds Section
st.markdown("### 📊 Betting Odds (Bet365)")
odds_col1, odds_col2, odds_col3 = st.columns(3)

with odds_col1:
    B365H = st.number_input(
        "Home Win Odds",
        min_value=1.0,
        max_value=100.0,
        value=2.5,
        step=0.1
    )

with odds_col2:
    B365D = st.number_input(
        "Draw Odds",
        min_value=1.0,
        max_value=100.0,
        value=3.5,
        step=0.1
    )

with odds_col3:
    B365A = st.number_input(
        "Away Win Odds",
        min_value=1.0,
        max_value=100.0,
        value=3.0,
        step=0.1
    )

# Separator
st.divider()

# Home Team Form Section
st.markdown("### 🏠 Home Team Form Metrics")
home_col1, home_col2, home_col3 = st.columns(3)

with home_col1:
    home_avg_goals_for = st.number_input(
        "Home Avg Goals For",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

with home_col2:
    home_avg_goals_against = st.number_input(
        "Home Avg Goals Against",
        min_value=0.0,
        max_value=5.0,
        value=1.2,
        step=0.1
    )

with home_col3:
    home_avg_points = st.number_input(
        "Home Avg Points/Match",
        min_value=0.0,
        max_value=3.0,
        value=1.8,
        step=0.1
    )

# Away Team Form Section
st.markdown("### ⭐ Away Team Form Metrics")
away_col1, away_col2, away_col3 = st.columns(3)

with away_col1:
    away_avg_goals_for = st.number_input(
        "Away Avg Goals For",
        min_value=0.0,
        max_value=5.0,
        value=1.2,
        step=0.1
    )

with away_col2:
    away_avg_goals_against = st.number_input(
        "Away Avg Goals Against",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

with away_col3:
    away_avg_points = st.number_input(
        "Away Avg Points/Match",
        min_value=0.0,
        max_value=3.0,
        value=1.4,
        step=0.1
    )

# Separator
st.divider()

# Form Difference Metrics
st.markdown("### 📈 Form Differences")
diff_col1, diff_col2, diff_col3 = st.columns(3)

with diff_col1:
    form_points_diff = st.number_input(
        "Points Difference",
        min_value=-3.0,
        max_value=3.0,
        value=0.4,
        step=0.1
    )

with diff_col2:
    form_goals_for_diff = st.number_input(
        "Goals For Difference",
        min_value=-3.0,
        max_value=3.0,
        value=0.3,
        step=0.1
    )

with diff_col3:
    form_goals_against_diff = st.number_input(
        "Goals Against Difference",
        min_value=-3.0,
        max_value=3.0,
        value=-0.3,
        step=0.1
    )

# Separator
st.divider()

# Predict Button
if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
    with st.spinner("Making prediction..."):
        result = make_prediction(
            home_team=home_team,
            away_team=away_team,
            B365H=B365H,
            B365D=B365D,
            B365A=B365A,
            home_avg_goals_for=home_avg_goals_for,
            home_avg_goals_against=home_avg_goals_against,
            home_avg_points=home_avg_points,
            away_avg_goals_for=away_avg_goals_for,
            away_avg_goals_against=away_avg_goals_against,
            away_avg_points=away_avg_points,
            form_points_diff=form_points_diff,
            form_goals_for_diff=form_goals_for_diff,
            form_goals_against_diff=form_goals_against_diff,
        )
    
    if result:
        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Matchup Header
        st.markdown(f"### {result['home_team']} vs {result['away_team']}")
        
        # Prediction Output
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        
        # Color coding for prediction
        if predicted_class == "H":
            color = "🟢"
            prediction_text = f"Home Win ({result['home_team']})"
        elif predicted_class == "D":
            color = "🟡"
            prediction_text = "Draw"
        else:  # "A"
            color = "🔴"
            prediction_text = f"Away Win ({result['away_team']})"
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"<div class='prediction-result'>"
                f"{color} <b>{prediction_text}</b><br>"
                f"Confidence: {confidence:.1%}"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Probabilities
        st.markdown("### 📊 Outcome Probabilities")
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.metric(
                f"🏠 {result['home_team']} Win",
                f"{result['home_win_probability']:.1%}",
                delta=None
            )
        
        with prob_col2:
            st.metric(
                "🤝 Draw",
                f"{result['draw_probability']:.1%}",
                delta=None
            )
        
        with prob_col3:
            st.metric(
                f"⭐ {result['away_team']} Win",
                f"{result['away_win_probability']:.1%}",
                delta=None
            )
        
        # Bar Chart
        st.markdown("### 📈 Probability Distribution")
        
        chart_data = pd.DataFrame({
            "Outcome": [
                f"{result['home_team']} (H)",
                "Draw (D)",
                f"{result['away_team']} (A)"
            ],
            "Probability": [
                result["home_win_probability"],
                result["draw_probability"],
                result["away_win_probability"]
            ]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=chart_data["Outcome"],
                y=chart_data["Probability"],
                marker_color=["#2E7D32", "#F57C00", "#C62828"],
                text=[f"{p:.1%}" for p in chart_data["Probability"]],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>"
            )
        ])
        
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Outcome",
            yaxis_title="Probability",
            yaxis=dict(tickformat=".0%"),
            height=400,
            showlegend=False,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw Response (for debugging)
        with st.expander("📋 Raw API Response"):
            st.json(result)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
### About DeepMatch AI
Deep learning model trained on Premier League historical data (1992-2026).
- **Model**: Neural Network with Team Embeddings
- **Features**: 13 numerical features + team embeddings
- **Training Data**: 4,096 matches across 34 seasons
- **Accuracy**: 48.66% on test set
""")
