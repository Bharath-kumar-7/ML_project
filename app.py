import streamlit as st
import numpy as np
import pandas as pd
import base64
import io
from PIL import Image

from model import IPLPredictor
from gemini_api import GeminiCricketAPI

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("🏏 IPL Match Win Predictor")

# ------------------ BACKGROUND ------------------
def set_bg(image_file):
    img = Image.open(image_file)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

try:
    set_bg("images/default.avif")
except:
    pass

# ------------------ INIT ------------------
predictor = IPLPredictor()
predictor.train()

api = GeminiCricketAPI()

teams = ["CSK", "MI", "RCB", "KKR", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]

# ------------------ CACHE ALL TEAM STATS ------------------
@st.cache_data(ttl=3600)
def preload_stats(api, teams):
    return {team: api.get_team_stats(team) for team in teams}

stats_cache = preload_stats(api, teams)

# ------------------ INPUTS ------------------
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

toss = st.radio("Toss Winner", [team1, team2])

venue = st.selectbox(
    "Venue",
    [
        "Wankhede Stadium",
        "MA Chidambaram Stadium",
        "M Chinnaswamy Stadium",
        "Eden Gardens",
        "Arun Jaitley Stadium",
        "Narendra Modi Stadium",
    ]
)

weather = st.selectbox("Weather", ["Clear", "Cloudy", "Hot", "Humid"])

# ------------------ PREDICTION ------------------
if st.button("Predict Winner"):
    with st.spinner("⚡ Crunching match analytics..."):

        team1_stats = stats_cache[team1]
        team2_stats = stats_cache[team2]

        features = np.array([[
            1 if toss == team1 else 0,
            team1_stats["batting_avg"],
            team1_stats["bowling_avg"],
            team1_stats["recent_form"],
            team2_stats["batting_avg"],
            team2_stats["bowling_avg"],
            team2_stats["recent_form"]
        ]])

        pred, prob = predictor.predict(features)

        winner = team1 if pred == 1 else team2
        confidence = max(prob) * 100

    # ------------------ OUTPUT ------------------
    st.success(f"🏆 Predicted Winner: **{winner}**")
    st.info(f"📊 Winning Probability: **{confidence:.2f}%**")

    df = pd.DataFrame({
        "Metric": ["Batting Avg", "Bowling Avg", "Recent Form"],
        team1: list(team1_stats.values()),
        team2: list(team2_stats.values())
    })

    st.subheader("📈 Team Comparison")
    st.table(df)
