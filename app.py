import streamlit as st
import numpy as np
import pandas as pd
import base64
import io
from PIL import Image

from model import IPLPredictor
from gemini_api import GeminiCricketAPI

# ---------------- PAGE ----------------
st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("🏏 IPL Match Win Predictor")

# ---------------- BACKGROUND ----------------
def set_bg(image_file):
    img = Image.open(image_file)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
            linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
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

# ---------------- INIT ----------------
predictor = IPLPredictor()
predictor.train()

api = GeminiCricketAPI()

teams = ["CSK", "MI", "RCB", "KKR", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]

# ---------------- INPUTS ----------------
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
        "Narendra Modi Stadium",
    ]
)
weather = st.selectbox("Weather", ["Clear", "Cloudy", "Hot", "Humid"])

# ---------------- PREDICT ----------------
if st.button("Predict Winner"):
    with st.spinner("⚡ Fetching team stats..."):

        # ✅ SAFE: cached inside gemini_api.py
        team1_stats = api.get_team_stats(team1)
        team2_stats = api.get_team_stats(team2)

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

    st.success(f"🏆 Predicted Winner: **{winner}**")
    st.info(f"📊 Winning Probability: **{confidence:.2f}%**")

    df = pd.DataFrame({
        "Metric": ["Batting Avg", "Bowling Avg", "Recent Form"],
        team1: list(team1_stats.values()),
        team2: list(team2_stats.values())
    })

    st.subheader("📈 Team Comparison")
    st.table(df)
