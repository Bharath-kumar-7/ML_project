import streamlit as st
import numpy as np
import pandas as pd
import base64
from model import IPLPredictor
from gemini_api import GeminiCricketAPI

# ---------- Helper: Background Image ----------
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- Initialize ----------
predictor = IPLPredictor()
predictor.train()
api = GeminiCricketAPI()

st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("🏏 IPL Match Win Predictor")

# ---------- ALL IPL TEAMS ----------
teams = [
    "CSK", "MI", "RCB", "KKR", "DC",
    "PBKS", "RR", "SRH", "GT", "LSG"
]

# ---------- Inputs ----------
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# ---------- Background Image ----------
image_path = f"images/{team1}_{team2}.jpg"
reverse_path = f"images/{team2}_{team1}.jpg"

try:
    set_bg(image_path)
except:
    try:
        set_bg(reverse_path)
    except:
        set_bg("images/default.jpg")

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
        "Rajiv Gandhi Stadium",
        "Sawai Mansingh Stadium",
        "Punjab Cricket Association Stadium",
        "Ekana Cricket Stadium"
    ]
)

weather = st.selectbox("Weather", ["Clear", "Cloudy", "Hot", "Humid"])

# ---------- Prediction ----------
if st.button("Predict Winner"):
    with st.spinner("Analyzing match conditions..."):
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

    # ---------- Stats Table ----------
    st.subheader("📈 Team Performance Statistics")

    df = pd.DataFrame({
        "Metric": ["Batting Average", "Bowling Average", "Recent Form"],
        team1: [
            team1_stats["batting_avg"],
            team1_stats["bowling_avg"],
            team1_stats["recent_form"]
        ],
        team2: [
            team2_stats["batting_avg"],
            team2_stats["bowling_avg"],
            team2_stats["recent_form"]
        ]
    })

    st.table(df)
