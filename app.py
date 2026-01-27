import streamlit as st
import numpy as np
import pandas as pd
import base64
import io
from PIL import Image
from model import IPLPredictor
from gemini_api import GeminiCricketAPI

# ---------- Background with Dark Overlay ----------
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
                linear-gradient(
                    rgba(0,0,0,0.75),
                    rgba(0,0,0,0.75)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        /* Inputs */
        div[data-baseweb="select"] > div {{
            background-color: #020617 !important;
            color: white !important;
        }}

        .stRadio > div {{
            background-color: #020617 !important;
            padding: 12px;
            border-radius: 10px;
        }}

        /* Table */
        div[data-testid="stTable"] {{
            background-color: #020617 !important;
            border-radius: 14px;
            padding: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.7);
        }}

        thead tr th {{
            background-color: #020617 !important;
            color: #38bdf8 !important;
            font-size: 16px;
        }}

        tbody tr td {{
            background-color: #020617 !important;
            color: #f8fafc !important;
            font-size: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- Init ----------
predictor = IPLPredictor()
predictor.train()
api = GeminiCricketAPI()

st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("🏏 IPL Match Win Predictor")

# ---------- Teams ----------
teams = ["CSK", "MI", "RCB", "KKR", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# ---------- Background Logic ----------
image_path = f"images/{team1}_{team2}.avif"
reverse_path = f"images/{team2}_{team1}.avif"

try:
    set_bg(image_path)
except:
    try:
        set_bg(reverse_path)
    except:
        set_bg("images/default.avif")

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

    # ---------- Result Cards ----------
    st.markdown(
        f"""
        <div style="
            background:#064e3b;
            padding:18px;
            border-radius:14px;
            color:#ecfdf5;
            font-size:20px;
            box-shadow:0 10px 30px rgba(0,0,0,0.7);
            margin-bottom:10px;
        ">
            🏆 <b>Predicted Winner:</b> {winner}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background:#1e3a8a;
            padding:18px;
            border-radius:14px;
            color:#eff6ff;
            font-size:20px;
            box-shadow:0 10px 30px rgba(0,0,0,0.7);
            margin-bottom:20px;
        ">
            📊 <b>Winning Probability:</b> {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

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
