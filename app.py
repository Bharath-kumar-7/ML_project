import streamlit as st
import numpy as np
import base64
import io
from PIL import Image


from model import IPLPredictor
from gemini_api import GeminiCricketAPI

# ---------------- PAGE ----------------
st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("🏏 IPL Match Win Predictor")

# ---------------- BACKGROUND (TEAM BASED) ----------------
def set_bg_by_team(team):
    path = f"images/{team}.avif"
    try:
        img = Image.open(path)
    except:
        img = Image.open("images/default.avif")

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
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- INIT ----------------
predictor = IPLPredictor()
predictor.train()

api = GeminiCricketAPI()

teams = ["CSK", "MI", "RCB", "KKR", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]

# ---------------- INPUTS ----------------
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# 🔥 Background updates instantly when team changes
set_bg_by_team(team1)

toss = st.radio("Toss Winner", [team1, team2])

venue = st.selectbox(
    "Venue",
    [
        "M Chinnaswamy Stadium, Bengaluru",
        "Wankhede Stadium, Mumbai",
        "MA Chidambaram Stadium, Chennai",
        "Arun Jaitley Stadium, Delhi",
        "Eden Gardens, Kolkata",
        "Narendra Modi Stadium, Ahmedabad",
        "Rajiv Gandhi International Stadium, Hyderabad",
        "Sawai Mansingh Stadium, Jaipur",
        "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Stadium, Lucknow",
        "DY Patil Stadium, Navi Mumbai",
        "Brabourne Stadium, Mumbai",
        "Dr YS Rajasekhara Reddy ACA-VDCA Stadium, Visakhapatnam",
        "Barsapara Cricket Stadium, Guwahati",
        "Holkar Cricket Stadium, Indore",
        "Greenfield International Stadium, Thiruvananthapuram"
    ]
)

weather = st.selectbox("Weather", ["Clear", "Cloudy", "Hot", "Humid"])

# ---------------- PREDICT ----------------
if st.button("Predict Winner"):
    with st.spinner("⚡ Fetching team stats..."):

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

    # ---------------- RESULT ----------------
    st.success(f"🏆 Predicted Winner: **{winner}**")
    st.info(f"📊 Winning Probability: **{confidence:.2f}%**")

    # ---------------- CLASSIC STATS TABLE ----------------
    st.subheader("📈 Team Comparison")

    st.markdown(
        f"""
        <div style="
            background:#020617;
            padding:18px;
            border-radius:16px;
            box-shadow:0 12px 30px rgba(0,0,0,0.6);
            margin-top:10px;
        ">
        <table style="width:100%; border-collapse:collapse; color:#f8fafc;">
            <tr style="border-bottom:1px solid #334155;">
                <th style="text-align:left; padding:10px;">Metric</th>
                <th style="padding:10px;">{team1}</th>
                <th style="padding:10px;">{team2}</th>
            </tr>
            <tr>
                <td style="padding:10px;">Batting Average</td>
                <td style="text-align:center;">{team1_stats["batting_avg"]}</td>
                <td style="text-align:center;">{team2_stats["batting_avg"]}</td>
            </tr>
            <tr>
                <td style="padding:10px;">Bowling Average</td>
                <td style="text-align:center;">{team1_stats["bowling_avg"]}</td>
                <td style="text-align:center;">{team2_stats["bowling_avg"]}</td>
            </tr>
            <tr>
                <td style="padding:10px;">Recent Form</td>
                <td style="text-align:center;">{team1_stats["recent_form"]}</td>
                <td style="text-align:center;">{team2_stats["recent_form"]}</td>
            </tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True
    )
