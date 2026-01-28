import os
import json
import re
import streamlit as st
import google.generativeai as genai

class GeminiCricketAPI:
    def __init__(self):
        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = genai.GenerativeModel("models/gemini-1.5-flash")

    @st.cache_data(ttl=3600)
    def get_team_stats(self, team):
        prompt = f"""
Give recent IPL performance stats for {team}.
Return ONLY valid JSON:

{{
  "batting_avg": 38.5,
  "bowling_avg": 26.4,
  "recent_form": 0.72
}}
"""
        try:
            response = self.model.generate_content(prompt)
            text = re.sub(r"```json|```", "", response.text).strip()
            return json.loads(text)
        except:
            return {
                "batting_avg": 35.0,
                "bowling_avg": 28.0,
                "recent_form": 0.5
            }
