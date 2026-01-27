import google.generativeai as genai
import json
import re


class GeminiCricketAPI:
    def __init__(self):
        # 🔐 Configure API key
        genai.configure(
            api_key="AIzaSyB5H6b-gUMM85dLzPatSEp004EDlyQPA74"
        )

        # ✅ UPDATED MODEL (IMPORTANT)
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash"
        )

    def get_team_stats(self, team):
        prompt = f"""
You are a cricket data analyst.

Give recent IPL performance stats for {team}.
Return ONLY valid JSON in this exact format:

{{
  "batting_avg": 38.5,
  "bowling_avg": 26.4,
  "recent_form": 0.72
}}
"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # 🧹 Remove markdown/code blocks if Gemini adds them
            text = re.sub(r"```json|```", "", text).strip()

            return json.loads(text)

        except Exception as e:
            print("Gemini parsing error:", e)

            # 🛑 Safe fallback (so app never crashes)
            return {
                "batting_avg": 35.0,
                "bowling_avg": 28.0,
                "recent_form": 0.5
            }


# 🧪 Test run
if __name__ == "__main__":
    api = GeminiCricketAPI()
    stats = api.get_team_stats("Mumbai Indians")
    print(stats)
