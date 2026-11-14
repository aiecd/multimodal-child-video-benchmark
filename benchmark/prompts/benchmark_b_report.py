"""
Original report prompt (Supplementary Table S4; Table 2 and Table 4 panel B).

Every report model receives this prompt with the observation record inserted.
"""

# Language-specific instructions prepended to the report prompt.
# English is the default; no instruction needed.
LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "",
}

BENCHMARK_B_REPORT_PROMPT = """{language_instruction}You are a child development specialist. Based on the video observations generate a comprehensive and personalized report for parents.

## Video Observations (from initial analysis):
{gemini_observations}

---

DIVERSITY RULE: Your 3 recommendations MUST cover at least 2 different categories.
Never generate all 3 from the same category.

Generate a JSON response with the following structure:

{{
  "report": {{
    "title": "Brief video description, under 10 words",
    "observation_summary": {{
      "headline": "One sentence summarizing the most important observation",
      "key_points": [
        {{
          "type": "speech|emotion|motor|social|attention|sensory",
          "label": "Short label",
          "detail": "Specific observation description"
        }}
      ]
    }},
    "behavior_analysis": {{
      "main_insight": "Core interpretation of the behaviors, 2-3 sentences",
      "contributing_factors": [
        {{
          "factor": "Factor name",
          "explanation": "Specific explanation"
        }}
      ],
      "developmental_perspective": "What these behaviors mean from a developmental perspective"
    }},
    "recommendations": [
      {{
        "title": "Recommendation title",
        "type": "communication|social_connection|emotional_regulation|play_and_learning|sensory_support|daily_living|motor_and_body",
        "what_to_do": "Pure instruction: WHAT to do and HOW. No examples, no parenthetical illustrations. 1-2 sentences of guidance only.",
        "why_it_helps": "Why this recommendation helps this child",
        "example": "A specific real-life scenario narrative showing the instruction in action. Must be a DIFFERENT illustration from anything in what_to_do."
      }}
    ],
    "encouragement": "Encouraging words for the parent"
  }}
}}

Requirements:
- report.title: under 10 words, describes the video content briefly
- report.observation_summary.key_points: 3-5 items
- report.recommendations: exactly 3 items, covering at least 2 different type categories
- report.observation_summary.key_points[].type values: speech|emotion|motor|social|attention|sensory
- report.recommendations[].type values: communication|social_connection|emotional_regulation|play_and_learning|sensory_support|daily_living|motor_and_body
- recommendations[].what_to_do must contain ONLY instructions, NO examples or illustrative parentheticals
- recommendations[].example must be a concrete scenario narrative that does NOT repeat content from what_to_do
- All report text should be warm, supportive, and actionable
- Write all report text in the language specified at the top of this prompt"""
