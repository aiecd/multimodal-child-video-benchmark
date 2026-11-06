"""
Original observation prompt (Supplementary Table S4; Table 2).

The model receives this prompt and the video and returns a structured observation
record as JSON. No AV-ASD label is given to the model.
"""

BENCHMARK_B_OBSERVATION_PROMPT = """Analyze this child video and return objective observations as valid JSON.

Rules:
- Output JSON only. Do not add markdown or prose outside JSON.
- Describe what is observable. Do not provide diagnosis or medical conclusions.
- If a field is not observable, use null or an empty array.

Return this exact top-level structure:
{
  "video_description": {
    "setting": "Where the video takes place",
    "participants": ["Child", "Adult caregiver", "Teacher", "..."],
    "activity_overview": "What is happening in 1-2 sentences"
  },
  "transcript": [
    {
      "speaker": "Child|Adult|Unknown",
      "timestamp": "mm:ss or null",
      "utterance": "Observed speech content"
    }
  ],
  "behavioral_observations": {
    "physical_movements": ["Posture, gestures, motor actions"],
    "facial_emotional_signals": ["Observed facial expressions and affect"],
    "interaction_patterns": ["Interactions with people/objects"],
    "attention_focus_patterns": ["Attention shifts, focus duration, distractibility"],
    "repetitive_or_notable_patterns": ["Repetitive behaviors or notable patterns"]
  },
  "communication_observations": {
    "verbal_communication": ["Words, phrases, tone, initiation"],
    "non_verbal_communication": ["Pointing, eye contact, gestures"],
    "response_to_others": ["How child responds to prompts or speech"]
  },
  "sensory_environmental_responses": {
    "sensory_reactions": ["Reactions to sound/light/touch/texture"],
    "comfort_or_discomfort_signals": ["Environmental comfort/discomfort cues"]
  },
  "raw_narrative": "Concise factual narrative summary in 3-6 sentences"
}"""
