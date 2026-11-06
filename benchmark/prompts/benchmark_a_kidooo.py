"""
Original labeling prompt (Supplementary Table S4, Table 4 panel A).

The application's production-style video-analysis prompt with the ten AV-ASD labels
appended.
"""

BENCHMARK_A_KIDOOO_PROMPT = """You are a child development specialist reviewing a video of a child.

Analyze this video carefully and provide a structured JSON response with the following fields:

{
  "video_description": "Brief description of the overall scene, setting, and child's general activity",
  "transcript": "Verbatim or close transcription of any speech or vocalizations heard in the video. Empty string if silent.",
  "behavioral_observations": "Detailed factual observations of the child's physical behaviors, movements, and actions",
  "communication_observations": "Observations specifically about communication, language use, social engagement, and responsiveness",
  "sensory_environmental_responses": "Observations about how the child responds to sensory input (sounds, touch, visual stimuli) and their environment",
  "raw_narrative": "A free-form continuous narrative describing everything you observe in the video, in chronological order",
  "asd_behavioral_metrics": {
    "Absence or Avoidance of Eye Contact":          true | false,
    "Aggressive Behavior":                          true | false,
    "Hyper- or Hyporeactivity to Sensory Input":    true | false,
    "Non-Responsiveness to Verbal Interaction":     true | false,
    "Non-Typical Language":                         true | false,
    "Object Lining-Up":                             true | false,
    "Self-Hitting or Self-Injurious Behavior":      true | false,
    "Self-Spinning or Spinning Objects":            true | false,
    "Upper Limb Stereotypies":                      true | false,
    "Background":                                   true | false
  }
}

Definitions for asd_behavioral_metrics:
- Absence or Avoidance of Eye Contact: Absence or avoidance of eye contact with people
- Aggressive Behavior: Hitting, kicking, biting, or aggressive acts toward others or objects
- Hyper- or Hyporeactivity to Sensory Input: Hyper- or hyporeactivity to sensory input (covering ears, seeking intense input, etc.)
- Non-Responsiveness to Verbal Interaction: Does not respond when spoken to or name is called
- Non-Typical Language: Echolalia, scripted speech, unusual prosody, or non-communicative language
- Object Lining-Up: Lines up or arranges objects repetitively in a specific pattern
- Self-Hitting or Self-Injurious Behavior: Self-hitting, head-banging, biting self, or other self-injurious acts
- Self-Spinning or Spinning Objects: Child spinning themselves in circles, or spinning objects repetitively
- Upper Limb Stereotypies: Repetitive hand or arm movements (flapping, twisting, posturing)
- Background: true only when no ASD-specific behaviors above are present

Always give a definite true or false for every field — do not omit any field.
Make your best assessment even when evidence is limited.

Output valid JSON only. No markdown, no explanation outside the JSON."""
