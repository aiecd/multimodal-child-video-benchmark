"""
Behavior-mention extraction (Table 5).

Reads one caregiver report and records which of the nine AV-ASD behaviors the report
says the child showed. The extractor sees neither the video nor the AV-ASD labels.
"""

NINE_BEHAVIORS = [
    "Absence or Avoidance of Eye Contact",
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Non-Responsiveness to Verbal Interaction",
    "Non-Typical Language",
    "Object Lining-Up",
    "Self-Hitting or Self-Injurious Behavior",
    "Self-Spinning or Spinning Objects",
    "Upper Limb Stereotypies",
]

BEHAVIOR_MENTIONS_PROMPT = """You will read a parent-facing report written about a short video of a child. Decide, for each behavior below, whether the report states that the child showed that behavior in the video.

Rules:
- Count a behavior only when the report describes the child doing it in this video. The report does not need to use the behavior's name; a concrete description that meets the definition counts (for example, "she flapped her hands repeatedly" counts as Upper Limb Stereotypies).
- Do not count: behaviors the report says did not occur or could not be seen; hypothetical or general statements ("some children spin..."); behaviors mentioned only in recommendations or examples of what a parent might try; behaviors attributed to someone other than the child.
- Judge only from the report text. Do not guess what the video showed.

Behavior definitions:
- Absence or Avoidance of Eye Contact: during someone's attempt to engage the child (speaking to, calling, showing something, face-to-face play), the child does not look toward the person's face, looks away, or does not orient to a person speaking to them.
- Aggressive Behavior: hitting, kicking, biting, pushing, throwing objects at others, attempted or blocked strikes at a person or object, or shouting hostile words at a person.
- Hyper- or Hyporeactivity to Sensory Input: unusual reactions to sensory input (covering ears, recoiling or distress at a sound, light, touch, or texture; no reaction to a salient non-speech sound, such as a loud noise, or to touch or pain) or unusual sensory seeking (spinning, rocking, or swinging for sensation; inspecting objects very close to the eyes or from the corner of the eye; repeatedly rubbing, pressing, or mouthing non-food objects).
- Non-Responsiveness to Verbal Interaction: the child does not respond, verbally or non-verbally, to speech directed to them, such as their name, a question, or a request. Not responding to speech counts here only, not as Hyper- or Hyporeactivity to Sensory Input.
- Non-Typical Language: echolalia, scripted or repeated phrases, unusual prosody or pitch, jargon, non-word vocalizations, squealing, or repetitive vocal sounds in place of expected speech, or speech not used to communicate.
- Object Lining-Up: lining up or arranging objects repetitively in a specific pattern.
- Self-Hitting or Self-Injurious Behavior: hitting oneself, head-banging, biting oneself, or other self-injurious acts.
- Self-Spinning or Spinning Objects: the child turns their whole body around (spinning, twirling), rotates on a swivel chair or spinning ride, or spins an object.
- Upper Limb Stereotypies: repetitive hand or arm movements such as hand flapping, finger flicking, wrist twisting, hand shaking, repeated opening and closing of the hands, or stiff hand or arm postures; not functional movements such as waving, pointing, or clapping in a game.

Report:
{report}

Return JSON only, with exactly these nine keys, each an object with "mentioned" (true or false) and "evidence" (a short quote copied word for word from the report, or "" when false):
{{
{keys}
}}"""


def build_mentions_prompt(report_text: str) -> str:
    keys = ",\n".join(f'  "{b}": {{"mentioned": false, "evidence": ""}}' for b in NINE_BEHAVIORS)
    return BEHAVIOR_MENTIONS_PROMPT.format(report=report_text, keys=keys)
