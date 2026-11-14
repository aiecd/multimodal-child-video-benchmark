"""
Revised labeling prompt (Supplementary Table S4, Table 4 panel A).

The original labeling prompt with the rules in Table 3 and Supplementary Note S1.
Same JSON output as the original.
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

Apply the following rules when filling asd_behavioral_metrics:
Context and evidence rules:
- Label behaviors from what is observable in this clip's video and audio, interpreted in the context of the child's activity and the interaction taking place.
- Do not infer a behavior from a category name, presumed diagnosis, or an event that may have happened outside the recording.
- Output true when the clip shows at least one qualifying event — an event that meets the behavior's definition below. One observable instance is enough; it does not need to be dramatic, sustained, or repeated.
- Output false when no qualifying event occurs, or when the relevant evidence cannot be observed at all (for example, the child's face is never visible, or the audio is unusable). Do not output false merely because an event you observed was brief or mild; when you saw a plausible qualifying event, make your best assessment.
- Your labels must agree with your observations: if any observation field describes an event that meets a definition below, set that label to true.
- One event can satisfy more than one label. For example, spinning that appears to seek sensory input counts for both Self-Spinning or Spinning Objects and Hyper- or Hyporeactivity to Sensory Input.
- For Absence or Avoidance of Eye Contact and Non-Responsiveness to Verbal Interaction, evaluate each engagement bid separately. A qualifying event in one bid supports true even if the child engages during another.
- These labels describe observations in this clip and do not establish a diagnosis. Provide a definite true or false for every field; do not omit fields, use null, or add keys.

Behavior definitions:
- Absence or Avoidance of Eye Contact: During an engagement bid, the child does not look toward the person's face or looks away from it.
  - An engagement bid is someone addressing the child, calling the child's name, offering or showing the child an object, or playing with the child face to face. The person may be off-screen if they are clearly audible.
  - Not orienting toward a person who calls the child's name or speaks to the child counts.
  - Does not count: looking at a toy or task during independent play with no one engaging the child, or not looking at the camera.
- Aggressive Behavior: Hitting, kicking, biting, pushing, throwing objects at others, or other aggressive acts toward people or objects.
  - Attempted or blocked strikes count: swinging, kicking, or lunging at a person or object. Contact is not required.
  - Shouting hostile words at a person (for example, "I hate you!") counts. A plain "No!" or a refusal does not.
  - Self-injury during a tantrum, meltdown, or angry outburst (hitting one's own head or face, banging the head against something, or biting oneself while crying, screaming, or protesting) also counts as Aggressive Behavior. Calm, rhythmic, or self-stimulatory self-injury is Self-Hitting or Self-Injurious Behavior only.
- Hyper- or Hyporeactivity to Sensory Input: Unusual reactions to, or unusual seeking of, sensory input.
  - Hyperreactivity: covering the ears, recoiling or becoming distressed at a sound, light, touch, or texture.
  - Hyporeactivity: no visible reaction to a clearly audible, salient sound or to clearly observed touch or pain.
  - Sensory seeking: spinning, rocking, or swinging for vestibular input; inspecting objects very close to the eyes, from the corner of the eye, or through other objects; repeatedly rubbing, pressing, or mouthing non-food objects or textures.
  - Does not count: a single ordinary startle, squint, or exploratory touch.
- Non-Responsiveness to Verbal Interaction: The child does not respond, verbally or non-verbally, to speech directed to the child — for example their name being called, a question, or a request.
  - Responses include turning or looking toward the speaker, nodding, pointing, gesturing, handing over an object, or doing what was asked; speech is not required.
  - Speech from a television or device, adult-to-adult conversation, and narration to the camera are not directed to the child unless it clearly turns into a request to the child.
  - While the child is crying, screaming, self-injuring, acting aggressively, or having a tantrum, these are not non-responsiveness: continuing the behavior after "Stop!", "No!", or another prohibition, even when it includes the child's name; protesting or refusing; and not answering reassurance, apologies, or scolding or rhetorical questions ("Why do you keep doing that?", "It's okay"). A plain name call, or a request for a new action ("come here", "sit down"), that gets no orienting, reply, or change in activity still counts.
  - If the clip ends right after the utterance, leaving no time to respond, output false.
- Non-Typical Language: Speech or vocalization that is unusual in form or use: echolalia, scripted or repeated phrases, unusual prosody or pitch, jargon, non-word vocalizations, squealing, or repetitive vocal sounds used in place of age-expected speech, or speech that is not used to communicate. Complete silence alone does not count.
- Object Lining-Up: Lines up or arranges objects repetitively in a specific pattern.
- Self-Hitting or Self-Injurious Behavior: Self-hitting, head-banging, biting self, or other self-injurious acts.
- Self-Spinning or Spinning Objects: The child turns their whole body around in a circle (spinning, twirling, pirouetting), including a single twirl during dancing or play; rotates themselves on a swivel chair, merry-go-round, spinning seat, or ride-on toy driven in circles; or spins or twirls an object.
  - Does not count: turning to face someone or something, or changing walking direction.
  - If your observations say the child spins, twirls, or swivels around, set this label to true.
- Upper Limb Stereotypies: Repetitive hand or arm movements such as hand flapping, finger flicking or wiggling near the face, wrist twisting, hand shaking, repeated opening and closing of the hands, or stiff hand or arm postures. These still count when the child is also holding or playing with a toy. Functional movements themselves — waving goodbye, pointing, clapping in a game, manipulating a toy, adjusting clothing — do not count.
- Background: true if and only if all nine behavior labels above are false; otherwise false. This is a clip-level label, not a statement about diagnosis.

Narrative rules:
- Keep descriptions factual. Do not invent speech, speakers, eye contact, intentions, or actions that cannot be observed. If a speech source cannot be identified, say so.
- Describe each engagement bid and the child's response, and note any visibility or audio limitations.
- Describe every time the child turns their whole body around.

Always give a definite true or false for every field — do not omit any field.
Make your best assessment from what the clip shows.

Output valid JSON only. No markdown, no explanation outside the JSON."""
