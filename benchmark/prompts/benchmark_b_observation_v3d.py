"""
Revised observation prompt (Supplementary Tables S2a and S4; Table 4 panel B and Table 5).

The observation record with the revised labeling rules written as recording rules.
No behavior labels are produced; JSON keys, nesting and value types are unchanged.
"""

BENCHMARK_B_OBSERVATION_PROMPT = """Analyze this child behavior clip and return objective, context-rich observations as valid JSON for a later parent report.

General rules:
- Output JSON only, using the exact keys, nesting, and value types below. Do not add fields, behavior labels, or true/false judgments about behaviors (for example, "Aggressive Behavior: true").
- First observe the whole scene and activity, then describe the relevant details. Record ordinary play, successful interactions, and actions that stand out in the clip according to what the clip actually shows.
- Describe visible or audible facts. Do not provide diagnoses, medical conclusions, recommendations, or judgments about developmental level. Do not infer intentions, internal states, stable traits, or causes from a short clip.
- An interaction bid is speech or action by another person directed to the child that invites attention or a response, such as calling the child's name, asking a question, making a request, or trying to engage face-to-face.
- Observation limits are anything that prevents assessing an event: unclear audio, poor visibility, camera angle, occlusion, a person off-screen, an uncertain speech source or recipient, or the clip ending too soon.
- For relevant events, record the action, surrounding context, subsequent response or outcome, and any observation limits. A single brief or mild event is worth recording; it does not need to be dramatic, sustained, or repeated. Include approximate timestamps, durations, or counts only when reliably assessable; do not invent precision.
- Preserve variation across events rather than summarizing the child as globally responsive, disengaged, or repetitive. Different responses can occur in different situations.
- Only video_description.setting and transcript timestamps may be null. Use an empty array when an array has no entries. Where it matters for an event, explain in the relevant observation string whether no event was observed during a period that could be assessed, no relevant opportunity occurred, or observation limits prevented assessment. Missing information is not evidence of absent behavior.

Behavior observation guides:
The nine headings below identify areas to notice, not labels to assign. Their guidance and examples are illustrative, not a checklist: do not search for every example, force each event into a heading, or write an absence statement for every heading. Their order and number of examples do not indicate importance. Include relevant observations outside these areas, including ordinary activity and successful interaction. Do not create nine category fields; place concrete descriptions in the existing JSON fields. No field needs a fixed number of entries.

1. Eye contact and gaze during interaction
- In non_verbal_communication and interaction_patterns, describe gaze direction, the other person's location, and whether an interaction bid occurs. Distinguish clearly visible eye contact from simply looking toward someone.
- Include activity context: attention to a toy, screen, or task, a brief glance away, or not looking at the camera does not by itself establish avoidance.
- A person who is off-screen but clearly audible can make an interaction bid. For each bid, record whether the child turns or looks toward the voice or the person's face, keeps looking elsewhere, or looks away.
- Evaluate each bid separately; the child may look during one bid and not during another. If the child's face or eyes cannot be seen, say so; do not infer either eye contact or its absence from that missing information.

2. Actions toward other people or objects
- In physical_movements and interaction_patterns, describe actions such as hitting, kicking, biting, pushing, grabbing, or throwing: the action, target, surrounding sequence, and visible outcome.
- Include visible context such as an accidental collision, pretend play, a game, reaching for an object, or an interaction before the action. Describe repeated or isolated actions as observed.
- Do not infer aggressive intent, anger, injury, or a cause solely from the movement. Distinguish visible contact, attempted contact, and contact that cannot be confirmed from the camera view.
- Record attempted or blocked strikes (swinging, kicking, or lunging at a person or object) even when no contact occurs. Quote hostile words shouted at a person (for example, "I hate you!"); a plain "No!" or refusal is speech, not an action toward others.

3. Sensory input and observable responses
- In sensory_reactions, record the observable stimulus, subsequent action, and timing. Examples include covering ears around a sound, shielding eyes around light, withdrawing from touch, rubbing a texture, or pressing against a surface.
- Also record looking at objects very close to the eyes or from the corner of the eye, repeatedly rubbing, pressing, or mouthing non-food objects, and spinning, rocking, or swinging that continues after its play purpose.
- Describe little or no visible reaction only when the stimulus and the period after it can be assessed, such as a clearly audible, sudden sound. Include ordinary exploration and task context; do not infer heightened/reduced sensitivity or a sensory-seeking motive.
- In comfort_or_discomfort_signals, record visible cues that occur in relation to a sensory stimulus or a change in the environment, such as crying, tensing, or moving away when a sound starts. Describe the cue, not the emotion it may reflect. State when the stimulus, response, or relationship between them is unclear.

4. Responses to interaction bids
- In verbal_communication and response_to_others, identify the speech source and recipient when supported. Distinguish speech directed to the child from television/device speech, narration to the camera, background conversation, or an unknown source.
- Describe the interaction bid, the child's activity, and any related verbal or non-verbal response: turning, nodding, gesturing, handing over an object, or carrying out a request. Silence alone does not establish no response.
- Record what kind of bid it is: a name call, a question, a request for a new action, a prohibition ("Stop!", "No!"), reassurance, or a rhetorical question. If the bid occurs while the child is crying, screaming, protesting, or having a tantrum, record that state with the bid and the response.
- If no response is visible, specify the period observed after the bid and any observation limits. If the clip ends right after the bid, say there was no time to respond. Do not infer deliberate ignoring, lack of understanding, or inability to hear.

5. Speech, vocalizations, repetition, and delivery
- In transcript and verbal_communication, record actual words/vocalizations and their use in requests, answers, naming, singing, imitation, or play when observable. Write [unintelligible] for speech that cannot be made out; do not invent words.
- If speech repeats an utterance heard earlier in the clip, identify both utterances and their sequence. Do not assume a script or source outside the clip; describe an unclear communicative purpose as unclear.
- Record non-word vocalizations, jargon, squealing, and repeated vocal sounds, and whether they are directed to someone. Describe relevant audible pitch, volume, rhythm, or stress concretely. Limited speech, repetition, or delivery alone does not establish a language delay or a developmental difficulty.

6. Object arrangement and manipulation
- In interaction_patterns and repetitive_or_notable_patterns, describe objects being placed in rows, grouped, stacked, ordered, or returned to an arrangement. Record the observable sequence and any repetition.
- Include activity context such as sorting, tidying, building, pretend play, or following a request. Describe varied object use as well as repeated arrangements when present.
- If an arrangement is changed, record the child's visible response only if that event occurs. Do not infer rigidity, distress about interruption, or a restricted interest solely from an orderly arrangement.

7. Actions involving the child's own body
- In physical_movements and repetitive_or_notable_patterns, describe self-directed contact such as a hand striking the head/body, biting one's own skin, or the head contacting a surface. Include body part, repetition, and context when assessable, including whether it happens while the child is crying, screaming, or protesting, or during calm or rhythmic activity.
- Describe visible outcomes only when observed, such as a mark, crying, withdrawal, or continuation of activity. Do not infer injury, pain, deliberate self-harm, or lack of pain.
- Include context such as scratching, rubbing the eyes, adjusting clothing, gentle tapping in play, or incidental contact. State when impact or its outcome cannot be seen clearly.

8. Body rotation and rotating objects
- In physical_movements and repetitive_or_notable_patterns, identify what rotates: the child's body, an object, or an object part. Describe initiation, repetition/duration when assessable, and whether the child watches the rotation.
- Include context such as dancing, turning to follow someone, a movement game, or operating a toy designed to rotate. Describe rotation within varied play or repeated episodes when the clip supports that distinction.
- Describe every time the child turns their whole body around in a circle (spinning, twirling, pirouetting), including a single twirl during dancing or play, and rotation on a swivel chair, spinning seat, merry-go-round, or ride-on toy driven in circles. Turning to face someone or changing walking direction is not a whole-body turn; describe it as turning toward or changing direction.
- Do not infer a sensory motive, a stable repetitive pattern, or preferences outside the clip. No fixed number of turns is required to document an observed action.

9. Hand, finger, and arm movement patterns
- In physical_movements and repetitive_or_notable_patterns, describe movement form, body part, one/both sides, and observed repetition. Examples include hand flapping, finger flicking, wrist/forearm rotation, hand shaking, opening/closing hands, or recurring stiff postures.
- Record repetitive movements even while the child is holding or playing with a toy. Include visible communication or task context such as waving, pointing, clapping in a game, handling a toy, or adjusting clothing. If function is unclear, do not assign a motive.
- Describe isolated and recurring movements accurately, with duration or count only when assessable. Appearance alone does not establish a stereotypy or a stable motor characteristic.

Whole-scene observations beyond these guides:
- Record play, initiation, turn-taking, help-seeking, and shared activity in interaction_patterns; attention targets, attention shifts, and focus intervals in attention_focus_patterns; and facial expressions and body cues not tied to a sensory stimulus in facial_emotional_signals, describing the cue rather than the emotion it may reflect. Describe event sequences rather than inferring shared attention, distractibility, or an internal emotion from a single cue.
- In participants, give a person's role (for example, parent or sibling) only when the clip makes it clear, such as by how the person is addressed; otherwise use a neutral description such as "adult" or "another child".
- The transcript covers live speakers only: people present in the scene, on or off camera. Do not transcribe television, device, or other recorded audio; note it in verbal_communication when relevant. Keep transcript.utterance for spoken content; record source/recipient context in communication_observations.
- In transcript.speaker, use Child only for the target child, Adult for any adult, and Unknown when the speaker cannot be identified or is another child; identify other children in verbal_communication. Do not put roles such as parent in the speaker field. Use null for timestamp when timing cannot be determined.
- In raw_narrative, summarize the actual activity and most informative observations in 3-6 factual sentences, including any important observation limits already recorded in other fields. Emphasis should reflect the clip, not the guide headings. Do not introduce claims absent from the other fields.

Return this exact structure. The strings below describe the information to supply; do not copy them as observations. Keep observation arrays as arrays of strings.
{
  "video_description": {
    "setting": "Observable setting, or null",
    "participants": ["Participants, with roles only when the clip makes them clear"],
    "activity_overview": "Main activity and context in 1-2 factual sentences"
  },
  "transcript": [
    {
      "speaker": "Child|Adult|Unknown",
      "timestamp": "mm:ss or null",
      "utterance": "Audible speech or vocalization content from a live speaker"
    }
  ],
  "behavioral_observations": {
    "physical_movements": ["Concrete movements, relevant context, and visible outcomes"],
    "facial_emotional_signals": ["Visible expressions and body cues with context, described as cues rather than emotions"],
    "interaction_patterns": ["Interactions with people or objects and their observable sequences"],
    "attention_focus_patterns": ["Observable attention targets, shifts, and focus intervals"],
    "repetitive_or_notable_patterns": ["Observed repetition or other actions that stand out in the clip, with activity context"]
  },
  "communication_observations": {
    "verbal_communication": ["Speech/vocalizations, context, and relevant source/recipient information"],
    "non_verbal_communication": ["Gestures, gaze, engagement, and relevant observation limits"],
    "response_to_others": ["Interaction bids and subsequent responses, with context, timing, and observation limits"]
  },
  "sensory_environmental_responses": {
    "sensory_reactions": ["Observable stimuli, subsequent actions, and relevant context or uncertainty"],
    "comfort_or_discomfort_signals": ["Visible cues related to a stimulus or environmental change, and their observable context"]
  },
  "raw_narrative": "Concise factual summary in 3-6 sentences"
}"""
