"""
Baseline prompt for Benchmark A.
All non-kidooo models receive this prompt — no model-specific framing.
Output keys match LABEL_KEYS (human-readable) so parsing requires no remapping.
"""

BENCHMARK_A_BASELINE_PROMPT = """Watch this child behavior video and identify which of the following behaviors are present.

For each behavior, output true (observed) or false (not observed).
Always give a definite answer — do not omit any field.

Behavior categories:
- Absence or Avoidance of Eye Contact
- Aggressive Behavior
- Hyper- or Hyporeactivity to Sensory Input
- Non-Responsiveness to Verbal Interaction
- Non-Typical Language
- Object Lining-Up
- Self-Hitting or Self-Injurious Behavior
- Self-Spinning or Spinning Objects
- Upper Limb Stereotypies
- Background: true only when no ASD-specific behaviors above are present

Output as JSON only. No explanation, no markdown:
{
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
}"""
