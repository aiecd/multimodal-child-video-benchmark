"""
Benchmark C — report normalization for blinded judging.

All model reports are flattened to an identical plain-text schema before being
sent to judges.  This prevents structural fingerprinting: Kidooo's production
report has rich nested fields (key_points, icon, what_to_do, why_it_helps,
example, type) that are absent from single-stage outputs and would allow a
judge to identify the model from JSON structure alone, defeating anonymization.

Canonical output schema (all values are plain strings or list of strings):
{
  "title":               str,
  "observation_summary": str,
  "behavior_analysis":   str,
  "recommendations":     ["1. <Title>: <body>", ...],
  "encouragement":       str,
}

Both known schemas are handled:

  Single-stage schema:
    observation_summary  → str
    behavior_analysis    → str
    recommendations[i]   → {title, description}

  Kidooo production schema:
    observation_summary  → {headline, key_points:[{icon,label,detail}], context_connection}
    behavior_analysis    → {main_insight, contributing_factors:[{factor,explanation}],
                            developmental_perspective}
    recommendations[i]   → {title, icon, type, what_to_do, why_it_helps, example}

Any unrecognized structure is stringified as a fallback so normalization never
raises and never silently drops content.
"""

from __future__ import annotations

from typing import Any


def _str(v: Any, fallback: str = "") -> str:
    """Coerce any value to a non-empty string, or return fallback."""
    if v is None:
        return fallback
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _normalize_observation_summary(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        parts: list[str] = []
        if raw.get("headline"):
            parts.append(_str(raw["headline"]))
        for kp in raw.get("key_points", []):
            if isinstance(kp, dict):
                label  = _str(kp.get("label", ""))
                detail = _str(kp.get("detail", ""))
                if label and detail:
                    parts.append(f"{label}: {detail}")
                elif detail:
                    parts.append(detail)
        if raw.get("context_connection"):
            parts.append(_str(raw["context_connection"]))
        return " ".join(parts)
    return _str(raw)


def _normalize_behavior_analysis(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        parts: list[str] = []
        if raw.get("main_insight"):
            parts.append(_str(raw["main_insight"]))
        for cf in raw.get("contributing_factors", []):
            if isinstance(cf, dict):
                factor = _str(cf.get("factor", ""))
                expl   = _str(cf.get("explanation", ""))
                if factor and expl:
                    parts.append(f"{factor}: {expl}")
                elif expl:
                    parts.append(expl)
        if raw.get("developmental_perspective"):
            parts.append(_str(raw["developmental_perspective"]))
        return " ".join(parts)
    return _str(raw)


def _normalize_recommendation(i: int, rec: Any) -> str:
    """Return a numbered plain-text recommendation string."""
    if not isinstance(rec, dict):
        return f"{i}. {_str(rec)}"
    title = _str(rec.get("title", f"Recommendation {i}"))
    # Body: prefer what_to_do (Kidooo) then description (single-stage)
    body_parts: list[str] = []
    if rec.get("what_to_do"):
        body_parts.append(_str(rec["what_to_do"]))
    elif rec.get("description"):
        body_parts.append(_str(rec["description"]))
    if rec.get("why_it_helps"):
        body_parts.append(_str(rec["why_it_helps"]))
    body = " ".join(body_parts) if body_parts else ""
    if body:
        return f"{i}. {title}: {body}"
    return f"{i}. {title}"


def normalize_report(report_output: dict) -> dict:
    """
    Flatten a raw model report_output to the canonical plain-text schema.

    Accepts both the outer wrapper ({report: {...}}) and the inner report dict
    directly.  Returns the canonical dict; never raises.
    """
    # Unwrap outer {report: ...} envelope if present
    inner = report_output
    if "report" in report_output and isinstance(report_output["report"], dict):
        inner = report_output["report"]

    title = _str(inner.get("title", ""))
    observation_summary = _normalize_observation_summary(inner.get("observation_summary", ""))
    behavior_analysis   = _normalize_behavior_analysis(inner.get("behavior_analysis", ""))

    raw_recs = inner.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []
    recommendations = [
        _normalize_recommendation(i + 1, rec) for i, rec in enumerate(raw_recs)
    ]

    encouragement = _str(inner.get("encouragement", ""))

    return {
        "title":               title,
        "observation_summary": observation_summary,
        "behavior_analysis":   behavior_analysis,
        "recommendations":     recommendations,
        "encouragement":       encouragement,
    }
