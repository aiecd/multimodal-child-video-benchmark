"""
Gemini adapters for Benchmark B — Stage 2 (report) and Stage 3 (judge).

Text-only input (observations JSON) — no video upload needed for either stage.
"""

from __future__ import annotations

import importlib
import json
import os
import re

from google import genai
from google.genai import types

from benchmark.models.base_text import JudgeModel, ReportModel
from benchmark.prompts.benchmark_b_judge import build_judge_prompt
from benchmark.prompts.benchmark_b_report import BENCHMARK_B_REPORT_PROMPT, LANGUAGE_INSTRUCTIONS

try:
    from google.api_core import exceptions as _gac

    _GOOGLE_RETRYABLE: tuple[type[Exception], ...] = (
        _gac.ServiceUnavailable,
        _gac.InternalServerError,
        _gac.ResourceExhausted,
        _gac.DeadlineExceeded,
    )
    _GOOGLE_RATE_LIMIT: tuple[type[Exception], ...] = (_gac.ResourceExhausted,)
except ImportError:
    _GOOGLE_RETRYABLE = (ConnectionError, TimeoutError)
    _GOOGLE_RATE_LIMIT = ()


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Extract JSON from markdown code block if present (complete fence: opening + closing)
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    elif text.startswith("```"):
        # Opening fence present but closing fence missing (truncated response).
        # Strip just the opening fence header and attempt to parse the JSON body.
        text = re.sub(r"^```(?:json)?\s*", "", text).strip()
    # Fix trailing commas before closing braces/brackets (Gemini quirk)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    result = None
    last_err: json.JSONDecodeError | None = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        last_err = e

    if result is None:
        # Handle "Extra data": Gemini sometimes appends prose after the closing brace.
        try:
            result, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            pass

    if result is None:
        raise ValueError(
            f"Failed to parse Gemini response as JSON: {last_err}\n"
            f"Response ({len(text)} chars total):\n{text[:800]}"
        )

    # Gemini sometimes wraps the object in an array: [{...}] or [{"Model A": {...}}, ...]
    # Merge all dict elements into one flat dict.
    if isinstance(result, list):
        merged: dict = {}
        for item in result:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return result


def _token_costs(usage) -> dict:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


class GeminiReportModel(ReportModel):
    """
    Stage 2: generate a caregiver report from behavioral observations using Gemini.

    Text-only (no video). Uses synchronous client; the base class async default
    (asyncio.to_thread) wraps it for the concurrent runner.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _GOOGLE_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _GOOGLE_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        self._client = genai.Client(api_key=api_key)
        self._max_tokens: int | None = config.get("max_tokens", None)
        self._temperature: float = config.get("temperature", 0.3)
        # report_prompt_module selects the report prompt file (Table 5). Default is
        # benchmark_b_report.
        prompt_module = config.get("report_prompt_module")
        self._report_prompt: str = (
            importlib.import_module(f"benchmark.prompts.{prompt_module}").BENCHMARK_B_REPORT_PROMPT
            if prompt_module else BENCHMARK_B_REPORT_PROMPT
        )

    def generate_report(self, observations: dict, language: str = "en") -> tuple[dict, str, dict]:
        """
        Generate a structured caregiver report from behavioral observations.

        Returns:
            (output_dict, raw_response_str, token_costs)
            output_dict: {"report": {...}}
        """
        prompt = self._report_prompt.format(
            language_instruction=LANGUAGE_INSTRUCTIONS.get(language, ""),
            gemini_observations=json.dumps(observations, ensure_ascii=False, indent=2),
        )
        cfg_kwargs: dict = {
            "temperature": self._temperature,
            "response_mime_type": "application/json",
        }
        if self._max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self._max_tokens
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        raw = response.text
        output = _extract_json(raw)
        usage = getattr(response, "usage_metadata", None)
        return output, raw, _token_costs(usage)


class GeminiJudge(JudgeModel):
    """
    Stage 3: score all model reports for a video using Gemini as judge.

    Receives anonymized reports (Model A, B, C) and original observations.
    Does not receive the AV-ASD labels.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _GOOGLE_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _GOOGLE_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        self._client = genai.Client(api_key=api_key)
        self._max_tokens: int | None = config.get("max_tokens", None)
        self._temperature: float = config.get("temperature", 0.3)

    def score(
        self,
        observations: dict,
        reports: dict[str, dict],
        language: str = "en",
    ) -> tuple[dict, str, dict]:
        """
        Score all model reports. Reports are already anonymized by the runner.

        Returns:
            (scores_dict, raw_response_str, token_costs)
        """
        reports_text = json.dumps(reports, ensure_ascii=False, indent=2)
        observations_text = json.dumps(observations, ensure_ascii=False, indent=2)
        prompt = build_judge_prompt(language, observations_text, reports_text)
        # No response_mime_type: constraining to JSON causes Gemini to wrap output
        # in an array or truncate. Plain text + code block extraction is more reliable.
        # Safety filters off: judge content contains medical/autism
        # terminology that triggers Gemini's default filters, causing mid-JSON truncation.
        gen_config_kwargs: dict = {
            "temperature": self._temperature,
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY",   threshold="OFF"),
            ],
        }
        if self._max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = self._max_tokens
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**gen_config_kwargs),
        )
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        usage = getattr(response, "usage_metadata", None)
        raw = response.text
        try:
            scores = _extract_json(raw)
        except ValueError as e:
            raise ValueError(f"{e}\nfinish_reason={finish_reason}") from None
        return scores, raw, _token_costs(usage)
