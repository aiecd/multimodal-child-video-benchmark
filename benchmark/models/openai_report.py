"""
OpenAI-compatible adapter for Benchmark B — Stage 2 (report generation).

Works with any OpenAI-compatible API (OpenAI, Moonshot, DashScope, Meta, DeepSeek).
Set base_url in config to override the default OpenAI endpoint.
"""

from __future__ import annotations

import importlib
import json
import os
import re

from openai import OpenAI

from benchmark.models.base_text import JudgeModel, ReportModel
from benchmark.prompts.benchmark_b_judge import build_judge_prompt
from benchmark.prompts.benchmark_b_report import BENCHMARK_B_REPORT_PROMPT, LANGUAGE_INSTRUCTIONS

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    _OPENAI_RETRYABLE: tuple[type[Exception], ...] = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    _OPENAI_RATE_LIMIT: tuple[type[Exception], ...] = (RateLimitError,)
except ImportError:
    _OPENAI_RETRYABLE = (ConnectionError, TimeoutError)
    _OPENAI_RATE_LIMIT = ()


def _normalize_json_punctuation(text: str) -> str:
    """Replace full-width/Chinese punctuation that LLMs sometimes emit in JSON
    structural positions with their ASCII equivalents.

    Safe to apply globally: commas/colons inside JSON string values are valid
    JSON regardless. Chinese quotation marks as string delimiters would break
    parsing, so they are also normalized.
    """
    return (
        text
        .replace("\uff0c", ",")   # ， FULLWIDTH COMMA
        .replace("\uff1a", ":")   # ： FULLWIDTH COLON
        .replace("\u201c", '"')   # " LEFT DOUBLE QUOTATION MARK
        .replace("\u201d", '"')   # " RIGHT DOUBLE QUOTATION MARK
        .replace("\uff1b", ";")   # ； FULLWIDTH SEMICOLON
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Retry after normalizing full-width punctuation (common in Chinese LLM output)
    normalized = _normalize_json_punctuation(text)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass
    # Handle "Extra data": model appended prose after the closing brace.
    # raw_decode parses the first complete JSON value and ignores trailing text.
    # strict=False also accepts raw control characters (literal newlines/tabs)
    # inside strings, which some models (e.g. DeepSeek) occasionally emit.
    for candidate in (normalized, text):
        for decoder in (json.JSONDecoder(), json.JSONDecoder(strict=False)):
            try:
                obj, _ = decoder.raw_decode(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse OpenAI response as JSON: {e}\n"
            f"Response: {text[:500]}"
        )
    raise ValueError("Failed to parse OpenAI response as JSON")


def _token_costs(usage) -> dict:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


class OpenAIReportModel(ReportModel):
    """
    Stage 2: generate a caregiver report from behavioral observations using OpenAI.

    Text-only input (observations JSON). Uses synchronous client; the base class
    async default (asyncio.to_thread) wraps it for the concurrent runner.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _OPENAI_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _OPENAI_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        base_url = config.get("base_url")
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._max_tokens: int | None = config.get("max_tokens", None)
        self._temperature: float | None = config.get("temperature", 0.3)
        # Some providers (e.g. Kimi) don't support json_object response_format
        self._use_json_format: bool = config.get("use_json_format", True)
        # Newer OpenAI models (gpt-5+) use max_completion_tokens instead of max_tokens
        self._use_max_completion_tokens: bool = config.get("use_max_completion_tokens", False)
        # Some providers (e.g. Kimi) reject any temperature other than 1; set skip_temperature: true to omit it
        self._skip_temperature: bool = config.get("skip_temperature", False)
        # report_prompt_module (a module name under benchmark/prompts/) pins a report
        # prompt revision — used by the Stage 2 prompt comparison. Default is
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
        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._max_tokens is not None:
            tokens_key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
            kwargs[tokens_key] = self._max_tokens
        if not self._skip_temperature:
            kwargs["temperature"] = self._temperature
        if self._use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        raw = choice.message.content
        try:
            output = _extract_json(raw)
        except ValueError as e:
            raise ValueError(f"{e}\nfinish_reason={finish_reason}") from None
        return output, raw, _token_costs(response.usage)


class OpenAIJudge(JudgeModel):
    """
    Stage 3: score all model reports for a video using an OpenAI-compatible model as judge.

    Receives anonymized reports (Model A, B, C) and original observations.
    Does not receive the AV-ASD labels.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _OPENAI_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _OPENAI_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        base_url = config.get("base_url")
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._max_tokens: int | None = config.get("max_tokens", None)
        self._temperature: float | None = config.get("temperature", 0.3)
        self._use_json_format: bool = config.get("use_json_format", True)
        self._use_max_completion_tokens: bool = config.get("use_max_completion_tokens", False)
        self._skip_temperature: bool = config.get("skip_temperature", False)

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
        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._max_tokens is not None:
            tokens_key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
            kwargs[tokens_key] = self._max_tokens
        if not self._skip_temperature:
            kwargs["temperature"] = self._temperature
        if self._use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        raw = choice.message.content
        try:
            scores = _extract_json(raw)
        except ValueError as e:
            raise ValueError(f"{e}\nfinish_reason={finish_reason}") from None
        return scores, raw, _token_costs(response.usage)
