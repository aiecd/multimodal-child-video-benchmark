"""
OpenAI-compatible judge for Benchmark C — Stage 3.

Scores Safety, Clinical Actionability, and Relational Empathy only.
Does not receive the observation records; reports are scored on their own
merit. The observations argument is accepted for interface compatibility but unused.

Works with any OpenAI-compatible provider.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI

from benchmark.models.base_text import JudgeModel
from benchmark.prompts.benchmark_c_judge import build_judge_prompt

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


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse OpenAI C judge response as JSON: {e}\nResponse: {text[:500]}"
        )


def _token_costs(usage) -> dict:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


class OpenAICJudge(JudgeModel):
    """
    OpenAI-compatible judge for Benchmark C.

    Scores Safety + Clinical Actionability + Relational Empathy.
    observations argument is accepted for interface compatibility but unused.
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
        self._max_tokens: Optional[int] = config.get("max_tokens")
        self._temperature: Optional[float] = config.get("temperature", 0.3)
        self._use_json_format: bool = config.get("use_json_format", True)
        self._use_max_completion_tokens: bool = config.get("use_max_completion_tokens", False)
        self._skip_temperature: bool = config.get("skip_temperature", False)

    def score(
        self,
        observations: dict,   # unused in C — kept for JudgeModel interface compatibility
        reports: dict[str, dict],
        language: str = "en", # unused in C — kept for JudgeModel interface compatibility
    ) -> tuple[dict, str, dict]:
        prompt = build_judge_prompt(reports)

        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._max_tokens is not None:
            key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
            kwargs[key] = self._max_tokens
        if not self._skip_temperature:
            kwargs["temperature"] = self._temperature
        if self._use_json_format:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        scores = _extract_json(raw)
        return scores, raw, _token_costs(response.usage)
