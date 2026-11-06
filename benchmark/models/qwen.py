"""
Qwen model adapter for Benchmark A.
Uses DashScope's OpenAI-compatible API for video understanding.

Video input
-----------
Qwen accepts video via URL in the OpenAI-compatible message format:
    {"type": "video_url", "video_url": {"url": "https://..."}}

When s3_bucket + cloudfront_domain are set in models.yaml, the runner passes
a CloudFront URL as clip_path — no local file handling needed in this adapter.

Async path
----------
predict_async() overrides the base class and uses AsyncOpenAI for true async
execution — no thread pool needed.

Exception classification
------------------------
Uses standard openai package exceptions for retry classification.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from benchmark.models.base import BehaviorPrediction, VideoModel
from benchmark.prompts.benchmark_a_baseline import BENCHMARK_A_BASELINE_PROMPT

_DEFAULT_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"

# --- Exception classification ---
try:
    import openai as _openai

    _OPENAI_RETRYABLE: tuple[type[Exception], ...] = (
        _openai.APIConnectionError,
        _openai.APITimeoutError,
        _openai.InternalServerError,
        _openai.RateLimitError,
    )
    _OPENAI_RATE_LIMIT: tuple[type[Exception], ...] = (
        _openai.RateLimitError,
    )
except ImportError:
    _OPENAI_RETRYABLE = ()
    _OPENAI_RATE_LIMIT = ()


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {e}\nResponse: {text[:500]}")


def _extract_token_costs(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


def _build_messages(clip_path: str) -> list:
    """
    Build OpenAI-compatible message list.
    clip_path is expected to be an HTTPS URL (CloudFront or other).
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": clip_path}},
                {"type": "text", "text": BENCHMARK_A_BASELINE_PROMPT},
            ],
        }
    ]


class QwenModel(VideoModel):
    """
    Qwen model via DashScope's OpenAI-compatible API.

    Both sync and async paths are implemented:
      predict()       — synchronous, for the sequential runner and dev runs
      predict_async() — true async using AsyncOpenAI
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _OPENAI_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _OPENAI_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "ALICLOUD_US_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        base_url = config.get("base_url", _DEFAULT_BASE_URL)

        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def predict(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        messages = _build_messages(clip_path)

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        raw = response.choices[0].message.content

        parsed = _extract_json(raw)
        prediction = BehaviorPrediction.from_dict(parsed)
        token_costs = _extract_token_costs(response)

        return prediction, raw, token_costs

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def predict_async(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        messages = _build_messages(clip_path)

        response = await self._async_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        raw = response.choices[0].message.content

        parsed = _extract_json(raw)
        prediction = BehaviorPrediction.from_dict(parsed)
        token_costs = _extract_token_costs(response)

        return prediction, raw, token_costs
