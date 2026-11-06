"""
Kimi model adapter for Benchmark A.
Uses Moonshot AI's OpenAI-compatible API with base64-encoded video input.

Video flow
----------
Moonshot does not accept external video URLs. The clip is read from disk,
base64-encoded, and sent inline as a data URI:

    data:video/mp4;base64,{b64}

This is passed via the video_url content type — no Files API upload or
polling required. The runner passes a local clip_path for this adapter
(no s3_bucket set in models.yaml).

Async path
----------
predict_async() overrides the base class default and uses AsyncOpenAI for true
async execution. The file read + encode runs via asyncio.to_thread so it does
not block the event loop.

Exception classification
------------------------
The openai package provides typed HTTP error classes used for retry classification.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re

from benchmark.models.base import BehaviorPrediction, VideoModel
from benchmark.prompts.benchmark_a_baseline import BENCHMARK_A_BASELINE_PROMPT

_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"

# --- Exception classification ---
try:
    import openai as _openai

    _OPENAI_RETRYABLE: tuple[type[Exception], ...] = (
        _openai.APIConnectionError,
        _openai.APITimeoutError,
        _openai.InternalServerError,    # 500
        _openai.RateLimitError,         # 429
    )
    _OPENAI_RATE_LIMIT: tuple[type[Exception], ...] = (
        _openai.RateLimitError,         # 429
    )
except ImportError:
    _OPENAI_RETRYABLE = ()
    _OPENAI_RATE_LIMIT = ()


def _extract_json(text: str) -> dict:
    """
    Parse JSON from a model response.
    Handles responses wrapped in ```json ... ``` markdown blocks.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {e}\nResponse: {text[:500]}")


def _extract_token_costs(response) -> dict:
    """Extract token usage from an OpenAI-compatible response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


def _encode_video(clip_path: str) -> str:
    """Read a video file and return a base64 data URI string."""
    ext = os.path.splitext(clip_path)[1].lstrip(".") or "mp4"
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/{ext};base64,{b64}"


def _build_messages(video_data_uri: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_data_uri}},
                {"type": "text", "text": BENCHMARK_A_BASELINE_PROMPT},
            ],
        }
    ]


class KimiModel(VideoModel):
    """
    Kimi model via Moonshot AI OpenAI-compatible API.

    Video is base64-encoded and sent inline as a data URI — Moonshot does not
    accept external URLs.

    Both sync and async paths are implemented:
      predict()       — synchronous, for the sequential runner and dev runs
      predict_async() — true async using AsyncOpenAI; file encoding runs via
                        asyncio.to_thread to avoid blocking the event loop
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _OPENAI_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _OPENAI_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "MOONSHOT_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        base_url = config.get("base_url", _MOONSHOT_BASE_URL)

        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def predict(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        Base64-encode the clip and run inference.

        Returns:
            (BehaviorPrediction, raw_response_str, token_costs)
        """
        video_data_uri = _encode_video(clip_path)
        messages = _build_messages(video_data_uri)

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
    # Async path — AsyncOpenAI client
    # ------------------------------------------------------------------

    async def predict_async(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        True async inference via AsyncOpenAI.
        File encoding runs in a thread pool to avoid blocking the event loop.
        """
        video_data_uri = await asyncio.to_thread(_encode_video, clip_path)
        messages = _build_messages(video_data_uri)

        response = await self._async_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        raw = response.choices[0].message.content

        parsed = _extract_json(raw)
        prediction = BehaviorPrediction.from_dict(parsed)
        token_costs = _extract_token_costs(response)

        return prediction, raw, token_costs
