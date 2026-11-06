"""
Muse Spark model adapter for Benchmark A.
Uses Meta's Model API (OpenAI SDK-compatible) Responses endpoint for video input.

Video flow
----------
Meta requires the video to be served as `video/mp4`. The CloudFront distribution
used by the Qwen/Pegasus adapters serves clips as `binary/octet-stream`, which the
API rejects:

    unsupported media type at input[0].content[1]: `binary/octet-stream`. Supported: MP4.

So the clip is read from disk, base64-encoded, and sent inline as a data URI
inside an `input_video` block — same approach as the Kimi adapter. The runner
passes a local clip_path for this adapter (no s3_bucket set in models.yaml).

Two size limits apply; only the first can be avoided:

  ~10 MB request   413 payload_too_large, raised only because the API defaults to
                   store=True and cannot persist a request that large. Sending
                   store=False removes the limit entirely — see __init__.

  50 MiB media     400 file_above_max_size (52428800 bytes), a hard cap on the
                   video itself. No request-level flag avoids it; the clip would
                   have to be re-encoded smaller. 12 of the 823 benchmark clips
                   exceed it and are simply unassessable for this model — they
                   fail with a non-retryable 400 and land in failed_clips.

Responses API, not Chat Completions
-----------------------------------
Video input lives on the Responses API (`client.responses.create`), so this
adapter builds an `input=[{type: message, ...}]` payload rather than `messages`.
Meta's endpoint is OpenAI SDK-compatible, so the standard OpenAI/AsyncOpenAI
clients drive it directly with only base_url swapped.

Muse Spark is a reasoning model: output token counts include reasoning tokens,
and the API default reasoning effort ("high") applies unless `reasoning_effort`
is set in models.yaml.

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
from typing import Optional

from benchmark.models.base import BehaviorPrediction, VideoModel
from benchmark.prompts.benchmark_a_baseline import BENCHMARK_A_BASELINE_PROMPT

_META_BASE_URL = "https://api.meta.ai/v1"

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
    """Extract token usage from a Responses API response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def _encode_video(clip_path: str) -> str:
    """Read a video file and return a base64 data URI string."""
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def _build_input(video_data_uri: str) -> list:
    """Build a Responses API input list with a video block and the prompt."""
    return [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": BENCHMARK_A_BASELINE_PROMPT},
                {"type": "input_video", "video_url": video_data_uri},
            ],
        }
    ]


class MuseSparkModel(VideoModel):
    """
    Muse Spark via Meta Model API (Responses endpoint, native video input).

    Video is base64-encoded and sent inline as a data URI — see module docstring
    for why a CloudFront URL is not used.

    Both sync and async paths are implemented:
      predict()       — synchronous, for the sequential runner and dev runs
      predict_async() — true async using AsyncOpenAI; file encoding runs via
                        asyncio.to_thread to avoid blocking the event loop
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _OPENAI_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _OPENAI_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "META_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        base_url = config.get("base_url", _META_BASE_URL)

        self._temperature: Optional[float] = config.get("temperature")
        self._reasoning_effort: Optional[str] = config.get("reasoning_effort")
        self._max_output_tokens: Optional[int] = config.get("max_tokens")
        self._use_json_format: bool = config.get("use_json_format", True)
        # store=False is required, not an optimization: with the API default
        # (store=True) Meta tries to persist the whole inlined request and rejects
        # anything over ~10 MB with 413 payload_too_large. Since a base64 video is
        # 4/3 the file size, that trips at ~7.5 MB of clip — 104 of the 823
        # benchmark clips. The benchmark never reads back stored responses, so
        # there is nothing to lose by disabling it.
        self._store: bool = config.get("store", False)

        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _request_kwargs(self, video_data_uri: str) -> dict:
        """Assemble create() kwargs, omitting any option not set in models.yaml."""
        kwargs: dict = {
            "model": self.model_id,
            "input": _build_input(video_data_uri),
            "store": self._store,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        if self._use_json_format:
            kwargs["text"] = {"format": {"type": "json_object"}}
        return kwargs

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

        response = self._client.responses.create(**self._request_kwargs(video_data_uri))
        raw = response.output_text

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

        response = await self._async_client.responses.create(
            **self._request_kwargs(video_data_uri)
        )
        raw = response.output_text

        parsed = _extract_json(raw)
        prediction = BehaviorPrediction.from_dict(parsed)
        token_costs = _extract_token_costs(response)

        return prediction, raw, token_costs
