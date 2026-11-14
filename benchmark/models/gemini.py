"""
Gemini model adapter for Benchmark A.
Uses the Google GenAI Files API for native video understanding.

Async path
----------
predict_async() overrides the base class default and uses the Gemini async
client (client.aio.*) for true async execution — no thread pool needed.
The synchronous predict() is kept intact for the sequential path and dev runs.

Exception classification
------------------------
google.api_core.exceptions (google-api-core) provides the typed HTTP error
classes. If it is not installed, these errors are not retried.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import time
from typing import Optional

from google import genai
from google.genai import types

from benchmark.models.base import BehaviorPrediction, LABEL_KEYS, VideoModel
from benchmark.prompts.benchmark_a_baseline import BENCHMARK_A_BASELINE_PROMPT
from benchmark.prompts.benchmark_a_kidooo import BENCHMARK_A_KIDOOO_PROMPT

_DEFAULT_POLL_INTERVAL = 3
_DEFAULT_POLL_TIMEOUT = 120

_PROMPTS = {
    "kidooo": BENCHMARK_A_KIDOOO_PROMPT,
    "baseline": BENCHMARK_A_BASELINE_PROMPT,
}

# Constant each prompt module exposes per prompt_style. A model entry can set
# prompt_module (a module name under benchmark/prompts/) to use another prompt file
# (Table 4 panel A).
_PROMPT_ATTRS = {
    "kidooo": "BENCHMARK_A_KIDOOO_PROMPT",
    "baseline": "BENCHMARK_A_BASELINE_PROMPT",
}

# --- Exception classification ---
# Typed HTTP errors from google-api-core; without it, these errors are not retried.
try:
    from google.api_core import exceptions as _gac

    _GOOGLE_RETRYABLE: tuple[type[Exception], ...] = (
        _gac.ServiceUnavailable,    # 503
        _gac.InternalServerError,   # 500
        _gac.ResourceExhausted,     # 429 — rate limit
        _gac.DeadlineExceeded,      # 504
    )
    _GOOGLE_RATE_LIMIT: tuple[type[Exception], ...] = (
        _gac.ResourceExhausted,     # 429
    )
except ImportError:
    _GOOGLE_RETRYABLE = ()
    _GOOGLE_RATE_LIMIT = ()


def _raise_if_blocked(response) -> str:
    """
    Return response.text, or raise a diagnostic error when Gemini produced none.

    response.text is None whenever the request yields no usable candidate — most
    often because the safety filter blocked the input outright
    (prompt_feedback.block_reason = PROHIBITED_CONTENT), in which case there are
    no candidates at all. Parsing straight through would fail deep inside
    _extract_json as "'NoneType' object has no attribute 'strip'", which says
    nothing about the real cause.

    Raised as RuntimeError, which is deliberately absent from
    RETRYABLE_EXCEPTIONS: a content block is a permanent property of the clip,
    so retrying only burns quota.
    """
    text = getattr(response, "text", None)
    if text is not None:
        return text

    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason is not None:
        message = getattr(feedback, "block_reason_message", None)
        raise RuntimeError(
            f"Gemini returned no output: prompt blocked "
            f"({getattr(block_reason, 'name', block_reason)})"
            + (f" — {message}" if message else "")
        )

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        reasons = [
            getattr(getattr(c, "finish_reason", None), "name", getattr(c, "finish_reason", None))
            for c in candidates
        ]
        raise RuntimeError(
            f"Gemini returned no output text; candidate finish_reason(s): {reasons}"
        )

    raise RuntimeError("Gemini returned no output text and no candidates.")


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


def _extract_metrics(parsed: dict, prompt_style: str) -> dict:
    """
    Extract the label dict from the parsed JSON response.
    - kidooo: labels are nested under 'asd_behavioral_metrics'
    - baseline: labels are at the top level (flat JSON)
    """
    if prompt_style == "kidooo":
        return parsed.get("asd_behavioral_metrics", {})
    return parsed


def _extract_token_costs(response) -> dict:
    """Extract token usage from a Gemini response object."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


class GeminiModel(VideoModel):
    """
    Gemini model via the Files API (native video input).

    Both sync and async paths are implemented:
      predict()       — synchronous, for the sequential runner and dev runs
      predict_async() — true async using client.aio, for the concurrent runner
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
        self._poll_interval: int = config.get("file_poll_interval_s", _DEFAULT_POLL_INTERVAL)
        self._poll_timeout: int = config.get("file_poll_timeout_s", _DEFAULT_POLL_TIMEOUT)
        self._temperature: Optional[float] = config.get("temperature")

        if self.prompt_style not in _PROMPTS:
            raise ValueError(
                f"Unknown prompt_style '{self.prompt_style}' for model '{self.model_id}'. "
                f"Valid options: {list(_PROMPTS.keys())}"
            )
        prompt_module = config.get("prompt_module")
        if prompt_module:
            module = importlib.import_module(f"benchmark.prompts.{prompt_module}")
            self._prompt = getattr(module, _PROMPT_ATTRS[self.prompt_style])
        else:
            self._prompt = _PROMPTS[self.prompt_style]

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def _upload_and_wait(self, clip_path: str) -> types.File:
        """Upload video to Files API and poll (blocking) until ACTIVE."""
        video_file = self._client.files.upload(file=clip_path)

        deadline = time.monotonic() + self._poll_timeout
        while video_file.state == types.FileState.PROCESSING:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Gemini file '{video_file.name}' stuck in PROCESSING "
                    f"after {self._poll_timeout}s"
                )
            time.sleep(self._poll_interval)
            video_file = self._client.files.get(name=video_file.name)

        if video_file.state == types.FileState.FAILED:
            raise RuntimeError(
                f"Gemini file upload failed for '{clip_path}': state={video_file.state.name}"
            )
        return video_file

    def predict(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        Upload clip to Gemini Files API, run inference, parse behavioral metrics.
        Deletes the uploaded file from Google's servers after inference.

        Returns:
            (BehaviorPrediction, raw_response_str, token_costs)
        """
        video_file: Optional[types.File] = None
        try:
            video_file = self._upload_and_wait(clip_path)

            gen_config: dict = {"response_mime_type": "application/json"}
            if self._temperature is not None:
                gen_config["temperature"] = self._temperature

            response = self._client.models.generate_content(
                model=self.model_id,
                contents=[video_file, self._prompt],
                config=types.GenerateContentConfig(**gen_config),
            )
            raw = _raise_if_blocked(response)

            parsed = _extract_json(raw)
            metrics_dict = _extract_metrics(parsed, self.prompt_style)
            prediction = BehaviorPrediction.from_dict(metrics_dict)
            token_costs = _extract_token_costs(response)

            return prediction, raw, token_costs

        finally:
            if video_file is not None:
                try:
                    self._client.files.delete(name=video_file.name)
                except Exception:
                    pass  # Non-fatal; Google auto-expires files after 48h

    # ------------------------------------------------------------------
    # Async path — true async using client.aio, no thread pool needed
    # ------------------------------------------------------------------

    async def _upload_and_wait_async(self, clip_path: str) -> types.File:
        """Upload video to Files API and poll (non-blocking) until ACTIVE."""
        video_file = await self._client.aio.files.upload(file=clip_path)

        deadline = time.monotonic() + self._poll_timeout
        while video_file.state == types.FileState.PROCESSING:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Gemini file '{video_file.name}' stuck in PROCESSING "
                    f"after {self._poll_timeout}s"
                )
            await asyncio.sleep(self._poll_interval)
            video_file = await self._client.aio.files.get(name=video_file.name)

        if video_file.state == types.FileState.FAILED:
            raise RuntimeError(
                f"Gemini file upload failed for '{clip_path}': state={video_file.state.name}"
            )
        return video_file

    async def predict_async(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        True async inference via client.aio. Overrides the base class thread-wrap
        default. The polling loop uses await asyncio.sleep() so it never blocks
        the event loop while waiting for a video to become ACTIVE.
        """
        video_file: Optional[types.File] = None
        try:
            video_file = await self._upload_and_wait_async(clip_path)

            gen_config: dict = {"response_mime_type": "application/json"}
            if self._temperature is not None:
                gen_config["temperature"] = self._temperature

            response = await self._client.aio.models.generate_content(
                model=self.model_id,
                contents=[video_file, self._prompt],
                config=types.GenerateContentConfig(**gen_config),
            )
            raw = _raise_if_blocked(response)

            parsed = _extract_json(raw)
            metrics_dict = _extract_metrics(parsed, self.prompt_style)
            prediction = BehaviorPrediction.from_dict(metrics_dict)
            token_costs = _extract_token_costs(response)

            return prediction, raw, token_costs

        finally:
            if video_file is not None:
                try:
                    await self._client.aio.files.delete(name=video_file.name)
                except Exception:
                    pass  # Non-fatal; Google auto-expires files after 48h
