"""
Gemini single-stage report model (Table 5, matched single-stage condition).

Uploads a video clip to the Gemini Files API and generates a caregiver report in a
single call, with no intermediate observation record.

Reuses the same upload/poll/delete pattern as GeminiObservationExtractor
and GeminiModel, but returns a report dict instead of observations or
BehaviorPrediction.
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

from benchmark.models.base_c import VideoReportModel
from benchmark.prompts.benchmark_e2e_matched_single_stage import MATCHED_SINGLE_STAGE_PROMPT

_DEFAULT_POLL_INTERVAL = 5
_DEFAULT_POLL_TIMEOUT = 180

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
    _GOOGLE_RETRYABLE = ()
    _GOOGLE_RATE_LIMIT = ()


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Gemini C baseline response as JSON: {e}\n"
            f"Response: {text[:500]}"
        )


def _token_costs(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


class GeminiVideoReportModel(VideoReportModel):
    """
    Gemini single-stage report model (Table 5).

    Uploads the clip via Files API, generates a caregiver report in one call
    using the matched single-stage prompt. Both sync and async paths implemented.
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
        self._poll_interval: int = config.get("poll_interval_s", _DEFAULT_POLL_INTERVAL)
        self._poll_timeout: int = config.get("poll_timeout_s", _DEFAULT_POLL_TIMEOUT)
        self._temperature: float = config.get("temperature", 0.3)
        # prompt_module + prompt_attr select the single-stage prompt file
        # (Table 5, matched single-stage condition).
        prompt_module = config.get("prompt_module")
        self._prompt: str = (
            getattr(importlib.import_module(f"benchmark.prompts.{prompt_module}"),
                    config.get("prompt_attr", "MATCHED_SINGLE_STAGE_PROMPT"))
            if prompt_module else MATCHED_SINGLE_STAGE_PROMPT
        )

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def _upload_and_wait(self, clip_path: str) -> types.File:
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
                f"Gemini file upload failed for '{clip_path}': "
                f"state={video_file.state.name}"
            )
        return video_file

    def predict(self, clip_path: str) -> tuple[dict, str, dict]:
        video_file: Optional[types.File] = None
        try:
            video_file = self._upload_and_wait(clip_path)
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=[video_file, self._prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=self._temperature,
                ),
            )
            raw = response.text
            output = _extract_json(raw)
            return output, raw, _token_costs(response)
        finally:
            if video_file is not None:
                try:
                    self._client.files.delete(name=video_file.name)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def _upload_and_wait_async(self, clip_path: str) -> types.File:
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
                f"Gemini file upload failed for '{clip_path}': "
                f"state={video_file.state.name}"
            )
        return video_file

    async def predict_async(self, clip_path: str) -> tuple[dict, str, dict]:
        video_file: Optional[types.File] = None
        try:
            video_file = await self._upload_and_wait_async(clip_path)
            response = await self._client.aio.models.generate_content(
                model=self.model_id,
                contents=[video_file, self._prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=self._temperature,
                ),
            )
            raw = response.text
            output = _extract_json(raw)
            return output, raw, _token_costs(response)
        finally:
            if video_file is not None:
                try:
                    await self._client.aio.files.delete(name=video_file.name)
                except Exception:
                    pass
