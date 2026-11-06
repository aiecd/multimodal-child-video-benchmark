"""
Gemini observation extractor for Benchmark B — Stage 1.

Uploads a clip to the Google GenAI Files API, waits for processing,
then calls generate_content with the observation prompt to extract structured
behavioral observations.

Same file upload / polling pattern as GeminiModel in gemini.py.
Returns a raw observations dict instead of BehaviorPrediction.
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

from benchmark.models.base_text import ObservationExtractor
from benchmark.prompts.benchmark_b_observation import BENCHMARK_B_OBSERVATION_PROMPT

_DEFAULT_POLL_INTERVAL = 3
_DEFAULT_POLL_TIMEOUT = 180

try:
    from google.api_core import exceptions as _gac

    _GOOGLE_RETRYABLE: tuple[type[Exception], ...] = (
        _gac.ServiceUnavailable,
        _gac.InternalServerError,
        _gac.ResourceExhausted,
        _gac.DeadlineExceeded,
    )
    _GOOGLE_RATE_LIMIT: tuple[type[Exception], ...] = (
        _gac.ResourceExhausted,
    )
except ImportError:
    _GOOGLE_RETRYABLE = ()
    _GOOGLE_RATE_LIMIT = ()


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Handle "Extra data": Gemini sometimes appends prose after the closing brace.
    # raw_decode parses the first complete JSON value and ignores trailing text.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Re-raise with the original json.loads error message for clarity.
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Gemini observation response as JSON: {e}\n"
            f"Response: {text[:500]}"
        )
    raise ValueError("Failed to parse Gemini observation response as JSON")


def _extract_token_costs(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


class GeminiObservationExtractor(ObservationExtractor):
    """
    Stage 1 extractor: video → structured behavioral observations.

    Uses the Gemini Files API (native video input). Both sync and async paths
    are implemented following the same pattern as GeminiModel in gemini.py.
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
        self._temperature: Optional[float] = config.get("temperature", None)
        # prompt_module (a module name under benchmark/prompts/) selects another
        # observation prompt file; the default is the original observation prompt.
        prompt_module = config.get("prompt_module")
        self._prompt: str = (
            importlib.import_module(f"benchmark.prompts.{prompt_module}").BENCHMARK_B_OBSERVATION_PROMPT
            if prompt_module else BENCHMARK_B_OBSERVATION_PROMPT
        )

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def _upload_and_wait(self, video_path: str) -> types.File:
        video_file = self._client.files.upload(file=video_path)
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
                f"Gemini file upload failed for '{video_path}': "
                f"state={video_file.state.name}"
            )
        return video_file

    def extract(self, video_path: str) -> tuple[dict, str, dict]:
        """
        Upload video to Gemini Files API, extract behavioral observations.
        Deletes the uploaded file after inference.

        Returns:
            (observations_dict, raw_response_str, token_costs)
        """
        video_file: Optional[types.File] = None
        try:
            video_file = self._upload_and_wait(video_path)
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=[video_file, self._prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=self._temperature,
                ),
            )
            raw = response.text
            observations = _extract_json(raw)
            token_costs = _extract_token_costs(response)
            return observations, raw, token_costs
        finally:
            if video_file is not None:
                try:
                    self._client.files.delete(name=video_file.name)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def _upload_and_wait_async(self, video_path: str) -> types.File:
        video_file = await self._client.aio.files.upload(file=video_path)
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
                f"Gemini file upload failed for '{video_path}': "
                f"state={video_file.state.name}"
            )
        return video_file

    async def extract_async(self, video_path: str) -> tuple[dict, str, dict]:
        """True async via client.aio — no thread pool overhead."""
        video_file: Optional[types.File] = None
        try:
            video_file = await self._upload_and_wait_async(video_path)
            response = await self._client.aio.models.generate_content(
                model=self.model_id,
                contents=[video_file, self._prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=self._temperature,
                ),
            )
            raw = response.text
            observations = _extract_json(raw)
            token_costs = _extract_token_costs(response)
            return observations, raw, token_costs
        finally:
            if video_file is not None:
                try:
                    await self._client.aio.files.delete(name=video_file.name)
                except Exception:
                    pass
