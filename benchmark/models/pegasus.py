"""
TwelveLabs Pegasus model adapter for Benchmark A.
Uses the synchronous /analyze endpoint — no index, no ingestion task, no polling.

Video flow
----------
Pegasus 1.5 accepts a video inline on the analyze call, so the indexed workflow
(create index → upload task → poll → generate) is not used here. Note that the
/indexes endpoint only accepts marengo3.0 / pegasus1.2 — pegasus1.5 is reachable
only through /analyze.

Two delivery modes, chosen by what the runner hands to predict():
    HTTPS URL   → {"type": "url", "url": ...}            (s3_bucket set in models.yaml)
    local path  → {"type": "base64_string", ...}          (no s3_bucket)

Not an OpenAI-compatible API
----------------------------
TwelveLabs uses its own REST shape (`x-api-key` header, `model_name`, a `video`
object, and a flat `{"data": "..."}` response), so this adapter speaks raw HTTP
via httpx rather than reusing the OpenAI client.

max_tokens is constrained by the API to 512..98304 for this model and mode; the
default below sits at the low end since the benchmark expects a small JSON object.

Async path
----------
predict_async() overrides the base class default and uses httpx.AsyncClient for
true async execution. Base64 encoding runs via asyncio.to_thread so it does not
block the event loop.

Exception classification
------------------------
HTTP status codes are mapped onto the two module-level exception types below so
the shared retry helper can classify 429s separately from 5xx errors.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Optional

import httpx

from benchmark.models.base import BehaviorPrediction, VideoModel
from benchmark.prompts.benchmark_a_baseline import BENCHMARK_A_BASELINE_PROMPT

_TWELVELABS_BASE_URL = "https://api.twelvelabs.io/v1.3"
_DEFAULT_MAX_TOKENS = 512      # API floor for pegasus1.5 on /analyze
_DEFAULT_TIMEOUT_S = 300.0     # video analysis is slow; well above the default 5s


class PegasusServerError(Exception):
    """5xx from the TwelveLabs API — transient, worth retrying."""


class PegasusRateLimitError(PegasusServerError):
    """429 from the TwelveLabs API — triggers the cross-worker rate-limit pause."""


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


def _extract_token_costs(payload: dict) -> dict:
    """Extract token usage from an /analyze response body."""
    usage = payload.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def _video_block(clip_path: str) -> dict:
    """
    Build the `video` object for /analyze.
    An HTTPS clip_path is passed through as a URL; anything else is read from
    disk and base64-encoded.
    """
    if clip_path.startswith(("http://", "https://")):
        return {"type": "url", "url": clip_path}
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "base64_string", "base64_string": b64}


def _raise_for_status(response: httpx.Response) -> None:
    """
    Map HTTP errors onto the module exception types so retry can classify them.
    4xx other than 429 are permanent (bad request, auth, unsupported model) and
    raise a plain RuntimeError so the runner fails the clip without retrying.
    """
    if response.status_code < 400:
        return
    body = response.text[:500]
    if response.status_code == 429:
        raise PegasusRateLimitError(f"429 from TwelveLabs /analyze: {body}")
    if response.status_code >= 500:
        raise PegasusServerError(f"{response.status_code} from TwelveLabs /analyze: {body}")
    raise RuntimeError(f"{response.status_code} from TwelveLabs /analyze: {body}")


class PegasusModel(VideoModel):
    """
    TwelveLabs Pegasus via the synchronous /analyze endpoint.

    Both sync and async paths are implemented:
      predict()       — synchronous httpx client, for the sequential runner
      predict_async() — httpx.AsyncClient; base64 encoding (URL mode aside) runs
                        via asyncio.to_thread to avoid blocking the event loop
    """

    RETRYABLE_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        PegasusServerError,          # includes PegasusRateLimitError
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
    )
    RATE_LIMIT_EXCEPTIONS = (PegasusRateLimitError,)

    def __init__(self, config: dict):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "TWELVELABS_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Add it to your .env file or export it before running."
            )
        base_url = config.get("base_url", _TWELVELABS_BASE_URL).rstrip("/")
        self._analyze_url = f"{base_url}/analyze"
        self._headers = {"x-api-key": api_key, "Content-Type": "application/json"}

        self._temperature: Optional[float] = config.get("temperature")
        self._max_tokens: int = config.get("max_tokens", _DEFAULT_MAX_TOKENS)
        self._timeout_s: float = config.get("request_timeout_s", _DEFAULT_TIMEOUT_S)

    def _payload(self, video: dict) -> dict:
        """Assemble the /analyze request body."""
        payload: dict = {
            "model_name": self.model_id,
            "video": video,
            "prompt": BENCHMARK_A_BASELINE_PROMPT,
            "stream": False,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        return payload

    @staticmethod
    def _parse(payload: dict) -> tuple[BehaviorPrediction, str, dict]:
        raw = payload.get("data") or ""
        parsed = _extract_json(raw)
        prediction = BehaviorPrediction.from_dict(parsed)
        return prediction, raw, _extract_token_costs(payload)

    # ------------------------------------------------------------------
    # Synchronous path
    # ------------------------------------------------------------------

    def predict(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        Run inference on a clip URL or local file.

        Returns:
            (BehaviorPrediction, raw_response_str, token_costs)
        """
        video = _video_block(clip_path)

        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(
                self._analyze_url, headers=self._headers, json=self._payload(video)
            )
        _raise_for_status(response)

        return self._parse(response.json())

    # ------------------------------------------------------------------
    # Async path — httpx.AsyncClient
    # ------------------------------------------------------------------

    async def predict_async(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        True async inference via httpx.AsyncClient.
        Base64 encoding (local-file mode) runs in a thread pool so it does not
        block the event loop.
        """
        video = await asyncio.to_thread(_video_block, clip_path)

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                self._analyze_url, headers=self._headers, json=self._payload(video)
            )
        _raise_for_status(response)

        return self._parse(response.json())
