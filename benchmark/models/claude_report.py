"""
Claude adapters for Benchmark B — Stage 2 (report) and Stage 3 (judge).

Two classes:
  ClaudeReportModel — Stage 2: generates a caregiver report from observations
  ClaudeJudge       — Stage 3: scores all model reports for a video

Both use the Anthropic SDK.
"""

from __future__ import annotations

import json
import os
import re

import anthropic

from benchmark.models.base_text import JudgeModel, ReportModel
from benchmark.prompts.benchmark_b_judge import build_judge_prompt
from benchmark.prompts.benchmark_b_report import BENCHMARK_B_REPORT_PROMPT, LANGUAGE_INSTRUCTIONS

try:
    _ANTHROPIC_RETRYABLE: tuple[type[Exception], ...] = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
        anthropic.RateLimitError,
    )
    _ANTHROPIC_RATE_LIMIT: tuple[type[Exception], ...] = (
        anthropic.RateLimitError,
    )
except AttributeError:
    _ANTHROPIC_RETRYABLE = (ConnectionError, TimeoutError)
    _ANTHROPIC_RATE_LIMIT = ()


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Claude response as JSON: {e}\n"
            f"Response: {text[:500]}"
        )


def _response_text(message) -> str:
    """Join the text blocks; thinking blocks come first on models with thinking always on."""
    if message.stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        raise ValueError(f"Claude declined the request (refusal category: {category})")
    if message.stop_reason == "max_tokens":
        raise ValueError("Claude response hit max_tokens before finishing; raise max_tokens")
    return "".join(block.text for block in message.content if block.type == "text")


def _create_kwargs(config: dict) -> dict:
    """Optional request parameters taken from the model's config entry."""
    effort = config.get("effort")
    return {"output_config": {"effort": effort}} if effort else {}


def _make_client(config: dict) -> anthropic.Anthropic:
    """
    Build the Anthropic client from the model's config entry.

    An API key that is not scoped to a workspace is rejected unless every request
    names a workspace in the anthropic-workspace-id header. Set the workspace ID in
    the env var named by workspace_id_env (default ANTHROPIC_WORKSPACE_ID) to send
    it; leave it unset for a workspace-scoped key.
    """
    api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            f"Add it to your .env file or export it before running."
        )
    workspace_id = os.environ.get(config.get("workspace_id_env", "ANTHROPIC_WORKSPACE_ID"))
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic.Anthropic(api_key=api_key, default_headers=headers)


def _create_streamed(client, model_id: str, max_tokens: int, prompt: str, create_kwargs: dict):
    """
    Send one request over the streaming endpoint and return the final message.

    Judges need a max_tokens well above what a non-streaming request allows: thinking
    is always on and counts toward max_tokens, and the judge schema is long. The SDK
    refuses non-streaming requests whose max_tokens implies a >10 min response, so
    judges stream and collect the final message instead.
    """
    with client.messages.stream(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **create_kwargs,
    ) as stream:
        return stream.get_final_message()


def _token_costs(usage) -> dict:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


class ClaudeReportModel(ReportModel):
    """
    Stage 2: generate a caregiver report from behavioral observations using Claude.

    Uses the synchronous Anthropic client. The base class async default
    (asyncio.to_thread) is sufficient since the Anthropic SDK handles
    the HTTP connection pool internally.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _ANTHROPIC_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _ANTHROPIC_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = _make_client(config)
        self._max_tokens: int = config.get("max_tokens", 16000)  # required by Anthropic API
        self._create_kwargs = _create_kwargs(config)

    def generate_report(self, observations: dict, language: str = "en") -> tuple[dict, str, dict]:
        """
        Generate a structured caregiver report from behavioral observations.

        Returns:
            (output_dict, raw_response_str, token_costs)
            output_dict: {"report": {...}}
        """
        prompt = BENCHMARK_B_REPORT_PROMPT.format(
            language_instruction=LANGUAGE_INSTRUCTIONS.get(language, ""),
            gemini_observations=json.dumps(observations, ensure_ascii=False, indent=2),
        )
        message = self._client.messages.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._create_kwargs,
        )
        raw = _response_text(message)
        output = _extract_json(raw)
        return output, raw, _token_costs(message.usage)


class ClaudeJudge(JudgeModel):
    """
    Stage 3: score all model reports for a video using Claude as judge.

    Receives anonymized reports (Model A, B, C...) and original observations.
    Does not receive the AV-ASD labels.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _ANTHROPIC_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _ANTHROPIC_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = _make_client(config)
        self._max_tokens: int = config.get("max_tokens", 16000)  # required by Anthropic API
        self._create_kwargs = _create_kwargs(config)

    def score(
        self,
        observations: dict,
        reports: dict[str, dict],
        language: str = "en",
    ) -> tuple[dict, str, dict]:
        """
        Score all model reports. Reports are already anonymized by the runner.

        Args:
            observations: Stage 1 observations dict.
            reports: {"Model A": report_output, "Model B": report_output, ...}

        Returns:
            (scores_dict, raw_response_str, token_costs)
        """
        reports_text = json.dumps(reports, ensure_ascii=False, indent=2)
        observations_text = json.dumps(observations, ensure_ascii=False, indent=2)
        prompt = build_judge_prompt(language, observations_text, reports_text)
        message = _create_streamed(
            self._client, self.model_id, self._max_tokens, prompt, self._create_kwargs
        )
        raw = _response_text(message)
        scores = _extract_json(raw)
        return scores, raw, _token_costs(message.usage)
