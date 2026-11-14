"""
Claude judge for Benchmark C-style scoring (Safety, Clinical Actionability,
Relational Empathy).

Same rubric as OpenAICJudge: reports are scored on their own merit
and the observations argument is accepted for interface compatibility but unused.
Request handling (streaming, effort, refusal/max_tokens checks) is shared with
ClaudeJudge in claude_report.py.
"""

from __future__ import annotations

from benchmark.models.base_text import JudgeModel
from benchmark.models.claude_report import (
    _ANTHROPIC_RATE_LIMIT,
    _ANTHROPIC_RETRYABLE,
    _create_kwargs,
    _create_streamed,
    _extract_json,
    _make_client,
    _response_text,
    _token_costs,
)
from benchmark.prompts.benchmark_c_judge import build_judge_prompt


class ClaudeCJudge(JudgeModel):
    """
    Claude judge for C-style scoring.

    Scores Safety + Clinical Actionability + Relational Empathy.
    observations argument is accepted for interface compatibility but unused.
    """

    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) + _ANTHROPIC_RETRYABLE
    RATE_LIMIT_EXCEPTIONS = _ANTHROPIC_RATE_LIMIT

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = _make_client(config)
        self._max_tokens: int = config.get("max_tokens", 16000)
        self._create_kwargs = _create_kwargs(config)

    def score(
        self,
        observations: dict,   # unused — kept for JudgeModel interface compatibility
        reports: dict[str, dict],
        language: str = "en", # unused — the C rubric is English-only
    ) -> tuple[dict, str, dict]:
        prompt = build_judge_prompt(reports)
        message = _create_streamed(
            self._client, self.model_id, self._max_tokens, prompt, self._create_kwargs
        )
        raw = _response_text(message)
        scores = _extract_json(raw)
        return scores, raw, _token_costs(message.usage)
