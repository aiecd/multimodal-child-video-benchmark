"""
Behavior-mention extractor for Table 5 (OpenAI-compatible, text only).

Same client setup as OpenAICJudge; one call per report returns, for each of the nine
AV-ASD behaviors, whether the report says the child showed it.
"""

from __future__ import annotations

import asyncio
import json

from benchmark.models.openai_c_judge import OpenAICJudge, _extract_json, _token_costs
from benchmark.prompts.benchmark_e2e_behavior_mentions import NINE_BEHAVIORS, build_mentions_prompt


class OpenAIBehaviorMentions(OpenAICJudge):

    def extract_mentions(self, report: dict) -> tuple[dict, str, dict]:
        prompt = build_mentions_prompt(json.dumps(report, ensure_ascii=False, indent=2))
        kwargs: dict = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}]}
        if self._max_tokens is not None:
            key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
            kwargs[key] = self._max_tokens
        if not self._skip_temperature:
            kwargs["temperature"] = self._temperature
        if self._use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        out = _extract_json(raw)
        missing = [b for b in NINE_BEHAVIORS
                   if not isinstance(out.get(b), dict) or not isinstance(out[b].get("mentioned"), bool)]
        if missing:
            raise ValueError(f"mention output missing or non-boolean for: {missing}")
        return {b: out[b] for b in NINE_BEHAVIORS}, raw, _token_costs(response.usage)

    async def extract_mentions_async(self, report: dict) -> tuple[dict, str, dict]:
        return await asyncio.to_thread(self.extract_mentions, report)
