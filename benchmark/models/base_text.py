"""
Abstract base classes for benchmark_b text-based pipeline models.

Three roles in the pipeline:
  ObservationExtractor  — Stage 1: video → structured observations (Gemini)
  ReportModel           — Stage 2: observations → caregiver-facing structured report
  JudgeModel            — Stage 3: observations + all reports → per-model scores

Scalability contract mirrors VideoModel in base.py:
  - Implement the sync method (extract / generate_report / score)
  - Declare RETRYABLE_EXCEPTIONS and RATE_LIMIT_EXCEPTIONS
  - Optionally override the _async variant for true async (no thread pool)
  - Store no mutable per-request state on self (thread-safety for to_thread)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class ObservationExtractor(ABC):
    """
    Stage 1: extract structured behavioral observations from a video file.
    Implemented by Gemini (native video understanding).
    """

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    RATE_LIMIT_EXCEPTIONS: tuple[type[Exception], ...] = ()

    def __init__(self, config: dict):
        self.config = config
        self.model_id: str = config["model_id"]
        self.display_name: str = config.get("display_name", self.model_id)

    @abstractmethod
    def extract(self, video_path: str) -> tuple[dict, str, dict]:
        """
        Extract structured observations from a video file.

        Returns:
            (observations_dict, raw_response_str, token_costs)
            observations_dict: structured JSON matching the Stage 1 schema.
            raw_response_str:  full model output, stored for debugging.
            token_costs:       {"input_tokens": int, "output_tokens": int} or {}.
        """
        ...

    async def extract_async(self, video_path: str) -> tuple[dict, str, dict]:
        """Async variant. Default wraps extract() via asyncio.to_thread."""
        return await asyncio.to_thread(self.extract, video_path)


class ReportModel(ABC):
    """
    Stage 2: generate a caregiver-facing report from structured observations.
    Implemented by each report model.
    """

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    RATE_LIMIT_EXCEPTIONS: tuple[type[Exception], ...] = ()

    def __init__(self, config: dict):
        self.config = config
        self.model_id: str = config["model_id"]
        self.display_name: str = config.get("display_name", self.model_id)

    @abstractmethod
    def generate_report(self, observations: dict, language: str = "en") -> tuple[dict, str, dict]:
        """
        Generate a structured caregiver report from behavioral observations.

        Args:
            observations: Stage 1 structured observations dict.
            language: Report language code ("en").

        Returns:
            (output_dict, raw_response_str, token_costs)
            output_dict: {"report": {...}}
        """
        ...

    async def generate_report_async(self, observations: dict, language: str = "en") -> tuple[dict, str, dict]:
        """Async variant. Default wraps generate_report() via asyncio.to_thread."""
        return await asyncio.to_thread(self.generate_report, observations, language)


class JudgeModel(ABC):
    """
    Stage 3: score all model reports for a single video.
    Receives anonymized reports (Model A, B, C) and the original observations.
    Does not receive the AV-ASD labels.
    """

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    RATE_LIMIT_EXCEPTIONS: tuple[type[Exception], ...] = ()

    def __init__(self, config: dict):
        self.config = config
        self.model_id: str = config["model_id"]
        self.display_name: str = config.get("display_name", self.model_id)

    @abstractmethod
    def score(
        self,
        observations: dict,
        reports: dict[str, dict],
        language: str = "en",
    ) -> tuple[dict, str, dict]:
        """
        Score multiple model reports against the observations.

        Args:
            observations: Stage 1 structured observations dict.
            reports: {anonymous_label → report_output_dict}
                     e.g. {"Model A": {...}, "Model B": {...}}
                     Labels are assigned by the runner (not model keys).

        Returns:
            (scores_dict, raw_response_str, token_costs)
            scores_dict: {
                "Model A": {
                    "clinical_accuracy": {
                        "total": int,              # 0-12 (6 DSM-5 domains × 0-2)
                        "reasoning": str,
                        ...sub-scores              # a1_..., a2_..., a3_..., b4_..., b5_..., b6_...
                    },
                    "relational_empathy": {
                        "total": int,              # 7-35 (7 CARE items × 1-5)
                        "reasoning": str,
                        ...sub-scores              # e1_... through e7_...
                    },
                    "clinical_actionability": {
                        "total": int,              # 0-7 (7 SBAR-LA items × 0-1)
                        "reasoning": str,
                        ...sub-scores              # s1_... through s7_...
                    },
                    "safety": {
                        "score": int,              # 0-3
                        "reasoning": str
                    }
                },
                ...
            }
        """
        ...

    async def score_async(
        self,
        observations: dict,
        reports: dict[str, dict],
        language: str = "en",
    ) -> tuple[dict, str, dict]:
        """Async variant. Default wraps score() via asyncio.to_thread."""
        return await asyncio.to_thread(self.score, observations, reports, language)
