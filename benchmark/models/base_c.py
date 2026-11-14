"""
Abstract base class for single-stage report models (Table 5).

VideoReportModel: video path → structured caregiver report dict.

Used by models that generate a report directly from a video
in a single call — no intermediate observation step.

Distinct from:
  VideoModel  (Benchmark A) — video path → BehaviorPrediction
  ReportModel (Benchmark B) — observations dict → report dict
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class VideoReportModel(ABC):

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    RATE_LIMIT_EXCEPTIONS: tuple[type[Exception], ...] = ()

    def __init__(self, config: dict):
        self.config = config
        self.model_id: str = config["model_id"]
        self.display_name: str = config.get("display_name", self.model_id)

    @abstractmethod
    def predict(self, clip_path: str) -> tuple[dict, str, dict]:
        """
        Generate a caregiver report directly from a video file.

        Returns:
            (report_dict, raw_response_str, token_costs)
            report_dict: {"report": {...}} matching the Kidooo report schema.
        """
        ...

    async def predict_async(self, clip_path: str) -> tuple[dict, str, dict]:
        """Async variant. Default wraps predict() via asyncio.to_thread."""
        return await asyncio.to_thread(self.predict, clip_path)
