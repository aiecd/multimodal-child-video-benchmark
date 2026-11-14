"""
Abstract base class for all video models in the benchmark.
Every model adapter must implement predict() and return a BehaviorPrediction.

Scalability contract for new adapters
--------------------------------------
What you must implement:
  predict(clip_path)          — synchronous inference, must be stateless
  RETRYABLE_EXCEPTIONS        — tuple of exception types worth retrying
  RATE_LIMIT_EXCEPTIONS       — subset of RETRYABLE_EXCEPTIONS for 429-equivalents

What you get for free:
  predict_async()             — default wraps predict() via asyncio.to_thread
  Resume/cache, retry with your exception attrs, observability fields,
  per-model summary, leaderboard merge, rate-limit coordination

Override predict_async() only if:
  - Your provider SDK has a native async client (eliminates thread overhead), or
  - Your adapter stores per-request state on self (thread-unsafe under concurrency)

Thread-safety note (max_concurrent > 1)
-----------------------------------------
The default predict_async() runs predict() in a thread pool. Multiple threads
may call predict() on the same model instance concurrently. Adapters must:
  1. Store no mutable per-request state on self (no self.last_response, etc.)
  2. Use a thread-safe provider SDK client (most HTTP clients qualify)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

# Canonical label keys — human-readable, match CSV column names and JSON output
LABEL_KEYS: list[str] = [
    "Absence or Avoidance of Eye Contact",
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Non-Responsiveness to Verbal Interaction",
    "Non-Typical Language",
    "Object Lining-Up",
    "Self-Hitting or Self-Injurious Behavior",
    "Self-Spinning or Spinning Objects",
    "Upper Limb Stereotypies",
    "Background",
]


class BehaviorPrediction:
    """
    Model output for a single clip.
    Each label in LABEL_KEYS maps to True (observed), False (not observed),
    or None (model abstained / did not produce a parseable answer for that label).
    """

    def __init__(self, values: dict[str, Optional[bool]]):
        self.values = values

    def to_dict(self) -> dict[str, Optional[bool]]:
        """Returns {human-readable label → bool|None} for all LABEL_KEYS."""
        return {label: self.values.get(label) for label in LABEL_KEYS}

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorPrediction":
        """
        Build from a dict keyed by human-readable label names.
        Accepts JSON bool values or string "true"/"false".
        null or missing values are preserved as None (model abstained).
        """
        values: dict[str, Optional[bool]] = {}
        for label in LABEL_KEYS:
            val = d.get(label)
            if isinstance(val, bool):
                values[label] = val
            elif isinstance(val, str):
                values[label] = val.lower() == "true"
            else:
                values[label] = None
        return cls(values)

    def __repr__(self) -> str:
        trues = [k for k, v in self.values.items() if v]
        return f"BehaviorPrediction(true={trues})"


class VideoModel(ABC):
    """
    Base class for all benchmark video models.
    Subclasses implement predict() for their specific provider.
    """

    # --- Exception classification (override in every subclass) ---
    # Only these exception types are retried. Everything else fails immediately.
    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    # Subset of RETRYABLE_EXCEPTIONS that represent 429-equivalent rate limits.
    # Used to set rate_limited=True in per-clip JSON and trigger cross-worker pause.
    RATE_LIMIT_EXCEPTIONS: tuple[type[Exception], ...] = ()

    def __init__(self, config: dict):
        self.config = config
        self.model_id: str = config["model_id"]
        self.display_name: str = config.get("display_name", self.model_id)
        self.prompt_style: str = config.get("prompt_style", "baseline")
        self.input_type: str = config.get("input_type", "native_video")

    @abstractmethod
    def predict(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        Run inference on a video clip.

        Args:
            clip_path: Absolute path to an extracted .mp4 clip.

        Returns:
            (BehaviorPrediction, raw_response_str, token_costs)
            raw_response_str: full model output text, stored for debugging.
            token_costs: {"input_tokens": int, "output_tokens": int}, or {} if unavailable.

        Raises:
            Exception on unrecoverable error (caller handles retry + logging).

        Thread-safety: must be stateless — store no mutable state on self
        between calls. See class docstring.
        """
        ...

    async def predict_async(self, clip_path: str) -> tuple[BehaviorPrediction, str, dict]:
        """
        Async inference. Default implementation runs the synchronous predict()
        in a thread pool via asyncio.to_thread so it does not block the event loop.

        Adapters that provide a native async SDK (e.g. Gemini aio client) should
        override this method for true async execution without thread overhead.
        The synchronous predict() is always kept in place for the sequential path
        and single-clip dev runs.
        """
        return await asyncio.to_thread(self.predict, clip_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id}, prompt={self.prompt_style})"
