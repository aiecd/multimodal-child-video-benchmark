"""
Resume support: check whether a clip result already exists on disk.

result_path() is intentionally public so the runner can use it both for the
resume check and for the atomic in-progress claim (open with "x" flag).
"""

from __future__ import annotations

import json
import os


def result_exists(results_dir: str, model_key: str, clip_id: str) -> bool:
    """
    Return True only if the per-clip JSON exists, completed without error,
    and is NOT an in-progress placeholder.

    In-progress placeholders ({"status": "in_progress"}) are written by the
    runner before starting work on a clip. If the process crashed mid-inference
    the placeholder is left on disk. result_exists returns False for these so
    the next run re-claims and reprocesses them.

    Failed results ({"error": "..."}) also return False; the runner retries them
    only with --retry-failed.
    """
    path = result_path(results_dir, model_key, clip_id)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("status") != "in_progress"
            and data.get("error") is None
        )
    except Exception:
        return False


def prior_result_status(results_dir: str, model_key: str, clip_id: str) -> str | None:
    """
    Classify an existing per-clip result: "done", "failed", "in_progress",
    or None when there is no readable file.

    Used to explain why a clip is being skipped. result_exists() answers only
    "should this be reprocessed", which collapses "already finished" and
    "previously failed" into the same False — and since a durable error file
    still occupies the claim path, the runner would otherwise report a permanent
    failure as "claimed by another process".
    """
    path = result_path(results_dir, model_key, clip_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("status") == "in_progress":
        return "in_progress"
    if data.get("error") is not None:
        return "failed"
    return "done"


def result_path(results_dir: str, model_key: str, clip_id: str) -> str:
    """Absolute path for a clip's result JSON."""
    return os.path.join(results_dir, "benchmark_a", model_key, f"{clip_id}.json")
