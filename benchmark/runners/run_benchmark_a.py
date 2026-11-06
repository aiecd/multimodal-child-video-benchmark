"""
Benchmark A runner — CLI entrypoint.

Usage (from the repository root):
    python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash
    python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash --max-clips 10
    python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash --keep-clips
    python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash --score-only
    python -m benchmark.runners.run_benchmark_a --score-only   # reassemble leaderboard only
    python -m benchmark.runners.run_benchmark_a --score-only --leaderboard-models a,b,c
    python -m benchmark.runners.run_benchmark_a --all-models

Execution paths
---------------
Sequential (max_concurrent == 1):
    One clip at a time, synchronous predict(). Default — safe, simple, no async overhead.

Concurrent (max_concurrent > 1):
    asyncio worker pool. max_concurrent workers compete for a semaphore of the
    same size. Uses predict_async() on the model, which for Gemini is a true async
    implementation (no thread pool). Per-model max_concurrent is read from
    models.yaml with settings.yaml as global fallback.

Result file layout
------------------
    results/benchmark_a/
        {model_key}/
            {clip_id}.json            ← written immediately after each clip
            summary_{model_key}.json  ← written atomically after all clips done
        leaderboard.json              ← assembled after all models done

Resume / orphan cleanup
-----------------------
In-progress placeholder files ({status: in_progress}) left by a crashed run are
deleted at the start of each run so they are re-claimed and reprocessed.
Completed clips are cached and skipped, so re-running the same command resumes
without re-spending on work already done.

Clips that errored are NOT reprocessed by default — failures are durable so a
resumed run converges. Pass --retry-failed to re-attempt them once the cause is
fixed.

Interrupting a run
------------------
One Ctrl-C drains the clips currently in flight, writes the summary for the
running model, and then stops — it does not advance to the next model in
--all-models. SIGTERM behaves the same on POSIX.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import pathlib
import signal
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load .env from the repository root (or any parent) before anything reads env vars
load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")

from benchmark.data.clip_extractor import (
    async_temp_clip,
    extract_clip,
    temp_clip,
)
from benchmark.data.loader import load_clips, ClipRecord
from benchmark.models.base import BehaviorPrediction, VideoModel
from benchmark.scoring.benchmark_a import (
    assemble_leaderboard,
    print_summary_table,
    score_and_save_model,
)
from benchmark.utils.cache import prior_result_status, result_exists, result_path
from benchmark.utils.retry import with_retry, with_retry_async
from benchmark.utils.s3 import clip_s3_key, get_cloudfront_url, get_presigned_url

# Resolve the repository root relative to this file
_AV_ASD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Set by the SIGINT/SIGTERM handler in _run_single_model_async(). The handler only
# stops the *current* model — each model gets its own asyncio.run() — so main()
# reads this flag to stop instead of advancing to the next model. Without it, one
# Ctrl-C on --all-models drains the running model and then starts the next one,
# meaning an abort takes one Ctrl-C per remaining model (each up to
# graceful_shutdown_timeout_s) or a SIGKILL.
_SHUTDOWN_REQUESTED = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(rel_path: str) -> dict:
    abs_path = os.path.join(_AV_ASD_ROOT, rel_path)
    with open(abs_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(rel: str) -> str:
    return os.path.join(_AV_ASD_ROOT, rel)


def _instantiate_model(model_key: str, models_config: dict, settings: dict) -> VideoModel:
    """Dynamically instantiate a model from its config entry."""
    if model_key not in models_config["models"]:
        available = list(models_config["models"].keys())
        raise KeyError(
            f"Model '{model_key}' not found in models.yaml. "
            f"Available: {available}"
        )

    model_cfg = dict(models_config["models"][model_key])
    # Inject benchmark-level settings into config (model can override via its own keys)
    model_cfg.setdefault("file_poll_interval_s", settings["benchmark_a"]["file_poll_interval_s"])
    model_cfg.setdefault("file_poll_timeout_s", settings["benchmark_a"]["file_poll_timeout_s"])

    class_path: str = model_cfg["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(model_cfg)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_clip_result_atomic(path: str, data: dict) -> None:
    """Write result data atomically via a .tmp file then os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _compute_per_label_match(
    prediction: BehaviorPrediction,
    ground_truth: dict[str, bool],
) -> dict[str, Optional[bool]]:
    """
    True = prediction matches GT, False = mismatch, None = abstained (null prediction).
    """
    pred_dict = prediction.to_dict()
    result = {}
    for label, gt_val in ground_truth.items():
        pred_val = pred_dict.get(label)
        if pred_val is None:
            result[label] = None
        else:
            result[label] = (bool(pred_val) == bool(gt_val))
    return result


def _clean_orphaned_claims(
    results_dir: str,
    model_key: str,
    retry_failed: bool = False,
) -> tuple[int, int]:
    """
    Delete result files that should be reprocessed on this run.
    Called once at the start of each model run, before any workers begin.

    Always deleted:
      {"status": "in_progress"} placeholders — a crashed or interrupted run left
      the clip claimed but unfinished, so it must be re-claimed.

    Deleted only when retry_failed=True:
      files with {"error": ...}. Errors are durable by default so a run converges:
      a clip the provider permanently rejects (e.g. Pegasus refusing a sub-4s clip)
      would otherwise re-burn an API call on every single resume and never settle.
      Pass --retry-failed after fixing a transient cause to re-attempt them.

    Returns (placeholders_deleted, error_files_deleted).
    """
    model_dir = os.path.join(results_dir, "benchmark_a", model_key)
    if not os.path.isdir(model_dir):
        return 0, 0
    placeholders = 0
    errors = 0
    for fname in os.listdir(model_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(model_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "in_progress":
                os.remove(path)
                placeholders += 1
            elif data.get("error") is not None and retry_failed:
                os.remove(path)
                errors += 1
        except Exception:
            pass
    return placeholders, errors


def _report_cleanup(placeholders: int, errors: int) -> None:
    """Print what _clean_orphaned_claims removed, naming each kind accurately."""
    if placeholders:
        print(f"  Cleaned {placeholders} orphaned in-progress placeholder(s).")
    if errors:
        print(f"  Cleared {errors} previously-failed clip(s) for retry (--retry-failed).")


# ---------------------------------------------------------------------------
# Sequential runner  (max_concurrent == 1)
# ---------------------------------------------------------------------------

def _run_single_model(
    model_key: str,
    model: VideoModel,
    clips: list[ClipRecord],
    results_dir: str,
    original_videos_dir: str,
    keep_clips: bool,
    clips_cache_dir: str,
    retry_attempts: int,
    retry_delay_s: float,
    retry_backoff: float,
    retry_jitter: float,
    s3_config: Optional[dict] = None,
    retry_failed: bool = False,
) -> tuple[int, int, int]:
    """
    Sequential inference: one clip at a time. Uses the synchronous predict() path.
    Returns (processed, skipped_cached, failed).
    """
    _report_cleanup(*_clean_orphaned_claims(results_dir, model_key, retry_failed))

    processed = 0
    skipped = 0
    failed = 0
    failed_prior = 0     # durable failures from an earlier run, left untouched
    total = len(clips)

    print(f"\n{'='*60}")
    print(f"Model: {model.display_name} ({model_key})")
    print(f"Prompt style: {model.prompt_style}")
    print(f"Clips to process: {total}  [sequential]")
    print(f"{'='*60}")

    for i, clip in enumerate(clips, 1):
        prefix = f"[{i:>4}/{total}] {clip.clip_id}"

        clip_json_path = result_path(results_dir, model_key, clip.clip_id)
        out_dir = os.path.dirname(clip_json_path)
        os.makedirs(out_dir, exist_ok=True)

        # Resume: skip already-completed clips
        if result_exists(results_dir, model_key, clip.clip_id):
            print(f"{prefix}  (cached, skipping)")
            skipped += 1
            continue

        # Atomic claim: prevents double-processing if two processes run simultaneously
        try:
            with open(clip_json_path, "x", encoding="utf-8") as f:
                json.dump({"status": "in_progress"}, f)
        except FileExistsError:
            # A durable error file from an earlier run also occupies this path —
            # distinguish it from a genuine concurrent claim.
            if prior_result_status(results_dir, model_key, clip.clip_id) == "failed":
                print(f"{prefix}  (failed previously, skipping — use --retry-failed)")
                failed_prior += 1
            else:
                print(f"{prefix}  (claimed by another process, skipping)")
                skipped += 1
            continue

        print(f"{prefix}  extracting...", end="", flush=True)

        @contextmanager
        def _get_clip_ctx():
            if s3_config:
                key = clip_s3_key(s3_config["key_prefix"], clip.clip_id)
                if s3_config["cloudfront_domain"]:
                    yield get_cloudfront_url(s3_config["cloudfront_domain"], key)
                else:
                    yield get_presigned_url(s3_config["bucket"], key, s3_config["region"], s3_config["expiry_s"])
            elif keep_clips:
                clip_path = extract_clip(
                    video_id=clip.video_id,
                    start_sec=clip.start_sec,
                    end_sec=clip.end_sec,
                    original_videos_dir=original_videos_dir,
                    output_dir=clips_cache_dir,
                    clip_id=clip.clip_id,
                )
                yield clip_path
            else:
                with temp_clip(
                    video_id=clip.video_id,
                    start_sec=clip.start_sec,
                    end_sec=clip.end_sec,
                    original_videos_dir=original_videos_dir,
                    clip_id=clip.clip_id,
                ) as clip_path:
                    yield clip_path

        try:
            with _get_clip_ctx() as clip_path:
                print(f" inferring...", end="", flush=True)
                t0 = time.monotonic()

                (prediction, raw_response, token_costs), retry_log = with_retry(
                    fn=lambda: model.predict(clip_path),
                    attempts=retry_attempts,
                    delay_s=retry_delay_s,
                    backoff=retry_backoff,
                    jitter=retry_jitter,
                    label=clip.clip_id,
                    retryable_exceptions=model.RETRYABLE_EXCEPTIONS,
                    rate_limit_exceptions=model.RATE_LIMIT_EXCEPTIONS,
                )

                elapsed = round(time.monotonic() - t0, 2)
                attempt_count = len(retry_log) + 1
                rate_limited = any(r["rate_limited"] for r in retry_log)
                per_label_match = _compute_per_label_match(prediction, clip.ground_truth)

                result_data = {
                    "clip_id": clip.clip_id,
                    "model": model_key,
                    "display_name": model.display_name,
                    "prompt_style": model.prompt_style,
                    "asd_behavioral_metrics": prediction.to_dict(),
                    "ground_truth": clip.ground_truth,
                    "per_label_match": per_label_match,
                    "timing_s": elapsed,
                    "token_costs": token_costs,
                    "attempt_count": attempt_count,
                    "rate_limited": rate_limited,
                    "retry_reasons": retry_log,
                    "raw_response": raw_response,
                    "error": None,
                }
                _save_clip_result_atomic(clip_json_path, result_data)
                processed += 1

                n_null = sum(1 for v in prediction.to_dict().values() if v is None)
                n_match = sum(1 for v in per_label_match.values() if v is True)
                n_total = sum(1 for v in per_label_match.values() if v is not None)
                print(f" done ({elapsed}s | {n_match}/{n_total} match, {n_null} null)")

        except FileNotFoundError as e:
            print(f"\n  SKIP (video not found): {e}")
            _save_clip_result_atomic(clip_json_path, {
                "clip_id": clip.clip_id,
                "model": model_key,
                "error": f"FileNotFoundError: {e}",
            })
            failed += 1

        except Exception as e:
            print(f" FAIL — {type(e).__name__}: {e}")
            _save_clip_result_atomic(clip_json_path, {
                "clip_id": clip.clip_id,
                "model": model_key,
                "error": f"{type(e).__name__}: {e}",
            })
            failed += 1

    print(f"\n  Done: {processed} processed, {skipped} cached, {failed} failed"
          + (f", {failed_prior} failed earlier (skipped)" if failed_prior else ""))

    # Write per-model summary file atomically
    try:
        score_and_save_model(results_dir, model_key)
        print(f"  Summary saved: results/benchmark_a/{model_key}/summary_{model_key}.json")
    except Exception as e:
        print(f"  [scoring] Could not score model '{model_key}': {e}")

    return processed, skipped, failed


# ---------------------------------------------------------------------------
# Concurrent async runner  (max_concurrent > 1)
# ---------------------------------------------------------------------------

async def _run_single_model_async(
    model_key: str,
    model: VideoModel,
    clips: list[ClipRecord],
    results_dir: str,
    original_videos_dir: str,
    keep_clips: bool,
    clips_cache_dir: str,
    retry_attempts: int,
    retry_delay_s: float,
    retry_backoff: float,
    retry_jitter: float,
    max_concurrent: int,
    rate_limit_pause_s: float,
    graceful_shutdown_timeout_s: float,
    s3_config: Optional[dict] = None,
    retry_failed: bool = False,
) -> tuple[int, int, int]:
    """
    Concurrent inference using an asyncio worker pool.

    `max_concurrent` workers pull clips from a shared queue. Each worker:
      1. Checks result_exists() — skip if already done (no semaphore consumed)
      2. Claims the result file with open("x") — skips on FileExistsError
      3. Waits on rate_limit_ok event before acquiring the semaphore
      4. Acquires semaphore → extracts clip → calls predict_async() → saves result

    Signal handling:
      SIGINT (Ctrl-C) sets shutdown_event. Workers finish their current clip then
      stop accepting new work. After graceful_shutdown_timeout_s, any remaining
      in-flight tasks are hard-cancelled.
      On Windows only SIGINT is available; SIGTERM is registered on POSIX only.

    Returns (processed, skipped_cached, failed).
    """
    _report_cleanup(*_clean_orphaned_claims(results_dir, model_key, retry_failed))

    total = len(clips)

    print(f"\n{'='*60}")
    print(f"Model: {model.display_name} ({model_key})")
    print(f"Prompt style: {model.prompt_style}")
    print(f"Clips to process: {total}  [async, max_concurrent={max_concurrent}]")
    print(f"{'='*60}")

    # Shared async primitives
    queue: asyncio.Queue[ClipRecord] = asyncio.Queue()
    for clip in clips:
        await queue.put(clip)

    semaphore = asyncio.Semaphore(max_concurrent)
    # SET = workers may proceed; CLEARED = all workers pause (rate limit in effect)
    rate_limit_ok = asyncio.Event()
    rate_limit_ok.set()
    shutdown_event = asyncio.Event()

    # Counters — safe without a lock because asyncio is single-threaded/cooperative
    # failed_prior: durable failures from an earlier run, left untouched
    counters = {"processed": 0, "skipped": 0, "failed": 0, "failed_prior": 0, "started": 0}

    # --- Signal handling ---
    loop = asyncio.get_running_loop()

    def _on_shutdown(signum, frame):
        global _SHUTDOWN_REQUESTED
        _SHUTDOWN_REQUESTED = True
        print("\n[Benchmark] Shutdown signal received. Finishing in-flight clips...")
        loop.call_soon_threadsafe(shutdown_event.set)

    saved_sigint = signal.signal(signal.SIGINT, _on_shutdown)
    saved_sigterm = None
    if hasattr(signal, "SIGTERM"):
        saved_sigterm = signal.signal(signal.SIGTERM, _on_shutdown)

    # --- Worker coroutine ---
    async def _worker() -> None:
        while not shutdown_event.is_set():
            try:
                clip = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            counters["started"] += 1
            idx = counters["started"]
            prefix = f"  [{idx:>4}/{total}] {clip.clip_id}"

            clip_json_path = result_path(results_dir, model_key, clip.clip_id)
            out_dir = os.path.dirname(clip_json_path)
            os.makedirs(out_dir, exist_ok=True)

            # Resume: skip already-completed clips (no semaphore consumed)
            if result_exists(results_dir, model_key, clip.clip_id):
                print(f"{prefix}  (cached)")
                counters["skipped"] += 1
                continue

            # Atomic claim (no semaphore consumed — fast path for multi-process safety)
            try:
                with open(clip_json_path, "x", encoding="utf-8") as f:
                    json.dump({"status": "in_progress"}, f)
            except FileExistsError:
                # A durable error file from an earlier run also occupies this path —
                # distinguish it from a genuine concurrent claim.
                if prior_result_status(results_dir, model_key, clip.clip_id) == "failed":
                    print(f"{prefix}  (failed previously, skipping — use --retry-failed)")
                    counters["failed_prior"] += 1
                else:
                    print(f"{prefix}  (claimed by another process)")
                    counters["skipped"] += 1
                continue

            # Pause if a rate-limit event is in effect across all workers
            await rate_limit_ok.wait()

            async with semaphore:
                t0 = time.monotonic()

                @asynccontextmanager
                async def _clip_ctx():
                    if s3_config:
                        key = clip_s3_key(s3_config["key_prefix"], clip.clip_id)
                        if s3_config["cloudfront_domain"]:
                            yield get_cloudfront_url(s3_config["cloudfront_domain"], key)
                        else:
                            url = await asyncio.to_thread(
                                get_presigned_url,
                                s3_config["bucket"], key, s3_config["region"], s3_config["expiry_s"],
                            )
                            yield url
                    elif keep_clips:
                        cp = await asyncio.to_thread(
                            extract_clip,
                            clip.video_id, clip.start_sec, clip.end_sec,
                            original_videos_dir, clips_cache_dir, clip.clip_id,
                        )
                        yield cp
                    else:
                        async with async_temp_clip(
                            video_id=clip.video_id,
                            start_sec=clip.start_sec,
                            end_sec=clip.end_sec,
                            original_videos_dir=original_videos_dir,
                            clip_id=clip.clip_id,
                        ) as cp:
                            yield cp

                try:
                    print(f"{prefix}  inferring...", flush=True)

                    async with _clip_ctx() as clip_path:
                        (prediction, raw_response, token_costs), retry_log = (
                            await with_retry_async(
                                fn=lambda: model.predict_async(clip_path),
                                attempts=retry_attempts,
                                delay_s=retry_delay_s,
                                backoff=retry_backoff,
                                jitter=retry_jitter,
                                label=clip.clip_id,
                                retryable_exceptions=model.RETRYABLE_EXCEPTIONS,
                                rate_limit_exceptions=model.RATE_LIMIT_EXCEPTIONS,
                                rate_limit_pause_s=rate_limit_pause_s,
                                rate_limit_ok=rate_limit_ok,
                            )
                        )

                    elapsed = round(time.monotonic() - t0, 2)
                    attempt_count = len(retry_log) + 1
                    rate_limited = any(r["rate_limited"] for r in retry_log)
                    per_label_match = _compute_per_label_match(prediction, clip.ground_truth)

                    result_data = {
                        "clip_id": clip.clip_id,
                        "model": model_key,
                        "display_name": model.display_name,
                        "prompt_style": model.prompt_style,
                        "asd_behavioral_metrics": prediction.to_dict(),
                        "ground_truth": clip.ground_truth,
                        "per_label_match": per_label_match,
                        "timing_s": elapsed,
                        "token_costs": token_costs,
                        "attempt_count": attempt_count,
                        "rate_limited": rate_limited,
                        "retry_reasons": retry_log,
                        "raw_response": raw_response,
                        "error": None,
                    }
                    _save_clip_result_atomic(clip_json_path, result_data)
                    counters["processed"] += 1

                    n_null = sum(1 for v in prediction.to_dict().values() if v is None)
                    n_match = sum(1 for v in per_label_match.values() if v is True)
                    n_total = sum(1 for v in per_label_match.values() if v is not None)
                    print(
                        f"{prefix}  done "
                        f"({elapsed}s | {n_match}/{n_total} match, {n_null} null)",
                        flush=True,
                    )

                except FileNotFoundError as e:
                    print(f"{prefix}  SKIP (video not found): {e}", flush=True)
                    _save_clip_result_atomic(clip_json_path, {
                        "clip_id": clip.clip_id,
                        "model": model_key,
                        "error": f"FileNotFoundError: {e}",
                    })
                    counters["failed"] += 1

                except Exception as e:
                    print(f"{prefix}  FAIL — {type(e).__name__}: {e}", flush=True)
                    _save_clip_result_atomic(clip_json_path, {
                        "clip_id": clip.clip_id,
                        "model": model_key,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    counters["failed"] += 1

    # --- Graceful shutdown watcher ---
    tasks: list[asyncio.Task] = []

    async def _shutdown_watcher() -> None:
        await shutdown_event.wait()
        print(
            f"[Benchmark] Waiting up to {graceful_shutdown_timeout_s}s "
            f"for in-flight clips to finish..."
        )
        await asyncio.sleep(graceful_shutdown_timeout_s)
        still_running = [t for t in tasks if not t.done()]
        if still_running:
            print(f"[Benchmark] Hard-cancelling {len(still_running)} remaining task(s).")
            for t in still_running:
                t.cancel()

    # --- Launch ---
    tasks = [asyncio.create_task(_worker()) for _ in range(max_concurrent)]
    watcher = asyncio.create_task(_shutdown_watcher())

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # Cancel watcher; release any stuck rate_limit_ok; restore signal handlers
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        rate_limit_ok.set()

        signal.signal(signal.SIGINT, saved_sigint)
        if saved_sigterm is not None:
            signal.signal(signal.SIGTERM, saved_sigterm)

        processed = counters["processed"]
        skipped = counters["skipped"]
        failed = counters["failed"]
        failed_prior = counters["failed_prior"]
        print(f"\n  Done: {processed} processed, {skipped} cached, {failed} failed"
              + (f", {failed_prior} failed earlier (skipped)" if failed_prior else ""))

        # Write per-model summary atomically (runs even on Ctrl-C)
        try:
            score_and_save_model(results_dir, model_key)
            print(f"  Summary saved: results/benchmark_a/{model_key}/summary_{model_key}.json")
        except Exception as e:
            print(f"  [scoring] Could not score model '{model_key}': {e}")

    return counters["processed"], counters["skipped"], counters["failed"]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Benchmark A: ASD behavioral observation quality evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash
  python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash --max-clips 10
  python -m benchmark.runners.run_benchmark_a --all-models --keep-clips
  python -m benchmark.runners.run_benchmark_a --model gemini-3.7-flash --score-only
  python -m benchmark.runners.run_benchmark_a --score-only
  python -m benchmark.runners.run_benchmark_a --all-models --retry-failed
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model key from models.yaml (e.g. 'gemini-3.7-flash')",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all video-capable models from models.yaml",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit number of clips (useful for dev/testing)",
    )
    parser.add_argument(
        "--keep-clips",
        action="store_true",
        help="Cache extracted clips to disk (avoids re-extraction across runs)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=(
            "Skip inference. With --model: re-score that model and update leaderboard. "
            "Without --model: reassemble leaderboard from existing summary files only."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override the CSV split to use (default: test.csv from settings.yaml)",
    )
    parser.add_argument(
        "--clips-dir",
        type=str,
        default=None,
        help=(
            "Override the pre-cut clip store (default: clips_cache_dir from "
            "settings.yaml). Only affects models that read local files; models with "
            "s3_bucket set fetch by clip_id from CloudFront and ignore this."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Re-attempt clips that previously errored. Off by default so failures "
            "are durable and a resumed run converges — without it, permanently "
            "rejected clips would re-burn an API call on every resume."
        ),
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys from models.yaml (alternative to --model)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Override paths.results_dir from settings.yaml; results go to "
            "<results-dir>/benchmark_a. Use a separate directory to keep an "
            "experiment (e.g. a prompt comparison) out of the main leaderboard."
        ),
    )

    parser.add_argument(
        "--leaderboard-models",
        type=str,
        default=None,
        help=(
            "Comma-separated model keys to include in the leaderboard (default: every "
            "model with a summary in the results dir). Models are ranked on the clips "
            "all included models completed, so leaving out a model that ran on few "
            "clips compares the rest on more."
        ),
    )

    args = parser.parse_args(argv)
    leaderboard_models = (
        [k.strip() for k in args.leaderboard_models.split(",") if k.strip()]
        if args.leaderboard_models else None
    )

    if args.models:
        if args.model:
            parser.error("Use either --model or --models, not both")
        args.model = None
    if not args.model and not args.models and not args.all_models and not args.score_only:
        parser.error("Specify --model <key>, --models <k1,k2>, --all-models, or --score-only")

    # Load configs
    settings = _load_yaml("benchmark/config/settings.yaml")
    models_config = _load_yaml("benchmark/config/models.yaml")

    # Resolve paths
    original_videos_dir = _resolve_path(settings["paths"]["original_videos"])
    results_dir = _resolve_path(args.results_dir or settings["paths"]["results_dir"])
    clips_cache_dir = _resolve_path(args.clips_dir or settings["paths"]["clips_cache_dir"])
    csv_path = _resolve_path(args.csv or settings["paths"]["test_csv"])

    bench_cfg = settings["benchmark_a"]
    max_clips = args.max_clips if args.max_clips is not None else bench_cfg.get("max_clips")
    keep_clips = args.keep_clips or bench_cfg.get("keep_clips", False)
    retry_attempts = bench_cfg.get("retry_attempts", 3)
    retry_delay_s = bench_cfg.get("retry_delay_s", 5.0)
    retry_backoff = bench_cfg.get("retry_backoff", 2.0)
    retry_jitter = bench_cfg.get("retry_jitter", 0.5)
    rate_limit_pause_s = bench_cfg.get("rate_limit_pause_s", 15.0)
    graceful_shutdown_timeout_s = bench_cfg.get("graceful_shutdown_timeout_s", 30.0)
    assessability_threshold = bench_cfg.get("assessability_threshold", 0.85)
    global_max_concurrent = bench_cfg.get("max_concurrent", 1)

    # --- score-only without --model: just reassemble the leaderboard ---
    if args.score_only and not args.model and not args.models and not args.all_models:
        print("Reassembling leaderboard from existing summary files...")
        try:
            leaderboard = assemble_leaderboard(results_dir, leaderboard_models)
            print(f"Leaderboard saved: {os.path.relpath(os.path.join(results_dir, 'benchmark_a', 'leaderboard.json'), _AV_ASD_ROOT)}")
            print_summary_table(leaderboard)
            _warn_assessability(leaderboard, assessability_threshold)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        return

    # --- Determine which models to operate on ---
    if args.all_models:
        model_keys = [
            k for k, v in models_config["models"].items()
            if v.get("video_capable", False)
        ]
    elif args.models:
        model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    else:
        model_keys = [args.model]

    if not model_keys:
        print("No models to process.")
        sys.exit(0)

    # --- score-only with --model: re-score and update leaderboard ---
    if args.score_only:
        for model_key in model_keys:
            print(f"Scoring '{model_key}' from existing results...")
            try:
                score_and_save_model(results_dir, model_key)
                print(f"  Summary saved: results/benchmark_a/{model_key}/summary_{model_key}.json")
            except Exception as e:
                print(f"  [ERROR] Could not score '{model_key}': {e}")
        leaderboard = assemble_leaderboard(results_dir, leaderboard_models)
        print(f"Leaderboard saved: {os.path.relpath(os.path.join(results_dir, 'benchmark_a', 'leaderboard.json'), _AV_ASD_ROOT)}")
        print_summary_table(leaderboard)
        _warn_assessability(leaderboard, assessability_threshold)
        return

    # --- Full inference run ---
    clips = load_clips(csv_path, max_clips=max_clips)
    if not clips:
        print("No clips loaded. Check your CSV path and settings.")
        sys.exit(1)

    for i, model_key in enumerate(model_keys):
        # A Ctrl-C during the previous model stops the whole run, not just that model.
        if _SHUTDOWN_REQUESTED:
            remaining = model_keys[i:]
            print(
                f"\n[Benchmark] Shutdown requested — not starting "
                f"{len(remaining)} remaining model(s): {', '.join(remaining)}"
            )
            print("  Re-run the same command to resume; completed clips are cached.")
            break

        try:
            model = _instantiate_model(model_key, models_config, settings)
        except Exception as e:
            print(f"\n[ERROR] Cannot instantiate model '{model_key}': {e}")
            continue

        # Per-model max_concurrent: models.yaml overrides settings.yaml
        model_cfg = models_config["models"][model_key]
        max_concurrent = model_cfg.get("max_concurrent", global_max_concurrent)

        # Per-model S3/CloudFront config — set s3_bucket in models.yaml to enable
        s3_bucket = model_cfg.get("s3_bucket")
        s3_config: Optional[dict] = None
        if s3_bucket:
            cloudfront_domain = model_cfg.get("cloudfront_domain", "").strip()
            s3_config = {
                "bucket": s3_bucket,
                "region": model_cfg.get("s3_region", "us-east-1"),
                "key_prefix": model_cfg.get("s3_key_prefix", "benchmark-clips/"),
                "expiry_s": model_cfg.get("s3_presigned_url_expiry_s", 3600),
                "cloudfront_domain": cloudfront_domain,
            }
            source = f"cloudfront ({cloudfront_domain})" if cloudfront_domain else f"s3://{s3_bucket} (pre-signed)"
            print(f"[video source] {model_key}: {source}")


        if max_concurrent > 1:
            asyncio.run(
                _run_single_model_async(
                    model_key=model_key,
                    model=model,
                    clips=clips,
                    results_dir=results_dir,
                    original_videos_dir=original_videos_dir,
                    keep_clips=keep_clips,
                    clips_cache_dir=clips_cache_dir,
                    retry_attempts=retry_attempts,
                    retry_delay_s=retry_delay_s,
                    retry_backoff=retry_backoff,
                    retry_jitter=retry_jitter,
                    max_concurrent=max_concurrent,
                    rate_limit_pause_s=rate_limit_pause_s,
                    graceful_shutdown_timeout_s=graceful_shutdown_timeout_s,
                    s3_config=s3_config,
                    retry_failed=args.retry_failed,
                )
            )
        else:
            _run_single_model(
                model_key=model_key,
                model=model,
                clips=clips,
                results_dir=results_dir,
                original_videos_dir=original_videos_dir,
                keep_clips=keep_clips,
                clips_cache_dir=clips_cache_dir,
                retry_attempts=retry_attempts,
                retry_delay_s=retry_delay_s,
                retry_backoff=retry_backoff,
                retry_jitter=retry_jitter,
                s3_config=s3_config,
                retry_failed=args.retry_failed,
            )

    # Assemble leaderboard from all per-model summary files
    print("\nAssembling leaderboard...")
    try:
        leaderboard = assemble_leaderboard(results_dir, leaderboard_models)
        print(f"Leaderboard saved: {os.path.relpath(os.path.join(results_dir, 'benchmark_a', 'leaderboard.json'), _AV_ASD_ROOT)}")
        print_summary_table(leaderboard)
        _warn_assessability(leaderboard, assessability_threshold)
    except Exception as e:
        print(f"[WARNING] Could not assemble leaderboard: {e}")


def _warn_assessability(leaderboard: dict, threshold: float) -> None:
    for mk, ms in leaderboard.get("models", {}).items():
        rate = ms.get("macro_assessability_rate")
        if rate is not None and rate < threshold:
            print(
                f"  [WARNING] {mk}: macro_assessability_rate={rate:.4f} "
                f"is below threshold {threshold}"
            )


if __name__ == "__main__":
    main()
