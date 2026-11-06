"""
Benchmark B runner — CLI entrypoint.

Three-stage pipeline per clip:
  Stage 1: Gemini writes an observation record from the clip (cached)
  Stage 2: every report model writes a caregiver report from the record (each cached)
  Stage 3: the AI judges score all reports together (cached)

Usage (from the repository root):
    python -m benchmark.runners.run_benchmark_b
    python -m benchmark.runners.run_benchmark_b --max-videos 5
    python -m benchmark.runners.run_benchmark_b --score-only

Result layout:
    results/benchmark_b/
        observations/{clip_id}.json                     ← Stage 1 observation record
        reports/{language}/{model_key}/{clip_id}.json   ← Stage 2 report per model
        judge/{language}/{clip_id}.json                 ← Stage 3 scores for all models
        summary_{language}_{model_key}.json             ← aggregated scores per model
        leaderboard_{language}.json                     ← ranked leaderboard

Cache behavior:
    Each stage checks for its output file before calling the API, so a resumed run
    re-reads cached observations, reports and judge scores instead of paying for
    them again. Delete a file manually to force that stage to re-run.

    Interrupt safety:
      Orphaned in-progress placeholders and stale .lock sidecars are cleaned at
      startup so the slot is re-claimable.

      Stage 3 claims via a sidecar lock rather than overwriting its result file,
      because a partial judge retry carries already-completed judges forward in
      memory; overwriting would lose them if the process died before the merged
      write.

      Errored results are NOT retried by default — pass --retry-failed. Durable
      errors matter here because a Stage 2 report that fails then succeeds changes
      the model set for that video, invalidating its cached judge result and
      re-running every judge for it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import pathlib
import random
import signal
import sys
import time
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")

from benchmark.data.loader_b import VideoRecord, load_videos
from benchmark.models.base_text import JudgeModel, ObservationExtractor, ReportModel
from benchmark.scoring.benchmark_b import (
    assemble_leaderboard,
    print_summary_table,
    score_and_save_model,
)
from benchmark.utils.retry import with_retry_async

_AV_ASD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(rel_path: str) -> dict:
    abs_path = os.path.join(_AV_ASD_ROOT, rel_path)
    with open(abs_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(rel: str) -> str:
    return os.path.join(_AV_ASD_ROOT, rel)


def _instantiate(class_path: str, config: dict):
    """Dynamically instantiate a class from a dotted class path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def _instantiate_extractor(
    models_config: dict, settings: dict
) -> tuple[str, ObservationExtractor]:
    """Instantiate the Stage 1 observation extractor from models.yaml."""
    bench_cfg = settings.get("benchmark_b", {})
    extractor_key = bench_cfg.get("observation_model")
    if not extractor_key:
        raise ValueError(
            "benchmark_b.observation_model not set in settings.yaml. "
            "Specify the model key to use for Stage 1 observation extraction."
        )
    cfg = dict(models_config["models"][extractor_key])
    cfg.setdefault("file_poll_interval_s", bench_cfg.get("file_poll_interval_s", 3))
    cfg.setdefault("file_poll_timeout_s", bench_cfg.get("file_poll_timeout_s", 180))
    return extractor_key, _instantiate(cfg["class"], cfg)


def _instantiate_report_models(
    models_config: dict,
    tolerate_failures: bool = False,
    only: Optional[list[str]] = None,
) -> dict[str, Optional[ReportModel]]:
    """Instantiate all Stage 2 report models from models.yaml.

    tolerate_failures: used by --merge-judges, where report models are never
    called (Stage 2 is cache-only) but their keys are still needed to read
    cached reports. A model that cannot instantiate (e.g. missing API key on
    this machine) is kept with a None instance instead of aborting the run.

    only: optional subset of report model keys (from --models). Unknown keys, or
    keys whose entry is not role: report, are an error rather than silently skipped.
    """
    report_keys = [
        key for key, cfg in models_config.get("models", {}).items()
        if cfg.get("role") == "report"
    ]
    if only is not None:
        unknown = [k for k in only if k not in report_keys]
        if unknown:
            raise ValueError(
                f"--models: {unknown} not found as role: report in models.yaml. "
                f"Available: {sorted(report_keys)}"
            )
    result: dict[str, Optional[ReportModel]] = {}
    for key, cfg in models_config.get("models", {}).items():
        if cfg.get("role") != "report":
            continue
        if only is not None and key not in only:
            continue
        cfg = dict(cfg)
        # Models used in both benchmarks have report_class for Stage 2 (text-only)
        cls_path = cfg.get("report_class") or cfg["class"]
        try:
            result[key] = _instantiate(cls_path, cfg)
        except Exception as e:
            if not tolerate_failures:
                raise
            print(f"[WARN] Report model '{key}' not instantiated "
                  f"(cache-only in merge mode): {e}")
            result[key] = None
    if not result:
        raise ValueError(
            "No report models found in models.yaml under models: "
            "with role: report"
        )
    return result


def _instantiate_judges(
    models_config: dict, only: Optional[list[str]] = None
) -> dict[str, JudgeModel]:
    """Instantiate Stage 3 judge models from models.yaml (role: judge).

    only: optional subset of judge keys (from --judges). Unknown keys, or keys
    whose entry is not role: judge, are an error rather than silently skipped.
    """
    judge_cfgs = {
        key: cfg
        for key, cfg in models_config.get("models", {}).items()
        if cfg.get("role") == "judge"
    }
    if only is not None:
        unknown = [k for k in only if k not in judge_cfgs]
        if unknown:
            raise ValueError(
                f"--judges: {unknown} not found as role: judge in models.yaml. "
                f"Available: {sorted(judge_cfgs)}"
            )
        judge_cfgs = {k: judge_cfgs[k] for k in only}
    result: dict[str, JudgeModel] = {
        key: _instantiate(cfg["class"], dict(cfg))
        for key, cfg in judge_cfgs.items()
    }
    if not result:
        raise ValueError(
            "No judge models found in models.yaml under models: "
            "with role: judge"
        )
    return result


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def _obs_path(bench_dir: str, video_id: str) -> str:
    return os.path.join(bench_dir, "observations", f"{video_id}.json")


def _report_path(bench_dir: str, language: str, model_key: str, video_id: str) -> str:
    return os.path.join(bench_dir, "reports", language, model_key, f"{video_id}.json")


def _judge_path(bench_dir: str, language: str, video_id: str) -> str:
    return os.path.join(bench_dir, "judge", language, f"{video_id}.json")


def _is_done(path: str) -> bool:
    """Return True if file exists, is not in-progress, and has no error."""
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


def _save_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _claim(path: str) -> bool:
    """Atomically claim a result slot. Returns True if claim succeeded."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump({"status": "in_progress"}, f)
        return True
    except FileExistsError:
        return False


def _prior_error(path: str) -> Optional[str]:
    """
    Return the error string from a completed-but-failed result, else None.

    Needed because _is_done() collapses "never attempted" and "attempted and
    failed" into the same False. Without this the stage would fall through to
    _claim(), hit FileExistsError on the durable error file, and then sit in the
    60s "claimed by another worker" poll for a slot nobody is working on.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("status") == "in_progress":
        return None
    err = data.get("error")
    return err if err is not None else None


def _lock_path(result_path: str) -> str:
    return result_path + ".lock"


def _claim_lock(result_path: str) -> bool:
    """
    Claim a result slot without touching the result file itself.

    Stage 3 merges newly-run judges on top of judges carried over from a previous
    partial run, so the existing file holds work we must not lose. Claiming by
    overwriting it (the _claim path) would delete those scores, leaving them only
    in memory until the final write — an interrupt in that window loses them
    permanently. A sidecar lock keeps the result file intact until the merged
    write replaces it atomically.
    """
    lock = _lock_path(result_path)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        with open(lock, "x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def _release_lock(result_path: str) -> None:
    try:
        os.remove(_lock_path(result_path))
    except Exception:
        pass


def _clean_orphans(bench_dir: str, retry_failed: bool = False) -> tuple[int, int, int]:
    """
    Remove state left behind by an interrupted run.

    Always removed:
      in-progress placeholders, and stale .lock sidecars — both mean a worker died
      mid-item, so the slot must be re-claimable.

    Removed only when retry_failed=True:
      files recording an error. Errors are durable by default so a resumed run
      converges instead of re-attempting a permanently failing item every time.
      This matters more in Benchmark B than in A: a Stage 2 report that failed in
      one run and succeeds on retry changes the model set for that video, which
      invalidates the whole video's cached judge result and forces every judge to
      re-run (see the model-set check in Stage 3).

    Returns (placeholders, error_files, locks) removed.
    """
    placeholders = errors = locks = 0
    for dirpath, _, filenames in os.walk(bench_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.endswith(".lock"):
                try:
                    os.remove(fpath)
                    locks += 1
                except Exception:
                    pass
                continue
            if not fname.endswith(".json"):
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "in_progress":
                    os.remove(fpath)
                    placeholders += 1
                elif data.get("error") is not None and retry_failed:
                    os.remove(fpath)
                    errors += 1
            except Exception:
                pass
    return placeholders, errors, locks


# ---------------------------------------------------------------------------
# Anonymous label helpers (Model A, B, C...)
# ---------------------------------------------------------------------------

def _assign_labels(model_keys: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Randomly assign anonymous labels to model keys.
    Labels are generated dynamically ("Model A", "Model B", ...) so any number
    of models is supported.

    Returns:
        (key_to_label, label_to_key)
        e.g. {"gpt-5.6-sol": "Model B", "kimi-k3": "Model A"}
    """
    shuffled = list(model_keys)
    random.shuffle(shuffled)
    labels = [f"Model {chr(ord('A') + i)}" for i in range(len(shuffled))]
    key_to_label = dict(zip(shuffled, labels))
    label_to_key = {v: k for k, v in key_to_label.items()}
    return key_to_label, label_to_key


# ---------------------------------------------------------------------------
# Per-video async pipeline
# ---------------------------------------------------------------------------

async def _process_video(
    video: VideoRecord,
    bench_dir: str,
    language: str,
    extractor: ObservationExtractor,
    report_models: dict[str, ReportModel],
    judges: dict[str, JudgeModel],
    model_semaphores: dict[str, asyncio.Semaphore],
    judge_semaphores: dict[str, asyncio.Semaphore],
    extractor_semaphore: asyncio.Semaphore,
    retry_attempts: int,
    retry_delay_s: float,
    retry_backoff: float,
    retry_jitter: float,
    rate_limit_pause_s: float,
    rate_limit_events: dict[str, asyncio.Event],
    retry_failed: bool = False,
    merge_judges: bool = False,
    configured_judge_keys: Optional[set] = None,
) -> None:
    """Run all three pipeline stages for a single video.

    merge_judges: additive Stage 3 mode. Instead of invalidating a cached judge
    result whose judge set differs from the current one, reuse its anonymization
    mapping and scores, run only the current judges that have no cached scores,
    prune scores of judges no longer configured (role: judge in models.yaml,
    passed as configured_judge_keys), and write the merged set. Requires the
    cached model set to match this run's reports — on mismatch the video is
    skipped with a warning rather than re-judged, so existing scores survive.
    """

    prefix = f"  [{video.video_id}]"

    # ── Stage 1: Observation extraction ────────────────────────────────────
    obs_path = _obs_path(bench_dir, video.video_id)
    observations: Optional[dict] = None

    if _is_done(obs_path):
        with open(obs_path, encoding="utf-8") as f:
            obs_file = json.load(f)
        observations = obs_file.get("observations")
        # Patch ground_truth_label if we now have it but the cached file doesn't
        if video.ground_truth is not None and obs_file.get("ground_truth_label") is None:
            obs_file["ground_truth_label"] = video.ground_truth
            _save_atomic(obs_path, obs_file)
        print(f"{prefix} Stage 1: cached", flush=True)
    else:
        if merge_judges:
            # Merge mode is strictly additive over existing artifacts: never
            # extract (or record errors for) videos with no cached observations.
            print(f"{prefix} Stage 1: SKIP (merge) — no cached observations",
                  flush=True)
            return
        prior = _prior_error(obs_path)
        if prior is not None and not retry_failed:
            print(f"{prefix} Stage 1: failed previously, skipping "
                  f"(use --retry-failed) — {prior[:90]}", flush=True)
            return
        if not _claim(obs_path):
            print(f"{prefix} Stage 1: claimed by another worker, loading...", flush=True)
            # Wait briefly for the other worker to finish
            for _ in range(30):
                await asyncio.sleep(2)
                if _is_done(obs_path):
                    break
            if _is_done(obs_path):
                with open(obs_path, encoding="utf-8") as f:
                    obs_file = json.load(f)
                observations = obs_file.get("observations")
            else:
                print(f"{prefix} Stage 1: SKIP — could not load observation", flush=True)
                return
        else:
            if not os.path.isfile(video.video_path):
                _save_atomic(obs_path, {
                    "video_id": video.video_id,
                    "error": f"FileNotFoundError: {video.video_path}",
                })
                print(f"{prefix} Stage 1: SKIP — video file not found", flush=True)
                return

            print(f"{prefix} Stage 1: extracting...", flush=True)
            t0 = time.monotonic()
            try:
                await rate_limit_events["extractor"].wait()
                async with extractor_semaphore:
                    (observations, raw, costs), _ = await with_retry_async(
                        fn=lambda: extractor.extract_async(video.video_path),
                        attempts=retry_attempts,
                        delay_s=retry_delay_s,
                        backoff=retry_backoff,
                        jitter=retry_jitter,
                        label=f"{video.video_id}/obs",
                        retryable_exceptions=extractor.RETRYABLE_EXCEPTIONS,
                        rate_limit_exceptions=extractor.RATE_LIMIT_EXCEPTIONS,
                        rate_limit_pause_s=rate_limit_pause_s,
                        rate_limit_ok=rate_limit_events["extractor"],
                    )
                elapsed = round(time.monotonic() - t0, 2)
                _save_atomic(obs_path, {
                    "video_id": video.video_id,
                    "observations": observations,
                    "ground_truth_label": video.ground_truth,
                    "timing_s": elapsed,
                    "token_costs": costs,
                    "raw_response": raw,
                    "error": None,
                })
                print(f"{prefix} Stage 1: done ({elapsed}s)", flush=True)
            except Exception as e:
                _save_atomic(obs_path, {
                    "video_id": video.video_id,
                    "error": f"{type(e).__name__}: {e}",
                })
                print(f"{prefix} Stage 1: FAIL — {type(e).__name__}: {e}", flush=True)
                return

    if observations is None:
        print(f"{prefix} Stage 1: SKIP — no observations available", flush=True)
        return

    # ── Stage 2: Report generation (all models concurrently) ───────────────

    async def _generate_one(model_key: str, model: ReportModel) -> Optional[dict]:
        rpath = _report_path(bench_dir, language, model_key, video.video_id)

        if _is_done(rpath):
            with open(rpath, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("report_output")

        if merge_judges:
            # Merge mode never generates new reports — cache only. The judge
            # merge below only needs the reports named in each video's cached
            # anonymization mapping.
            return None

        prior = _prior_error(rpath)
        if prior is not None and not retry_failed:
            print(f"{prefix} Stage 2 [{model_key}]: failed previously, skipping "
                  f"(use --retry-failed)", flush=True)
            return None

        if not _claim(rpath):
            for _ in range(30):
                await asyncio.sleep(2)
                if _is_done(rpath):
                    break
            if _is_done(rpath):
                with open(rpath, encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("report_output")
            return None

        try:
            await rate_limit_events[model_key].wait()
            async with model_semaphores[model_key]:
                t0 = time.monotonic()
                (output, raw, costs), _ = await with_retry_async(
                    fn=lambda: model.generate_report_async(observations, language),
                    attempts=retry_attempts,
                    delay_s=retry_delay_s,
                    backoff=retry_backoff,
                    jitter=retry_jitter,
                    label=f"{video.video_id}/{model_key}",
                    retryable_exceptions=model.RETRYABLE_EXCEPTIONS,
                    rate_limit_exceptions=model.RATE_LIMIT_EXCEPTIONS,
                    rate_limit_pause_s=rate_limit_pause_s,
                    rate_limit_ok=rate_limit_events[model_key],
                )
                elapsed = round(time.monotonic() - t0, 2)
            _save_atomic(rpath, {
                "video_id": video.video_id,
                "model": model_key,
                "report_output": output,
                "timing_s": elapsed,
                "token_costs": costs,
                "raw_response": raw,
                "error": None,
            })
            print(f"{prefix} Stage 2 [{model_key}]: done ({elapsed}s)", flush=True)
            return output
        except Exception as e:
            _save_atomic(rpath, {
                "video_id": video.video_id,
                "model": model_key,
                "error": f"{type(e).__name__}: {e}",
            })
            print(
                f"{prefix} Stage 2 [{model_key}]: FAIL — {type(e).__name__}: {e}",
                flush=True,
            )
            return None

    report_results = await asyncio.gather(
        *[_generate_one(mk, m) for mk, m in report_models.items()]
    )
    all_reports: dict[str, dict] = {
        mk: r
        for mk, r in zip(report_models.keys(), report_results)
        if r is not None
    }

    if not all_reports:
        print(f"{prefix} Stage 3: SKIP — no reports generated", flush=True)
        return

    # ── Stage 3: Judge scoring (all judges concurrently) ───────────────────
    judge_path = _judge_path(bench_dir, language, video.video_id)

    if merge_judges and not os.path.isfile(judge_path):
        # Strictly additive: never create a fresh (single-judge-set) result for
        # a video the previous run didn't judge.
        print(f"{prefix} Stage 3: SKIP (merge) — no cached judge result",
              flush=True)
        return

    # Scores/metadata carried over from a previous partial run
    judges_to_run: dict[str, JudgeModel] = dict(judges)
    carry_scores: dict[str, dict] = {}
    carry_costs: dict[str, dict] = {}
    carry_raws: dict[str, str] = {}
    carry_errors: dict[str, Optional[str]] = {}
    reuse_key_to_label: Optional[dict] = None
    reuse_label_to_key: Optional[dict] = None

    if os.path.isfile(judge_path):
        try:
            with open(judge_path, encoding="utf-8") as f:
                existing = json.load(f)

            if existing.get("status") == "in_progress":
                print(f"{prefix} Stage 3: claimed by another worker", flush=True)
                return

            cached_model_keys = set(existing.get("model_key_to_label", {}).keys())
            cached_judge_keys = set(existing.get("judge_keys", []))

            if existing.get("error") is not None and not retry_failed:
                print(f"{prefix} Stage 3: failed previously, skipping "
                      f"(use --retry-failed)", flush=True)
                return

            if (merge_judges
                    and existing.get("error") is None
                    and cached_model_keys <= set(all_reports.keys())):
                # Additive merge: keep cached scores for still-configured judges,
                # run only current judges without scores, prune retired judges.
                # Subset (not equality): new judges score exactly the reports in
                # the cached anonymization mapping, so extra cached reports from
                # models added to the panel later are simply not judged here.
                cached_scores = existing.get("judge_scores", {})
                keep = configured_judge_keys or set(judges.keys())
                pruned = [jk for jk in cached_scores if jk not in keep]
                missing = [jk for jk in judges if jk not in cached_scores]
                if not missing and not pruned:
                    print(f"{prefix} Stage 3: cached (merge: nothing to add)",
                          flush=True)
                    return
                judges_to_run = {jk: judges[jk] for jk in missing}
                carry_scores = {k: v for k, v in cached_scores.items() if k in keep}
                carry_costs = {k: v for k, v in
                               (existing.get("judge_token_costs") or {}).items()
                               if k in keep}
                carry_raws = {k: v for k, v in
                              (existing.get("judge_raw_responses") or {}).items()
                              if k in keep}
                carry_errors = {k: v for k, v in
                                (existing.get("judge_errors") or {}).items()
                                if k in keep}
                reuse_key_to_label = existing.get("model_key_to_label")
                reuse_label_to_key = existing.get("label_to_model_key")
                msg = f"merging {len(missing)} new judge(s): {missing}"
                if pruned:
                    msg += f", pruning retired: {pruned}"
                print(f"{prefix} Stage 3: {msg}...", flush=True)
            elif (existing.get("error") is None
                    and cached_model_keys == set(all_reports.keys())
                    and cached_judge_keys == set(judges.keys())):
                # Identify which judges have no scores (failed or missing)
                failed_judge_keys = [
                    jk for jk in judges
                    if jk not in existing.get("judge_scores", {})
                ]
                if not failed_judge_keys:
                    print(f"{prefix} Stage 3: cached", flush=True)
                    return
                # Partial retry: only re-run failed judges; carry over the rest
                judges_to_run = {jk: judges[jk] for jk in failed_judge_keys}
                carry_scores = dict(existing.get("judge_scores", {}))
                carry_costs  = dict(existing.get("judge_token_costs", {}))
                carry_raws   = dict(existing.get("judge_raw_responses", {}))
                carry_errors = dict(existing.get("judge_errors", {}))
                reuse_key_to_label = existing.get("model_key_to_label")
                reuse_label_to_key = existing.get("label_to_model_key")
                print(
                    f"{prefix} Stage 3: retrying {len(judges_to_run)} failed judge(s): "
                    f"{list(judges_to_run.keys())}...",
                    flush=True,
                )
            elif merge_judges and existing.get("error") is None:
                # Reports for some judged models are gone. Re-judging would draw
                # a fresh mapping and discard cached scores — refuse instead.
                print(
                    f"{prefix} Stage 3: SKIP (merge) — cached model set "
                    f"{sorted(cached_model_keys)} not covered by available "
                    f"reports {sorted(all_reports.keys())}; not overwriting",
                    flush=True,
                )
                return
            # else: models/judges changed → full invalidation, fall through
        except Exception:
            pass  # corrupt file — fall through to full re-run

    # Claim via a sidecar lock, leaving any existing result file untouched: on a
    # partial retry it still holds the judges we are carrying over, and it must
    # survive an interrupt until the merged write replaces it atomically.
    if not _claim_lock(judge_path):
        print(f"{prefix} Stage 3: claimed by another worker", flush=True)
        return

    # Anonymization: reuse cached mapping for partial retry, fresh random for full run
    if reuse_key_to_label is not None:
        key_to_label = reuse_key_to_label
        label_to_key = reuse_label_to_key
    else:
        key_to_label, label_to_key = _assign_labels(list(all_reports.keys()))
    anon_reports = {
        label: all_reports[key]
        for key, label in key_to_label.items()
        if key in all_reports
    }

    n_running = len(judges_to_run)
    if not merge_judges:
        run_label = (
            f"judging ({n_running} judges)"
            if n_running == len(judges)
            else f"retrying {n_running}/{len(judges)} failed judges"
        )
        print(f"{prefix} Stage 3: {run_label}...", flush=True)
    t0 = time.monotonic()

    async def _run_one_judge(
        judge_key: str, judge: JudgeModel
    ) -> tuple[str, Optional[dict], str, dict, Optional[str]]:
        """Run a single judge; returns (key, scores, raw, costs, error_str)."""
        try:
            await rate_limit_events[judge_key].wait()
            async with judge_semaphores[judge_key]:
                (scores_by_label, raw, costs), _ = await with_retry_async(
                    fn=lambda: judge.score_async(observations, anon_reports, language),
                    attempts=retry_attempts,
                    delay_s=retry_delay_s,
                    backoff=retry_backoff,
                    jitter=retry_jitter,
                    label=f"{video.video_id}/{judge_key}",
                    retryable_exceptions=judge.RETRYABLE_EXCEPTIONS,
                    rate_limit_exceptions=judge.RATE_LIMIT_EXCEPTIONS,
                    rate_limit_pause_s=rate_limit_pause_s,
                    rate_limit_ok=rate_limit_events[judge_key],
                )
            return judge_key, scores_by_label, raw, costs, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"{prefix} Stage 3 [{judge_key}]: FAIL — {err}", flush=True)
            return judge_key, None, "", {}, err

    try:
        judge_results = await asyncio.gather(
            *[_run_one_judge(jk, jm) for jk, jm in judges_to_run.items()]
        )

        elapsed = round(time.monotonic() - t0, 2)

        # Merge new results on top of any carried-over scores
        judge_scores: dict[str, dict] = {**carry_scores}
        judge_costs:  dict[str, dict] = {**carry_costs}
        judge_errors: dict[str, Optional[str]] = {**carry_errors}
        judge_raws:   dict[str, str] = {**carry_raws}

        for jk, scores, raw, costs, err in judge_results:
            judge_errors[jk] = err
            if raw:
                judge_raws[jk] = raw
            if scores is not None:
                judge_scores[jk] = scores
                judge_costs[jk]  = costs

        if not judge_scores:
            # Only reachable when nothing was carried over, so no good scores are
            # being overwritten by this error record.
            _save_atomic(judge_path, {
                "video_id": video.video_id,
                "error": "All judges failed",
                "judge_errors": judge_errors,
            })
            print(f"{prefix} Stage 3: FAIL — all judges failed", flush=True)
            return

        # In merge mode the attempted set is carried judges + newly run ones;
        # otherwise it is exactly this run's judge set (unchanged behavior).
        if merge_judges:
            final_judge_keys = list(dict.fromkeys(
                [*carry_scores.keys(), *judges_to_run.keys()]
            ))
        else:
            final_judge_keys = list(judges.keys())

        _save_atomic(judge_path, {
            "video_id": video.video_id,
            "model_order": list(key_to_label.values()),
            "model_key_to_label": key_to_label,
            "label_to_model_key": label_to_key,
            "judge_keys": final_judge_keys,
            "judge_scores": judge_scores,
            "judge_raw_responses": judge_raws,
            "judge_errors": judge_errors,
            "judge_token_costs": judge_costs,
            "timing_s": elapsed,
            "error": None,
        })
        succeeded = sum(1 for s in judge_scores.values() if s)
        print(f"{prefix} Stage 3: done ({elapsed}s, "
              f"{succeeded}/{len(final_judge_keys)} judges)",
              flush=True)
    finally:
        # Released on every path, including CancelledError from a hard shutdown, so
        # the slot is re-claimable on resume. If we are cancelled the existing result
        # file is still on disk with its previously-completed judges intact.
        _release_lock(judge_path)


# ---------------------------------------------------------------------------
# Async runner — concurrent videos
# ---------------------------------------------------------------------------

async def _run_pipeline(
    videos: list[VideoRecord],
    bench_dir: str,
    language: str,
    extractor: ObservationExtractor,
    report_models: dict[str, ReportModel],
    judges: dict[str, JudgeModel],
    max_concurrent_videos: int,
    extractor_max_concurrent: int,
    judge_max_concurrent: dict[str, int],
    model_max_concurrent: dict[str, int],
    retry_attempts: int,
    retry_delay_s: float,
    retry_backoff: float,
    retry_jitter: float,
    rate_limit_pause_s: float,
    graceful_shutdown_timeout_s: float,
    retry_failed: bool = False,
    merge_judges: bool = False,
    configured_judge_keys: Optional[set] = None,
) -> None:
    total = len(videos)
    print(f"\n{'='*60}")
    print(f"Benchmark B — {total} videos  [language: {language}]")
    print(f"Report models: {list(report_models.keys())}")
    print(f"Judges: {list(judges.keys())}"
          + ("  [merge mode]" if merge_judges else ""))
    print(f"Max concurrent videos: {max_concurrent_videos}")
    print(f"{'='*60}\n")

    video_semaphore = asyncio.Semaphore(max_concurrent_videos)
    extractor_semaphore = asyncio.Semaphore(extractor_max_concurrent)
    judge_semaphores = {
        jk: asyncio.Semaphore(judge_max_concurrent.get(jk, 2))
        for jk in judges
    }
    model_semaphores = {
        mk: asyncio.Semaphore(model_max_concurrent.get(mk, 5))
        for mk in report_models
    }

    # Per-component rate-limit events (SET = ok to proceed)
    rate_limit_events: dict[str, asyncio.Event] = {
        "extractor": asyncio.Event(),
        **{jk: asyncio.Event() for jk in judges},
        **{mk: asyncio.Event() for mk in report_models},
    }
    for e in rate_limit_events.values():
        e.set()

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_shutdown(signum, frame):
        print("\n[Benchmark B] Shutdown signal received. Finishing in-flight videos...")
        loop.call_soon_threadsafe(shutdown_event.set)

    saved_sigint = signal.signal(signal.SIGINT, _on_shutdown)
    saved_sigterm = None
    if hasattr(signal, "SIGTERM"):
        saved_sigterm = signal.signal(signal.SIGTERM, _on_shutdown)

    async def _process_with_semaphore(video: VideoRecord) -> None:
        if shutdown_event.is_set():
            return
        async with video_semaphore:
            await _process_video(
                video=video,
                bench_dir=bench_dir,
                language=language,
                extractor=extractor,
                report_models=report_models,
                judges=judges,
                model_semaphores=model_semaphores,
                judge_semaphores=judge_semaphores,
                extractor_semaphore=extractor_semaphore,
                retry_attempts=retry_attempts,
                retry_delay_s=retry_delay_s,
                retry_backoff=retry_backoff,
                retry_jitter=retry_jitter,
                rate_limit_pause_s=rate_limit_pause_s,
                rate_limit_events=rate_limit_events,
                retry_failed=retry_failed,
                merge_judges=merge_judges,
                configured_judge_keys=configured_judge_keys,
            )

    tasks = [asyncio.create_task(_process_with_semaphore(v)) for v in videos]

    async def _watcher() -> None:
        await shutdown_event.wait()
        print(
            f"[Benchmark B] Waiting up to {graceful_shutdown_timeout_s}s "
            f"for in-flight videos..."
        )
        await asyncio.sleep(graceful_shutdown_timeout_s)
        still_running = [t for t in tasks if not t.done()]
        if still_running:
            print(f"[Benchmark B] Hard-cancelling {len(still_running)} task(s).")
            for t in still_running:
                t.cancel()

    watcher = asyncio.create_task(_watcher())
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        for e in rate_limit_events.values():
            e.set()
        signal.signal(signal.SIGINT, saved_sigint)
        if saved_sigterm is not None:
            signal.signal(signal.SIGTERM, saved_sigterm)

    print("\n[Benchmark B] Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Benchmark B: LLM report quality evaluation on AV-ASD videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.runners.run_benchmark_b
  python -m benchmark.runners.run_benchmark_b --max-videos 5
  python -m benchmark.runners.run_benchmark_b --score-only
  python -m benchmark.runners.run_benchmark_b --csv dataset/csvs/val.csv
        """,
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos (useful for dev/testing)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip inference. Re-score from existing results and update leaderboard.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override the CSV split to use (default: test_csv from settings.yaml)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Re-attempt stages that previously errored. Off by default so failures "
            "are durable: a Stage 2 report that fails then succeeds on retry changes "
            "the model set for that video, which invalidates its cached judge result "
            "and re-runs every judge."
        ),
    )
    parser.add_argument(
        "--language",
        choices=["en"],
        default="en",
        help="Report language (English)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Override paths.results_dir from settings.yaml; results go to "
            "<results-dir>/benchmark_b. Use a separate directory to run a different "
            "model set without overwriting another run's cached judge results."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of role: report keys from models.yaml to run "
            "(default: all). Only the listed models are instantiated, so missing "
            "API keys for the others don't block the run. NOTE: the Stage 3 cache "
            "is keyed on each video's report model set — adding models later "
            "re-judges every video."
        ),
    )
    parser.add_argument(
        "--judges",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of role: judge keys from models.yaml to run "
            "(default: all). Only the listed judges are instantiated, so missing "
            "API keys for the others don't block the run. NOTE: the Stage 3 cache "
            "is keyed on the exact judge set — a run with a subset invalidates "
            "cached full-set results and rewrites judge files with only the "
            "subset's scores (and vice versa). Combine with --merge-judges to "
            "add scores instead."
        ),
    )
    parser.add_argument(
        "--merge-judges",
        action="store_true",
        help=(
            "Additive Stage 3: keep cached judge scores (reusing each video's "
            "anonymization mapping), run only judges in the current set that "
            "have no cached scores, and prune scores of judges no longer "
            "configured as role: judge in models.yaml. Videos whose cached "
            "model set differs from this run's reports are skipped with a "
            "warning instead of re-judged. Typical use: --judges new-judge "
            "--merge-judges to add one judge on top of an existing run."
        ),
    )

    args = parser.parse_args(argv)

    language: str = args.language

    settings = _load_yaml("benchmark/config/settings.yaml")
    models_config = _load_yaml("benchmark/config/models.yaml")

    results_dir = _resolve_path(args.results_dir or settings["paths"]["results_dir"])
    bench_dir = os.path.join(results_dir, "benchmark_b")

    bench_cfg = settings.get("benchmark_b", {})
    # One clip file per record (see loader_b).
    clips_dir = _resolve_path(
        bench_cfg.get("clips_dir") or settings["paths"]["clips_cache_dir"]
    )
    # Resolution order: --csv flag > benchmark_b.csv_path > paths.test_csv
    csv_path = _resolve_path(
        args.csv
        or bench_cfg.get("csv_path")
        or settings["paths"]["test_csv"]
    )
    retry_attempts = bench_cfg.get("retry_attempts", 3)
    retry_delay_s = bench_cfg.get("retry_delay_s", 5.0)
    retry_backoff = bench_cfg.get("retry_backoff", 2.0)
    retry_jitter = bench_cfg.get("retry_jitter", 0.5)
    rate_limit_pause_s = bench_cfg.get("rate_limit_pause_s", 15.0)
    graceful_shutdown_timeout_s = bench_cfg.get("graceful_shutdown_timeout_s", 30.0)
    max_concurrent_videos = bench_cfg.get("max_concurrent_videos", 3)
    label_column = bench_cfg.get("label_column")
    invert_label: bool = bench_cfg.get("invert_label", False)

    models_only: Optional[list[str]] = (
        [k.strip() for k in args.models.split(",") if k.strip()]
        if args.models else None
    )

    # --- score-only: re-score from existing results ---
    if args.score_only:
        # Scoring only reads result files — model keys suffice, instances are
        # never called — so missing API keys must not block it.
        report_models = _instantiate_report_models(
            models_config, tolerate_failures=True, only=models_only
        )
        print(f"Scoring from existing results  [language: {language}]...")
        for model_key in report_models:
            try:
                score_and_save_model(results_dir, model_key, language=language)
                print(f"  Summary saved: benchmark_b/summary_{language}_{model_key}.json")
            except Exception as e:
                print(f"  [ERROR] Could not score '{model_key}': {e}")
        leaderboard = assemble_leaderboard(results_dir, language=language)
        print(f"Leaderboard saved: benchmark_b/leaderboard_{language}.json")
        print_summary_table(leaderboard)
        return

    # --- Full pipeline run ---
    videos = load_videos(
        csv_path=csv_path,
        clips_dir=clips_dir,
        label_column=label_column,
        invert_label=invert_label,
        max_videos=args.max_videos,
    )
    if not videos:
        print("No clips loaded. Check your CSV path and settings.")
        sys.exit(1)

    # Instantiate models
    try:
        _, extractor = _instantiate_extractor(models_config, settings)
    except Exception as e:
        print(f"[ERROR] Cannot instantiate observation extractor: {e}")
        sys.exit(1)

    try:
        report_models = _instantiate_report_models(
            models_config, tolerate_failures=args.merge_judges, only=models_only
        )
    except Exception as e:
        print(f"[ERROR] Cannot instantiate report models: {e}")
        sys.exit(1)

    judges_only: Optional[list[str]] = (
        [k.strip() for k in args.judges.split(",") if k.strip()]
        if args.judges else None
    )
    try:
        judges = _instantiate_judges(models_config, only=judges_only)
    except Exception as e:
        print(f"[ERROR] Cannot instantiate judges: {e}")
        sys.exit(1)

    # Full configured judge set (regardless of --judges filter) — merge mode
    # keeps cached scores for these and prunes everything else (retired judges).
    configured_judge_keys: set = {
        k for k, cfg in models_config.get("models", {}).items()
        if cfg.get("role") == "judge"
    }

    # Clean orphaned placeholders from any previous crashed run
    placeholders, cleared_errors, locks = _clean_orphans(
        bench_dir, retry_failed=args.retry_failed)
    if placeholders:
        print(f"Cleaned {placeholders} orphaned in-progress placeholder(s).")
    if locks:
        print(f"Cleared {locks} stale lock(s) from an interrupted run.")
    if cleared_errors:
        print(f"Cleared {cleared_errors} previously-failed result(s) for retry "
              f"(--retry-failed).")

    # Per-model max_concurrent from models.yaml
    all_models_cfg = models_config.get("models", {})
    extractor_max = bench_cfg.get("extractor_max_concurrent", 3)
    _judge_max_default = bench_cfg.get("judge_max_concurrent", 2)
    judge_max: dict[str, int] = {
        jk: all_models_cfg.get(jk, {}).get("max_concurrent", _judge_max_default)
        for jk in judges
    }
    model_max: dict[str, int] = {
        mk: all_models_cfg.get(mk, {}).get("max_concurrent", 5)
        for mk in report_models
    }

    asyncio.run(
        _run_pipeline(
            videos=videos,
            bench_dir=bench_dir,
            language=language,
            extractor=extractor,
            report_models=report_models,
            judges=judges,
            max_concurrent_videos=max_concurrent_videos,
            extractor_max_concurrent=extractor_max,
            judge_max_concurrent=judge_max,
            model_max_concurrent=model_max,
            retry_attempts=retry_attempts,
            retry_delay_s=retry_delay_s,
            retry_backoff=retry_backoff,
            retry_jitter=retry_jitter,
            rate_limit_pause_s=rate_limit_pause_s,
            graceful_shutdown_timeout_s=graceful_shutdown_timeout_s,
            retry_failed=args.retry_failed,
            merge_judges=args.merge_judges,
            configured_judge_keys=configured_judge_keys,
        )
    )

    # Score and build leaderboard
    print(f"\nScoring results  [language: {language}]...")
    for model_key in report_models:
        try:
            score_and_save_model(results_dir, model_key, language=language)
            print(f"  Summary saved: benchmark_b/summary_{language}_{model_key}.json")
        except Exception as e:
            print(f"  [WARNING] Could not score '{model_key}': {e}")

    try:
        leaderboard = assemble_leaderboard(results_dir, language=language)
        print(f"Leaderboard saved: benchmark_b/leaderboard_{language}.json")
        print_summary_table(leaderboard)
    except Exception as e:
        print(f"[WARNING] Could not assemble leaderboard: {e}")


if __name__ == "__main__":
    main()
