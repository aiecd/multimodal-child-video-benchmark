"""
Shared machinery for the Table 4 panel B and Table 5 runners.

Both runners read and write one Stage 1/2 cache (settings.yaml →
compare_common.compare_cache_dir):

    observations/{extractor_key}/{clip_id}.json
    reports/{language}/{arm_key}/{clip_id}.json
        arm_key = "{extractor_key}__{report_model_key}" for a two-step pipeline,
                  or the model key for a single-call video → report baseline

Concurrency safety
------------------
The two runners may run at the same time against the shared cache. Every claim
(in-progress placeholder or judge .lock) records the owning PID, and startup
cleanup removes only claims whose process is gone. Benchmark B's cleanup removes
every claim unconditionally, which would delete a live run's work here.

Failures are durable, as in Benchmark B: a stage that errored is skipped on the
next run unless --retry-failed is passed. Judges that failed on a clip are always
re-attempted on the next run (their scores are missing, not errored).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from benchmark.utils.retry import with_retry_async

BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_CLAIM_WAIT_POLLS = 120      # × _CLAIM_POLL_S — how long to wait on another run's claim
_CLAIM_POLL_S = 5


# ---------------------------------------------------------------------------
# Paths and instantiation
# ---------------------------------------------------------------------------

def resolve_path(rel: str) -> str:
    return os.path.join(BENCH_ROOT, rel)


def obs_path(cache_dir: str, extractor_key: str, clip_id: str) -> str:
    return os.path.join(cache_dir, "observations", extractor_key, f"{clip_id}.json")


def report_path(cache_dir: str, language: str, arm_key: str, clip_id: str) -> str:
    return os.path.join(cache_dir, "reports", language, arm_key, f"{clip_id}.json")


def pipeline_arm_key(extractor_key: str, report_model_key: str) -> str:
    return f"{extractor_key}__{report_model_key}"


def instantiate(models_config: dict, key: str, common_cfg: dict, field_name: str = "class"):
    """Instantiate a models.yaml entry, injecting the shared Files API polling settings."""
    models = models_config.get("models", {})
    if key not in models:
        raise KeyError(f"Model '{key}' not found in models.yaml")
    cfg = dict(models[key])
    # GeminiObservationExtractor reads file_poll_*, GeminiVideoReportModel reads poll_*
    cfg.setdefault("file_poll_interval_s", common_cfg.get("file_poll_interval_s", 5))
    cfg.setdefault("file_poll_timeout_s", common_cfg.get("file_poll_timeout_s", 180))
    cfg.setdefault("poll_interval_s", cfg["file_poll_interval_s"])
    cfg.setdefault("poll_timeout_s", cfg["file_poll_timeout_s"])
    class_path = cfg.get(field_name) or cfg["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)(cfg)


# ---------------------------------------------------------------------------
# Result files, claims and cleanup
# ---------------------------------------------------------------------------

def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_done(path: str) -> bool:
    data = load_json(path)
    return data is not None and data.get("status") != "in_progress" and data.get("error") is None


def prior_error(path: str) -> Optional[str]:
    data = load_json(path)
    if data is None or data.get("status") == "in_progress":
        return None
    return data.get("error")


def save_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def claim(path: str) -> bool:
    """Atomically claim a result slot with a placeholder that records our PID."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:
            json.dump({"status": "in_progress", "pid": os.getpid()}, f)
        return True
    except FileExistsError:
        return False


def _lock_path(path: str) -> str:
    return path + ".lock"


def claim_lock(path: str) -> bool:
    """Claim a judge slot via a sidecar lock, leaving the result file (and the
    judge scores it may already hold) untouched until the merged write."""
    lock = _lock_path(path)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        with open(lock, "x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def release_lock(path: str) -> None:
    try:
        os.remove(_lock_path(path))
    except FileNotFoundError:
        pass


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def clean_stale(root: str, retry_failed: bool = False) -> tuple[int, int, int]:
    """
    Remove claims left by dead processes (and, with retry_failed, error files).
    Claims held by a live process — e.g. the other comparison runner sharing the
    cache — are left alone. Returns (placeholders, error_files, locks) removed.
    """
    placeholders = errors = locks = 0
    if not os.path.isdir(root):
        return 0, 0, 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.endswith(".lock"):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        pid = int(f.read().strip() or 0)
                except Exception:
                    pid = 0
                if not _pid_alive(pid):
                    try:
                        os.remove(fpath)
                        locks += 1
                    except FileNotFoundError:
                        pass
                continue
            if fname.endswith(".tmp"):
                continue
            if not fname.endswith(".json"):
                continue
            data = load_json(fpath)
            if data is None:
                continue
            if data.get("status") == "in_progress":
                if not _pid_alive(data.get("pid")):
                    os.remove(fpath)
                    placeholders += 1
            elif data.get("error") is not None and retry_failed:
                os.remove(fpath)
                errors += 1
    return placeholders, errors, locks


def report_cleanup(label: str, counts: tuple[int, int, int]) -> None:
    placeholders, errors, locks = counts
    if placeholders:
        print(f"[{label}] Cleaned {placeholders} orphaned in-progress placeholder(s).")
    if locks:
        print(f"[{label}] Cleared {locks} stale lock(s).")
    if errors:
        print(f"[{label}] Cleared {errors} previously-failed result(s) for retry (--retry-failed).")


# ---------------------------------------------------------------------------
# Calling models with per-component concurrency, retry and rate-limit pauses
# ---------------------------------------------------------------------------

@dataclass
class RunCtx:
    retry_attempts: int
    retry_delay_s: float
    retry_backoff: float
    retry_jitter: float
    rate_limit_pause_s: float
    retry_failed: bool
    limits: dict[str, int]                 # component key → max concurrent calls
    semaphores: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    events: dict[str, asyncio.Event] = field(default_factory=dict)

    def init_async(self) -> None:
        """Create asyncio primitives — must run inside the event loop."""
        for key, n in self.limits.items():
            self.semaphores[key] = asyncio.Semaphore(n)
            ev = asyncio.Event()
            ev.set()
            self.events[key] = ev


def make_ctx(common_cfg: dict, limits: dict[str, int], retry_failed: bool) -> RunCtx:
    return RunCtx(
        retry_attempts=common_cfg.get("retry_attempts", 3),
        retry_delay_s=common_cfg.get("retry_delay_s", 5.0),
        retry_backoff=common_cfg.get("retry_backoff", 2.0),
        retry_jitter=common_cfg.get("retry_jitter", 0.5),
        rate_limit_pause_s=common_cfg.get("rate_limit_pause_s", 15.0),
        retry_failed=retry_failed,
        limits=limits,
    )


def component_limit(models_config: dict, key: str, default: int) -> int:
    return models_config.get("models", {}).get(key, {}).get("max_concurrent", default)


async def call_model(
    ctx: RunCtx,
    component: str,
    model: Any,
    fn: Callable[[], Awaitable[tuple]],
    label: str,
) -> tuple[Any, str, dict, float]:
    """Run fn under the component's semaphore with retry. Returns (out, raw, costs, secs)."""
    await ctx.events[component].wait()
    async with ctx.semaphores[component]:
        t0 = time.monotonic()
        (out, raw, costs), _ = await with_retry_async(
            fn=fn,
            attempts=ctx.retry_attempts,
            delay_s=ctx.retry_delay_s,
            backoff=ctx.retry_backoff,
            jitter=ctx.retry_jitter,
            label=label,
            retryable_exceptions=model.RETRYABLE_EXCEPTIONS,
            rate_limit_exceptions=model.RATE_LIMIT_EXCEPTIONS,
            rate_limit_pause_s=ctx.rate_limit_pause_s,
            rate_limit_ok=ctx.events[component],
        )
        return out, raw, costs, round(time.monotonic() - t0, 2)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

async def cached_stage(
    ctx: RunCtx,
    path: str,
    result_key: str,
    meta: dict,
    component: str,
    model: Any,
    fn: Callable[[], Awaitable[tuple]],
    label: str,
) -> Optional[Any]:
    """
    One cached model call (Stage 1 observations or Stage 2 report).
    Returns the stored result, or None when it is unavailable this run.
    """
    if is_done(path):
        return load_json(path).get(result_key)

    err = prior_error(path)
    if err is not None and not ctx.retry_failed:
        print(f"  [{label}] failed previously, skipping (use --retry-failed) — {err[:90]}",
              flush=True)
        return None

    if not claim(path):
        # Held by another live worker or the other comparison runner
        for _ in range(_CLAIM_WAIT_POLLS):
            await asyncio.sleep(_CLAIM_POLL_S)
            if is_done(path):
                return load_json(path).get(result_key)
            if prior_error(path) is not None:
                return None
        print(f"  [{label}] still claimed by another process, skipping this run", flush=True)
        return None

    try:
        out, raw, costs, secs = await call_model(ctx, component, model, fn, label)
    except Exception as e:
        save_atomic(path, {**meta, "error": f"{type(e).__name__}: {e}"})
        print(f"  [{label}] FAIL — {type(e).__name__}: {e}", flush=True)
        return None
    except BaseException:
        # Cancelled (hard shutdown): release the claim so a resume re-runs it
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        raise

    save_atomic(path, {
        **meta,
        result_key: out,
        "timing_s": secs,
        "token_costs": costs,
        "raw_response": raw,
        "error": None,
    })
    print(f"  [{label}] done ({secs}s)", flush=True)
    return out


async def judge_stage(
    ctx: RunCtx,
    path: str,
    judges: dict[str, Any],
    label: str,
    compatible: Callable[[dict], bool],
    base_record: Callable[[Optional[dict]], dict],
    make_call: Callable[[str, Any, dict], Callable[[], Awaitable[tuple]]],
) -> None:
    """
    Run every judge on one judging unit, reusing cached per-judge scores.

    compatible(existing)    → may the cached file's scores be reused for this unit?
    base_record(existing)   → identifying fields to write (existing is the compatible
                              cached record or None) — e.g. the anonymization map,
                              which must be reused when carrying scores over.
    make_call(jk, judge, record) → zero-arg coroutine factory returning
                              (scores, raw, costs) for that judge.

    Only judges in `judges` are kept, so retired judges' scores are pruned.
    """
    carry_scores: dict[str, dict] = {}
    carry_raws: dict[str, str] = {}
    carry_costs: dict[str, dict] = {}
    reuse: Optional[dict] = None

    existing = load_json(path) if os.path.isfile(path) else None
    if existing is not None:
        if existing.get("error") is not None and not ctx.retry_failed:
            print(f"  [{label}] failed previously, skipping (use --retry-failed)", flush=True)
            return
        if existing.get("error") is None and compatible(existing):
            reuse = existing
            carry_scores = {k: v for k, v in (existing.get("judge_scores") or {}).items() if k in judges}
            carry_raws = {k: v for k, v in (existing.get("judge_raw_responses") or {}).items() if k in judges}
            carry_costs = {k: v for k, v in (existing.get("judge_token_costs") or {}).items() if k in judges}

    to_run = {jk: j for jk, j in judges.items() if jk not in carry_scores}
    stale = existing is not None and reuse is not None and set(existing.get("judge_scores") or {}) - set(judges)
    if not to_run and not stale:
        return

    if not claim_lock(path):
        print(f"  [{label}] claimed by another process", flush=True)
        return

    try:
        record = base_record(reuse)
        if to_run:
            print(f"  [{label}] judging ({len(to_run)}/{len(judges)} judges)...", flush=True)

        async def _one(jk: str, judge: Any):
            try:
                scores, raw, costs, _ = await call_model(
                    ctx, jk, judge, make_call(jk, judge, record), f"{label}/{jk}")
                return jk, scores, raw, costs, None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"  [{label}/{jk}] FAIL — {err[:200]}", flush=True)
                return jk, None, "", {}, err

        t0 = time.monotonic()
        results = await asyncio.gather(*[_one(jk, j) for jk, j in to_run.items()])
        secs = round(time.monotonic() - t0, 2)

        scores = dict(carry_scores)
        raws = dict(carry_raws)
        costs = dict(carry_costs)
        errors: dict[str, Optional[str]] = {jk: None for jk in carry_scores}
        for jk, s, raw, c, err in results:
            errors[jk] = err
            if raw:
                raws[jk] = raw
            if s is not None:
                scores[jk] = s
                costs[jk] = c

        if not scores:
            save_atomic(path, {**record, "error": "All judges failed", "judge_errors": errors})
            print(f"  [{label}] FAIL — all judges failed", flush=True)
            return

        save_atomic(path, {
            **record,
            "judge_keys": list(judges),
            "judge_scores": scores,
            "judge_raw_responses": raws,
            "judge_errors": errors,
            "judge_token_costs": costs,
            "timing_s": secs,
            "error": None,
        })
        print(f"  [{label}] judged ({len(scores)}/{len(judges)} judges, {secs}s)", flush=True)
    finally:
        release_lock(path)


# ---------------------------------------------------------------------------
# Clip loop with graceful shutdown
# ---------------------------------------------------------------------------

async def run_clips(
    clips: list,
    process: Callable[[Any], Awaitable[None]],
    ctx: RunCtx,
    max_concurrent: int,
    graceful_shutdown_timeout_s: float,
    title: str,
) -> bool:
    """Process clips concurrently. Returns True if a shutdown signal interrupted the run."""
    ctx.init_async()
    clip_sem = asyncio.Semaphore(max_concurrent)
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_shutdown(signum, frame):
        print(f"\n[{title}] Shutdown signal received. Finishing in-flight clips...")
        loop.call_soon_threadsafe(shutdown.set)

    saved_int = signal.signal(signal.SIGINT, _on_shutdown)
    saved_term = signal.signal(signal.SIGTERM, _on_shutdown) if hasattr(signal, "SIGTERM") else None

    async def _guarded(clip) -> None:
        if shutdown.is_set():
            return
        async with clip_sem:
            if shutdown.is_set():
                return
            try:
                await process(clip)
            except Exception as e:  # never let one clip take down the run
                print(f"  [{getattr(clip, 'clip_id', clip)}] ERROR — {type(e).__name__}: {e}",
                      flush=True)

    tasks = [asyncio.create_task(_guarded(c)) for c in clips]

    async def _watcher() -> None:
        await shutdown.wait()
        print(f"[{title}] Waiting up to {graceful_shutdown_timeout_s}s for in-flight clips...")
        await asyncio.sleep(graceful_shutdown_timeout_s)
        running = [t for t in tasks if not t.done()]
        if running:
            print(f"[{title}] Hard-cancelling {len(running)} task(s).")
            for t in running:
                t.cancel()

    watcher = asyncio.create_task(_watcher())
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        for ev in ctx.events.values():
            ev.set()
        signal.signal(signal.SIGINT, saved_int)
        if saved_term is not None:
            signal.signal(signal.SIGTERM, saved_term)
    print(f"\n[{title}] Pipeline complete.")
    return shutdown.is_set()
