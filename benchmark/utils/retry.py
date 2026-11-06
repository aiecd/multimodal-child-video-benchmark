"""
Exponential backoff retry for API calls.

Provides both sync (with_retry) and async (with_retry_async) variants.
Both return (result, retry_log) so the runner can record observability fields
(attempt_count, rate_limited, retry_reasons) in every per-clip JSON.

Error classification
--------------------
Only exceptions listed in `retryable_exceptions` are retried.
All other exceptions propagate immediately, consuming zero retry budget.
This prevents wasting the retry budget on permanent failures (bad API key,
missing file, malformed JSON) that will never succeed on a retry.

`rate_limit_exceptions` must be a subset of `retryable_exceptions`.
When a rate-limit exception is caught it is still retried, but it also sets
the `rate_limited` flag in the retry log entry, and in the async path it
clears the shared `rate_limit_ok` event to pause all other workers.

Jitter formula
--------------
  wait = current_delay + uniform(0, jitter * base_delay)

Jitter is always added to the current (backoff-scaled) delay. The random
component is bounded by the *initial* delay so jitter stays proportional
and does not grow unboundedly.

retry_log contract
------------------
Each entry covers one failed attempt that triggered a retry (not the final
raising attempt). When with_retry/with_retry_async returns successfully:
  attempt_count = len(retry_log) + 1
  rate_limited  = any(r["rate_limited"] for r in retry_log)
  retry_reasons = retry_log
Empty list when the first attempt succeeds (attempt_count == 1).
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    attempts: int = 3,
    delay_s: float = 5.0,
    backoff: float = 2.0,
    jitter: float = 0.5,
    label: str = "",
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    rate_limit_exceptions: tuple[type[Exception], ...] = (),
) -> tuple[T, list[dict]]:
    """
    Call fn() up to `attempts` times with exponential backoff.

    Only exceptions that are instances of `retryable_exceptions` are retried.
    All other exceptions propagate immediately without consuming retry budget.

    Returns:
        (result, retry_log) on success.
        retry_log: one dict per failed attempt that triggered a retry.
        Empty list when the first attempt succeeds.

    Raises:
        The last retryable exception after all attempts are exhausted.
        Any non-retryable exception immediately on first occurrence.
    """
    retry_log: list[dict] = []
    current_delay = delay_s

    for attempt in range(1, attempts + 1):
        try:
            return fn(), retry_log
        except Exception as e:
            if not isinstance(e, retryable_exceptions):
                raise  # permanent failure — fail immediately, no retry

            is_rate_limited = (
                bool(rate_limit_exceptions) and isinstance(e, rate_limit_exceptions)
            )
            tag = f"[{label}] " if label else ""

            if attempt < attempts:
                wait = current_delay + random.uniform(0, jitter * delay_s)
                retry_log.append({
                    "attempt": attempt,
                    "exception": type(e).__name__,
                    "message": str(e),
                    "delay_s": round(wait, 2),
                    "rate_limited": is_rate_limited,
                })
                kind = "rate-limited" if is_rate_limited else type(e).__name__
                print(
                    f"  {tag}Attempt {attempt}/{attempts} failed ({kind}). "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                current_delay *= backoff
            else:
                print(f"  {tag}All {attempts} attempts failed.")
                raise


async def with_retry_async(
    fn: Callable[[], Coroutine[Any, Any, T]],
    attempts: int = 3,
    delay_s: float = 5.0,
    backoff: float = 2.0,
    jitter: float = 0.5,
    label: str = "",
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    rate_limit_exceptions: tuple[type[Exception], ...] = (),
    rate_limit_pause_s: float = 15.0,
    rate_limit_ok: asyncio.Event | None = None,
) -> tuple[T, list[dict]]:
    """
    Async counterpart of with_retry(). Uses await asyncio.sleep() so it never
    blocks the event loop during backoff waits.

    When a rate-limit exception occurs and rate_limit_ok is provided:
      - rate_limit_ok is cleared  → all other workers pause before their next request
      - Sleeps for rate_limit_pause_s + jitter
      - rate_limit_ok is set in a finally block → always released, even on cancellation

    Returns and raises: same contract as with_retry().
    """
    retry_log: list[dict] = []
    current_delay = delay_s

    for attempt in range(1, attempts + 1):
        try:
            result = await fn()
            return result, retry_log
        except Exception as e:
            if not isinstance(e, retryable_exceptions):
                raise  # permanent failure — fail immediately

            is_rate_limited = (
                bool(rate_limit_exceptions) and isinstance(e, rate_limit_exceptions)
            )
            tag = f"[{label}] " if label else ""

            if attempt < attempts:
                if is_rate_limited and rate_limit_ok is not None:
                    # Use the longer cross-worker pause for rate limits
                    wait = rate_limit_pause_s + random.uniform(0, jitter * delay_s)
                    retry_log.append({
                        "attempt": attempt,
                        "exception": type(e).__name__,
                        "message": str(e),
                        "delay_s": round(wait, 2),
                        "rate_limited": True,
                    })
                    print(
                        f"  {tag}Attempt {attempt}/{attempts} rate-limited. "
                        f"Pausing all workers for {wait:.1f}s..."
                    )
                    rate_limit_ok.clear()
                    try:
                        await asyncio.sleep(wait)
                    finally:
                        rate_limit_ok.set()  # always released — even on task cancellation
                else:
                    wait = current_delay + random.uniform(0, jitter * delay_s)
                    retry_log.append({
                        "attempt": attempt,
                        "exception": type(e).__name__,
                        "message": str(e),
                        "delay_s": round(wait, 2),
                        "rate_limited": is_rate_limited,
                    })
                    print(
                        f"  {tag}Attempt {attempt}/{attempts} failed "
                        f"({type(e).__name__}). Retrying in {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
                    current_delay *= backoff
            else:
                print(f"  {tag}All {attempts} attempts failed.")
                raise
