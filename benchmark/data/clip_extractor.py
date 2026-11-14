"""
On-demand clip extraction using ffmpeg.
Reuses the same logic as scripts/data_preparation/extract_clips.py
but as a function-based API with temp-dir and cache support.

Both sync and async variants are provided:
  extract_clip / temp_clip          — synchronous, for the sequential runner
  async_extract_clip / async_temp_clip — async, for the concurrent runner

The async variants use asyncio.create_subprocess_exec (no shell=True, args as list)
so ffmpeg never blocks the event loop. On Windows this requires ProactorEventLoop,
which is the default since Python 3.8.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".webm"]


def find_original_video(video_id: str, original_videos_dir: str) -> Optional[str]:
    """Search for a video file by ID across supported extensions."""
    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(original_videos_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def _is_cached(path: str) -> bool:
    """True only for a non-empty existing clip.

    A bare os.path.exists() check is not enough: a failed run can leave a
    0-byte .mp4 behind, and treating that as a cache hit would hand an empty
    file to the model instead of re-cutting it.
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def _discard_partial(output_path: str) -> None:
    """Delete an empty output file left behind by a failed ffmpeg run.

    ffmpeg creates the output file before it discovers it cannot decode the
    requested range (e.g. a truncated source that hits EOF before the start
    offset). Leaving the 0-byte file behind would make _is_cached treat the
    failure as a finished clip on the next run.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
        os.remove(output_path)


def _check_output(output_path: str, clip_id: str) -> None:
    """Raise if ffmpeg reported success but produced nothing usable."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return
    _discard_partial(output_path)
    raise RuntimeError(
        f"ffmpeg reported success but wrote no data for '{clip_id}' — the "
        f"source video is likely truncated or shorter than the clip offset."
    )


# ------------------------------------------------------------------
# Synchronous path
# ------------------------------------------------------------------

def extract_clip(
    video_id: str,
    start_sec: int,
    end_sec: int,
    original_videos_dir: str,
    output_dir: str,
    clip_id: str,
) -> str:
    """
    Extract a clip segment from an original video using ffmpeg.

    Args:
        video_id: Base video filename without extension (e.g. '_gEps3r7FfQ').
        start_sec: Clip start in seconds.
        end_sec: Clip end in seconds.
        original_videos_dir: Directory containing the source video files.
        output_dir: Directory to write the extracted clip.
        clip_id: Used as the output filename stem (e.g. '_gEps3r7FfQ_50_53').

    Returns:
        Absolute path to the extracted clip .mp4 file.

    Raises:
        FileNotFoundError: Clip not cached and source video not available.
        RuntimeError: ffmpeg failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_id}.mp4")

    # Cache check comes first: a clip already cut in output_dir is used without
    # looking up the original video.
    if _is_cached(output_path):
        return output_path

    original_path = find_original_video(video_id, original_videos_dir)
    if not original_path:
        raise FileNotFoundError(
            f"Clip '{clip_id}' is not in '{output_dir}' and the original video "
            f"'{video_id}' is not in '{original_videos_dir}' to re-cut it from"
        )

    duration = end_sec - start_sec
    command = (
        f'ffmpeg -y -ss {start_sec} -t {duration} '
        f'-i "{original_path}" '
        f'-c:v libx264 -c:a aac '
        f'"{output_path}"'
    )

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _discard_partial(output_path)
        stderr_snippet = result.stderr.strip().splitlines()
        # Show last 5 lines of ffmpeg output — most relevant error is at the bottom
        stderr_tail = "\n".join(stderr_snippet[-5:]) if stderr_snippet else "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) for '{clip_id}':\n{stderr_tail}"
        )

    _check_output(output_path, clip_id)
    return output_path


@contextmanager
def temp_clip(
    video_id: str,
    start_sec: int,
    end_sec: int,
    original_videos_dir: str,
    clip_id: str,
):
    """
    Context manager that extracts a clip to a temporary directory and cleans
    it up on exit. Use this when --keep-clips is False.

    Usage:
        with temp_clip(video_id, start, end, videos_dir, clip_id) as clip_path:
            model.predict(clip_path)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = extract_clip(
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            original_videos_dir=original_videos_dir,
            output_dir=tmp_dir,
            clip_id=clip_id,
        )
        yield clip_path


# ------------------------------------------------------------------
# Async path
# ------------------------------------------------------------------

async def async_extract_clip(
    video_id: str,
    start_sec: int,
    end_sec: int,
    original_videos_dir: str,
    output_dir: str,
    clip_id: str,
) -> str:
    """
    Async version of extract_clip. Uses asyncio.create_subprocess_exec so
    ffmpeg runs as a non-blocking subprocess and does not block the event loop.

    Args and returns: same as extract_clip().

    Raises:
        FileNotFoundError: Clip not cached and source video not available.
        RuntimeError: ffmpeg failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_id}.mp4")

    # Cache check comes first — see extract_clip() for why.
    if _is_cached(output_path):
        return output_path

    original_path = find_original_video(video_id, original_videos_dir)
    if not original_path:
        raise FileNotFoundError(
            f"Clip '{clip_id}' is not in '{output_dir}' and the original video "
            f"'{video_id}' is not in '{original_videos_dir}' to re-cut it from"
        )

    duration = end_sec - start_sec
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", original_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_bytes = await proc.communicate()

    if proc.returncode != 0:
        _discard_partial(output_path)
        stderr_lines = stderr_bytes.decode(errors="replace").strip().splitlines()
        stderr_tail = "\n".join(stderr_lines[-5:]) if stderr_lines else "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}) for '{clip_id}':\n{stderr_tail}"
        )

    _check_output(output_path, clip_id)
    return output_path


@asynccontextmanager
async def async_temp_clip(
    video_id: str,
    start_sec: int,
    end_sec: int,
    original_videos_dir: str,
    clip_id: str,
):
    """
    Async context manager that extracts a clip to a temporary directory and
    cleans it up on exit. Use this in the async runner when --keep-clips is False.

    Usage:
        async with async_temp_clip(video_id, start, end, videos_dir, clip_id) as clip_path:
            await model.predict_async(clip_path)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = await async_extract_clip(
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            original_videos_dir=original_videos_dir,
            output_dir=tmp_dir,
            clip_id=clip_id,
        )
        yield clip_path
