"""
Benchmark C data loader.

Scans the local clips directory for .mp4 files and returns a list of ClipRecord
objects — one per valid (non-empty) clip.

Zero-byte files are skipped with a warning; they represent failed downloads
from the AV-ASD source dataset.

clip_id is the filename without extension (e.g. "0g51c2BCqu8_0_166").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClipRecord:
    clip_id: str
    video_path: str   # absolute path to .mp4 file


def load_clips(
    clips_dir: str,
    max_clips: Optional[int] = None,
) -> list[ClipRecord]:
    """
    Load all valid clips from clips_dir.

    Args:
        clips_dir:  Directory containing .mp4 clip files.
        max_clips:  If set, limit to this many clips (dev/testing).

    Returns:
        List of ClipRecord, sorted by clip_id for reproducibility.
    """
    if not os.path.isdir(clips_dir):
        raise FileNotFoundError(
            f"Clips directory not found: '{clips_dir}'. "
            f"Run the download step first (see BENCHMARK_C.md § Implementation Plan)."
        )

    records: list[ClipRecord] = []
    skipped_empty = 0
    skipped_other = 0

    for fname in sorted(os.listdir(clips_dir)):
        if not fname.lower().endswith(".mp4"):
            continue
        fpath = os.path.join(clips_dir, fname)
        if not os.path.isfile(fpath):
            skipped_other += 1
            continue
        if os.path.getsize(fpath) == 0:
            skipped_empty += 1
            continue
        clip_id = os.path.splitext(fname)[0]
        records.append(ClipRecord(clip_id=clip_id, video_path=fpath))
        if max_clips is not None and len(records) >= max_clips:
            break

    if skipped_empty:
        print(
            f"  [loader_c] Skipped {skipped_empty} zero-byte clip(s) "
            f"(failed downloads from AV-ASD source dataset)."
        )
    if skipped_other:
        print(f"  [loader_c] Skipped {skipped_other} non-file entries.")

    print(
        f"  [loader_c] Loaded {len(records)} clips from {clips_dir}"
        + (f" (limited to {max_clips})" if max_clips else "")
    )
    return records
