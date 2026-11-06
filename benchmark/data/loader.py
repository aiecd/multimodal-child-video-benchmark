"""
Loads AV-ASD CSV splits into ClipRecord objects.
Handles clip_id parsing (format: {video_id}_{start_sec}_{end_sec}).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from benchmark.models.base import LABEL_KEYS


@dataclass
class ClipRecord:
    clip_id: str
    video_id: str
    start_sec: int
    end_sec: int
    ground_truth: dict[str, bool]  # human-readable label → True/False

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec


def parse_clip_id(clip_id: str) -> tuple[str, int, int]:
    """
    Parse clip_id like '_gEps3r7FfQ_50_53' into (video_id, start_sec, end_sec).
    Uses rsplit from the right so video_ids with underscores are handled correctly.
    Raises ValueError if the format is invalid.
    """
    parts = clip_id.rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Cannot parse clip_id '{clip_id}': expected at least 3 underscore-separated parts")
    video_id = "_".join(parts[:-2])
    try:
        start_sec = int(parts[-2])
        end_sec = int(parts[-1])
    except ValueError:
        raise ValueError(f"Cannot parse clip_id '{clip_id}': last two parts must be integers")
    if end_sec <= start_sec:
        raise ValueError(f"Invalid clip_id '{clip_id}': end_sec ({end_sec}) must be > start_sec ({start_sec})")
    return video_id, start_sec, end_sec


def load_clips(
    csv_path: str,
    max_clips: Optional[int] = None,
) -> list[ClipRecord]:
    """
    Load clips from a CSV split file (test.csv / val.csv / train.csv).

    Args:
        csv_path: Path to the CSV file.
        max_clips: If set, only load this many clips (useful for dev runs).

    Returns:
        List of ClipRecord objects with parsed video info and ground truth labels.
        ground_truth keys are human-readable label names matching LABEL_KEYS.
    """
    df = pd.read_csv(csv_path)
    if max_clips is not None:
        df = df.head(max_clips)

    records: list[ClipRecord] = []
    skipped = 0

    for _, row in df.iterrows():
        clip_id = str(row["Video_ID"]).strip()
        try:
            video_id, start_sec, end_sec = parse_clip_id(clip_id)
        except ValueError as e:
            print(f"  [loader] Skipping invalid clip_id: {e}")
            skipped += 1
            continue

        # CSV column names match LABEL_KEYS (both are human-readable)
        ground_truth: dict[str, bool] = {}
        for label in LABEL_KEYS:
            if label not in row.index:
                print(f"  [loader] Warning: expected column '{label}' not found in CSV")
                ground_truth[label] = False
            else:
                ground_truth[label] = bool(int(row[label]))

        records.append(ClipRecord(
            clip_id=clip_id,
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            ground_truth=ground_truth,
        ))

    print(f"  [loader] Loaded {len(records)} clips ({skipped} skipped) from {csv_path}")
    return records
