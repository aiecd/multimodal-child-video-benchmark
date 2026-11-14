"""
Loads AV-ASD CSV splits into ClipRecord objects.
Handles clip_id parsing (format: {video_id}_{start_sec}_{end_sec}).
"""

from __future__ import annotations

import os
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


def read_clip_list(path: str) -> list[str]:
    """Clip ids from a one-id-per-line file such as dataset/clips_screened_pass.txt."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_clips(
    csv_path: str,
    max_clips: Optional[int] = None,
    labels_csv: Optional[str] = None,
    include: Optional[list[str]] = None,
) -> list[ClipRecord]:
    """
    Load clips from a CSV with a Video_ID column.

    Args:
        csv_path:   CSV naming the clips (dataset.csv, or a clip list such as
                    dataset/csvs/a80_test.csv, which has only Video_ID).
        max_clips:  If set, only load this many clips (useful for dev runs).
        labels_csv: AV-ASD annotations (dataset/csvs/dataset.csv). Used when csv_path
                    has no label columns, so a clip list gets its reference labels.
        include:    If set, keep only these clip ids (the 701 included clips).

    Returns:
        List of ClipRecord objects with parsed video info and ground truth labels.
        ground_truth keys are human-readable label names matching LABEL_KEYS.
    """
    df = pd.read_csv(csv_path)
    df["Video_ID"] = df["Video_ID"].astype(str).str.strip()
    if include is not None:
        keep = set(include)
        n_before = len(df)
        df = df[df["Video_ID"].isin(keep)]
        print(f"  [loader] {len(df)} of {n_before} clips are in the included list")

    missing_labels = [l for l in LABEL_KEYS if l not in df.columns]
    if missing_labels and labels_csv is not None:
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(
                f"'{csv_path}' has no label columns and '{labels_csv}' is missing. "
                f"Run: python -m benchmark.scripts.fetch_labels"
            )
        labels = pd.read_csv(labels_csv)
        labels["Video_ID"] = labels["Video_ID"].astype(str).str.strip()
        labels = labels.drop_duplicates("Video_ID").set_index("Video_ID")[LABEL_KEYS]
        unlabeled = sorted(set(df["Video_ID"]) - set(labels.index))
        if unlabeled:
            raise ValueError(f"{len(unlabeled)} clip(s) in '{csv_path}' have no labels in "
                             f"'{labels_csv}', e.g. {unlabeled[:3]}")
        df = df.drop(columns=[l for l in LABEL_KEYS if l in df.columns]).join(labels, on="Video_ID")

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
