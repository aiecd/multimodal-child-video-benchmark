"""
Table 2 data loader: one ClipRecord per included clip.

Clip files are expected at {clips_dir}/{clip_id}.mp4, where clip_id is the CSV's
Video_ID column (`{youtube_id}_{start}_{end}`). No clip-level label is used
(settings.yaml: benchmark_b.label_column is null).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ClipRecord:
    video_id: str                   # clip_id — `{youtube_id}_{start}_{end}`; keys all B artifacts
    video_path: str                 # absolute path to the clip .mp4
    ground_truth: Optional[bool]    # optional clip-level label; None when label_column is null


# Name used by run_benchmark_b.
VideoRecord = ClipRecord


def load_videos(
    csv_path: str,
    clips_dir: str,
    label_column: Optional[str] = None,
    invert_label: bool = False,
    max_videos: Optional[int] = None,
    video_extension: str = ".mp4",
) -> list[ClipRecord]:
    """
    Load one clip record per row of the AV-ASD dataset CSV.

    Args:
        csv_path:        Path to the CSV file (dataset.csv / test.csv / val.csv).
        clips_dir:       Directory containing the extracted clip files.
        label_column:    Optional CSV column to attach as a clip-level label.
                         None (the setting used) loads no label.
        invert_label:    If True, invert the boolean label.
        max_videos:      If set, limit to this many clips (dev runs).
        video_extension: File extension for clip files (default: .mp4).

    Returns:
        List of ClipRecord objects, one per clip.
    """
    df = pd.read_csv(csv_path)

    # Video_ID is the clip id verbatim. Rows without the _{start}_{end} suffix
    # cannot be resolved to a clip file, so drop them.
    def _is_clip_id(clip_id: str) -> bool:
        return len(str(clip_id).rsplit("_", 2)) == 3

    total_rows = len(df)
    df = df[df["Video_ID"].apply(_is_clip_id)]
    if len(df) < total_rows:
        print(
            f"  [loader_b] Skipped {total_rows - len(df)} row(s) whose Video_ID "
            f"does not match '{{id}}_{{start}}_{{end}}'."
        )

    has_label = label_column is not None and label_column in df.columns
    if label_column is not None and not has_label:
        print(
            f"  [loader_b] Warning: label column '{label_column}' not found in CSV. "
            f"ground_truth will be None for all clips."
        )

    records: list[ClipRecord] = []
    seen: set[str] = set()
    missing_files: int = 0

    for _, row in df.iterrows():
        clip_id = str(row["Video_ID"])
        # The CSV should hold one row per clip; guard against duplicate rows so
        # a clip is never scored twice.
        if clip_id in seen:
            continue
        seen.add(clip_id)

        ground_truth: Optional[bool] = None
        if has_label:
            raw_label = row[label_column]
            try:
                ground_truth = bool(int(raw_label))
            except (ValueError, TypeError):
                if isinstance(raw_label, str):
                    ground_truth = raw_label.strip().lower() in ("true", "1")
            if ground_truth is not None and invert_label:
                ground_truth = not ground_truth

        clip_path = os.path.join(clips_dir, f"{clip_id}{video_extension}")
        if not os.path.isfile(clip_path):
            missing_files += 1

        records.append(ClipRecord(
            video_id=clip_id,
            video_path=clip_path,
            ground_truth=ground_truth,
        ))

        if max_videos is not None and len(records) >= max_videos:
            break

    if missing_files:
        print(
            f"  [loader_b] Warning: {missing_files}/{len(records)} clip file(s) "
            f"not found in '{clips_dir}'. Missing files will fail at inference time. "
            f"Run scripts/data_preparation/extract_clips.py to cut the rest."
        )

    gt_count = sum(1 for r in records if r.ground_truth is not None)
    print(
        f"  [loader_b] Loaded {len(records)} clips "
        f"({gt_count} with ground truth labels) from {csv_path}"
    )
    return records
