#!/usr/bin/env python3
"""
Statistical analysis of the extracted clip set in dataset/clips_video.

Answers three questions that matter before a benchmark run:

    1. What is actually on disk?   duration / size / bitrate / codec / audio
    2. Is any of it broken?        0-byte, unprobeable, silent, or truncated
                                   clips (a clip is "truncated" when its real
                                   duration is materially shorter than the
                                   start/end range encoded in its filename)
    3. Does it still represent the dataset?
                                   coverage against dataset/csvs/*.csv and the
                                   label prevalence drift caused by whatever is
                                   missing

Clip filenames are `{video_id}_{start}_{end}.mp4`, where video_id may itself
contain underscores, so the offsets are parsed from the right.

ffprobe results are cached in the clips directory (keyed by size + mtime), so
re-running after adding a few hundred clips only probes the new ones.

Usage
-----
    python scripts/analyze_clips.py

    # Machine-readable summary alongside the printed report
    python scripts/analyze_clips.py --json dataset/clip_stats.json

    # Ignore the cache, re-probe everything
    python scripts/analyze_clips.py --no-cache --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import pandas as pd

CACHE_NAME = ".probe_cache.json"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")

# Below this the file is almost certainly a truncated download rather than a
# real clip; extract_clips.py uses the same reasoning on the source videos.
MIN_PLAUSIBLE_KBPS = 60

# A clip is flagged as short when it loses more than this much of the range its
# filename claims. ffmpeg keyframe seeking routinely shaves a fraction of a
# second, so a small absolute tolerance is applied as well.
DURATION_TOLERANCE_FRAC = 0.10
DURATION_TOLERANCE_SEC = 1.0

LABEL_SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


# --------------------------------------------------------------------------- #
# probing
# --------------------------------------------------------------------------- #

def list_clips(clips_dir: str) -> list[str]:
    """Clip filenames on disk, sorted, ignoring dotfiles and the cache."""
    return sorted(
        name
        for name in os.listdir(clips_dir)
        if not name.startswith(".") and name.lower().endswith(VIDEO_EXTENSIONS)
    )


def probe(path: str) -> dict:
    """Read one clip's container and stream properties via ffprobe.

    Returns a flat record; `error` is set (and the rest left empty) when the
    file cannot be read, which is itself a finding rather than a crash.
    """
    size = os.path.getsize(path)
    record: dict = {"size": size, "mtime": os.path.getmtime(path), "error": None}

    if size == 0:
        record["error"] = "empty file"
        return record

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        record["error"] = (result.stderr.strip().splitlines() or ["ffprobe failed"])[-1]
        return record

    try:
        meta = json.loads(result.stdout)
    except json.JSONDecodeError:
        record["error"] = "unparseable ffprobe output"
        return record

    fmt = meta.get("format", {})
    streams = meta.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    record["duration"] = _as_float(fmt.get("duration"))
    record["bitrate"] = _as_float(fmt.get("bit_rate"))
    record["n_video_streams"] = sum(1 for s in streams if s.get("codec_type") == "video")
    record["n_audio_streams"] = sum(1 for s in streams if s.get("codec_type") == "audio")

    if video is not None:
        record["width"] = video.get("width")
        record["height"] = video.get("height")
        record["video_codec"] = video.get("codec_name")
        record["pix_fmt"] = video.get("pix_fmt")
        record["fps"] = _as_fraction(video.get("avg_frame_rate"))
        record["n_frames"] = _as_int(video.get("nb_frames"))
        if record["duration"] is None:
            record["duration"] = _as_float(video.get("duration"))
    if audio is not None:
        record["audio_codec"] = audio.get("codec_name")
        record["sample_rate"] = _as_int(audio.get("sample_rate"))
        record["channels"] = audio.get("channels")

    if record.get("duration") is None:
        record["error"] = "no duration reported"
    return record


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_fraction(value):
    """ffprobe reports frame rates as 'num/den'; 0/0 means unknown."""
    if not value or "/" not in str(value):
        return _as_float(value)
    num, _, den = str(value).partition("/")
    num, den = _as_float(num), _as_float(den)
    if not num or not den:
        return None
    return num / den


def probe_all(clips_dir: str, names: list[str], workers: int, use_cache: bool) -> dict:
    """Probe every clip, reusing cached records whose size and mtime match."""
    cache_path = os.path.join(clips_dir, CACHE_NAME)
    cache: dict = {}
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError):
            cache = {}

    records: dict = {}
    todo = []
    for name in names:
        path = os.path.join(clips_dir, name)
        cached = cache.get(name)
        if (
            use_cache
            and cached
            and cached.get("size") == os.path.getsize(path)
            and cached.get("mtime") == os.path.getmtime(path)
        ):
            records[name] = cached
        else:
            todo.append(name)

    if todo:
        print(f"Probing {len(todo)} clip(s) ({len(records)} from cache)...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(probe, os.path.join(clips_dir, name)): name for name in todo
            }
            for done, future in enumerate(as_completed(futures), 1):
                name = futures[future]
                try:
                    records[name] = future.result()
                except Exception as exc:  # a probe must never abort the report
                    records[name] = {"size": None, "mtime": None, "error": repr(exc)}
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)}", file=sys.stderr)
        try:
            with open(cache_path, "w") as handle:
                json.dump(records, handle)
        except OSError as exc:
            print(f"Warning: could not write probe cache: {exc}", file=sys.stderr)

    return records


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #

def parse_clip_id(name: str) -> tuple[str, float | None, float | None]:
    """Split `{video_id}_{start}_{end}.mp4` from the right; video ids have '_'."""
    clip_id = os.path.splitext(name)[0]
    parts = clip_id.rsplit("_", 2)
    if len(parts) == 3:
        start, end = _as_float(parts[1]), _as_float(parts[2])
        if start is not None and end is not None:
            return clip_id, start, end
    return clip_id, None, None


def human_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def describe(series: pd.Series, fmt=lambda v: f"{v:.2f}") -> str:
    """One-line five-number summary plus mean and standard deviation."""
    clean = series.dropna()
    if clean.empty:
        return "no data"
    q = clean.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return (
        f"min {fmt(clean.min())} | p05 {fmt(q[0.05])} | p25 {fmt(q[0.25])} | "
        f"median {fmt(q[0.50])} | p75 {fmt(q[0.75])} | p95 {fmt(q[0.95])} | "
        f"max {fmt(clean.max())} | mean {fmt(clean.mean())} | sd {fmt(clean.std())}"
    )


def counter_lines(counts: Counter, total: int, limit: int = 10) -> list[str]:
    lines = []
    for value, count in counts.most_common(limit):
        lines.append(f"    {str(value):<28} {count:>5}  ({count / total:6.1%})")
    if len(counts) > limit:
        lines.append(f"    ... and {len(counts) - limit} more distinct value(s)")
    return lines


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def build_frame(records: dict) -> pd.DataFrame:
    rows = []
    for name, record in records.items():
        clip_id, start, end = parse_clip_id(name)
        expected = end - start if start is not None and end is not None else None
        duration = record.get("duration")
        rows.append(
            {
                "file": name,
                "clip_id": clip_id,
                "video_id": clip_id.rsplit("_", 2)[0] if start is not None else clip_id,
                "expected_duration": expected,
                "duration": duration,
                "duration_delta": None if (expected is None or duration is None) else duration - expected,
                "size": record.get("size"),
                "bitrate_kbps": None if record.get("bitrate") is None else record["bitrate"] / 1000,
                "width": record.get("width"),
                "height": record.get("height"),
                "fps": record.get("fps"),
                "n_frames": record.get("n_frames"),
                "video_codec": record.get("video_codec"),
                "pix_fmt": record.get("pix_fmt"),
                "audio_codec": record.get("audio_codec"),
                "sample_rate": record.get("sample_rate"),
                "channels": record.get("channels"),
                "n_audio_streams": record.get("n_audio_streams", 0),
                "n_video_streams": record.get("n_video_streams", 0),
                "error": record.get("error"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["resolution"] = frame.apply(
        lambda r: f"{int(r.width)}x{int(r.height)}" if pd.notna(r.width) and pd.notna(r.height) else None,
        axis=1,
    )
    return frame.sort_values("clip_id").reset_index(drop=True)


def report_inventory(frame: pd.DataFrame) -> dict:
    ok = frame[frame.error.isna()]
    section("1. Inventory")
    print(f"  Clips on disk             {len(frame)}")
    print(f"  Successfully probed       {len(ok)}")
    print(f"  Failed to probe           {len(frame) - len(ok)}")
    print(f"  Total size                {human_bytes(frame['size'].sum())}")
    print(f"  Total duration            {human_seconds(ok['duration'].sum())}")
    print(f"  Distinct source videos    {frame['video_id'].nunique()}")
    per_video = frame.groupby("video_id").size()
    print(f"  Clips per source video    {describe(per_video, lambda v: f'{v:.1f}')}")
    return {
        "n_clips": len(frame),
        "n_probed": len(ok),
        "n_probe_failures": len(frame) - len(ok),
        "total_bytes": int(frame["size"].fillna(0).sum()),
        "total_duration_sec": float(ok["duration"].sum()),
        "n_source_videos": int(frame["video_id"].nunique()),
    }


def report_distributions(frame: pd.DataFrame) -> dict:
    ok = frame[frame.error.isna()]
    section("2. Duration, size and bitrate (seconds / MB / kbps)")
    print(f"  duration   {describe(ok['duration'])}")
    print(f"  size       {describe(ok['size'] / 1e6)}")
    print(f"  bitrate    {describe(ok['bitrate_kbps'], lambda v: f'{v:.0f}')}")

    buckets = pd.cut(
        ok["duration"],
        bins=[0, 5, 10, 20, 30, 60, 120, float("inf")],
        labels=["0-5s", "5-10s", "10-20s", "20-30s", "30-60s", "60-120s", ">120s"],
        right=False,
    )
    print("\n  Duration buckets")
    for label, count in buckets.value_counts().sort_index().items():
        print(f"    {str(label):<28} {count:>5}  ({count / len(ok):6.1%})")

    section("3. Video and audio properties")
    total = len(ok)
    print("  Resolution")
    print("\n".join(counter_lines(Counter(ok["resolution"].dropna()), total)))
    print("  Video codec")
    print("\n".join(counter_lines(Counter(ok["video_codec"].dropna()), total)))
    print("  Pixel format")
    print("\n".join(counter_lines(Counter(ok["pix_fmt"].dropna()), total, limit=5)))
    print(f"  Frame rate   {describe(ok['fps'], lambda v: f'{v:.2f}')}")
    print(f"  Frame count  {describe(ok['n_frames'], lambda v: f'{v:.0f}')}")
    with_audio = int((ok["n_audio_streams"] > 0).sum())
    print(f"\n  Clips with an audio track {with_audio}/{total} ({with_audio / total:.1%})")
    if with_audio:
        audio = ok[ok["n_audio_streams"] > 0]
        print("  Audio codec")
        print("\n".join(counter_lines(Counter(audio["audio_codec"].dropna()), with_audio, limit=5)))
        print("  Sample rate")
        print("\n".join(counter_lines(Counter(audio["sample_rate"].dropna()), with_audio, limit=5)))
        print("  Channels")
        print("\n".join(counter_lines(Counter(audio["channels"].dropna()), with_audio, limit=5)))

    return {
        "duration_sec": _summary(ok["duration"]),
        "size_bytes": _summary(ok["size"]),
        "bitrate_kbps": _summary(ok["bitrate_kbps"]),
        "fps": _summary(ok["fps"]),
        "resolution_counts": dict(Counter(ok["resolution"].dropna())),
        "video_codec_counts": dict(Counter(ok["video_codec"].dropna())),
        "audio_codec_counts": dict(Counter(ok["audio_codec"].dropna())),
        "n_with_audio": with_audio,
    }


def _summary(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {}
    q = clean.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": int(clean.size),
        "min": float(clean.min()),
        "p05": float(q[0.05]),
        "p25": float(q[0.25]),
        "median": float(q[0.50]),
        "p75": float(q[0.75]),
        "p95": float(q[0.95]),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
    }


def report_integrity(frame: pd.DataFrame, show: int) -> dict:
    section("4. Integrity")
    ok = frame[frame.error.isna()]

    issues = {
        "probe failures": frame[frame.error.notna()],
        "zero-byte files": frame[frame["size"] == 0],
        "no video stream": ok[ok["n_video_streams"] == 0],
        "no audio stream": ok[ok["n_audio_streams"] == 0],
        f"bitrate < {MIN_PLAUSIBLE_KBPS} kbps": ok[ok["bitrate_kbps"] < MIN_PLAUSIBLE_KBPS],
        "shorter than filename range": ok[
            ok["duration_delta"].notna()
            & (
                -ok["duration_delta"]
                > (ok["expected_duration"] * DURATION_TOLERANCE_FRAC).clip(
                    lower=DURATION_TOLERANCE_SEC
                )
            )
        ],
        "unparseable filename": frame[frame["expected_duration"].isna()],
    }

    summary = {}
    for label, subset in issues.items():
        print(f"  {label:<32} {len(subset):>5}")
        summary[label] = [str(f) for f in subset["file"]]
        for _, row in subset.head(show).iterrows():
            if pd.notna(row["error"]):
                detail = row["error"]
            elif pd.notna(row["duration_delta"]):
                detail = f"{row['duration']:.1f}s vs {row['expected_duration']:.0f}s expected"
            else:
                detail = ""
            print(f"      {row['file']}  {detail}")
        if len(subset) > show:
            print(f"      ... {len(subset) - show} more")

    delta = ok["duration_delta"].dropna()
    if not delta.empty:
        print(f"\n  Actual minus expected duration (s): {describe(delta)}")
    return summary


def report_coverage(frame: pd.DataFrame, csv_dir: str, show: int) -> dict:
    dataset_path = os.path.join(csv_dir, "dataset.csv")
    if not os.path.exists(dataset_path):
        section("5. Dataset coverage")
        print(f"  {dataset_path} not found; skipping coverage and label analysis.")
        return {}

    full = pd.read_csv(dataset_path)
    labels = [c for c in full.columns if c != "Video_ID"]
    on_disk = set(frame.loc[frame.error.isna(), "clip_id"])
    full["present"] = full["Video_ID"].isin(on_disk)

    section("5. Dataset coverage")
    print(f"  Clips listed in dataset.csv   {len(full)}")
    print(f"  Present and readable on disk  {int(full['present'].sum())}  "
          f"({full['present'].mean():.1%})")
    print(f"  Missing                       {int((~full['present']).sum())}")
    extra = sorted(on_disk - set(full["Video_ID"]))
    print(f"  On disk but not in dataset.csv {len(extra)}")
    for name in extra[:show]:
        print(f"      {name}")
    if len(extra) > show:
        print(f"      ... {len(extra) - show} more")

    split_rows = []
    for split, filename in LABEL_SPLIT_FILES.items():
        path = os.path.join(csv_dir, filename)
        if not os.path.exists(path):
            continue
        split_df = pd.read_csv(path)
        present = split_df["Video_ID"].isin(on_disk).sum()
        split_rows.append((split, len(split_df), int(present)))
    if split_rows:
        print("\n  Per split")
        for split, total, present in split_rows:
            print(f"    {split:<8} {present:>4}/{total:<4} ({present / total:6.1%})")

    section("6. Label prevalence (positive rate)")
    print(f"  {'label':<46} {'full':>8} {'on disk':>9} {'drift':>8}")
    label_stats = {}
    for label in labels:
        full_rate = full[label].mean()
        disk_rate = full.loc[full["present"], label].mean() if full["present"].any() else float("nan")
        print(f"  {label:<46} {full_rate:>7.1%} {disk_rate:>8.1%} {disk_rate - full_rate:>+7.1%}")
        label_stats[label] = {
            "full_positive": int(full[label].sum()),
            "full_rate": float(full_rate),
            "on_disk_positive": int(full.loc[full["present"], label].sum()),
            "on_disk_rate": float(disk_rate),
        }

    present = full[full["present"]]
    per_clip = present[labels].sum(axis=1)
    print(f"\n  Labels per clip (on disk): {describe(per_clip, lambda v: f'{v:.2f}')}")
    for count, n in per_clip.value_counts().sort_index().items():
        print(f"    {int(count)} label(s){'':<18} {n:>5}  ({n / len(present):6.1%})")

    pairs = Counter()
    for _, row in present[labels].iterrows():
        positives = [l for l in labels if row[l] == 1]
        pairs.update(combinations(sorted(positives), 2))
    if pairs:
        print("\n  Most frequent label co-occurrences (on disk)")
        for (a, b), n in pairs.most_common(8):
            print(f"    {n:>4}  {a} + {b}")

    return {
        "n_in_csv": len(full),
        "n_present": int(full["present"].sum()),
        "n_missing": int((~full["present"]).sum()),
        "n_not_in_csv": len(extra),
        "per_split": {s: {"total": t, "present": p} for s, t, p in split_rows},
        "labels": label_stats,
        "labels_per_clip": _summary(per_clip),
    }


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", default=os.path.join(root, "dataset", "clips_video"))
    parser.add_argument("--csv-dir", default=os.path.join(root, "dataset", "csvs"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", help="re-probe every clip")
    parser.add_argument("--show", type=int, default=5, help="examples printed per issue")
    parser.add_argument("--json", dest="json_out", help="write the summary as JSON")
    parser.add_argument("--csv", dest="csv_out", help="write the per-clip table as CSV")
    args = parser.parse_args()

    if not os.path.isdir(args.clips_dir):
        print(f"Clips directory not found: {args.clips_dir}", file=sys.stderr)
        return 1

    names = list_clips(args.clips_dir)
    if not names:
        print(f"No clips found in {args.clips_dir}", file=sys.stderr)
        return 1

    records = probe_all(args.clips_dir, names, args.workers, use_cache=not args.no_cache)
    frame = build_frame(records)

    print(f"\nClip analysis — {args.clips_dir}")
    print("=" * 72)
    summary = {"clips_dir": args.clips_dir}
    summary["inventory"] = report_inventory(frame)
    summary["distributions"] = report_distributions(frame)
    summary["integrity"] = report_integrity(frame, args.show)
    summary["coverage"] = report_coverage(frame, args.csv_dir, args.show)
    print()

    if args.csv_out:
        frame.to_csv(args.csv_out, index=False)
        print(f"Per-clip table written to {args.csv_out}")
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
