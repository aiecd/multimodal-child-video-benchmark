"""
Build the clip dataset without ever holding the full original video library.

For each source video, in order:
    download original -> cut all of its clips -> delete the original

Peak disk is therefore (workers x largest original) + the clip store, rather
than the ~33 GB the complete original library would occupy. The clips are
byte-for-byte what extract_clips.py produces, so clips from either script can be
mixed freely.

Resume:
    A video is skipped when all of its clips already exist, not when the
    original is on disk; the original is expected to be gone.

Safety:
    An original is only deleted once every one of its clips has been cut
    successfully. If any clip fails, the original is kept so the failure can be
    retried or inspected. Use --keep-originals to disable deletion entirely and
    build the full archive instead.

Usage
-----
    # See what would happen, download nothing
    python scripts/data_preparation/fetch_clips.py --dry-run

    # Build all remaining clips, discarding originals as they are consumed
    python scripts/data_preparation/fetch_clips.py --workers 3

    # Same, but keep the originals
    python scripts/data_preparation/fetch_clips.py --keep-originals
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_clips import extract, find_original, usable  # noqa: E402

# A real 360p+ video is far above this. The AV-ASD set contains a few truncated
# downloads whose container advertises the full duration while holding almost no
# data (e.g. 176 KB for 462 s); ffmpeg hits EOF before the clip offset and cuts
# nothing. Re-downloading is the only fix, so treat these as absent.
MIN_PLAUSIBLE_KBPS = 60


def clips_by_original(df):
    """Map each source video id to the list of clip ids cut from it."""
    grouped = OrderedDict()
    for clip_id in df['Video_ID']:
        parts = str(clip_id).rsplit('_', 2)
        if len(parts) < 3:
            continue
        grouped.setdefault('_'.join(parts[:-2]), []).append(str(clip_id))
    return grouped


def is_truncated(path):
    """True if the file decodes to an implausibly low bitrate for its duration."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', path],
            capture_output=True, text=True,
        ).stdout.strip()
        duration = float(out)
    except (ValueError, OSError):
        return True
    if duration <= 0:
        return True
    return (os.path.getsize(path) * 8 / duration / 1000) < MIN_PLAUSIBLE_KBPS


def download_original(video_id, path_data):
    """Fetch one source video. Returns True on success."""
    quarantined = original_path = None
    existing = find_original(video_id, path_data)
    if existing:
        if not is_truncated(existing):
            return True
        # Move the bad copy aside rather than deleting it: the source may have
        # been taken down, in which case a failed re-download would otherwise
        # leave us with nothing at all.
        print(f"  {video_id}: on-disk copy is truncated, re-downloading")
        original_path = existing
        quarantined = existing + ".truncated"
        os.replace(existing, quarantined)

    file_path = os.path.join(path_data, video_id) + ".%(ext)s"
    # Force mp4 so the clip cutter finds the file without a container convert.
    sources = [
        f'yt-dlp -f mp4 "https://www.youtube.com/watch?v={video_id}" -o "{file_path}"',
        f'yt-dlp -f mp4 "https://www.youtube.com/shorts/{video_id}" -o "{file_path}"',
        f'yt-dlp "https://www.facebook.com/watch/?v={video_id}" -o "{file_path}"',
    ]
    for command in sources:
        if subprocess.run(command, shell=True, capture_output=True).returncode == 0:
            path = find_original(video_id, path_data)
            if path and not is_truncated(path):
                if quarantined:
                    os.remove(quarantined)
                return True
            if path:
                # A truncated result from this source; let the next one try.
                os.remove(path)

    if quarantined:
        # Nothing better is available — put the truncated copy back so a future
        # run can still try, and so its state is visible on disk.
        os.replace(quarantined, original_path)
    return False


def process_original(video_id, clip_ids, originals_dir, clips_dir, keep_originals):
    """Download, cut every clip, then discard the original.

    Returns a dict describing what happened, for the run summary.
    """
    pending = [c for c in clip_ids if not usable(os.path.join(clips_dir, f"{c}.mp4"))]
    if not pending:
        return {'video_id': video_id, 'status': 'complete', 'extracted': 0, 'failed': []}

    if not download_original(video_id, originals_dir):
        return {'video_id': video_id, 'status': 'download_failed',
                'extracted': 0, 'failed': pending}

    extracted, failed = 0, []
    for clip_id in pending:
        status, _, detail = extract(clip_id, originals_dir, clips_dir)
        if status == 'extracted':
            extracted += 1
        elif status != 'skipped':
            failed.append((clip_id, detail))

    # Keep the original whenever anything failed, so the failure is retryable.
    freed = 0
    if not failed and not keep_originals:
        path = find_original(video_id, originals_dir)
        if path:
            freed = os.path.getsize(path)
            os.remove(path)

    return {
        'video_id': video_id,
        'status': 'extracted' if not failed else 'partial',
        'extracted': extracted,
        'failed': failed,
        'freed': freed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download originals, cut their clips, and delete the originals.",
    )
    parser.add_argument('--csv', default='dataset/csvs/dataset.csv')
    parser.add_argument('--originals-dir', default='dataset/original_videos',
                        help="Scratch space for in-flight originals (default: dataset/original_videos)")
    parser.add_argument('--clips-dir', default='dataset/clips_video')
    parser.add_argument('--workers', type=int, default=3,
                        help="Videos in flight at once. Peak disk is roughly "
                             "workers x largest original (default: 3)")
    parser.add_argument('--max-videos', type=int, default=None,
                        help="Limit number of source videos to process (default: all)")
    parser.add_argument('--keep-originals', action='store_true',
                        help="Do not delete originals after cutting their clips")
    parser.add_argument('--prune-existing', action='store_true',
                        help="Delete originals already on disk whose clips are all cut, "
                             "then exit. Downloads nothing. Combine with --dry-run to preview.")
    parser.add_argument('--min-free-gb', type=float, default=5.0,
                        help="Abort before starting if free space is below this (default: 5)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Report the plan and exit without downloading")
    args = parser.parse_args()

    os.makedirs(args.originals_dir, exist_ok=True)
    os.makedirs(args.clips_dir, exist_ok=True)

    grouped = clips_by_original(pd.read_csv(args.csv))

    todo = OrderedDict()
    done_videos = done_clips = 0
    for video_id, clip_ids in grouped.items():
        pending = [c for c in clip_ids
                   if not usable(os.path.join(args.clips_dir, f"{c}.mp4"))]
        done_clips += len(clip_ids) - len(pending)
        if pending:
            todo[video_id] = clip_ids
        else:
            done_videos += 1

    if args.max_videos is not None:
        todo = OrderedDict(list(todo.items())[:args.max_videos])

    total_clips = sum(len(v) for v in grouped.values())
    pending_clips = sum(
        1 for v in todo.values() for c in v
        if not usable(os.path.join(args.clips_dir, f"{c}.mp4"))
    )
    print(f"{len(grouped)} source video(s), {total_clips} clip(s) in {args.csv}")
    print(f"  {done_clips} clip(s) already cut  ({done_videos} video(s) fully done, nothing to fetch)")
    print(f"  {pending_clips} clip(s) to cut from {len(todo)} video(s) to download")
    print(f"  originals will be {'KEPT' if args.keep_originals else 'DELETED after cutting'}")

    if args.prune_existing:
        drained = [
            (vid, find_original(vid, args.originals_dir))
            for vid, clips in grouped.items()
            if vid not in todo and find_original(vid, args.originals_dir)
        ]
        reclaim = sum(os.path.getsize(p) for _, p in drained)
        verb = "Would delete" if args.dry_run else "Deleting"
        print(f"\n{verb} {len(drained)} fully-drained original(s), "
              f"reclaiming {reclaim/1e9:.1f} GB")
        if not args.dry_run:
            for _, path in drained:
                os.remove(path)
            remaining = len(os.listdir(args.originals_dir))
            print(f"  {remaining} original(s) left on disk "
                  f"(these still have clips to cut)")
        return

    if args.dry_run or not todo:
        return

    free_gb = shutil.disk_usage(args.clips_dir).free / 1e9
    if free_gb < args.min_free_gb:
        print(f"\n[ABORT] only {free_gb:.1f} GB free, need {args.min_free_gb} GB. "
              f"Use --min-free-gb to override.")
        sys.exit(1)

    print(f"\nStarting with {args.workers} worker(s), {free_gb:.0f} GB free...\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_original, vid, clips,
                            args.originals_dir, args.clips_dir, args.keep_originals): vid
            for vid, clips in todo.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            freed = r.get('freed', 0)
            note = f", freed {freed/1e6:.0f} MB" if freed else ""
            print(f"[{i}/{len(futures)}] {r['video_id']}: {r['status']} "
                  f"(+{r['extracted']} clip(s){note})")

    total_freed = sum(r.get('freed', 0) for r in results)
    unavailable = [r for r in results if r['status'] == 'download_failed']
    partial = [r for r in results if r['status'] == 'partial']

    print("\nDone.")
    print(f"  clips cut:              {sum(r['extracted'] for r in results)}")
    print(f"  originals discarded:    {sum(1 for r in results if r.get('freed'))} "
          f"({total_freed/1e9:.1f} GB reclaimed)")
    print(f"  could not be found:     {len(unavailable)}")
    print(f"  partial (kept on disk): {len(partial)}")

    if unavailable:
        print("\nNot available online (no clips cut):")
        for r in unavailable:
            print(f"  {r['video_id']}  ({len(r['failed'])} clip(s) blocked)")
    if partial:
        print("\nPartially cut — original kept for retry:")
        for r in partial:
            for clip_id, detail in r['failed']:
                print(f"  {clip_id}: {detail}")


if __name__ == '__main__':
    main()
