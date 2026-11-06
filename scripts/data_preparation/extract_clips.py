import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.webm']


def find_original(video_id, original_videos_dir):
    """Return the path of the downloaded original video, or None."""
    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(original_videos_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def usable(path):
    """A clip counts as done only if it exists and is non-empty.

    Failed ffmpeg runs leave 0-byte files behind; with `ffmpeg -n` those would
    block every later attempt, so they are treated as missing and overwritten.
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def run_ffmpeg(command):
    return subprocess.run(
        command, shell=True, capture_output=True, text=True
    )


def extract(row, original_videos_dir, clips_video_dir):
    """Extract one clip. Returns (status, video_id, detail).

    The clip keeps its audio track (-c:a aac); no separate .wav is written,
    because nothing in the benchmark reads one.
    """
    parts = row.rsplit('_', 2)
    if len(parts) < 3:
        return 'bad_id', row, 'Video_ID does not match <id>_<start>_<end>'

    video_id = '_'.join(parts[:-2])
    start_time = int(parts[-2])
    end_time = int(parts[-1])
    duration = end_time - start_time

    original_video_path = find_original(video_id, original_videos_dir)
    if not original_video_path:
        return 'no_original', row, video_id

    video_clip_path = os.path.join(clips_video_dir, f"{row}.mp4")

    if usable(video_clip_path):
        return 'skipped', row, 'already extracted'

    # -y because existence is gated above; -nostdin so parallel ffmpeg
    # processes don't fight over the terminal.
    command = (
        f'ffmpeg -y -nostdin -loglevel error -ss {start_time} -t {duration} '
        f'-i "{original_video_path}" -c:v libx264 -c:a aac "{video_clip_path}"'
    )

    result = run_ffmpeg(command)
    # A truncated original makes ffmpeg exit 0 while writing nothing, so the
    # size check is load-bearing — returncode alone would report success.
    if result.returncode != 0 or not usable(video_clip_path):
        # Don't leave a 0-byte file to be mistaken for a real clip later.
        if os.path.exists(video_clip_path) and os.path.getsize(video_clip_path) == 0:
            os.remove(video_clip_path)
        detail = (result.stderr or '').strip().splitlines()
        return 'failed', row, detail[-1] if detail else 'ffmpeg produced an empty file'

    return 'extracted', row, ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cut clips out of the downloaded original videos. Safe to re-run: "
                    "clips that already exist are skipped, so this resumes after new "
                    "originals are downloaded."
    )
    parser.add_argument('--csv', default='dataset/csvs/dataset.csv')
    parser.add_argument('--original-videos-dir', default='dataset/original_videos')
    parser.add_argument('--clips-video-dir', default='dataset/clips_video')
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel ffmpeg jobs (default: 4)")
    parser.add_argument('--max-clips', type=int, default=None,
                        help="Limit number of clips to process (default: all)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Report what would be extracted and exit")
    args = parser.parse_args()

    os.makedirs(args.clips_video_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    video_ids = df['Video_ID'].tolist()
    if args.max_clips is not None:
        video_ids = video_ids[:args.max_clips]

    if args.dry_run:
        todo = missing_original = done = 0
        for row in video_ids:
            parts = row.rsplit('_', 2)
            if len(parts) < 3:
                continue
            if not find_original('_'.join(parts[:-2]), args.original_videos_dir):
                missing_original += 1
            elif usable(os.path.join(args.clips_video_dir, f"{row}.mp4")):
                done += 1
            else:
                todo += 1
        print(f"{len(video_ids)} clip(s) in {args.csv}")
        print(f"  {done} already extracted")
        print(f"  {todo} ready to extract now")
        print(f"  {missing_original} waiting on an original video download")
        raise SystemExit(0)

    print(f"Processing {len(video_ids)} clip(s) with {args.workers} worker(s)...")

    counts = {}
    problems = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                extract, row, args.original_videos_dir, args.clips_video_dir,
            )
            for row in video_ids
        ]
        for i, future in enumerate(as_completed(futures), 1):
            status, row, detail = future.result()
            counts[status] = counts.get(status, 0) + 1
            if status in ('failed', 'bad_id'):
                problems.append((row, detail))
            if status == 'extracted':
                print(f"[{i}/{len(futures)}] extracted {row}")

    print("\nProcessing completed.")
    print(f"  extracted:            {counts.get('extracted', 0)}")
    print(f"  already present:      {counts.get('skipped', 0)}")
    print(f"  original not on disk: {counts.get('no_original', 0)}")
    print(f"  failed:               {counts.get('failed', 0) + counts.get('bad_id', 0)}")
    if problems:
        print("\nFailures:")
        for row, detail in problems:
            print(f"  {row}: {detail}")
