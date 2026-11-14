#!/usr/bin/env python3
"""
Upload all locally cached clips to a private S3 bucket.

Clips are stored permanently — re-running this script skips clips that are
already uploaded (cheap head_object check per clip).
Pre-signed URLs are generated at inference time by the benchmark runner.

Usage
-----
    python scripts/upload_clips_to_s3.py --bucket my-bucket --region us-east-1

    # Force re-upload even if already on S3
    python scripts/upload_clips_to_s3.py --bucket my-bucket --force

    # Custom clips directory or key prefix
    python scripts/upload_clips_to_s3.py --bucket my-bucket \
        --clips-dir dataset/clips_video \
        --prefix benchmark-clips/

AWS credentials
---------------
Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment or .env,
or use an IAM role / ~/.aws/credentials.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.utils.s3 import clip_s3_key, upload_clip


def bucket_problems(bucket: str, region: str) -> list[str]:
    """Return why the bucket is not private; empty when public access is fully blocked."""
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=region)
    try:
        cfg = s3.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"]
    except ClientError as e:
        return [f"cannot read the public access block ({e.response['Error']['Code']})"]
    if not all(cfg.get(k) for k in
               ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")):
        return [f"public access is not fully blocked: {cfg}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload benchmark clips to a private S3 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument(
        "--prefix",
        default="benchmark-clips/",
        help="S3 key prefix (default: benchmark-clips/)",
    )
    parser.add_argument(
        "--clips-dir",
        default="dataset/clips_video",
        help="Local directory containing .mp4 clips (default: dataset/clips_video)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload clips that already exist on S3",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel uploads (default: 4)",
    )
    parser.add_argument(
        "--skip-safety-check",
        action="store_true",
        help="Upload even if the bucket does not block public access",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the bucket and report the plan without uploading",
    )
    args = parser.parse_args()

    clips_dir = os.path.abspath(args.clips_dir)
    clips = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))

    if not clips:
        print(f"[ERROR] No .mp4 files found in: {clips_dir}")
        sys.exit(1)

    total_bytes = sum(os.path.getsize(c) for c in clips)
    print(f"Found {len(clips)} file(s) in {clips_dir}  ({total_bytes/1e9:.1f} GB)")
    print(f"Target: s3://{args.bucket}/{args.prefix.rstrip('/')}/")
    print()

    print("Checking that the bucket blocks public access...")
    problems = bucket_problems(args.bucket, args.region)
    if problems:
        for p in problems:
            print(f"  [BLOCKED] {p}")
        if not args.skip_safety_check:
            print("\nRefusing to upload. Fix the finding above, or pass --skip-safety-check.")
            sys.exit(1)
        print("\n  --skip-safety-check set; continuing.")
    else:
        print("  OK: public access blocked.")
    print()

    if args.dry_run:
        print(f"--dry-run: would upload {len(clips)} file(s), {total_bytes/1e9:.1f} GB.")
        return

    def _upload_one(clip_path: str) -> tuple[str, Optional[str]]:
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        try:
            upload_clip(
                local_path=clip_path,
                bucket=args.bucket,
                key=clip_s3_key(args.prefix, clip_id),
                region=args.region,
                skip_if_exists=not args.force,
            )
            return clip_id, None
        except Exception as e:
            return clip_id, str(e)

    uploaded = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_upload_one, c) for c in clips]
        for i, future in enumerate(as_completed(futures), 1):
            clip_id, error = future.result()
            if error is None:
                uploaded += 1
                print(f"  [{i:>5}/{len(clips)}] {clip_id}  ok")
            else:
                failures.append((clip_id, error))
                print(f"  [{i:>5}/{len(clips)}] {clip_id}  FAIL — {error}")

    print()
    print(f"Done. {uploaded} uploaded/verified, {len(failures)} failed.")
    if failures:
        print("\nFailures:")
        for clip_id, error in failures:
            print(f"  {clip_id}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
