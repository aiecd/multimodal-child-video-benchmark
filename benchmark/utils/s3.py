"""
S3 utilities for benchmark clip storage.

Clips are uploaded to a private bucket. At inference time, a pre-signed URL
is generated per clip (default expiry: 1 hour) so the model API can fetch it
without the bucket being publicly accessible.

Usage
-----
    from benchmark.utils.s3 import clip_s3_key, get_presigned_url, upload_clip

    key = clip_s3_key("benchmark-clips/", clip_id)
    upload_clip(local_path, bucket, key, region)      # skips if already uploaded
    url = get_presigned_url(bucket, key, region, expiry_s=3600)
"""

from __future__ import annotations


def clip_s3_key(key_prefix: str, clip_id: str) -> str:
    """Return the S3 object key for a clip. Consistent across upload and lookup."""
    return f"{key_prefix.rstrip('/')}/{clip_id}.mp4"


def upload_clip(
    local_path: str,
    bucket: str,
    key: str,
    region: str,
    skip_if_exists: bool = True,
) -> None:
    """
    Upload a clip to a private S3 bucket.

    If skip_if_exists=True (default), checks for the object first and skips
    the upload if it already exists — safe and cheap to call on every run.

    Requires boto3 and AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    in environment, or an IAM role / ~/.aws/credentials).
    """
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=region)

    if skip_if_exists:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return  # already exists, skip upload
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise  # unexpected error — propagate

    s3.upload_file(local_path, bucket, key)


def get_presigned_url(
    bucket: str,
    key: str,
    region: str,
    expiry_s: int = 3600,
) -> str:
    """
    Generate a pre-signed GET URL for a private S3 object.

    The URL is valid for expiry_s seconds (default 1 hour). After expiry,
    the model API can no longer fetch the clip — but inference is typically
    done within minutes so 1 hour is more than sufficient.
    """
    import boto3

    s3 = boto3.client("s3", region_name=region)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_s,
    )


def get_cloudfront_url(cloudfront_domain: str, key: str) -> str:
    """
    Return an HTTPS URL through a CloudFront distribution you control.

    The URL has no query parameters, so it is accepted by model APIs that reject
    pre-signed URLs (e.g. DashScope).
    """
    return f"https://{cloudfront_domain.rstrip('/')}/{key}"
