"""
Download the AV-ASD annotation files from the dataset authors' repository at a fixed commit.

The labels are not redistributed here: the AV-ASD repository has no license file, so this
script fetches them from the source and checks each file against its SHA-256 checksum.

Usage (run from the repository root):
    python -m benchmark.scripts.fetch_labels
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

COMMIT = "1dca5b2775ade23f30fa3220f3037a1ed0fd25f7"  # ShijianDeng/AV-ASD, 8 July 2024
BASE = f"https://raw.githubusercontent.com/ShijianDeng/AV-ASD/{COMMIT}/dataset/csvs/"
FILES = {
    "dataset.csv": "7cfd3d162c49020937b28696a3456f374dd329e36e27cef70915cfc8bee47e38",
    "train.csv": "bce45fe7fd12a1e7033ee00faa5c634ffd21106769652c3084910ba833dabe86",
    "val.csv": "197925f14c084e1a018ce38c99415a84df7dc74c16eb0f67afd77136097a6e7d",
    "test.csv": "dab30726ce40893bbaa2677ce576281f440fc31459a9958afcc11d34a6c8dfe1",
}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "dataset", "csvs")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, sha in FILES.items():
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == sha:
            print(f"{name}: present, checksum OK")
            continue
        with urllib.request.urlopen(BASE + name, timeout=60) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            print(f"{name}: checksum mismatch ({got}); not written", file=sys.stderr)
            return 1
        with open(dst, "wb") as f:
            f.write(data)
        print(f"{name}: downloaded, checksum OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
