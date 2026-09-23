"""Download the raw Phase 1 phishing-email dataset from Zenodo.

The raw file is intentionally kept out of Git history. This script downloads the
official merged dataset and verifies its MD5 checksum before it is used.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RECORD_ID = "17314806"
FILENAME = "Merged_Dataset.csv"
DOWNLOAD_URL = f"https://zenodo.org/records/{RECORD_ID}/files/{FILENAME}?download=1"
EXPECTED_MD5 = "a692ca69ffd7dbc9df991292035d29a9"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / FILENAME


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"{FILENAME} already exists. Verifying checksum...")
        checksum = md5sum(OUTPUT_PATH)
        if checksum == EXPECTED_MD5:
            print("Dataset is already downloaded and verified.")
            return
        print("Existing file failed checksum verification. Re-downloading it.")
        OUTPUT_PATH.unlink()

    print(f"Downloading {FILENAME} from Zenodo...")
    print("This file is about 391 MB, so the download may take a while.")

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, OUTPUT_PATH)
    except Exception:
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        raise

    print("Download complete. Verifying MD5 checksum...")
    checksum = md5sum(OUTPUT_PATH)

    if checksum != EXPECTED_MD5:
        OUTPUT_PATH.unlink(missing_ok=True)
        print(
            f"Checksum verification failed. Expected {EXPECTED_MD5}, got {checksum}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Verified successfully: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
