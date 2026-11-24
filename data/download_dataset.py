"""Download the original archive without committing large data."""
from pathlib import Path
import gzip
import hashlib
import gdown

TARGET = Path(__file__).resolve().parent / "merged_sentiment_dataset.csv.gz"
FILE_ID = "1ShejwvJSUKxZ6mYEuJkrtqjwS0BiLZy_"

def main():
    if not TARGET.exists():
        tmp = TARGET.with_suffix(".part")
        if not gdown.download(id=FILE_ID, output=str(tmp), quiet=False):
            raise RuntimeError("Download failed; check Drive permissions")
        with gzip.open(tmp, "rb") as f:
            while f.read(1024 * 1024):
                pass
        tmp.replace(TARGET)
    with TARGET.open("rb") as f:
        print("SHA256:", hashlib.file_digest(f, "sha256").hexdigest())
    print(TARGET, TARGET.stat().st_size)

if __name__ == "__main__":
    main()
