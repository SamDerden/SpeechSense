from pathlib import Path
import hashlib
import gzip
import os
import gdown

TARGET = Path(__file__).resolve().parent / "merged_sentiment_dataset.csv.gz"
URL = "https://drive.google.com/file/d/1ShejwvJSUKxZ6mYEuJkrtqjwS0BiLZy_/view?usp=drivesdk"
EXPECTED_SIZE = 116681042

def verify(path):
    if path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected archive size: {path.stat().st_size}")
    with gzip.open(path, "rb") as f:
        while f.read(1024 * 1024):
            pass
    with path.open("rb") as f:
        print("SHA256:", hashlib.file_digest(f, "sha256").hexdigest())

def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        verify(TARGET)
        print(TARGET)
        return
    tmp = TARGET.with_suffix(".part")
    try:
        result = gdown.download(id="1ShejwvJSUKxZ6mYEuJkrtqjwS0BiLZy_", output=str(tmp), quiet=False)
        if not result or not tmp.exists():
            raise RuntimeError("Google Drive download failed. Check sharing permissions and quota.")
        verify(tmp)
        os.replace(tmp, TARGET)
        print("Downloaded:", TARGET)
    except Exception as exc:
        raise RuntimeError(f"Dataset download failed: {exc}") from exc

if __name__ == "__main__":
    main()
