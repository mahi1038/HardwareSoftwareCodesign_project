"""Download and unpack UCI HAR into data/. Works on Windows, macOS, Linux.

Usage:
    python data/download.py              # normal download
    python data/download.py --insecure   # skip SSL verification (last resort)
If data/har_outer.zip already exists (manual download), it is used directly.
"""
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA = Path(__file__).resolve().parent
TARGET = DATA / "UCI HAR Dataset"
CHECK = TARGET / "train" / "Inertial Signals" / "total_acc_x_train.txt"


def download(outer: Path, insecure: bool):
    if insecure:
        ctx = ssl._create_unverified_context()
    else:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
    with urllib.request.urlopen(URL, context=ctx) as r, open(outer, "wb") as f:
        shutil.copyfileobj(r, f)


def main():
    if CHECK.exists():
        print("UCI HAR already present, skipping.")
        return
    outer = DATA / "har_outer.zip"
    if outer.exists():
        print("Using existing data/har_outer.zip")
    else:
        print("Downloading...")
        download(outer, insecure="--insecure" in sys.argv)

    with zipfile.ZipFile(outer) as z:
        z.extractall(DATA)
    inner = DATA / "UCI HAR Dataset.zip"      # the archive may contain a nested zip
    if inner.exists():
        with zipfile.ZipFile(inner) as z:
            z.extractall(DATA)
        inner.unlink()
    outer.unlink()
    shutil.rmtree(DATA / "__MACOSX", ignore_errors=True)
    if not CHECK.exists():
        sys.exit("ERROR: raw Inertial Signals not found after extraction")
    print(f"OK: UCI HAR is in {TARGET}")


if __name__ == "__main__":
    main()