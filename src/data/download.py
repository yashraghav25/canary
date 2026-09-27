"""
Explicit, one-dataset-at-a-time downloader.

Usage:
    python download.py --key subf
    python download.py --key ai_mechanic
    python download.py --list

Deliberately does NOT support "download everything" in one call —
several of these datasets are multi-GB (see download_manifest.py),
so each pull should be a conscious choice, not a side effect.

Kaggle datasets require a Kaggle API token at ~/.kaggle/kaggle.json
(https://www.kaggle.com/docs/api) before this script can use them.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import requests

from download_manifest import MANIFEST, DatasetEntry

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _find(key: str) -> DatasetEntry:
    for e in MANIFEST:
        if e.key == key:
            return e
    raise SystemExit(f"Unknown dataset key '{key}'. Run --list to see valid keys.")


def download_kaggle(entry: DatasetEntry, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", entry.location, "-p", str(dest), "--unzip"],
        check=True,
    )


def download_http(entry: DatasetEntry, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if entry.key == "paderborn" or entry.key == "cwru":
        print(
            f"'{entry.key}' is served from an interactive university page, not a single file: "
            f"{entry.location}\nOpen it in a browser and select the specific .mat files you need "
            f"(start with 2-3 bearings, not the full set) into {dest}"
        )
        return
    if entry.key == "vehicle_interior_sound":
        print(
            f"Zenodo record {entry.location} — download the archive manually via the "
            f"'Download' button on that page into {dest} (Zenodo rate-limits scripted pulls)."
        )
        return
    raise SystemExit(f"No generic http handler wired up for '{entry.key}' yet.")


def download_s3(entry: DatasetEntry, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "ims_bearings.zip"
    print(f"Downloading {entry.location} -> {out_file}")
    with requests.get(entry.location, stream=True) as r:
        r.raise_for_status()
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Done. Unzip manually to inspect structure before wiring into the loader.")


def download_git(entry: DatasetEntry, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} already populated, skipping clone.")
        return
    subprocess.run(["git", "clone", "--depth", "1", entry.location, str(dest)], check=True)


HANDLERS = {
    "kaggle": download_kaggle,
    "http": download_http,
    "s3": download_s3,
    "git": download_git,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="dataset key from download_manifest.py")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or not args.key:
        for e in MANIFEST:
            print(f"{e.key:28s} [{e.pool:9s}] {e.approx_size}")
        return

    entry = _find(args.key)
    dest = RAW_DIR / entry.key
    print(f"About to fetch '{entry.key}' ({entry.approx_size}) via {entry.source} into {dest}")
    HANDLERS[entry.source](entry, dest)


if __name__ == "__main__":
    sys.exit(main())
