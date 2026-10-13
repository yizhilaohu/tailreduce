"""Download the public index data (pinned commits, checksums) and build the loss series.

Usage:  python scripts/download_data.py [--force]
Outputs: data/raw/*.rda, data/processed/<NAME>_losses.csv, data/processed/summary.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tailreduce import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = list(data.MAIN_DATASETS)
    data.download(ROOT / "data" / "raw", names=names, force=args.force)
    summary = data.build(ROOT / "data" / "raw", ROOT / "data" / "processed", names=names)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
