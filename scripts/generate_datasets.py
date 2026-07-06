#!/usr/bin/env python3
"""Download real-world datasets into datasets/raw/ ."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_datasets import download_all  # noqa: E402


def main() -> None:
    (ROOT / "datasets" / "processed").mkdir(parents=True, exist_ok=True)
    paths = download_all()
    print(f"Prepared {len(paths)} datasets in {ROOT / 'datasets' / 'raw'}")


if __name__ == "__main__":
    main()
