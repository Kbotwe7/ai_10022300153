"""
Student: Denzel Nyarko | Index: 10022300153
Download required CSV and PDF into data/raw/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_DIR  # noqa: E402


CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
)
PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "Ghana_Election_Result.csv"
    pdf_path = RAW_DIR / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    print("Downloading CSV...")
    _download(CSV_URL, csv_path)
    print("Downloading PDF (may take a while)...")
    _download(PDF_URL, pdf_path)
    print("Done:", csv_path, pdf_path)


if __name__ == "__main__":
    main()
