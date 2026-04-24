"""
Student: Denzel Nyarko | Index: 10022300153
Print comparative chunking statistics (build both indexes first).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import INDEX_DIR  # noqa: E402
from src.vector_store import FaissVectorStore  # noqa: E402


def summarize(strategy: str) -> dict:
    store = FaissVectorStore.load(INDEX_DIR / strategy)
    lens = [len(c.text) for c in store.chunks]
    avg = sum(lens) / max(1, len(lens))
    return {
        "strategy": strategy,
        "num_chunks": len(store.chunks),
        "avg_chars": round(avg, 1),
        "min_chars": min(lens) if lens else 0,
        "max_chars": max(lens) if lens else 0,
    }


def main() -> None:
    rows = []
    for s in ("sentence", "fixed"):
        p = INDEX_DIR / s
        if not (p / "faiss.index").exists():
            print(f"Missing index for `{s}` — run: python scripts/build_index.py --strategy {s}")
            continue
        rows.append(summarize(s))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
