"""
Student: Denzel Nyarko | Index: 10022300153
Build FAISS index from cleaned documents (manual RAG indexing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import StrategyName, chunk_many  # noqa: E402
from src.config import CHUNK_STRATEGY, INDEX_DIR, RAW_DIR  # noqa: E402
from src.data_loader import load_budget_pdf, load_election_csv  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.vector_store import FaissVectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["sentence", "fixed"],
        default=CHUNK_STRATEGY,
        help="Chunking strategy to index",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/index/<strategy>)",
    )
    args = parser.parse_args()
    strategy: StrategyName = args.strategy  # type: ignore[assignment]

    csv_path = RAW_DIR / "Ghana_Election_Result.csv"
    pdf_path = RAW_DIR / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    if not csv_path.exists() or not pdf_path.exists():
        raise SystemExit("Missing raw files. Run: python scripts/download_data.py")

    election = load_election_csv(
        csv_path,
        "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv",
    )
    budget = load_budget_pdf(
        pdf_path,
        "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf",
    )

    docs = [
        (election.doc_id, election.text, election.source_uri),
        (budget.doc_id, budget.text, budget.source_uri),
    ]
    chunks = chunk_many(docs, strategy=strategy)
    print(f"Chunks: {len(chunks)} (strategy={strategy})")

    embedder = Embedder()
    vectors = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(dim=embedder.dim)
    store.add(chunks, vectors)

    out_dir = args.out or (INDEX_DIR / strategy)
    store.save(out_dir)
    print("Saved index to", out_dir)


if __name__ == "__main__":
    main()
