"""
Student: Denzel Nyarko | Index: 10022300153
Full RAG pipeline with structured stage logging.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LOGS_DIR, STUDENT_INDEX, STUDENT_NAME
from .llm_client import chat_complete
from .prompts import PromptVariant, build_messages
from .retrieval import HybridRetriever, RetrievedChunk


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_rag(
    query: str,
    retriever: HybridRetriever,
    variant: PromptVariant,
    top_k: int,
    use_retrieval: bool,
    hybrid_alpha: float | None = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    log: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "student_index": STUDENT_INDEX,
        "student_name": STUDENT_NAME,
        "query": query,
        "variant": variant.name,
        "top_k": top_k,
        "use_retrieval": use_retrieval,
        "hybrid_alpha": hybrid_alpha,
        "stages": {},
    }

    if use_retrieval:
        ranked = retriever.hybrid_search(query, k=top_k, alpha=hybrid_alpha)
        log["stages"]["retrieval"] = {
            "method": "hybrid_bm25_faiss_feedback",
            "hits": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "dense": r.dense_score,
                    "bm25": r.bm25_score,
                    "hybrid": r.hybrid_score,
                    "preview": r.chunk.text[:240].replace("\n", " "),
                }
                for r in ranked
            ],
        }
    else:
        ranked = []
        log["stages"]["retrieval"] = {"skipped": True}

    if use_retrieval:
        messages, final_prompt = build_messages(variant, query, ranked)
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer from general knowledge. "
                    "This run intentionally has NO document context (ablation baseline)."
                ),
            },
            {"role": "user", "content": query},
        ]
        final_prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
    log["stages"]["context_selection"] = {
        "packed_chars": len(messages[1]["content"]),
        "chunks_used": len(ranked),
    }
    log["stages"]["prompt"] = {"final_prompt": final_prompt}

    answer = chat_complete(messages)
    log["stages"]["generation"] = {"answer": answer}
    log["latency_sec"] = round(time.time() - t0, 3)

    lp = log_path or (LOGS_DIR / "rag_runs.jsonl")
    _append_jsonl(lp, log)
    return {
        "answer": answer,
        "ranked": ranked,
        "final_prompt": final_prompt,
        "messages": messages,
        "log": log,
    }


def serialize_ranked(ranked: List[RetrievedChunk]) -> List[dict]:
    out = []
    for r in ranked:
        out.append(
            {
                "rank": r.rank,
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "dense_score": r.dense_score,
                "bm25_score": r.bm25_score,
                "hybrid_score": r.hybrid_score,
                "text": r.chunk.text,
                "metadata": r.chunk.metadata,
            }
        )
    return out
