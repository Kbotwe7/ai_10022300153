"""
Student: Denzel Nyarko | Index: 10022300153
Top-k retrieval, similarity scoring, hybrid keyword (BM25) + dense (FAISS), and feedback boosts.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import TextChunk
from .config import FEEDBACK_PATH, HYBRID_ALPHA
from .embedder import Embedder
from .vector_store import FaissVectorStore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if math.isclose(hi, lo):
        return [1.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


@dataclass
class RetrievedChunk:
    chunk: TextChunk
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rank: int


class HybridRetriever:
    def __init__(
        self,
        store: FaissVectorStore,
        embedder: Embedder,
        chunks: Sequence[TextChunk],
        feedback_path: Path | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunks = list(chunks)
        self.corpus_tokens = [_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.feedback_path = feedback_path or FEEDBACK_PATH
        self._feedback = self._load_feedback()

    def _load_feedback(self) -> Dict[str, float]:
        if not self.feedback_path.exists():
            return {}
        try:
            raw = json.loads(self.feedback_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        out: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def reload_feedback(self) -> None:
        self._feedback = self._load_feedback()

    def dense_search(self, query: str, k: int) -> Tuple[List[TextChunk], List[float]]:
        q = self.embedder.encode_query(query)
        return self.store.search(q, k)

    def bm25_search(self, query: str, k: int) -> Tuple[List[TextChunk], List[float]]:
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        idx = np.argsort(-scores)[:k]
        hits = [self.chunks[int(i)] for i in idx]
        sims = [float(scores[int(i)]) for i in idx]
        return hits, sims

    def hybrid_search(self, query: str, k: int, alpha: float | None = None) -> List[RetrievedChunk]:
        a = HYBRID_ALPHA if alpha is None else alpha
        k2 = min(len(self.chunks), max(k * 5, k))
        d_hits, d_scores = self.dense_search(query, k2)
        b_hits, b_scores = self.bm25_search(query, k2)
        dense_map: Dict[str, float] = {}
        for c, s in zip(d_hits, d_scores):
            dense_map[c.chunk_id] = max(dense_map.get(c.chunk_id, float("-inf")), s)
        bm_map: Dict[str, float] = {}
        for c, s in zip(b_hits, b_scores):
            bm_map[c.chunk_id] = max(bm_map.get(c.chunk_id, float("-inf")), s)

        ids = set(dense_map) | set(bm_map)
        dense_list = [dense_map.get(i, 0.0) for i in ids]
        bm_list = [bm_map.get(i, 0.0) for i in ids]
        dn = _minmax(dense_list)
        bn = _minmax(bm_list)
        id_list = list(ids)
        hybrid: List[Tuple[str, float, float, float]] = []
        self.reload_feedback()
        for cid, ds, bs, dsn, bsn in zip(id_list, dense_list, bm_list, dn, bn):
            base = a * dsn + (1.0 - a) * bsn
            boost = 1.0 + float(self._feedback.get(cid, 0.0))
            hybrid.append((cid, ds, bs, base * boost))
        hybrid.sort(key=lambda x: x[3], reverse=True)
        chunk_by_id = {c.chunk_id: c for c in self.chunks}
        out: List[RetrievedChunk] = []
        for rank, (cid, ds, bs, hs) in enumerate(hybrid[:k], start=1):
            ch = chunk_by_id.get(cid)
            if ch is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk=ch,
                    dense_score=ds,
                    bm25_score=bs,
                    hybrid_score=hs,
                    rank=rank,
                )
            )
        return out


def record_chunk_feedback(
    chunk_id: str, delta: float, path: Path | None = None
) -> None:
    """Innovation: simple feedback loop — positive delta boosts future hybrid scores."""
    p = path or FEEDBACK_PATH
    data: Dict[str, float] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[str(chunk_id)] = float(data.get(str(chunk_id), 0.0)) + float(delta)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
