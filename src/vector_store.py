"""
Student: Denzel Nyarko | Index: 10022300153
FAISS vector store with manual save/load (L2-normalized vectors => cosine via inner product).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from .chunking import TextChunk


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[TextChunk] = []

    def add(self, chunks: Sequence[TextChunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunks/vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError("vector dimension mismatch")
        self.chunks.extend(list(chunks))
        self.index.add(np.ascontiguousarray(vectors.astype(np.float32)))

    def search(self, query_vec: np.ndarray, k: int) -> Tuple[List[TextChunk], List[float]]:
        if query_vec.ndim == 1:
            q = query_vec.reshape(1, -1).astype(np.float32)
        else:
            q = query_vec.astype(np.float32)
        scores, idx = self.index.search(q, k)
        hits: List[TextChunk] = []
        sims: List[float] = []
        for i, s in zip(idx[0], scores[0]):
            if i < 0:
                continue
            hits.append(self.chunks[int(i)])
            sims.append(float(s))
        return hits, sims

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        meta = [asdict(c) for c in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [
            TextChunk(
                chunk_id=m["chunk_id"],
                doc_id=m["doc_id"],
                text=m["text"],
                strategy=m["strategy"],
                char_start=m["char_start"],
                char_end=m["char_end"],
                metadata=m.get("metadata") or {},
            )
            for m in meta
        ]
        index = faiss.read_index(str(directory / "faiss.index"))
        vs = cls(dim=index.d)
        vs.chunks = chunks
        vs.index = index
        return vs
