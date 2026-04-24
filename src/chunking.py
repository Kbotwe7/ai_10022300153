"""
Student: Denzel Nyarko | Index: 10022300153
Manual chunking strategies (fixed windows vs sentence-aware boundaries).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Literal, Sequence

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


StrategyName = Literal["fixed", "sentence"]


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: StrategyName
    char_start: int
    char_end: int
    metadata: dict


def _stable_chunk_id(doc_id: str, text: str, start: int, end: int) -> str:
    h = hashlib.sha256(f"{doc_id}|{start}|{end}|{text[:200]}".encode("utf-8")).hexdigest()
    return h[:16]


def chunk_fixed(text: str, doc_id: str, size: int, overlap: int) -> List[TextChunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(1, size - overlap)
    chunks: List[TextChunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + size)
        piece = text[start:end].strip()
        if piece:
            cid = _stable_chunk_id(doc_id, piece, start, end)
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    text=piece,
                    strategy="fixed",
                    char_start=start,
                    char_end=end,
                    metadata={},
                )
            )
        if end == n:
            break
        start += step
    return chunks


def _split_sentences(text: str) -> List[str]:
    pieces = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_sentence_bounded(
    text: str, doc_id: str, max_chars: int, overlap_chars: int
) -> List[TextChunk]:
    sentences = _split_sentences(text)
    chunks: List[TextChunk] = []
    i = 0
    pos = 0
    while i < len(sentences):
        cur: List[str] = []
        length = 0
        j = i
        while j < len(sentences):
            needed = len(sentences[j]) + (1 if cur else 0)
            if length + needed > max_chars:
                break
            cur.append(sentences[j])
            length += needed
            j += 1
        if not cur:
            long = sentences[i]
            for k in range(0, len(long), max_chars):
                piece = long[k : k + max_chars]
                start = pos + k
                end = start + len(piece)
                cid = _stable_chunk_id(doc_id, piece, start, end)
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        text=piece,
                        strategy="sentence",
                        char_start=start,
                        char_end=end,
                        metadata={"split": "hard"},
                    )
                )
            pos += len(long) + 1
            i += 1
            continue
        piece = " ".join(cur)
        start = pos
        end = start + len(piece)
        cid = _stable_chunk_id(doc_id, piece, start, end)
        chunks.append(
            TextChunk(
                chunk_id=cid,
                doc_id=doc_id,
                text=piece,
                strategy="sentence",
                char_start=start,
                char_end=end,
                metadata={"sentence_span": [i, j]},
            )
        )
        pos = end + 1
        if j >= len(sentences):
            break
        if overlap_chars <= 0:
            i = j
            continue
        nxt = j - 1
        while nxt > i and len(" ".join(sentences[nxt:j])) < overlap_chars:
            nxt -= 1
        # Guarantee forward progress; otherwise large overlaps can stall at same i.
        i = max(i + 1, nxt)
    return chunks


def chunk_document(
    text: str,
    doc_id: str,
    strategy: StrategyName,
    size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[TextChunk]:
    if strategy == "fixed":
        return chunk_fixed(text, doc_id, size, overlap)
    if strategy == "sentence":
        return chunk_sentence_bounded(text, doc_id, size, overlap)
    raise ValueError(f"Unknown strategy: {strategy}")


def chunk_many(
    documents: Sequence[tuple[str, str, str]], strategy: StrategyName
) -> List[TextChunk]:
    out: List[TextChunk] = []
    for doc_id, text, uri in documents:
        for c in chunk_document(text, doc_id, strategy):
            c.metadata["source_uri"] = uri
            out.append(c)
    return out
