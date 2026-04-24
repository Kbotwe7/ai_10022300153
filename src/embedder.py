"""
Student: Denzel Nyarko | Index: 10022300153
Manual embedding pipeline using Sentence-Transformers (no LangChain).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
