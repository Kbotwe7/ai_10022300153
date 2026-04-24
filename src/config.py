"""
Introduction to Artificial Intelligence (2026)
Student: Denzel Nyarko | Index: 10022300153
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
LOGS_DIR = PROJECT_ROOT / "logs"

STUDENT_NAME = os.environ.get("STUDENT_NAME", "Denzel Nyarko")
STUDENT_INDEX = os.environ.get("STUDENT_INDEX", "10022300153")

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking defaults (justified in docs/PART_A_CHUNKING.md)
CHUNK_SIZE_CHARS = int(os.environ.get("CHUNK_SIZE_CHARS", "900"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "150"))
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "sentence")  # "sentence" | "fixed"

# Hybrid retrieval
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.65"))  # weight on dense similarity

# Context window (approximate chars for prompt budget)
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "6000"))

FEEDBACK_PATH = DATA_DIR / "chunk_feedback.json"

for p in (RAW_DIR, INDEX_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)
