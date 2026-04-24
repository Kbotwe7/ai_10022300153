"""
Student: Denzel Nyarko | Index: 10022300153
Load and clean Ghana election CSV and 2025 budget PDF (manual pipeline; no LangChain).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pandas as pd


@dataclass
class RawDocument:
    doc_id: str
    title: str
    text: str
    source_uri: str


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_election_csv(path: Path, source_uri: str) -> RawDocument:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Drop fully empty rows / duplicates
    df = df.dropna(how="all").drop_duplicates()
    # Normalize percentage strings and votes
    if "Votes(%)" in df.columns:
        df["Votes(%)"] = df["Votes(%)"].astype(str).str.replace("%", "", regex=False).str.strip()
    if "Votes" in df.columns:
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
    df = df.dropna(subset=[c for c in df.columns if c in ("Year", "New Region", "Candidate")])
    lines: List[str] = []
    for _, row in df.iterrows():
        parts = [f"{k}: {v}" for k, v in row.items() if pd.notna(v) and str(v).strip()]
        lines.append(" | ".join(parts))
    body = "\n".join(lines)
    return RawDocument(
        doc_id="ghana_election_results",
        title="Ghana Presidential Election Results (dataset rows)",
        text=_clean_text(body),
        source_uri=source_uri,
    )


def load_budget_pdf(path: Path, source_uri: str) -> RawDocument:
    doc = fitz.open(path)
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    body = "\n\n".join(parts)
    return RawDocument(
        doc_id="ghana_2025_budget_statement",
        title="2025 Budget Statement and Economic Policy (Ghana)",
        text=_clean_text(body),
        source_uri=source_uri,
    )
