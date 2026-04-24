"""
Student: Denzel Nyarko | Index: 10022300153
Prompt templates, hallucination guardrails, and context packing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import MAX_CONTEXT_CHARS
from .retrieval import RetrievedChunk


@dataclass
class PromptVariant:
    name: str
    system: str
    user_template: str


VARIANT_BASE = PromptVariant(
    name="base",
    system=(
        "You are an assistant for Academic City University College. "
        "Answer using ONLY the provided CONTEXT when it is sufficient. "
        "If CONTEXT is insufficient, say you do not have enough information in the corpus "
        "and list what is missing. Do not invent statistics, dates, or figures not present in CONTEXT."
    ),
    user_template=(
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
        "Instructions: Cite supporting lines implicitly (no fake citations). "
        "If multiple regions/years conflict, report both and state the ambiguity."
    ),
)

VARIANT_STRICT = PromptVariant(
    name="strict",
    system=(
        "You are a careful analyst. Use ONLY verbatim facts supported by CONTEXT. "
        "If unsure, respond with: 'Not found in provided documents.' "
        "Never use outside knowledge for numbers."
    ),
    user_template="CONTEXT:\n{context}\n\nQUESTION:\n{question}",
)

VARIANT_STRUCTURED = PromptVariant(
    name="structured",
    system=VARIANT_BASE.system,
    user_template=(
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
        "Answer format:\n"
        "1) Direct answer (1-3 sentences)\n"
        "2) Evidence bullets (quote short phrases from CONTEXT)\n"
        "3) Confidence: High/Medium/Low with reason\n"
    ),
)


def pack_context(
    ranked: Sequence[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, List[RetrievedChunk]]:
    """Greedy packing by hybrid rank until char budget; drops lowest-ranked overflow."""
    blocks: List[str] = []
    used = 0
    kept: List[RetrievedChunk] = []
    for rc in sorted(ranked, key=lambda x: x.rank):
        header = f"[doc={rc.chunk.doc_id} | rank={rc.rank} | hybrid={rc.hybrid_score:.3f}]\n"
        body = rc.chunk.text.strip()
        block = header + body + "\n\n"
        if used + len(block) > max_chars:
            continue
        blocks.append(block)
        used += len(block)
        kept.append(rc)
    if not blocks and ranked:
        rc = sorted(ranked, key=lambda x: x.rank)[0]
        snippet = rc.chunk.text[: max_chars - 120]
        blocks.append(
            f"[doc={rc.chunk.doc_id} | rank={rc.rank} | truncated]\n{snippet}\n\n"
        )
        kept = [rc]
    return "".join(blocks), kept


def build_messages(
    variant: PromptVariant, question: str, ranked: Sequence[RetrievedChunk]
) -> tuple[list[dict], str]:
    context, kept = pack_context(ranked)
    user = variant.user_template.format(context=context, question=question.strip())
    messages = [
        {"role": "system", "content": variant.system},
        {"role": "user", "content": user},
    ]
    final_prompt = f"SYSTEM:\n{variant.system}\n\nUSER:\n{user}"
    return messages, final_prompt
