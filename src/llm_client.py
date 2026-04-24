"""
Student: Denzel Nyarko | Index: 10022300153
Direct Groq Chat Completions API (no LangChain).
"""
from __future__ import annotations

import os
from typing import List, Sequence

from groq import Groq


def chat_complete(
    messages: Sequence[dict],
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    mdl = model or os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
    resp = client.chat.completions.create(
        model=mdl,
        messages=list(messages),
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_complete_many(
    messages: Sequence[dict],
    n: int,
    model: str | None = None,
    temperature: float = 0.2,
) -> List[str]:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    mdl = model or os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
    resp = client.chat.completions.create(
        model=mdl,
        messages=list(messages),
        temperature=temperature,
        n=n,
    )
    return [(c.message.content or "").strip() for c in resp.choices]
