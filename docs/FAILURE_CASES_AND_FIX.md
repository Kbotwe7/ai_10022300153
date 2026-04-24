# Failure cases — retrieval + implemented fix

**Student:** Denzel Nyarko | **Index:** 10022300153  

## Example failure (vector-only mental model)

**Query (ambiguous):** “What happened with revenue in 2020?”

**Symptom:** dense retrieval may return **election turnout / vote share** chunks because the year token “2020” and vague “revenue” language accidentally align with unrelated numeric-heavy passages.

## Why it fails

Pure dense similarity can conflate **shared surface patterns** (years, percentages, long digit spans) without requiring **keyword agreement** with fiscal vocabulary (“tax”, “GDP”, “budget”, “GRA”, “receipts”, …).

## Fix implemented in code

**Hybrid retrieval (dense + BM25)** in `src/retrieval.py` fuses normalised dense and sparse scores:

\[
s_{\text{hybrid}} = \alpha \cdot \text{dense\_norm} + (1-\alpha)\cdot \text{bm25\_norm}
\]

Default \(\alpha = 0.65\) (tunable in the Streamlit sidebar / env). BM25 rewards explicit lexical overlap, pulling the system toward budget sections when queries contain fiscal terms, while dense retrieval still supports paraphrases.

## What you should add for submission evidence

1. Paste **before/after** retrieved chunk lists for the same query (export from UI or `logs/rag_runs.jsonl`).  
2. Sweep \(\alpha \in \{0.35, 0.65, 0.9\}\) and note precision@k *by your manual judgement* — that is acceptable “evidence” for this coursework as long as you are transparent.
