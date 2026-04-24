# Manual experiment log (template)

**Course:** Introduction to Artificial Intelligence (2026)  
**Student:** Denzel Nyarko  
**Index:** 10022300153  
**Date / time:** YYYY-MM-DD HH:MM (local)

> Fill tables by hand after you run experiments. Do **not** paste raw AI “summary of my experiments” as your submission.

---

## Run metadata

| Field | Value |
|------|-------|
| Model | e.g. gpt-4o-mini |
| Temperature | |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Chunking strategy loaded | sentence / fixed |
| Hybrid alpha | |
| Top-k | |

---

## Experiment 1 — Chunking impact on retrieval

**Query:**

| Strategy | Rank-1 chunk preview (first 120 chars) | Relevant? (Y/N/Partial) | Notes |
|---------|------------------------------------------|---------------------------|-------|
| sentence | | | |
| fixed | | | |

---

## Experiment 2 — Prompt variant comparison (same query)

**Query:**

| Variant (base/strict/structured) | Answer (short excerpt) | Hallucination observed? | Preference |
|----------------------------------|------------------------|-------------------------|------------|
| | | | |
| | | | |

---

## Experiment 3 — Hybrid alpha sweep

**Query:**

| Alpha | Top-3 chunk doc_ids | Useful? | Notes |
|------|---------------------|---------|-------|
| 0.35 | | | |
| 0.65 | | | |
| 0.90 | | | |

---

## Experiment 4 — RAG ON vs OFF (ablation)

**Query:**

| Mode | Retrieved? | Answer excerpt | Factual issues |
|------|--------------|----------------|----------------|
| RAG ON | | | |
| RAG OFF | n/a | | |

---

## Experiment 5 — Feedback loop (innovation)

**Query:**

| Step | Action | Observed top-1 chunk after rerun |
|------|--------|----------------------------------|
| 1 | baseline | |
| 2 | 👎 irrelevant chunk id: ____ | |
| 3 | 👍 relevant chunk id: ____ | |

---

## Sign-off

I confirm this log was written manually by the student named above.

Signature / initials: ________
