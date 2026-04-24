# Part A — Chunking design & comparative notes

**Student:** Denzel Nyarko | **Index:** 10022300153  

## Chunk size & overlap (defaults)

Configured in `src/config.py`:

- `CHUNK_SIZE_CHARS = 900`  
- `CHUNK_OVERLAP_CHARS = 150`  

### Justification

- **900 characters** (~200–230 tokens depending on language) is large enough to retain a **small table segment** or a **short policy subsection** from the budget PDF, while staying below typical embedding model sweet spots where noise dominates.  
- **150-character overlap (~17%)** reduces boundary loss when a fact spans two windows (common in legal/fiscal prose and CSV-derived rows that reference the same region across consecutive lines).

## Implemented strategies

1. **`sentence` (default):** accumulates sentences until the char budget is reached, then rewinds sentence indices to satisfy overlap. Long single sentences are hard-split.  
2. **`fixed`:** classic sliding window with step `size - overlap`.

## Comparative analysis (how *you* should complete this for marks)

Run both indexes:

```powershell
python scripts/build_index.py --strategy sentence
python scripts/build_index.py --strategy fixed
python scripts/chunking_comparison.py
```

Then, for **the same 5–10 evaluation queries** you design, record in `experiments/MANUAL_LOG_TEMPLATE.md`:

- retrieved chunk IDs / previews  
- which strategy surfaced more **on-topic** evidence  
- failure cases (e.g., numeric tables split badly)

**Important:** the marker expects *your* manual reasoning — paste tables and write analysis yourself; do not submit an AI-only “summary” as your log.
