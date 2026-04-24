# Academic City RAG Chatbot - Final Project Documentation

**Student Name:** Denzel Nyarko  
**Index Number:** 10022300153  
**Repository Name:** `ai_10022300153`  
**Course:** Introduction to Artificial Intelligence (2026)  

## 1) Project Overview

This project implements a **manual Retrieval-Augmented Generation (RAG) chatbot** for Academic City using:

- Ghana election dataset (CSV)
- Ghana 2025 budget statement (PDF)

The system is built **without LangChain, LlamaIndex, or any pre-built RAG pipeline**.  
All key RAG stages are implemented directly in Python:

- data cleaning
- chunking
- embedding pipeline
- vector storage
- retrieval and scoring
- context packing
- prompt construction
- LLM generation
- stage-by-stage logging

---

## 2) Dataset and Preparation (Part A)

### Sources

- Election CSV: [Ghana_Election_Result.csv](https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv)
- Budget PDF: [2025 Budget Statement](https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf)

### Cleaning

Implemented in `src/data_loader.py`:

- CSV normalization and row-to-text linearization
- PDF text extraction using PyMuPDF
- source tracking and metadata attachment

### Chunking Strategies Implemented

Implemented in `src/chunking.py`:

1. **Sentence-bounded chunking** (`sentence`)
   - packs sentence units up to max character budget
   - preserves semantic continuity better for policy paragraphs
2. **Fixed-size chunking** (`fixed`)
   - sliding window with overlap
   - stable and simple baseline

### Chunking Configuration and Justification

Configured in `src/config.py`:

- `CHUNK_SIZE_CHARS = 900`
- `CHUNK_OVERLAP_CHARS = 150`

Justification:

- 900 chars balances context richness and embedding quality.
- 150-char overlap reduces information loss at chunk boundaries.

### Comparative Analysis Procedure

Run:

```powershell
python scripts/build_index.py --strategy sentence
python scripts/build_index.py --strategy fixed
python scripts/chunking_comparison.py
```

Then manually compare relevance quality in `experiments/MANUAL_LOG_TEMPLATE.md`.

---

## 3) Custom Retrieval System (Part B)

### Embedding Pipeline

Implemented in `src/embedder.py` using `sentence-transformers/all-MiniLM-L6-v2`.

### Vector Store

Implemented in `src/vector_store.py` using FAISS `IndexFlatIP`:

- stores dense vectors
- persists vectors + chunk metadata (`faiss.index`, `chunks.json`)

### Top-k Retrieval and Similarity Scoring

Implemented in `src/retrieval.py`:

- dense retrieval (FAISS cosine via normalized vectors)
- BM25 keyword retrieval (`rank-bm25`)
- score normalization and fusion

### Extension Chosen: Hybrid Search (Keyword + Vector)

Hybrid scoring:

`hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm`

Also includes innovation-ready score adjustment using chunk feedback.

### Failure Case + Fix

Failure example:

- ambiguous numeric query can retrieve irrelevant election chunks for budget intent.

Fix implemented:

- hybrid fusion improves lexical grounding
- tunable `alpha` in sidebar
- optional feedback loop to down-rank repeatedly irrelevant chunks

Evidence format is documented in `docs/FAILURE_CASES_AND_FIX.md`.

---

## 4) Prompt Engineering and Generation (Part C)

Implemented in `src/prompts.py`.

### Prompt Features

- retrieved context injection
- explicit anti-hallucination instruction
- uncertainty behavior ("not enough information in corpus")

### Prompt Variants

- `base`
- `strict`
- `structured`

### Context Window Management

Implemented via greedy rank-first packing with max character budget (`MAX_CONTEXT_CHARS`).

### Experiment Method

Use same query with different prompt variants and compare:

- factual grounding
- hallucination behavior
- answer structure/clarity

Log results manually in `experiments/MANUAL_LOG_TEMPLATE.md`.

---

## 5) Full RAG Pipeline (Part D)

Implemented in `src/rag_pipeline.py` and exposed in `streamlit_app.py`.

Pipeline:

`User Query -> Retrieval -> Context Selection -> Prompt -> LLM -> Response`

### Logging and Transparency

The app displays:

- retrieved chunks
- dense/BM25/hybrid scores
- final prompt sent to model
- structured run JSON

Each run is persisted to:

- `logs/rag_runs.jsonl`

---

## 6) Critical Evaluation and Adversarial Testing (Part E)

Framework is provided in `docs/PART_E_ADVERSARIAL.md`.

### Adversarial Types Covered

- ambiguous query
- misleading/incomplete query

### Metrics Used

- accuracy
- hallucination rate
- response consistency

### RAG vs Pure LLM Comparison

Implemented via sidebar toggle:

- `Enable retrieval (RAG)` ON/OFF

This allows evidence-based ablation comparisons using identical queries.

---

## 7) Architecture and Design Rationale (Part F)

See detailed architecture in:

- `docs/ARCHITECTURE.md`

Includes:

- component diagram
- end-to-end data flow
- interaction of ingest, retrieval, prompting, and UI
- why hybrid retrieval fits election/budget mixed-domain queries

---

## 8) Innovation Component (Part G)

Implemented innovation: **Chunk-level feedback loop** in `src/retrieval.py` + UI controls.

How it works:

- user marks chunk as relevant/irrelevant in Streamlit
- feedback is stored in `data/chunk_feedback.json`
- future hybrid scores are adjusted with feedback multipliers

Value:

- lightweight retrieval tuning without retraining
- practical domain adaptation for repeated coursework queries

---

## 9) Application Deliverables

### Functional UI

Implemented in `streamlit_app.py` with:

- query input
- retrieved chunk display
- scoring display
- final answer display
- final prompt visibility
- brown custom theme

### Run Commands

```powershell
cd "C:\Users\LENOVO FLEX 5\Desktop\AI exam"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_index.py --strategy sentence
python scripts/build_index.py --strategy fixed
python -m streamlit run streamlit_app.py
```

---

## 10) Submission Checklist

1. Push project to GitHub repository named: `ai_10022300153`
2. Deploy app to cloud and copy live URL
3. Record and submit video walkthrough (max 2 minutes)
4. Complete manual experiment log (not AI-generated summary)
5. Include architecture and documentation files from `docs/`
6. Add collaborator:
   - `godwin.danso@acity.edu.gh` or `GodwinDansoAcity`
7. Send email to `godwin.danso@acity.edu.gh` with:
   - GitHub link
   - deployed URL
   - documentation/video references
   - subject exactly:
     `CS4241-Introduction to Artificial Intelligence-2026:[10022300153 Denzel Nyarko]`

---

## 11) Originality Statement

This implementation is a custom, manually engineered RAG system.  
Manual experiments, interpretations, and final reporting should be written by the student to satisfy academic integrity requirements.
