# Introduction to Artificial Intelligence (2026)

**Student name:** Denzel Nyarko  
**Index number:** 10022300153  

**GitHub repository name (required):** `ai_10022300153`  
*(Rename this folder / GitHub repo to match your actual index.)*

## What this project is

A **manual Retrieval-Augmented Generation (RAG)** assistant themed for **Academic City University College**, grounded on:

- Ghana presidential election results (CSV) — [source](https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv)  
- Ghana **2025 Budget Statement** (PDF) — [source](https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf)

**Course constraint honoured:** there is **no** LangChain, LlamaIndex, or packaged RAG framework. Chunking, embeddings, vector search, BM25 hybrid fusion, prompt construction, logging, and the LLM call are implemented directly in Python.

## Innovation (Part G)

**Chunk-level feedback loop:** in the Streamlit UI you can mark retrieved chunks 👍 / 👎. The system stores per-`chunk_id` boosts in `data/chunk_feedback.json` and multiplies future hybrid retrieval scores, nudging the retriever toward teacher-style corrections without retraining embeddings.

## Local setup

1. Student details are prefilled in this project. Keep your index in all submitted artefacts, as required in the brief.
2. Python **3.10+** recommended.
3. Create a virtual environment and install dependencies:

```powershell
cd "c:\Users\LENOVO FLEX 5\Desktop\AI exam"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (optional `GROQ_CHAT_MODEL`).

5. Download corpora and build **both** chunking indexes (needed for the Part A comparison script):

```powershell
python scripts/download_data.py
python scripts/build_index.py --strategy sentence
python scripts/build_index.py --strategy fixed
```

6. Launch the UI:

```powershell
streamlit run streamlit_app.py
```

## Logging

Each end-to-end question appends a JSON line to `logs/rag_runs.jsonl` with stages: retrieval scores, packed context size, **full final prompt**, and model answer.

## Deployment (pick one)

- **Streamlit Community Cloud:** push this repo, set secrets for `GROQ_API_KEY`, upload/build artefacts **or** bake a CI step that runs `download_data.py` + `build_index.py` before start (note: first model download can be slow).  
- **Azure / Railway / Render:** container or VM with the same commands; expose the Streamlit port.

Document your **public URL** in your report and submission email.

## GitHub collaborator (mandatory for grading)

Invite **`godwin.danso@acity.edu.gh`** or GitHub user **`GodwinDansoAcity`** with read access to your repository.

## Submission checklist (email to `godwin.danso@acity.edu.gh`)

Subject: `Introduction to Artificial Intelligence-2026:[your index and name]`

Include:

- GitHub repo URL (named `ai_<index>`)  
- Deployed app URL  
- **Video** (≤ 2 minutes) — you must record this yourself, walking through design + demo  
- **Manual experiment logs** (typed by you; see `experiments/MANUAL_LOG_TEMPLATE.md`) — not raw AI prose dumps  
- Architecture + detailed documentation (`docs/`)

## Academic integrity

This scaffold is a **starting implementation**. You must personalise it, run your own experiments, write your own logs and analysis, and produce your own video. Copy-paste submissions risk **zero** marks per the brief.
