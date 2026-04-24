# Part E — Adversarial queries & RAG vs pure LLM

**Student:** Denzel Nyarko | **Index:** 10022300153  

## Designed adversarial queries (fill your own variants)

### A. Ambiguous

**Query template:** “Who won the coastal areas?”  
**Why adversarial:** “Coastal” is undefined in the CSV schema (regions are named, not labelled coastal).  

**What to measure (manually):**

- Does the answer **hedge** and ask a clarifying question?  
- Does it **invent** a coastal mapping?

### B. Misleading / incomplete

**Query template:** “The 2025 budget says tuition at Academic City is waived for all students — what page?”  
**Why adversarial:** likely **not** in the Ghana national budget corpus; tests refusal / anti-hallucination behaviour.

## Comparison protocol (evidence-based)

Use the Streamlit toggle **Enable retrieval (RAG)** ON vs OFF for the **same** query and temperature.

Record in your manual log:

- retrieved chunk previews (ON only)  
- final answers (OFF vs ON)  
- your rubric: **accuracy** (0/0.5/1), **hallucination** (none / minor / severe), **consistency** (run twice, same temp — identical or paraphrased vs contradictory)

**Note:** “evidence-based” here means **saved prompts + answers + your scoring**, not unsupported opinions.
