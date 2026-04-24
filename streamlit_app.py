"""
Introduction to Artificial Intelligence (2026)
Student: Denzel Nyarko | Index: 10022300153

Academic City RAG Assistant — Streamlit UI (manual RAG; no LangChain / LlamaIndex).
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import INDEX_DIR, STUDENT_INDEX, STUDENT_NAME  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.prompts import VARIANT_BASE, VARIANT_STRICT, VARIANT_STRUCTURED  # noqa: E402
from src.rag_pipeline import run_rag  # noqa: E402
from src.retrieval import HybridRetriever, record_chunk_feedback  # noqa: E402
from src.vector_store import FaissVectorStore  # noqa: E402


@st.cache_resource(show_spinner=True)
def load_stack(strategy: str):
    embedder = Embedder()
    store = FaissVectorStore.load(INDEX_DIR / strategy)
    retriever = HybridRetriever(store, embedder, store.chunks)
    return embedder, store, retriever


def main() -> None:
    st.set_page_config(page_title=f"Academic City RAG — {STUDENT_INDEX}", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6eee3 0%, #e6d0b6 100%);
            color: #3e2723;
        }
        [data-testid="stSidebar"] {
            background-color: #6d4c41;
        }
        [data-testid="stSidebar"] * {
            color: #f8f1e8 !important;
        }
        .stButton > button {
            background-color: #7b4f2c;
            color: #fff8ee;
            border: 1px solid #5d3a1a;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background-color: #5d3a1a;
            color: #fff8ee;
        }
        .stTextArea textarea, .stTextInput input {
            background-color: #fff9f2;
            color: #3e2723;
            border: 1px solid #a1887f;
        }
        [data-testid="stExpander"] {
            background-color: rgba(255, 248, 238, 0.7);
            border: 1px solid #bcaaa4;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Academic City RAG Assistant")
    st.caption(f"Student: {STUDENT_NAME} | Index: {STUDENT_INDEX}")

    if not os.environ.get("GROQ_API_KEY"):
        st.error("Set GROQ_API_KEY in a .env file or environment before running.")
        st.stop()

    with st.sidebar:
        strategy = st.selectbox("Indexed chunking strategy", ["sentence", "fixed"], index=0)
        top_k = st.slider("Top-k retrieval", 3, 15, 6)
        alpha = st.slider("Hybrid alpha (dense weight)", 0.0, 1.0, 0.65, 0.05)
        variant_name = st.selectbox(
            "Prompt variant",
            ["base", "strict", "structured"],
            index=0,
        )
        use_rag = st.toggle("Enable retrieval (RAG)", value=True)
        st.markdown("---")
        st.markdown(
            "**Innovation:** mark chunks 👍/👎 to adjust future hybrid scores (stored locally)."
        )

    try:
        _, _, retriever = load_stack(strategy)
    except Exception as e:
        st.error(f"Failed to load index for strategy `{strategy}`: {e}")
        st.info("Run: `python scripts/download_data.py` then `python scripts/build_index.py`")
        st.stop()

    variants = {
        "base": VARIANT_BASE,
        "strict": VARIANT_STRICT,
        "structured": VARIANT_STRUCTURED,
    }
    variant = variants[variant_name]

    query = st.text_area("Your question", height=120)
    run = st.button("Run pipeline", type="primary")

    if run and query.strip():
        with st.spinner("Running retrieval → prompt → LLM…"):
            out = run_rag(
                query.strip(),
                retriever,
                variant,
                top_k=top_k,
                use_retrieval=use_rag,
                hybrid_alpha=alpha,
            )

        st.subheader("Final answer")
        st.write(out["answer"])

        st.subheader("Retrieved chunks and scores")
        if use_rag:
            for r in out["ranked"]:
                with st.expander(f"Rank {r.rank} | hybrid={r.hybrid_score:.4f} | {r.chunk.doc_id}"):
                    st.write(
                        f"dense={r.dense_score:.4f}, bm25={r.bm25_score:.4f}, chunk_id=`{r.chunk.chunk_id}`"
                    )
                    st.text_area("chunk text", r.chunk.text, height=160, key=f"t_{r.chunk.chunk_id}")
                    c1, c2 = st.columns(2)
                    if c1.button("👍 relevant", key=f"up_{r.chunk.chunk_id}"):
                        record_chunk_feedback(r.chunk.chunk_id, +0.25)
                        st.success("Feedback saved (+0.25 boost)")
                    if c2.button("👎 irrelevant", key=f"dn_{r.chunk.chunk_id}"):
                        record_chunk_feedback(r.chunk.chunk_id, -0.15)
                        st.warning("Feedback saved (-0.15 boost)")
        else:
            st.info("Retrieval disabled — baseline LLM only.")

        st.subheader("Final prompt sent to the LLM")
        st.code(out["final_prompt"], language="text")

        with st.expander("Structured log (JSON)"):
            st.json(out["log"])


if __name__ == "__main__":
    main()
