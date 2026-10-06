#!/usr/bin/env python3
"""
End-to-end RAG inference pipeline.

Pipeline:
  1. Embed the query → retrieve top-10 chunks from FAISS (cosine similarity).
  2. Re-rank the 10 candidates with a cross-encoder for relevance.
  3. Build a prompt from the top re-ranked chunks and call the LLM to answer.

Usage:
    python scripts/inference.py "What is photosynthesis?"
    python scripts/inference.py --interactive
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Add project root ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Lazy imports (heavy libs loaded only when needed) ────────────────
_embedding_model = None
_cross_encoder = None
_faiss_index = None
_metadata: dict[int, dict] | None = None
_chunk_text_cache: dict[str, str] | None = None
_openai_client = None


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

INDEX_PATH = PROJECT_ROOT / "data" / "vector_store" / "faiss_index.bin"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "index_metadata.json"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

# These match rag_config.yaml
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
TOP_K_RETRIEVE = 10  # fetch 10, then rerank
TOP_K_RERANK = 5      # keep top 5 after reranking

SYSTEM_PROMPT = (
    "You are a helpful NEET exam tutor. Answer questions using ONLY the "
    "provided NCERT textbook context. If the context doesn't contain enough "
    "information, say so honestly. Provide clear, concise explanations "
    "suitable for NEET preparation. When relevant, mention the class and "
    "chapter the information comes from."
)


# ═══════════════════════════════════════════════════════════════════════
#  LAZY LOADERS
# ═══════════════════════════════════════════════════════════════════════

def _get_embedding_model():
    """Lazy-load the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_cross_encoder():
    """Lazy-load the cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    return _cross_encoder


def _get_faiss_index():
    """Lazy-load the FAISS index."""
    global _faiss_index
    if _faiss_index is None:
        import faiss
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python scripts/create_db.py"
            )
        _faiss_index = faiss.read_index(str(INDEX_PATH))
    return _faiss_index


def _get_metadata() -> dict[int, dict]:
    """Lazy-load chunk metadata (index → {subject, class, chapter, …})."""
    global _metadata
    if _metadata is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {METADATA_PATH}. "
                "Run: python scripts/create_db.py"
            )
        with open(METADATA_PATH) as f:
            raw = json.load(f)
        # Keys are stored as strings in JSON; convert to int
        _metadata = {int(k): v for k, v in raw.items()}
    return _metadata


def _get_openai_client():
    """Lazy-load the OpenAI client (reads OPENAI_API_KEY from env)."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ═══════════════════════════════════════════════════════════════════════
#  CHUNK TEXT LOADING
# ═══════════════════════════════════════════════════════════════════════

def _load_chunk_text(folder: str, pdf_stem: str, chunk_num: int) -> str:
    """Read the raw text of a single chunk from data/chunks/<folder>/<pdf_stem>_<chunk_num>.json."""
    chunk_file = CHUNKS_DIR / folder / f"{pdf_stem}_{chunk_num}.json"
    if not chunk_file.exists():
        return ""
    with open(chunk_file, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("text", "")


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1 — DENSE RETRIEVAL  (top-10)
# ═══════════════════════════════════════════════════════════════════════

def retrieve_top_k(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    """
    Embed the query, search FAISS, return top_k results with metadata + text.

    Each result dict:
        {index, score, subject, class, chapter, page, folder, pdf_stem, text}
    """
    model = _get_embedding_model()
    index = _get_faiss_index()
    metadata = _get_metadata()

    # Embed & normalize (FAISS IndexFlatIP expects normalized vectors for cosine)
    q_embed = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(q_embed, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata.get(int(idx), {})
        folder = meta.get("folder", "")
        pdf_stem = meta.get("pdf_stem", "")
        chunk_num = meta.get("page", 0)

        text = _load_chunk_text(folder, pdf_stem, chunk_num)

        results.append({
            "index": int(idx),
            "score": float(score),
            "subject": meta.get("subject", "N/A"),
            "class": meta.get("class", "N/A"),
            "chapter": meta.get("chapter", "N/A"),
            "page": chunk_num,
            "folder": folder,
            "pdf_stem": pdf_stem,
            "text": text,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2 — CROSS-ENCODER RE-RANKING
# ═══════════════════════════════════════════════════════════════════════

def rerank_chunks(query: str, chunks: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """
    Re-rank candidate chunks using a cross-encoder.

    The cross-encoder scores each (query, chunk_text) pair directly.
    Returns the top_k chunks sorted by relevance (descending).
    """
    if not chunks:
        return []

    cross_encoder = _get_cross_encoder()

    # Build (query, text) pairs
    pairs = [(query, ch["text"]) for ch in chunks]

    # Get relevance scores
    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    # Attach rerank_score and sort
    for ch, score in zip(chunks, scores):
        ch["rerank_score"] = float(score)

    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks[:top_k]


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3 — BUILD PROMPT & CALL LLM
# ═══════════════════════════════════════════════════════════════════════

def _build_context(chunks: list[dict]) -> str:
    """Build a formatted context string from the re-ranked chunks."""
    parts: list[str] = []
    for i, ch in enumerate(chunks, 1):
        src = (
            f"[Source {i}] "
            f"Class {ch['class']} {ch['subject'].title()}, "
            f"Chapter {ch['chapter']} "
            f"(relevance: {ch['rerank_score']:.3f})"
        )
        parts.append(f"{src}\n{ch['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, ranked_chunks: list[dict]) -> str:
    """Call the LLM with the query and re-ranked context to produce an answer."""
    client = _get_openai_client()
    context = _build_context(ranked_chunks)

    user_message = (
        f"Context from NCERT textbooks:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE  (the only public function you need)
# ═══════════════════════════════════════════════════════════════════════

def answer_query(query: str, verbose: bool = False) -> str:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    Args:
        query: The user's natural-language question.
        verbose: If True, print intermediate retrieval/rerank details.

    Returns:
        The LLM-generated answer string.
    """
    # ── 1. Retrieve top-10 ──
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}\n📥 Retrieving top-{TOP_K_RETRIEVE} chunks …")

    candidates = retrieve_top_k(query, top_k=TOP_K_RETRIEVE)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"{'Rank':<5} {'FAISS':<8} {'Subject':<12} {'Class':<7} {'Ch':<5}")
        print(f"{'─'*60}")
        for i, c in enumerate(candidates, 1):
            print(f"{i:<5} {c['score']:<8.4f} {c['subject']:<12} "
                  f"{c['class']!s:<7} {c['chapter']:<5}")

    if not candidates:
        return "No relevant content found in the NCERT textbooks for your query."

    # ── 2. Re-rank ──
    if verbose:
        print(f"\n🔄 Re-ranking with cross-encoder …")

    ranked = rerank_chunks(query, candidates, top_k=TOP_K_RERANK)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"After re-ranking (top {len(ranked)}):")
        print(f"{'Rank':<5} {'Rerank':<8} {'Subject':<12} {'Class':<7} {'Ch':<5}")
        print(f"{'─'*60}")
        for i, c in enumerate(ranked, 1):
            print(f"{i:<5} {c['rerank_score']:<8.4f} {c['subject']:<12} "
                  f"{c['class']!s:<7} {c['chapter']:<5}")

    # ── 3. Generate ──
    if verbose:
        print(f"\n🤖 Generating answer with {LLM_MODEL} …\n")

    answer = generate_answer(query, ranked)
    return answer


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAG inference: retrieve → rerank → generate"
    )
    parser.add_argument(
        "query", nargs="?", type=str,
        help="Question to answer (omit for interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start an interactive Q&A session",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show retrieval and reranking details",
    )
    args = parser.parse_args()

    if args.interactive:
        print("\n📚 NCERT RAG — Interactive Q&A")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                q = input("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if q.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not q:
                continue
            answer = answer_query(q, verbose=args.verbose)
            print(f"\n{answer}\n")
    elif args.query:
        answer = answer_query(args.query, verbose=args.verbose)
        print(answer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()