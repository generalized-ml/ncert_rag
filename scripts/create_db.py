#!/usr/bin/env python3
"""
Build a FAISS vector database from pre-generated embeddings.

Embeddings are read from:  data/embeddings/*.npy
FAISS index saved to:     data/vector_store/faiss_index.bin
Metadata saved to:        data/metadata/index_metadata.json

Usage:
    python scripts/create_db.py
    python scripts/create_db.py --subject physics --class 11
    python scripts/create_db.py --query "What is photosynthesis?"
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = METADATA_DIR / "index_metadata.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "processing_config.json"

# ── Helpers ─────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def parse_embedding_filename(filename: str) -> dict | None:
    """
    Parse filenames like:
      kebo101_10_embeddings.npy   → chapter 01
      lech1a1_16_embeddings.npy   → appendix 1
      lech2ps_10_embeddings.npy   → postscript
      kech1an_5_embeddings.npy    → answers
    Returns: {folder, pdf_stem, chunk_num, file_type} or None.
    """
    m = re.match(r"^([a-z]+\d+[a-z0-9]+)_(\d+)_embeddings\.npy$", filename)
    if not m:
        return None
    pdf_stem = m.group(1)       # e.g. kebo101, lech1a1, lech2ps
    chunk_num = int(m.group(2)) # e.g. 10
    # Folder name = first 5 chars of pdf_stem + "dd"
    folder = pdf_stem[:5] + "dd"

    # Detect file type from the suffix after the part number
    suffix = pdf_stem[6:]  # e.g. "01", "a1", "ps", "an"
    if suffix == "ps":
        file_type = "postscript"
    elif suffix == "an":
        file_type = "answers"
    elif re.match(r"^a\d+$", suffix):
        file_type = "appendix"
    else:
        file_type = "chapter"

    return {
        "folder": folder,
        "pdf_stem": pdf_stem,
        "chunk_num": chunk_num,
        "file_type": file_type,
    }


def build_metadata_entry(folder: str, pdf_stem: str, chunk_num: int, file_type: str) -> dict:
    """Return {subject, page, chapter, class, file_type} for one chunk."""
    config = load_config()
    info = config["folders"].get(folder, {})

    # Chapter = last 2 digits of pdf_stem (e.g. kebo101 → "01")
    # For appendix/answers/postscript, use the suffix (e.g. "a1", "an", "ps")
    suffix = pdf_stem[6:]
    if file_type == "chapter":
        chapter = suffix if suffix.isdigit() else "00"
    else:
        chapter = suffix  # "a1", "an", "ps"

    return {
        "subject": info.get("subject", "N/A"),
        "page": chunk_num,
        "chapter": chapter,
        "class": info.get("class", "N/A"),
        "file_type": file_type,
    }


# ── Build FAISS Index ───────────────────────────────────────────────
def build_index() -> tuple:
    """Load pre-generated .npy embeddings, build FAISS index, save metadata."""
    import faiss

    config = load_config()
    
    # ── 1. Collect embedding files ──
    emb_files = sorted(EMBEDDINGS_DIR.glob("*.npy"))
    if not emb_files:
        print(f"No .npy files found in {EMBEDDINGS_DIR}")
        sys.exit(1)

    print(f"Found {len(emb_files)} embedding files in {EMBEDDINGS_DIR}")

    # ── 2. Load embeddings & build metadata ──
    all_embeddings: list[np.ndarray] = []
    metadata: dict[int, dict] = {}  # index → {subject, page, chapter, class}
    idx = 0

    for emb_file in emb_files:
        parsed = parse_embedding_filename(emb_file.name)

        if parsed is None:
            print(f"  ⚠ Skipping unrecognized file: {emb_file.name}")
            continue

        folder = parsed["folder"]
        if folder not in config["folders"]:
            continue

        # Load the numpy embedding
        emb = np.load(emb_file).astype("float32")

        # Handle both 1D and 2D arrays
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        print(emb.shape)
        all_embeddings.append(emb)
        metadata[idx] = build_metadata_entry(
            folder, parsed["pdf_stem"], parsed["chunk_num"], parsed["file_type"]
        )
        idx += 1
        

    if not all_embeddings:
        print("No embeddings matched the filters. Check --subject / --class.")
        sys.exit(1)

    print(f"Loaded {len(all_embeddings)} embedding arrays, total vectors: {idx}")

    print(all_embeddings[0].shape)
    print(all_embeddings[0])

    embeddings = np.vstack(all_embeddings)
    dim = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} vectors, dim={dim}")

    # ── 3. Build FAISS index ──
    print("Building FAISS index (IndexFlatIP for cosine similarity) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors")

    # ── 4. Save ──
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ FAISS index saved → {INDEX_PATH}")
    print(f"✓ Metadata saved    → {METADATA_PATH}  ({len(metadata)} entries)")
    return index, metadata


# ── Query ────────────────────────────────────────────────────────────
def query_index(query: str, top_k: int = 5) -> list[dict]:
    """Search the FAISS index and return top-k results with metadata."""
    import faiss
    from sentence_transformers import SentenceTransformer

    if not INDEX_PATH.exists():
        print("Index not found! Run: python scripts/create_db.py  (without --query)")
        sys.exit(1)
    if not METADATA_PATH.exists():
        print("Metadata not found! Run: python scripts/create_db.py  (without --query)")
        sys.exit(1)

    print("Loading embedding model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Embed query
    q_embed = model.encode([query], normalize_embeddings=True).astype("float32")

    # Search
    scores, indices = index.search(q_embed, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata.get(str(idx), {})
        results.append({
            "score": float(score),
            "index": int(idx),
            "subject": meta.get("subject"),
            "class": meta.get("class"),
            "chapter": meta.get("chapter"),
            "page": meta.get("page"),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build / query FAISS vector DB")
    parser.add_argument("--subject", choices=["physics", "chemistry", "biology"])
    parser.add_argument("--class", type=int, dest="class_num", choices=[11, 12])
    parser.add_argument("--query", type=str, help="Search the index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        results = query_index(args.query, top_k=args.top_k)
        print(f"\n🔍 Query: \"{args.query}\"\n")
        print(f"{'Rank':<5} {'Score':<8} {'Subject':<12} {'Class':<7} {'Ch':<5} {'Page':<6}")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"{i:<5} {r['score']:<8.4f} {r['subject'] or '?':<12} "
                  f"{r['class'] or '?':<7} {r['chapter']:<5} {r['page'] or '?':<6}")
    else:
        build_index()


if __name__ == "__main__":
    main() 