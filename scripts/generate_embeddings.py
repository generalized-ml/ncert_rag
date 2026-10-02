#!/usr/bin/env python3
"""
Generate embeddings from processed chunks and store in vector database.

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --reset
"""

import argparse
import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for RAG")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing vector store before generating")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding generation")
    args = parser.parse_args()

    print("Generating embeddings...")

    # lets read the chunks from the processed JSON files data/chunks
    chunks_dir = Path("data/chunks")
    if not chunks_dir.exists():
        print(f"Chunks directory {chunks_dir} does not exist. Please run process_pdfs.py first.")
        sys.exit(1)

    #creating embeddings directory if it does not exist
    os.makedirs("data/embeddings", exist_ok=True)

    #TODO: Implement embedding generation logic
    # 1. Load chunks from data/chunks/folder_name/*.json
    # 2. Generate embeddings using a model (e.g., OpenAI, SentenceTransformers)
    # 3. Store embeddings in a vector database (e.g., FAISS, Milvus, Pinecone)
    # 4. Save metadata (e.g., class, subject, page number) alongside embeddings for retrieval


    


if __name__ == "__main__":
    main()