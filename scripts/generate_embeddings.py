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
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
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
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #TODO: Implement embedding generation logic
    # 1. Load chunks from data/chunks/folder_name/*.json
    # 2. Generate embeddings using a model (e.g., OpenAI, SentenceTransformers)
    # 3. Store embeddings in a vector database (e.g., FAISS, Milvus, Pinecone)
    # 4. Save metadata (e.g., class, subject, chapter, page number) alongside embeddings for retrieval
    for chunk_file in chunks_dir.glob("**/*.json"):
        print(f"Processing {chunk_file.stem}...")
        # Load the chunk data
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)


        # # Extract text from chunks
        texts = chunk_data["text"]

        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            batch_embeddings = model.encode(batch_texts)
            embeddings.append(batch_embeddings)

        print(f"Generated {len(embeddings)} embeddings for {chunk_file.stem}")

        #take mean of embeddings to get a single embedding for the chunk embeddings is a list of numpy arrays of shape (384,) 
        # for each chunk. We will take mean of all the embeddings to get a single embedding of shape (384,) for the chunk
        embeddings = np.mean(embeddings, axis=0)
        # Save embeddings to a file
        embedding_file = Path("data/embeddings") / f"{chunk_file.stem}_embeddings.npy"
        np.save(embedding_file, embeddings)
        print(f"Saved embeddings to {embedding_file}")

if __name__ == "__main__":
    main()