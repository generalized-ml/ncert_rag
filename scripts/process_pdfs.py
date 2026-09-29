#!/usr/bin/env python3
"""
Process all NCERT PDFs: extract text, clean, and chunk.

Usage:
    python scripts/process_pdfs.py
    python scripts/process_pdfs.py --subject physics --class 11
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Process NCERT PDFs for RAG")
    parser.add_argument("--subject", type=str, choices=["physics", "chemistry", "biology"],
                        help="Filter by subject")
    parser.add_argument("--class", type=int, dest="class_num", choices=[11, 12],
                        help="Filter by class")
    parser.add_argument("--dry-run", action="store_true",
                        help="List PDFs without processing")
    args = parser.parse_args()

    pdf_dir = Path("dataset/pdfs")
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    # Collect PDF folders
    folders = sorted(pdf_dir.glob("*dd"))
    print(f"Found {len(folders)} PDF folder(s):")
    for folder in folders:
        print(f"  - {folder.name}")

    if args.dry_run:
        return

    print("\nProcessing PDFs...")
    # TODO: Implement PDF processing pipeline
    # 1. Extract text from PDFs
    # 2. Clean and preprocess text
    # 3. Chunk text into segments
    # 4. Save processed chunks to data/chunks/


if __name__ == "__main__":
    main()