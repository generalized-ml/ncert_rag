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

import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Process NCERT PDFs for RAG")
    parser.add_argument("--subject", type=str, choices=["physics", "chemistry", "biology"],
                        help="Filter by subject")
    parser.add_argument("--class", type=int, dest="class_num", choices=[11, 12],
                        help="Filter by class")
    parser.add_argument("--dry-run", action="store_true",
                        help="List PDFs without processing", default = False)
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

    #read config file and get the subjects and classes to process
    config_path = Path("configs/processing_config.json")

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {"subjects": ["physics", "chemistry", "biology"], "classes": [11, 12]}


    # print("\nProcessing PDFs...", config["folders"])

    for folder in folders:
        #join the folder name with the pdf_dir to get the full path
        pdf_path =folder

        #check if the folder name is in the config file
        if folder.name not in config["folders"]:
            print(f"Skipping folder {folder.name} as it is not in the config file")
            continue

        # Process PDFs in the folder
        #list all the pdf files in the folder
        print(pdf_path)
        pdf_files = sorted(pdf_path.glob("*.pdf"))
        print(f"\nProcessing {len(pdf_files)} PDF(s) in folder: {folder.name}")
        for pdf_file in pdf_files:
            print(f"  - Processing PDF: {pdf_file.name}")
            




if __name__ == "__main__":
    main()