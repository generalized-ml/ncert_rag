#!/usr/bin/env python3
"""
Batch evaluation: run the RAG pipeline on all test questions and save results to CSV.

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --input data/test_data/test_questions.json
    python scripts/evaluate_rag.py --output data/test_data/results.csv
    python scripts/evaluate_rag.py --limit 10
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Add project root ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference import answer_query


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_data" / "test_questions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test_data" / "evaluation_results.csv"


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch RAG evaluation: run all test questions → CSV"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to test questions JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N questions (0 = all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each question and answer as they are processed",
    )
    args = parser.parse_args()

    # ── Load questions ──
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Test questions file not found: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)
    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"📋 Loaded {len(questions)} / {total} questions (limit={args.limit})")
    else:
        print(f"📋 Loaded {len(questions)} questions")

    # ── Prepare output ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "question", "subject", "class", "chapter", "type",
        "answer", "retrieval_time_s", "rerank_time_s",
        "generation_time_s", "total_time_s", "timestamp",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        overall_start = time.perf_counter()

        for i, q in enumerate(questions, 1):
            qid = q["id"]
            question_text = q["question"]
            subject = q.get("subject", "")
            qclass = q.get("class", "")
            chapter = q.get("chapter", "")
            qtype = q.get("type", "")

            print(f"\n[{i}/{len(questions)}] Q{qid}: {question_text[:80]}...")

            try:
                result = answer_query(question_text, verbose=False)
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                result = {
                    "answer": f"ERROR: {e}",
                    "retrieval_time_s": 0,
                    "rerank_time_s": 0,
                    "generation_time_s": 0,
                    "total_time_s": 0,
                }

            row = {
                "id": qid,
                "question": question_text,
                "subject": subject,
                "class": qclass,
                "chapter": chapter,
                "type": qtype,
                "answer": result["answer"],
                "retrieval_time_s": result["retrieval_time_s"],
                "rerank_time_s": result["rerank_time_s"],
                "generation_time_s": result["generation_time_s"],
                "total_time_s": result["total_time_s"],
                "timestamp": datetime.now().isoformat(),
            }
            writer.writerow(row)

            if args.verbose:
                print(f"  ⏱️  total={result['total_time_s']}s | "
                      f"retrieval={result['retrieval_time_s']}s | "
                      f"rerank={result['rerank_time_s']}s | "
                      f"generation={result['generation_time_s']}s")
                print(f"  📝 {result['answer'][:200]}...")
            else:
                print(f"  ✅ total={result['total_time_s']}s "
                      f"(retrieval={result['retrieval_time_s']}s, "
                      f"rerank={result['rerank_time_s']}s, "
                      f"gen={result['generation_time_s']}s)")

        overall_elapsed = time.perf_counter() - overall_start

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"   Questions processed: {len(questions)}")
    print(f"   Total time:          {overall_elapsed:.1f}s")
    print(f"   Avg time/question:   {overall_elapsed / len(questions):.2f}s")
    print(f"   Results saved to:    {output_path}")


if __name__ == "__main__":
    main()