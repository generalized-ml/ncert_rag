#!/usr/bin/env python3
"""
Batch evaluation with Opik tracking + RAGAS metrics (LLM-as-a-judge).

Pipeline:
  1. Load test questions from JSON.
  2. Run the RAG pipeline on each question (retrieve -> rerank -> generate).
  3. Score each answer with RAGAS:
     - AnswerRelevancy
     - Faithfulness
     - ContextPrecision
  4. Track everything in Opik (localhost) and save results to CSV.

Usage:
    # Start Opik locally first (in another terminal):
    #   opik server

    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --limit 10
    python scripts/evaluate_rag.py --experiment "neet-eval-v1"
    python scripts/evaluate_rag.py --skip-opik   # faster, no Opik tracking
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from opik import track
from opik.evaluation.metrics import (
    AnswerRelevance,
    ContextPrecision,
    Hallucination,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from scripts.inference import answer_query


# =========================================================================
# CONFIG
# =========================================================================

DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_data" / "test_questions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test_data" / "evaluation_results.csv"

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Ensure litellm/Opik LLM judges find the OpenAI-compatible key and base url
if OPENROUTER_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
if OPENROUTER_BASE_URL and not os.getenv("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = OPENROUTER_BASE_URL


# =========================================================================
# RAGAS METRICS FACTORY
# =========================================================================

def get_metrics():
    """
    Create Opik built-in metric scorers (no RAGAS dependency).

    Returns a list of 3 Opik metric instances:
      - AnswerRelevance   : how relevant is the answer to the question?
      - Hallucination     : is the answer faithful to the retrieved contexts?
      - ContextPrecision  : are retrieved contexts relevant to the question?
    """
    return [
        AnswerRelevance(),
        Hallucination(),
        ContextPrecision(),
    ]


# =========================================================================
# EVALUATION TASK (for Opik)
# =========================================================================

def evaluation_task(dataset_item: dict) -> dict:
    """Opik evaluation task: take a dataset item, run RAG pipeline."""
    question = dataset_item["question"]
    result = answer_query(question, verbose=False)

    expected_output = dataset_item.get("expected_output") or dataset_item.get("expected_answer") or result["answer"]

    return {
        "input": question,
        "output": result["answer"],
        "context": result.get("contexts", []),
        "expected_output": expected_output,
        "reference": expected_output,
    }


# =========================================================================
# ASYNC RAGAS SCORING (for CSV-only mode)
# =========================================================================

async def _score_single(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """Score one Q&A pair with AnswerRelevance, Hallucination, ContextPrecision."""
    from opik.evaluation.metrics import (
        AnswerRelevance,
        ContextPrecision,
        Hallucination,
    )

    scores: dict[str, float] = {}

    # 1. AnswerRelevance (user_input, response, context)
    try:
        scorer = AnswerRelevance()
        r = await scorer.ascore(user_input=question, response=answer, context=contexts)
        scores["answer_relevancy"] = round(float(r.value), 4)
    except Exception as e:
        print(f"    answer_relevancy failed: {e}")
        scores["answer_relevancy"] = -1.0

    # 2. Hallucination (user_input, response, context)
    try:
        scorer = Hallucination()
        r = await scorer.ascore(user_input=question, response=answer, context=contexts)
        scores["faithfulness"] = round(float(r.value), 4)
    except Exception as e:
        print(f"    faithfulness failed: {e}")
        scores["faithfulness"] = -1.0

    # 3. ContextPrecision (user_input, response, context, expected_output)
    try:
        scorer = ContextPrecision()
        r = await scorer.ascore(
            user_input=question,
            response=answer,
            context=contexts,
            expected_output=answer,
        )
        scores["context_precision"] = round(float(r.value), 4)
    except Exception as e:
        print(f"    context_precision failed: {e}")
        scores["context_precision"] = -1.0

    return scores


async def _score_batch(results: list[dict]) -> list[dict[str, float]]:
    """Score a batch sequentially (avoid rate limits)."""
    all_scores: list[dict[str, float]] = []
    for i, r in enumerate(results):
        print(f"  [{i + 1}/{len(results)}] scoring ...", end=" ", flush=True)
        s = await _score_single(r["question"], r["answer"], r.get("contexts", []))
        print(
            f"rel={s['answer_relevancy']:.3f} faith={s['faithfulness']:.3f} "
            f"prec={s['context_precision']:.3f}"
        )
        all_scores.append(s)
    return all_scores


# =========================================================================
# MAIN
# =========================================================================
@track
def main():
    parser = argparse.ArgumentParser(
        description="RAG evaluation with Opik tracking + RAGAS metrics"
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
        "--experiment", type=str, default=None,
        help="Experiment name for Opik (default: auto-generated)",
    )
    parser.add_argument(
        "--skip-opik", action="store_true",
        help="Skip Opik tracking (faster, just generate answers to CSV with RAGAS)",
    )
    args = parser.parse_args()

    # -- Load questions --
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Test questions file not found: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        questions = json.load(f)

    if args.limit > 0:
        questions = questions[:args.limit]

    total = len(questions)
    print(f"Loaded {total} questions")

    # -- Prepare output CSV --
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "question", "subject", "class", "chapter", "type",
        "answer", "retrieval_time_s", "rerank_time_s",
        "generation_time_s", "total_time_s", "timestamp",
        "answer_relevancy", "faithfulness", "context_precision",
    ]

    # -- Run evaluation --
    if args.skip_opik:
        print("Opik skipped -- generating answers with RAGAS scoring to CSV")
        _run_csv_with_ragas(questions, output_path, fieldnames)
    else:
        project_name = os.getenv("OPIK_PROJECT_NAME", "NEET-RAG")
        print(f"Opik evaluation -- project: {project_name}, experiment: {args.experiment or 'auto'}\n")

        try:
            from opik import Opik
            from opik.evaluation import evaluate

            client = Opik(project_name=project_name)
            dataset = client.get_or_create_dataset(name="ncert-neet-test-questions")

            dataset_items = [
                {
                    "question": q["question"],
                    "subject": q.get("subject", ""),
                    "class": q.get("class", ""),
                    "chapter": q.get("chapter", ""),
                    "type": q.get("type", ""),
                    "question_id": str(q["id"]),
                }
                for q in questions
            ]
            dataset.insert(dataset_items)
            print(f"   Dataset 'ncert-neet-test-questions' with {len(dataset_items)} items")

            eval_results = evaluate(
                dataset=dataset,
                task=evaluation_task,
                scoring_metrics=get_metrics(),
                experiment_name=args.experiment,
                project_name=project_name,
                verbose=1,
                task_threads=1,
            )

            print("\nOpik evaluation complete!")
            print("   View results in Opik Cloud dashboard.")

            _save_opik_results_to_csv(eval_results, questions, output_path, fieldnames)

        except Exception as e:
            print(f"\nOpik evaluation failed: {e}")
            print("   Falling back to CSV-only mode with RAGAS ...\n")
            _run_csv_with_ragas(questions, output_path, fieldnames)


def _run_csv_with_ragas(questions, output_path, fieldnames):
    """Run RAG pipeline + RAGAS scoring, save to CSV."""
    overall_start = time.perf_counter()

    # Step 1: Generate all answers
    all_results: list[dict] = []
    print("\nGenerating answers...")
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]

        print(f"\n[{i}/{len(questions)}] Q{qid}: {question_text[:80]}...")

        try:
            result = answer_query(question_text, verbose=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "answer": f"ERROR: {e}",
                "contexts": [],
                "retrieval_time_s": 0,
                "rerank_time_s": 0,
                "generation_time_s": 0,
                "total_time_s": 0,
            }

        row = {
            "id": qid,
            "question": question_text,
            "subject": q.get("subject", ""),
            "class": q.get("class", ""),
            "chapter": q.get("chapter", ""),
            "type": q.get("type", ""),
            "answer": result["answer"],
            "retrieval_time_s": result["retrieval_time_s"],
            "rerank_time_s": result["rerank_time_s"],
            "generation_time_s": result["generation_time_s"],
            "total_time_s": result["total_time_s"],
            "timestamp": datetime.now().isoformat(),
            "answer_relevancy": -1.0,
            "faithfulness": -1.0,
            "context_precision": -1.0,
        }
        all_results.append({**row, "contexts": result.get("contexts", [])})

        print(f"  total={result['total_time_s']}s "
              f"(retrieval={result['retrieval_time_s']}s, "
              f"rerank={result['rerank_time_s']}s, "
              f"gen={result['generation_time_s']}s)")

    # Step 2: RAGAS scoring
    print("\nComputing RAGAS metrics (LLM-as-a-judge) ...")
    print("   Metrics: AnswerRelevancy | Faithfulness | ContextPrecision\n")

    scores_list = asyncio.run(_score_batch(all_results))

    for row, scores in zip(all_results, scores_list):
        row["answer_relevancy"] = scores.get("answer_relevancy", -1.0)
        row["faithfulness"] = scores.get("faithfulness", -1.0)
        row["context_precision"] = scores.get("context_precision", -1.0)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})

    elapsed = time.perf_counter() - overall_start
    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"   Questions processed: {len(questions)}")
    print(f"   Total time:          {elapsed:.1f}s")
    print(f"   Avg time/question:   {elapsed / len(questions):.2f}s")

    # Print RAGAS averages
    valid = [r for r in all_results if r.get("answer_relevancy", -1) >= 0]
    if valid:
        avg_rel = np.mean([r["answer_relevancy"] for r in valid if r.get("answer_relevancy", -1) >= 0])
        avg_faith = np.mean([r["faithfulness"] for r in valid if r.get("faithfulness", -1) >= 0])
        avg_prec = np.mean([r["context_precision"] for r in valid if r.get("context_precision", -1) >= 0])
        print(f"\n   Average RAGAS scores ({len(valid)} valid / {len(all_results)} total):")
        print(f"   Avg AnswerRelevancy:  {avg_rel:.4f}")
        print(f"   Avg Faithfulness:     {avg_faith:.4f}")
        print(f"   Avg ContextPrecision: {avg_prec:.4f}")

    print(f"\n   Results saved to:    {output_path}")


def _save_opik_results_to_csv(eval_results, questions, output_path, fieldnames):
    """Save Opik evaluation results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, q in enumerate(questions):
            test_result = eval_results.test_results[i] if (hasattr(eval_results, "test_results") and i < len(eval_results.test_results)) else None

            # Opik TestResult has output in test_result.output, or test_result.evaluation_task_output
            answer = ""
            if test_result:
                if hasattr(test_result, "output"):
                    answer = test_result.output
                elif hasattr(test_result, "evaluation_task_output"):
                    out = test_result.evaluation_task_output
                    answer = out.get("output", "") if isinstance(out, dict) else str(out)

            row = {
                "id": q["id"],
                "question": q["question"],
                "subject": q.get("subject", ""),
                "class": q.get("class", ""),
                "chapter": q.get("chapter", ""),
                "type": q.get("type", ""),
                "answer": answer,
                "retrieval_time_s": 0,
                "rerank_time_s": 0,
                "generation_time_s": 0,
                "total_time_s": 0,
                "timestamp": datetime.now().isoformat(),
                "answer_relevancy": _get_score(test_result, "answer_relevance"),
                "faithfulness": _get_score(test_result, "hallucination"),
                "context_precision": _get_score(test_result, "context_precision"),
            }
            writer.writerow(row)

    print(f"   CSV saved -> {output_path}")


def _get_score(test_result, metric_name):
    """Extract a metric score from an Opik test result."""
    if test_result is None:
        return -1.0
    try:
        for score_result in test_result.score_results:
            if score_result.name.lower() == metric_name.lower():
                return round(float(score_result.value), 4)
    except Exception:
        pass
    return -1.0


if __name__ == "__main__":
    main()
