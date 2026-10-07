#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# run_evaluation.sh
#
# Runs the full RAG evaluation pipeline:
#   1. Activates the Python virtual environment
#   2. Runs evaluate_rag.py on all 110 test questions
#   3. Saves results to data/test_data/evaluation_results.csv
#
# Usage:
#   chmod +x run_evaluation.sh
#   ./run_evaluation.sh
#   ./run_evaluation.sh --limit 10          # test with first 10 only
#   ./run_evaluation.sh --verbose           # show detailed output
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
INPUT_FILE="$PROJECT_ROOT/data/test_data/test_questions.json"
OUTPUT_FILE="$PROJECT_ROOT/data/test_data/evaluation_results.csv"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Banner ───────────────────────────────────────────────────────────
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         NCERT RAG — Batch Evaluation Pipeline           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Pre-flight checks ────────────────────────────────────────────────
echo -e "${YELLOW}[1/4]${NC} Checking prerequisites..."

if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_PYTHON${NC}"
    echo "   Create it with: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Test questions file not found at $INPUT_FILE${NC}"
    exit 1
fi

QUESTION_COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT_FILE'))))")
echo -e "   ✅ Python venv:     $VENV_PYTHON"
echo -e "   ✅ Test questions:  $QUESTION_COUNT questions"
echo -e "   ✅ Output will go to: $OUTPUT_FILE"

# ── Check .env ────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/4]${NC} Checking environment variables..."
if [ -f "$PROJECT_ROOT/.env" ]; then
    if grep -q "OPENROUTER_API_KEY=sk-" "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo -e "   ✅ OPENROUTER_API_KEY found in .env"
    else
        echo -e "${RED}❌ OPENROUTER_API_KEY not set in .env${NC}"
        echo "   Add your OpenRouter key to .env: OPENROUTER_API_KEY=sk-or-v1-..."
        exit 1
    fi
else
    echo -e "${RED}❌ .env file not found at $PROJECT_ROOT/.env${NC}"
    exit 1
fi

# ── Run evaluation ────────────────────────────────────────────────────
echo -e "${YELLOW}[3/4]${NC} Running evaluation on $QUESTION_COUNT questions..."
echo ""

# Pass through any extra arguments (--limit, --verbose, etc.)
"$VENV_PYTHON" "$PROJECT_ROOT/scripts/evaluate_rag.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    "$@"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/4]${NC} Results summary:"

if [ -f "$OUTPUT_FILE" ]; then
    ROW_COUNT=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
    echo -e "   📄 CSV rows:        $ROW_COUNT"
    echo -e "   📁 Output file:     $OUTPUT_FILE"

    # Compute average times using awk
    echo ""
    echo -e "${CYAN}   Average timings:${NC}"
    awk -F',' 'NR>1 {
        ret+=$8; rerank+=$9; gen+=$10; total+=$11; count++
    }
    END {
        if (count > 0) {
            printf "   retrieval:    %.2fs\n", ret/count
            printf "   rerank:       %.2fs\n", rerank/count
            printf "   generation:   %.2fs\n", gen/count
            printf "   total:        %.2fs\n", total/count
        }
    }' "$OUTPUT_FILE"
else
    echo -e "${RED}❌ Output file was not created${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Evaluation complete!${NC}"