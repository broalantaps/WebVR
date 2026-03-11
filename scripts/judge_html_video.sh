#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HTML_DIR="${HTML_DIR:-}"
RUBRIC_JSONL="${RUBRIC_JSONL:-}"
JUDGER_MODEL="${JUDGER_MODEL:-kimi-k2.5}"
WORKERS="${WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$ROOT_DIR/outputs/judger/${JUDGER_MODEL}/results.jsonl}"
EXTRA_ARGS=("$@")

if [[ -z "$HTML_DIR" ]]; then
  echo "HTML_DIR is required"
  echo "Example:"
  echo "  HTML_DIR=/path/to/html_outputs RUBRIC_JSONL=/path/to/rubric.jsonl bash scripts/judge_html_video.sh"
  exit 1
fi

if [[ -z "$RUBRIC_JSONL" ]]; then
  echo "RUBRIC_JSONL is required"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_JSONL")"

cd "$ROOT_DIR"
python3 -m webvr_eval.judger \
  --html_dir "$HTML_DIR" \
  --rubric_jsonl "$RUBRIC_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --model "$JUDGER_MODEL" \
  --workers "$WORKERS" \
  --use_video \
  --max_tokens "$MAX_TOKENS" \
  --resume \
  "${EXTRA_ARGS[@]}"
