#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HTML_DIR="${HTML_DIR:-}"
WORKERS="${WORKERS:-8}"
RECORD_FORMAT="${RECORD_FORMAT:-webm}"
EXTRA_ARGS=("$@")

if [[ -z "$HTML_DIR" ]]; then
  echo "HTML_DIR is required"
  echo "Example:"
  echo "  HTML_DIR=/path/to/html_outputs bash scripts/capture_video.sh"
  exit 1
fi

cd "$ROOT_DIR"
python3 -m webvr_eval.record_html_dir \
  --html_dir "$HTML_DIR" \
  --workers "$WORKERS" \
  --record_format "$RECORD_FORMAT" \
  --test_hover \
  --skip_existing \
  "${EXTRA_ARGS[@]}"
