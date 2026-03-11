#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEO_INDEX_JSON="${VIDEO_INDEX_JSON:-}"
MODEL="${MODEL:-gemini-3-flash-native}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/model_generation_config.json}"
WORKERS="${WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/inference/$MODEL}"
EXTRA_ARGS=("$@")

if [[ -z "$VIDEO_INDEX_JSON" ]]; then
  echo "VIDEO_INDEX_JSON is required"
  echo "Example:"
  echo "  VIDEO_INDEX_JSON=/path/to/input_all_image_urls.json bash scripts/inference.sh"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
python3 -m webvr_eval.inference \
  --video "$VIDEO_INDEX_JSON" \
  --model "$MODEL" \
  --config "$CONFIG_PATH" \
  --workers "$WORKERS" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
