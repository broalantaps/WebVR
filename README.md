# WebVR

WebVR is a lightweight benchmark workspace for webpage recreation from videos. This repository contains self-contained inference and judging entrypoints that can be run from one place.

## Repository layout

```text
WebVR/
├── configs/
│   └── model_generation_config.json
├── scripts/
│   ├── capture_video.sh
│   ├── inference.sh
│   └── judge_html_video.sh
└── webvr_eval/
    ├── capture_video.py
    ├── __init__.py
    ├── inference.py
    ├── judger.py
    ├── prompts.py
    └── record_html_dir.py
```

## What was moved here

- `webvr_eval/inference.py`
  Generates webpage code from video inputs or an index JSON file.
- `webvr_eval/capture_video.py`
  Records a local HTML page into `idx_recorded.webm` or `idx_recorded.mp4`.
- `webvr_eval/record_html_dir.py`
  Batch entrypoint for recording every `idx.html` in an output directory.
- `webvr_eval/judger.py`
  Scores generated HTML against rubric JSONL files, optionally using recorded webpage videos.
- `scripts/capture_video.sh`
  Thin wrapper for batch webpage recording.
- `scripts/inference.sh`
  Thin wrapper for batch inference.
- `scripts/judge_html_video.sh`
  Thin wrapper for HTML+video judging.
- `configs/model_generation_config.json`
  Default generation parameters used by `inference.py`.
- `webvr_eval/prompts.py`
  Prompt templates shared by inference and judging.

## Environment

Python 3.10+ is recommended.

Install the Python dependencies used by the two entrypoints:

```bash
pip install openai google-genai httpx opencv-python numpy tqdm playwright
playwright install chromium
```

Set the API key before running either pipeline:

```bash
export OPENAI_API_KEY=...
```

Optional environment variables:

```bash
export OPENAI_BASE_URL=...
export GEMINI_BASE_URL=...
```

## Inference

`webvr_eval.inference` supports:

- a single video file via `--video /path/to/sample.webm`
- a JSON index file via `--video /path/to/input_all_image_urls.json`

The batch JSON mode is the main workflow. The JSON should map each sample id to its metadata. The current implementation expects each sample to resolve to a source video and optionally aligned image assets.

Run directly:

```bash
cd /path/to/WebVR

python3 -m webvr_eval.inference \
  --video /path/to/input_all_image_urls.json \
  --model gemini-3-flash-native \
  --config ./configs/model_generation_config.json \
  --workers 8 \
  --output ./outputs/inference/gemini-3-flash-native
```

Or use the wrapper:

```bash
cd /path/to/WebVR

VIDEO_INDEX_JSON=/path/to/input_all_image_urls.json \
MODEL=gemini-3-flash-native \
WORKERS=8 \
OUTPUT_DIR=./outputs/inference/gemini-3-flash-native \
bash scripts/inference.sh
```

Useful flags:

- `--print_prompt_example`
  Preview the final prompt without calling the model.
- `--prompt_example_output`
  Save that prompt preview to a file.
- `--resume` / `--no_resume`
  Control whether existing outputs are skipped.

## Recording HTML to webpage videos

Recording is a separate step after inference. The recorder expects a directory of HTML files named like `idx.html` and writes videos next to them as `idx_recorded.webm` by default.

Run directly:

```bash
cd /path/to/WebVR

python3 -m webvr_eval.record_html_dir \
  --html_dir ./outputs/inference/gemini-3-flash-native \
  --workers 8 \
  --test_hover \
  --record_format webm \
  --skip_existing
```

Or use the wrapper:

```bash
cd /path/to/WebVR

HTML_DIR=./outputs/inference/gemini-3-flash-native \
WORKERS=8 \
bash scripts/capture_video.sh
```

Useful flags:

- `--record_format webm|mp4`
  Keep the original WebM recording or transcode to MP4.
- `--skip_existing`
  Skip samples that already have `idx_recorded.webm` or `idx_recorded.mp4`.
- `--test_hover`
  Run a lightweight hover pass across common interactive elements while recording.
- `--headed`
  Show the browser window while recording.
- `--limit`
  Record only the first N HTML files in the directory.

For single-file debugging:

```bash
cd /path/to/WebVR

python3 -m webvr_eval.capture_video \
  --html_path ./outputs/inference/gemini-3-flash-native/101.html \
  --test_hover \
  --record_format webm
```

## Judging HTML + Video

`webvr_eval.judger` reads an HTML directory, matches files by `idx.html`, loads a rubric JSONL, and optionally attaches `idx_recorded.webm` from the same directory or a separate `--video_dir`.

Run directly:

```bash
cd /path/to/WebVR

python3 -m webvr_eval.judger \
  --html_dir /path/to/html_outputs \
  --rubric_jsonl /path/to/rubric.jsonl \
  --output_jsonl ./outputs/judger/kimi-k2.5/results.jsonl \
  --model kimi-k2.5 \
  --workers 8 \
  --use_video \
  --max_tokens 32768 \
  --resume
```

Or use the wrapper:

```bash
cd /path/to/WebVR

HTML_DIR=/path/to/html_outputs \
RUBRIC_JSONL=/path/to/rubric.jsonl \
JUDGER_MODEL=kimi-k2.5 \
WORKERS=8 \
MAX_TOKENS=32768 \
OUTPUT_JSONL=./outputs/judger/kimi-k2.5/results.jsonl \
bash scripts/judge_html_video.sh
```

Useful flags:

- `--use_video`
  Send the recorded webpage video together with HTML.
- `--video_only`
  Evaluate only from video instead of HTML+video.
- `--prompt_v2`
  Use the V2 judging prompt.
- `--summary_only`
  Recompute the summary JSON from an existing output JSONL.
## Input and output conventions

Inference outputs one HTML file per sample under the directory given by `--output`.

Judger expects:

- HTML files named `idx.html`
- optional recorded videos named `idx_recorded.webm`
- rubric JSONL rows containing `idx` or `line_index`, plus a `rubric` field

Judger outputs:

- `<output>.jsonl` with per-sample evaluation rows
- `<output>.summary.json` with aggregate metrics

## Notes

- This repo packages the evaluation code only.
- Datasets and generated outputs can stay outside the repo and be referenced by absolute or relative paths.
