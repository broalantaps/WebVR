import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from webvr_eval.prompts import (
    JUDGE_SYSTEM_PROMPT_ONLY_CODE_CHECKS_TEMPLATE,
    JUDGE_SYSTEM_PROMPT_ONLY_CODE_CHECKS_TEMPLATE_V2,
    JUDGE_SYSTEM_PROMPT_ONLY_VIDEO_CHECKS_TEMPLATE_V1,
    JUDGE_SYSTEM_PROMPT_VIDEO_CODE_CHECKS_TEMPLATE_V1,
    JUDGE_SYSTEM_PROMPT_VIDEO_CODE_CHECKS_TEMPLATE_V2,
)

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DEFAULT_MODEL = "gemini-3-pro-native"
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}

_thread_local = threading.local()
_request_lock = threading.Lock()
_last_request_time = 0.0
_write_lock = threading.Lock()

CATEGORY_NAMES = [
    "Global_Aesthetics",
    "Navigation_and_Footer",
    "Section_Specific_Layouts",
    "Interaction_and_Motion",
]


def _extract_idx_from_filename(path: Path) -> str:
    return path.stem.strip()


def _load_rubric_map(rubric_jsonl: Path) -> dict[str, Any]:
    if not rubric_jsonl.exists():
        raise FileNotFoundError(f"Rubric JSONL does not exist: {rubric_jsonl}")

    rubric_map: dict[str, Any] = {}
    invalid_lines = 0
    missing_rubric = 0

    with rubric_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                invalid_lines += 1
                continue

            # Support either idx or line_index
            idx = str(row.get("idx") or row.get("line_index", "")).strip()
            rubric = row.get("rubric")
            if not idx:
                invalid_lines += 1
                continue
            if not rubric:
                missing_rubric += 1
                continue
            rubric_map[idx] = rubric

    print(
        f"[rubric] loaded={len(rubric_map)} invalid_lines={invalid_lines} "
        f"missing_rubric={missing_rubric}"
    )
    return rubric_map


def _create_model_client(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing environment variable OPENAI_API_KEY")

    if "gemini" in model.lower():
        if genai is None or types is None:
            raise RuntimeError("google-genai is not installed in the current environment; Gemini requests are unavailable")
        http_options = {"api_version": "v1alpha"}
        gemini_base_url = os.environ.get("GEMINI_BASE_URL")
        if gemini_base_url:
            http_options["base_url"] = gemini_base_url
        return genai.Client(http_options=http_options, api_key=api_key)
    if OpenAI is None:
        raise RuntimeError("openai is not installed in the current environment; non-Gemini requests are unavailable")
    client_kwargs = {"api_key": api_key}
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    if openai_base_url:
        client_kwargs["base_url"] = openai_base_url
    return OpenAI(**client_kwargs)


def _get_thread_client(model: str):
    cached_model = getattr(_thread_local, "model", None)
    cached_client = getattr(_thread_local, "client", None)
    if cached_client is not None and cached_model == model:
        return cached_client
    client = _create_model_client(model)
    _thread_local.client = client
    _thread_local.model = model
    return client


def _rate_limit(delay: float):
    global _last_request_time
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _last_request_time = time.time()


def _map_model_name(model: str) -> str:
    return model


def _infer_text(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    delay: float,
    video_path: Path | None = None,
) -> str:
    _rate_limit(delay)
    # Map model name to actual API model name
    api_model = _map_model_name(model)

    if "gemini" in model.lower():
        contents = []
        if video_path and video_path.exists():
            video_bytes = video_path.read_bytes()
            suffix = video_path.suffix.lower()
            mime_type = VIDEO_MIME_TYPES.get(suffix, "video/webm")
            # video_size_mb = len(video_bytes) / (1024 * 1024)
            # print(f"    📹 Gemini video input: {video_path.name} ({video_size_mb:.2f} MB, {mime_type})")
            contents.append(types.Part.from_bytes(data=video_bytes, mime_type=mime_type))
        contents.append(types.Part.from_text(text=user_prompt))
        stream = client.models.generate_content_stream(
            model=api_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature
            ),
        )
        raw = ""
        for chunk in stream:
            raw += getattr(chunk, "text", "") or ""
        return raw

    if ("kimi" in model.lower() or "qwen" in model.lower()) and video_path and video_path.exists():
        print(f"Use kimi/qwen model for Judger....")
        suffix = video_path.suffix.lower()
        mime_type = VIDEO_MIME_TYPES.get(suffix, "video/webm")
        video_bytes = video_path.read_bytes()
        # video_size_mb = len(video_bytes) / (1024 * 1024)
        # print(f"    📹 Kimi video input: {video_path.name} ({video_size_mb:.2f} MB, {mime_type})")
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{video_base64}"}},
        ]
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        raw = ""
        for chunk in response:
            # Correctly handle the OpenAI streaming response format
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                raw += chunk.choices[0].delta.content
        return raw

    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not response.choices:
        return ""
    message = response.choices[0].message
    if not message:
        return ""

    content = message.content
    if isinstance(content, str):
        content = content.strip()
    elif isinstance(content, list):
        text_chunks = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_chunks.append(str(item.get("text", "")))
        content = "".join(text_chunks).strip()
    else:
        content = str(content or "").strip()

    if content:
        return content

    finish_reason = getattr(response.choices[0], "finish_reason", "")
    reasoning_content = getattr(message, "reasoning_content", "") or ""
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    raise RuntimeError(f"empty model content (finish_reason={finish_reason})")


def _strip_code_fence(text: str) -> str:
    content = (text or "").strip()
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return content


def _parse_json_payload(text: str) -> Any:
    content = _strip_code_fence(text)
    if not content:
        raise ValueError("The evaluation result is empty")

    try:
        return json.loads(content)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(content):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(content[idx:])
            return obj
        except Exception:
            continue

    raise ValueError("Failed to parse JSON from the model output")


def _extract_evaluation_content(text: str) -> str:
    """Extract the content inside <evaluation>...</evaluation> tags."""
    import re
    match = re.search(r"<evaluation>\s*(.*?)\s*</evaluation>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _build_system_prompt(rubric: Any, use_v2: bool = False, use_video: bool = False, video_only: bool = False) -> str:
    if video_only:
        return JUDGE_SYSTEM_PROMPT_ONLY_VIDEO_CHECKS_TEMPLATE_V1
    if use_video:
        return JUDGE_SYSTEM_PROMPT_VIDEO_CODE_CHECKS_TEMPLATE_V2
    if use_v2:
        return JUDGE_SYSTEM_PROMPT_ONLY_CODE_CHECKS_TEMPLATE_V2
    return JUDGE_SYSTEM_PROMPT_ONLY_CODE_CHECKS_TEMPLATE.replace(
        "{rubric_json}",
        json.dumps(rubric, ensure_ascii=False, indent=2),
    )


def _build_user_prompt(html_code: str, rubric: Any = None, use_v2: bool = False, use_video: bool = False, video_only: bool = False) -> str:
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2) if rubric else ""
    if video_only:
        return f"### Rubric JSON:\n{rubric_json}"
    if use_video:
        return f"### Rubric JSON:\n{rubric_json}\n\n### HTML Code:\n{html_code}"
    if use_v2 and rubric is not None:
        return f"### Rubric JSON:\n{rubric_json}\n\n### HTML Code:\n{html_code}"
    return html_code


def _count_rubric_criteria(rubric: Any) -> int:
    if isinstance(rubric, str):
        return 1
    if isinstance(rubric, list):
        return sum(_count_rubric_criteria(item) for item in rubric)
    if isinstance(rubric, dict):
        return sum(_count_rubric_criteria(value) for value in rubric.values())
    return 0


def _flatten_rubric_criteria(rubric: Any, prefix: str = "c") -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    def walk(node: Any, path: str):
        if isinstance(node, str):
            items.append({"id": path, "criterion": node})
            return
        if isinstance(node, list):
            for index, child in enumerate(node, start=1):
                walk(child, f"{path}.{index}")
            return
        if isinstance(node, dict):
            child_index = 0
            for _, child in node.items():
                child_index += 1
                walk(child, f"{path}.{child_index}")

    walk(rubric, prefix)
    return items


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _count_rubric_by_category(rubric: Any) -> dict[str, int]:
    counts = {name: 0 for name in CATEGORY_NAMES}
    if not isinstance(rubric, dict):
        return counts

    counts["Global_Aesthetics"] = _count_rubric_criteria(rubric.get("Global_Aesthetics"))
    counts["Navigation_and_Footer"] = _count_rubric_criteria(rubric.get("Navigation_and_Footer"))
    counts["Section_Specific_Layouts"] = _count_rubric_criteria(rubric.get("Section_Specific_Layouts"))
    counts["Interaction_and_Motion"] = _count_rubric_criteria(rubric.get("Interaction_and_Motion"))
    return counts


def _infer_category_from_check_id(check_id: str) -> str | None:
    normalized = (check_id or "").strip().lower().replace("-", "_")
    if not normalized:
        return None

    collapsed = normalized.replace("_", "")

    if normalized.startswith(("global", "ga_")):
        return "Global_Aesthetics"
    if normalized.startswith(("nav", "naf_", "header", "footer", "nf_")):
        return "Navigation_and_Footer"
    if normalized.startswith(("section", "layout", "hero", "catalog", "lab", "ssl_", "sls_")):
        return "Section_Specific_Layouts"
    if normalized.startswith(("interaction", "motion", "iam_", "hover", "scroll", "im_")):
        return "Interaction_and_Motion"

    if any(token in normalized for token in ("global", "aesthetic")):
        return "Global_Aesthetics"
    if any(token in normalized for token in ("navigation", "header", "footer", "menu")):
        return "Navigation_and_Footer"
    if (
        "ssl" in normalized
        or "sls" in normalized
        or collapsed.startswith("ssl")
        or collapsed.startswith("sls")
        or any(token in normalized for token in ("section", "layout", "hero", "catalog", "lab"))
    ):
        return "Section_Specific_Layouts"
    if any(token in normalized for token in ("interaction", "motion", "anim", "hover", "scroll", "transition")):
        return "Interaction_and_Motion"
    return None


def _extract_checks_from_video_output(parsed: Any, prefix: str = "") -> list[dict]:
    """Recursively extract all checks from categorized video-mode output."""
    checks = []
    if isinstance(parsed, list):
        for index, item in enumerate(parsed, start=1):
            item_prefix = f"{prefix}.{index}" if prefix else f"c.{index}"
            if isinstance(item, dict) and "criterion" in item:
                met_raw = item.get("met", False)
                if isinstance(met_raw, bool):
                    met = met_raw
                else:
                    met = str(met_raw).strip().lower() in {"1", "true", "yes", "met"}
                checks.append({
                    "id": item_prefix,
                    "criterion": str(item.get("criterion", "")).strip(),
                    "met": met,
                    "evidence": str(item.get("reason", "")).strip(),
                })
            else:
                checks.extend(_extract_checks_from_video_output(item, item_prefix))
    elif isinstance(parsed, dict):
        for key, value in parsed.items():
            key_prefix = f"{prefix}.{key}" if prefix else key
            checks.extend(_extract_checks_from_video_output(value, key_prefix))
    return checks


def _normalize_result(parsed: Any, rubric: Any) -> dict[str, Any]:
    expected_items = _flatten_rubric_criteria(rubric)
    checks = parsed.get("checks") if isinstance(parsed, dict) else None
    normalized_checks = []

    if isinstance(checks, list):
        for index, item in enumerate(checks, start=1):
            if not isinstance(item, dict):
                continue
            check_id = str(item.get("id", f"c.{index}")).strip() or f"c.{index}"
            criterion = str(item.get("criterion", "")).strip()
            met_raw = item.get("met", False)
            if isinstance(met_raw, bool):
                met = met_raw
            else:
                met = str(met_raw).strip().lower() in {"1", "true", "yes", "met"}
            evidence = str(item.get("evidence", "")).strip()
            normalized_checks.append(
                {
                    "id": check_id,
                    "criterion": criterion,
                    "met": met,
                    "evidence": evidence,
                }
            )

    # Video mode: extract checks from the categorized structure
    if not normalized_checks and isinstance(parsed, dict):
        video_checks = _extract_checks_from_video_output(parsed)
        if video_checks:
            normalized_checks = video_checks

    if not normalized_checks and expected_items:
        normalized_checks = [
            {
                "id": item["id"],
                "criterion": item["criterion"],
                "met": False,
                "evidence": "missing check from model output",
            }
            for item in expected_items
        ]

    total = len(normalized_checks)
    satisfied = sum(1 for item in normalized_checks if item["met"])
    if total <= 0:
        total = _count_rubric_criteria(rubric)
    score = (satisfied / total) if total > 0 else 0.0
    reason = str(parsed.get("reason", "")).strip() if isinstance(parsed, dict) else "invalid model output format"
    return {
        "checks": normalized_checks,
        "satisfied_count": satisfied,
        "total_count": total,
        "score": score,
        "reason": reason,
    }


def _load_processed_idx(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    done = set()
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            idx = str(row.get("idx", "")).strip()
            if idx:
                done.add(idx)
    return done


def _append_jsonl(output_jsonl: Path, row: dict[str, Any]):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with output_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


_RETRYABLE_HTTP_STATUS = {
    408,  # Request Timeout
    409,  # Conflict (transient in some proxies)
    424,  # Failed Dependency
    425,  # Too Early
    429,  # Too Many Requests
    500,
    502,
    503,
    504,
    520,
    522,
    524,
}


def _extract_http_status(error: Exception) -> int | None:
    message = str(error)
    for token in ("status code:", "status:", "Error code:", "HTTP"):
        if token not in message:
            continue
        # Find first 3-digit number after token.
        tail = message.split(token, 1)[-1]
        for chunk in tail.replace("/", " ").replace(":", " ").split():
            if chunk.isdigit() and len(chunk) == 3:
                return int(chunk)
    return None


def _evaluate_single_html(
    html_path: Path,
    rubric: Any,
    args,
):
    idx = _extract_idx_from_filename(html_path)
    client = _get_thread_client(args.model)
    use_v2 = getattr(args, "prompt_v2", False)
    use_video = getattr(args, "use_video", False)
    video_only = getattr(args, "video_only", False)

    video_path = None
    if use_video or video_only:
        video_dir = Path(args.video_dir) if getattr(args, "video_dir", None) else html_path.parent
        candidate = video_dir / f"{idx}_recorded.webm"
        if candidate.exists():
            video_path = candidate

    # video_only mode requires an input video
    if video_only and video_path is None:
        video_dir = Path(args.video_dir) if getattr(args, "video_dir", None) else html_path.parent
        expected_path = video_dir / f"{idx}_recorded.webm"
        return {
            "idx": idx,
            "html_path": str(html_path),
            "video_used": False,
            "status": "error",
            "reason": f"video_only mode but video not found: {expected_path}",
            "timestamp": int(time.time()),
        }

    html_code = "" if video_only else html_path.read_text(encoding="utf-8")
    system_prompt = _build_system_prompt(rubric, use_v2=use_v2, use_video=(video_path is not None and not video_only), video_only=video_only)
    user_prompt = _build_user_prompt(html_code, rubric=rubric, use_v2=use_v2, use_video=(video_path is not None and not video_only), video_only=video_only)

    zero_score_max_retries = 3
    final_result = None

    for zero_retry in range(1, zero_score_max_retries + 1):
        last_error = None
        parsed = None
        raw_response = ""

        for attempt in range(1, args.max_retries + 1):
            try:
                raw_response = _infer_text(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    delay=args.request_delay,
                    video_path=video_path,
                )
                print(raw_response)
                # In video mode, extract the <evaluation> payload first
                if video_path is not None:
                    raw_response = _extract_evaluation_content(raw_response)
                parsed = _parse_json_payload(raw_response)
                break
            except Exception as exc:
                last_error = str(exc)
                status = _extract_http_status(exc)
                is_retryable = status in _RETRYABLE_HTTP_STATUS if status is not None else True
                if not is_retryable:
                    break
                if attempt < args.max_retries:
                    if status is not None:
                        print(f"[{idx}] transient error HTTP {status}, retry {attempt}/{args.max_retries}")
                    time.sleep(args.retry_delay)

        if parsed is None:
            final_result = {
                "idx": idx,
                "html_path": str(html_path),
                "video_used": video_path is not None,
                "status": "error",
                "reason": f"inference failed: {last_error}",
                "raw_response": raw_response[-2000:],
                "timestamp": int(time.time()),
            }
            break

        normalized = _normalize_result(parsed, rubric)
        final_result = {
            "idx": idx,
            "html_path": str(html_path),
            "video_used": video_path is not None,
            "status": "ok",
            "model": args.model,
            "checks": normalized["checks"],
            "satisfied_count": normalized["satisfied_count"],
            "total_count": normalized["total_count"],
            "score": normalized["score"],
            "reason": normalized.get("reason", ""),
            "timestamp": int(time.time()),
        }

        if normalized["score"] > 0:
            break

        if zero_retry < zero_score_max_retries:
            print(f"[{idx}] score=0, retry {zero_retry}/{zero_score_max_retries}")
            time.sleep(args.retry_delay)

    return final_result


def _generate_summary_from_existing(output_jsonl: Path, rubric_jsonl: Path, html_dir: Path, args):
    """Generate a summary from an existing output_jsonl file."""
    if not output_jsonl.exists():
        raise FileNotFoundError(f"Output file does not exist: {output_jsonl}")

    rubric_map = _load_rubric_map(rubric_jsonl)

    ok_count = 0
    skip_count = 0
    fail_count = 0
    collected_scores = []

    category_met = {name: 0 for name in CATEGORY_NAMES}
    category_gt_total = {name: 0 for name in CATEGORY_NAMES}
    category_recognized_total = {name: 0 for name in CATEGORY_NAMES}
    category_macro_sum = {name: 0.0 for name in CATEGORY_NAMES}
    total_macro_samples = len(rubric_map) if rubric_map else 0
    unrecognized_check_count = 0
    unrecognized_check_case_count = 0
    missing_checks_case_count = 0
    missing_checks_total = 0
    missing_rubric_case_count = 0

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no} JSON parse error: {e}")
                continue
            status = row.get("status")
            if status == "ok":
                ok_count += 1
                collected_scores.append(float(row.get("score", 0.0)))

                idx = str(row.get("idx", "")).strip()
                rubric = rubric_map.get(idx)
                if rubric is None:
                    missing_rubric_case_count += 1
                    continue

                gt_counts = _count_rubric_by_category(rubric)
                for cat_name in CATEGORY_NAMES:
                    category_gt_total[cat_name] += gt_counts[cat_name]

                checks = row.get("checks", [])
                if not isinstance(checks, list):
                    checks = []

                case_unrecognized = False
                case_recognized_count = 0
                case_gt_total = sum(gt_counts.values())
                case_cat_met = {name: 0 for name in CATEGORY_NAMES}
                for check in checks:
                    if not isinstance(check, dict):
                        unrecognized_check_count += 1
                        case_unrecognized = True
                        continue
                    category = _infer_category_from_check_id(str(check.get("id", "")))
                    if category is None:
                        unrecognized_check_count += 1
                        case_unrecognized = True
                        continue
                    category_recognized_total[category] += 1
                    case_recognized_count += 1
                    if bool(check.get("met", False)):
                        category_met[category] += 1
                        case_cat_met[category] += 1

                if case_unrecognized:
                    unrecognized_check_case_count += 1

                missing_for_case = max(case_gt_total - case_recognized_count, 0)
                if missing_for_case > 0:
                    missing_checks_case_count += 1
                    missing_checks_total += missing_for_case

                for cat_name in CATEGORY_NAMES:
                    gt_total = gt_counts[cat_name]
                    if gt_total > 0:
                        category_macro_sum[cat_name] += (case_cat_met[cat_name] / gt_total)
                    else:
                        category_macro_sum[cat_name] += 0.0
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1

    avg_denominator = len(rubric_map) if rubric_map else len(collected_scores)
    avg_score_raw = (
        sum(collected_scores) / avg_denominator if avg_denominator > 0 else 0.0
    )
    avg_score = round(avg_score_raw * 100, 2)

    category_accuracy = {}
    for cat_name in CATEGORY_NAMES:
        met = category_met[cat_name]
        gt_total = category_gt_total[cat_name]
        recognized_total = category_recognized_total[cat_name]
        macro_count = total_macro_samples
        accuracy_raw = (category_macro_sum[cat_name] / macro_count) if macro_count > 0 else 0.0
        accuracy = round(accuracy_raw * 100, 2)
        category_accuracy[cat_name] = {
            "met_count": met,
            "ground_truth_total": gt_total,
            "recognized_check_total": recognized_total,
            "accuracy": accuracy,
            "macro_sample_count": macro_count,
        }

    summary = {
        "html_dir": str(html_dir),
        "rubric_jsonl": str(rubric_jsonl),
        "output_jsonl": str(output_jsonl),
        "model": args.model,
        "use_video": getattr(args, "use_video", False),
        "video_only": getattr(args, "video_only", False),
        "prompt_v2": getattr(args, "prompt_v2", False),
        "total_files": ok_count + skip_count + fail_count,
        "ok_count": ok_count,
        "skip_count": skip_count,
        "fail_count": fail_count,
        "avg_score": avg_score,
        "avg_score_denominator": avg_denominator,
        "avg_score_denominator_type": "rubric_count" if rubric_map else "ok_count",
        "category_accuracy_denominator": "sample_count",
        "category_accuracy_by_check_id": category_accuracy,
        "missing_and_unrecognized": {
            "missing_rubric_case_count": missing_rubric_case_count,
            "unrecognized_check_case_count": unrecognized_check_case_count,
            "unrecognized_check_count": unrecognized_check_count,
            "missing_checks_case_count": missing_checks_case_count,
            "missing_checks_total": missing_checks_total,
        },
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary_only] {json.dumps(summary, ensure_ascii=False)}")
    print(f"[summary_only] summary saved: {summary_path}")


def run(args):
    html_dir = Path(args.html_dir).resolve()
    rubric_jsonl = Path(args.rubric_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()

    # summary_only mode: generate the summary directly from existing results
    if getattr(args, "summary_only", False):
        _generate_summary_from_existing(output_jsonl, rubric_jsonl, html_dir, args)
        return

    if not html_dir.exists() or not html_dir.is_dir():
        raise NotADirectoryError(f"html_dir is not a valid directory: {html_dir}")

    rubric_map = _load_rubric_map(rubric_jsonl)
    html_paths = sorted(html_dir.glob("*.html"), key=lambda p: p.name)
    if args.limit > 0:
        html_paths = html_paths[: args.limit]

    # Preview prompt mode
    if getattr(args, "preview_prompt", False):
        if not html_paths:
            print("[preview] No HTML files were found")
            return
        html_path = html_paths[0]
        idx = _extract_idx_from_filename(html_path)
        rubric = rubric_map.get(idx)
        if not rubric:
            print(f"[preview] No rubric found for idx={idx}; trying the next one...")
            for hp in html_paths[1:]:
                idx = _extract_idx_from_filename(hp)
                rubric = rubric_map.get(idx)
                if rubric:
                    html_path = hp
                    break
        if not rubric:
            print("[preview] No rubric was found for any sample")
            return

        use_v2 = getattr(args, "prompt_v2", False)
        use_video = getattr(args, "use_video", False)
        video_only = getattr(args, "video_only", False)

        # Check whether the video file exists
        video_path = None
        if use_video or video_only:
            video_dir = Path(args.video_dir) if getattr(args, "video_dir", None) else html_path.parent
            candidate = video_dir / f"{idx}_recorded.webm"
            if candidate.exists():
                video_path = candidate

        html_code = "" if video_only else html_path.read_text(encoding="utf-8")
        system_prompt = _build_system_prompt(rubric, use_v2=use_v2, use_video=(video_path is not None and not video_only), video_only=video_only)
        user_prompt = _build_user_prompt(html_code, rubric=rubric, use_v2=use_v2, use_video=(video_path is not None and not video_only), video_only=video_only)

        # Determine the prompt mode
        if video_only:
            prompt_mode = "VIDEO_ONLY" if video_path else "VIDEO_ONLY (NO VIDEO FOUND!)"
        elif video_path:
            prompt_mode = "VIDEO+CODE"
        else:
            prompt_mode = "CODE_ONLY"

        print(f"[preview] idx={idx} use_v2={use_v2} use_video={use_video} video_only={video_only}")
        print(f"[preview] html_path={html_path}")
        if use_video or video_only:
            if video_path:
                video_size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"[preview] video_path={video_path} ({video_size_mb:.2f} MB) ✅ video found")
            else:
                video_dir = Path(args.video_dir) if getattr(args, "video_dir", None) else html_path.parent
                expected_path = video_dir / f"{idx}_recorded.webm"
                print(f"[preview] video_path=NOT FOUND ❌")
                print(f"[preview] expected path: {expected_path}")
                if video_only:
                    print(f"[preview] ⚠️  video_only mode requires a video")
                else:
                    print(f"[preview] ⚠️  Video not found; the prompt will fall back to text-only mode")
        print(f"[preview] prompt_mode={prompt_mode}")
        print("\n" + "=" * 60)
        print("SYSTEM PROMPT:")
        print("=" * 60)
        print(system_prompt)
        print("\n" + "=" * 60)
        print(f"USER PROMPT (length={len(user_prompt)}):")
        print("=" * 60)
        # Show only the first 2000 characters
        if len(user_prompt) > 2000:
            print(user_prompt[:2000])
            print(f"\n... [truncated, total {len(user_prompt)} chars]")
        else:
            print(user_prompt)
        return

    processed = _load_processed_idx(output_jsonl) if args.resume else set()
    total_files = len(html_paths)
    ok_count = 0
    skip_count = 0
    fail_count = 0
    collected_scores: list[float] = []

    print(
        f"[start] html_files={total_files} model={args.model} "
        f"output={output_jsonl}"
    )

    pending_tasks: list[Path] = []
    task_rubrics: dict[str, Any] = {}
    html_idx_set: set[str] = set()  # Track all idx values present in the HTML directory

    for rank, html_path in enumerate(html_paths, start=1):
        idx = _extract_idx_from_filename(html_path)
        html_idx_set.add(idx)  # Track every idx found in the HTML files
        if args.resume and idx in processed:
            print(f"[{rank}/{total_files}] skip idx={idx} (already processed)")
            skip_count += 1
            continue

        rubric = rubric_map.get(idx)
        if not rubric:
            row = {
                "idx": idx,
                "html_path": str(html_path),
                "status": "skip",
                "reason": "rubric not found for idx",
                "timestamp": int(time.time()),
            }
            _append_jsonl(output_jsonl, row)
            print(f"[{rank}/{total_files}] skip idx={idx} (missing rubric)")
            skip_count += 1
            continue

        pending_tasks.append(html_path)
        task_rubrics[idx] = rubric

    pending_total = len(pending_tasks)
    if pending_total > 0:
        max_workers = max(1, min(args.workers, pending_total))
        print(f"[run] pending={pending_total} workers={max_workers}")
        future_to_idx = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            idx_to_path = {}
            for html_path in pending_tasks:
                idx = _extract_idx_from_filename(html_path)
                idx_to_path[idx] = str(html_path)
                future = executor.submit(
                    _evaluate_single_html,
                    html_path,
                    task_rubrics[idx],
                    args,
                )
                future_to_idx[future] = idx

            completed = 0
            for future in as_completed(future_to_idx):
                completed += 1
                idx = future_to_idx[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "idx": idx,
                        "html_path": idx_to_path.get(idx, ""),
                        "status": "error",
                        "reason": f"worker exception: {exc}",
                        "timestamp": int(time.time()),
                    }

                _append_jsonl(output_jsonl, row)

                if row.get("status") == "ok":
                    score = float(row.get("score", 0.0))
                    satisfied = int(row.get("satisfied_count", 0))
                    total = int(row.get("total_count", 0))
                    video_flag = "+video" if row.get("video_used") else ""
                    collected_scores.append(score)
                    ok_count += 1
                    print(
                        f"[{completed}/{pending_total}] ok idx={idx} "
                        f"score={score:.4f} ({satisfied}/{total}) {video_flag}"
                    )
                elif row.get("status") == "skip":
                    skip_count += 1
                    print(f"[{completed}/{pending_total}] skip idx={idx}")
                else:
                    fail_count += 1
                    print(f"[{completed}/{pending_total}] error idx={idx}")

    # Handle idx values that exist in all_dataset but not in the HTML directory, assigning them a score of 0
    all_dataset_path = getattr(args, "all_dataset_path", None)
    if all_dataset_path and Path(all_dataset_path).exists():
        with open(all_dataset_path, "r", encoding="utf-8") as f:
            all_dataset = json.load(f)
        all_dataset_idx_set = set(str(k) for k in all_dataset.keys())

        missing_html_count = 0
        for dataset_idx in sorted(all_dataset_idx_set, key=lambda x: (len(x), x)):
            # Skip idx values that are already present in the HTML directory
            if dataset_idx in html_idx_set:
                continue
            # This idx exists in all_dataset but not in the HTML directory
            rubric = rubric_map.get(dataset_idx)
            total_criteria = _count_rubric_criteria(rubric) if rubric else 0
            row = {
                "idx": dataset_idx,
                "html_path": str(html_dir / f"{dataset_idx}.html"),
                "video_used": False,
                "status": "ok",
                "model": args.model,
                "checks": [],
                "satisfied_count": 0,
                "total_count": total_criteria,
                "score": 0.0,
                "reason": "HTML file not found, score set to 0",
                "timestamp": int(time.time()),
            }
            _append_jsonl(output_jsonl, row)
            missing_html_count += 1
            print(f"[missing_html] idx={dataset_idx} score=0 (0/{total_criteria})")

        if missing_html_count > 0:
            print(f"[info] missing_html_count={missing_html_count} (from all_dataset)")

    # Recalculate stats from jsonl file to handle --resume correctly
    ok_count = 0
    skip_count = 0
    fail_count = 0
    collected_scores = []

    category_met = {name: 0 for name in CATEGORY_NAMES}
    category_gt_total = {name: 0 for name in CATEGORY_NAMES}
    category_recognized_total = {name: 0 for name in CATEGORY_NAMES}
    category_macro_sum = {name: 0.0 for name in CATEGORY_NAMES}
    total_macro_samples = len(rubric_map) if rubric_map else 0
    unrecognized_check_count = 0
    unrecognized_check_case_count = 0
    missing_checks_case_count = 0
    missing_checks_total = 0
    missing_rubric_case_count = 0

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no} JSON parse error: {e}")
                continue
            status = row.get("status")
            if status == "ok":
                ok_count += 1
                collected_scores.append(float(row.get("score", 0.0)))

                idx = str(row.get("idx", "")).strip()
                rubric = rubric_map.get(idx)
                if rubric is None:
                    missing_rubric_case_count += 1
                    continue

                gt_counts = _count_rubric_by_category(rubric)
                for cat_name in CATEGORY_NAMES:
                    category_gt_total[cat_name] += gt_counts[cat_name]

                checks = row.get("checks", [])
                if not isinstance(checks, list):
                    checks = []

                case_unrecognized = False
                case_recognized_count = 0
                case_gt_total = sum(gt_counts.values())
                case_cat_met = {name: 0 for name in CATEGORY_NAMES}
                for check in checks:
                    if not isinstance(check, dict):
                        unrecognized_check_count += 1
                        case_unrecognized = True
                        continue
                    category = _infer_category_from_check_id(str(check.get("id", "")))
                    if category is None:
                        unrecognized_check_count += 1
                        case_unrecognized = True
                        continue
                    category_recognized_total[category] += 1
                    case_recognized_count += 1
                    if bool(check.get("met", False)):
                        category_met[category] += 1
                        case_cat_met[category] += 1

                if case_unrecognized:
                    unrecognized_check_case_count += 1

                missing_for_case = max(case_gt_total - case_recognized_count, 0)
                if missing_for_case > 0:
                    missing_checks_case_count += 1
                    missing_checks_total += missing_for_case

                for cat_name in CATEGORY_NAMES:
                    gt_total = gt_counts[cat_name]
                    if gt_total > 0:
                        category_macro_sum[cat_name] += (case_cat_met[cat_name] / gt_total)
                    else:
                        category_macro_sum[cat_name] += 0.0
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
    # Use rubric count as denominator to penalize missing HTML entries (treated as 0 score).
    avg_denominator = len(rubric_map) if rubric_map else len(collected_scores)
    avg_score_raw = (
        sum(collected_scores) / avg_denominator if avg_denominator > 0 else 0.0
    )
    avg_score = round(avg_score_raw * 100, 2)  # Convert to percentage with 2 decimal places

    category_accuracy = {}
    for cat_name in CATEGORY_NAMES:
        met = category_met[cat_name]
        gt_total = category_gt_total[cat_name]
        recognized_total = category_recognized_total[cat_name]
        macro_count = total_macro_samples
        accuracy_raw = (category_macro_sum[cat_name] / macro_count) if macro_count > 0 else 0.0
        accuracy = round(accuracy_raw * 100, 2)  # Convert to percentage with 2 decimal places
        category_accuracy[cat_name] = {
            "met_count": met,
            "ground_truth_total": gt_total,
            "recognized_check_total": recognized_total,
            "accuracy": accuracy,
            "macro_sample_count": macro_count,
        }

    summary = {
        "html_dir": str(html_dir),
        "rubric_jsonl": str(rubric_jsonl),
        "output_jsonl": str(output_jsonl),
        "model": args.model,
        "use_video": getattr(args, "use_video", False),
        "video_only": getattr(args, "video_only", False),
        "prompt_v2": getattr(args, "prompt_v2", False),
        "total_files": total_files,
        "ok_count": ok_count,
        "skip_count": skip_count,
        "fail_count": fail_count,
        "avg_score": avg_score,
        "avg_score_denominator": avg_denominator,
        "avg_score_denominator_type": "rubric_count" if rubric_map else "ok_count",
        "category_accuracy_denominator": "sample_count",
        "category_accuracy_by_check_id": category_accuracy,
        "missing_and_unrecognized": {
            "missing_rubric_case_count": missing_rubric_case_count,
            "unrecognized_check_case_count": unrecognized_check_case_count,
            "unrecognized_check_count": unrecognized_check_count,
            "missing_checks_case_count": missing_checks_case_count,
            "missing_checks_total": missing_checks_total,
        },
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {json.dumps(summary, ensure_ascii=False)}")
    print(f"[done] summary saved: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Traverse the HTML directory and output checks for each file based on the rubric.",
    )
    parser.add_argument(
        "--html_dir",
        required=True,
        help="Directory containing the HTML files to evaluate (filenames should be idx.html)",
    )
    parser.add_argument(
        "--rubric_jsonl",
        required=True,
        help="Path to the JSONL file containing idx and rubric fields",
    )
    parser.add_argument(
        "--output_jsonl",
        default="judger_results.jsonl",
        help="Output JSONL path for evaluation results",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Evaluation model name, for example gemini-3-pro-native / kimi-k2.5-thinking",
    )
    parser.add_argument("--max_tokens", type=int, default=64800)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--request_delay", type=float, default=0.2, help="Delay between requests in seconds")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N HTML files; 0 means all")
    parser.add_argument("--resume", action="store_true", help="Skip idx values that are already present in output_jsonl")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--use_video", action="store_true", help="Also provide the idx_recorded.webm video to the judger")
    parser.add_argument("--video_dir", help="Directory containing video files (defaults to html_dir)")
    parser.add_argument("--prompt_v2", action="store_true", help="Use the V2 prompt template, placing the rubric in user_prompt")
    parser.add_argument("--video_only", action="store_true", help="Judge using only video+rubric, without HTML code")
    parser.add_argument("--preview_prompt", action="store_true", help="Preview the first sample prompt and exit without running evaluation")
    parser.add_argument(
        "--all_dataset_path",
        default="",
        help="Path to the full dataset JSON file, used to compare idx values and assign missing idx values a score of 0",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Generate the summary only from the existing output_jsonl file without running evaluation",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    run(parsed_args)
