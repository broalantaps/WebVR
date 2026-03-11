import argparse
import base64
import concurrent.futures
from datetime import datetime
import json
import os
import re
import sys
import time
import traceback
import cv2
import numpy as np
from google import genai
from google.genai import types
from openai import OpenAI
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from webvr_eval.prompts import REPLICATE_VIDEO_PROMPT, REPLICATE_VIDEO_CLAUDE_PROMPT

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
DEFAULT_GENERATION_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "model_generation_config.json",
)
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def extract_by_fps_fast(video_path: str, target_fps: int = 1):
    video = cv2.VideoCapture(video_path)
    src_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        video.release()
        return []
    frame_interval = max(1, int(src_fps / target_fps))
    frames = []
    for idx in range(0, total_frames, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    return frames


def extract_uniform_frames(video_path: str, num_frames: int = 32):
    """Uniformly sample a fixed number of frames from a video."""
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        video.release()
        return []
    # Uniformly sampled frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    return frames


def encode_image_to_jpeg_bytes(frame) -> bytes:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    return buffer.tobytes()


def resize_frame_if_needed(frame, max_size: int = 1500):
    """Clamp the longest edge of a frame to max_size pixels."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_base64_image_if_needed(data_url: str, max_size: int) -> str:
    """Clamp the longest edge of a base64 data URL image to max_size pixels."""
    if not data_url.startswith("data:") or ";base64," not in data_url:
        return data_url
    _, encoded = data_url.split(";base64,", 1)
    image_bytes = base64.b64decode(encoded)
    # Decode into a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return data_url
    # Check whether resizing is needed
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return data_url
    # Resize
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Re-encode the resized image
    _, buffer = cv2.imencode(".jpg", resized)
    new_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{new_base64}"


def encode_image_to_base64(frame) -> str:
    return base64.b64encode(encode_image_to_jpeg_bytes(frame)).decode("utf-8")


def encode_video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def is_video_file(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


def is_json_file(path: str) -> bool:
    return path.lower().endswith(".json")


def find_image_path_by_index(images_dir: str, index: int) -> str:
    for extension in IMAGE_EXTENSIONS:
        image_path = os.path.join(images_dir, f"{index}{extension}")
        if os.path.isfile(image_path):
            return image_path
    return ""


def build_image_resource_content(sample_id: str, sample_dir: str, image_urls):
    if not sample_dir or not image_urls:
        return []

    images_dir = os.path.join(sample_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"⚠️  样本 {sample_id} 缺少 images 目录，跳过图片资源注入")
        return []

    url_lines = [
        "Input order note: I will provide the video first, then provide images corresponding to the URLs below.",
        "Image URL list (order-aligned with the image inputs below):",
    ]
    image_content = []
    for index, url in enumerate(image_urls, start=1):
        url_lines.append(f"{index}. {url}")
        image_path = find_image_path_by_index(images_dir, index)
        if not image_path:
            print(f"⚠️  样本 {sample_id} 缺少第 {index} 张图片文件，已跳过该 image_url 输入")
            continue
        extension = os.path.splitext(image_path)[1].lower()
        mime_type = IMAGE_MIME_TYPES.get(extension, "image/jpeg")
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            if not image_bytes:
                print(f"⚠️  样本 {sample_id} 第 {index} 张图片文件为空，已跳过")
                continue
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            }
        )

    if not image_content:
        return []

    return [{"type": "text", "text": "\n".join(url_lines)}] + image_content


def build_prompt_preview_text(image_resource_content, has_video: bool = False) -> str:
    preview_parts = [REPLICATE_VIDEO_PROMPT]
    for item in image_resource_content:
        if item.get("type") == "text":
            preview_parts.append(item.get("text", ""))
    modality_markers = []
    if has_video:
        modality_markers.append("[video]")
    if any(item.get("type") == "image_url" for item in image_resource_content):
        modality_markers.append("[image]")
    if modality_markers:
        preview_parts.append("Modalities: " + " ".join(modality_markers))
    return "\n\n".join([part for part in preview_parts if part])


def parse_data_url_to_bytes(data_url: str):
    if not data_url.startswith("data:") or ";base64," not in data_url:
        return None, None
    header, encoded = data_url.split(";base64,", 1)
    mime_type = header.replace("data:", "", 1).strip() or "image/jpeg"
    return mime_type, base64.b64decode(encoded)


def apply_gemini_media_resolution(part, enabled: bool):
    if not enabled:
        return part
    part_media_resolution_cls = getattr(types, "PartMediaResolution", None)
    if part_media_resolution_cls is None:
        return part
    part.media_resolution = part_media_resolution_cls(level="media_resolution_high")
    return part


def build_gemini_image_parts(image_resource_content, use_media_resolution: bool = False):
    parts = []
    for item in image_resource_content or []:
        item_type = item.get("type")
        if item_type == "text":
            parts.append(types.Part.from_text(text=item.get("text", "")))
            continue
        if item_type != "image_url":
            continue
        image_url = (item.get("image_url") or {}).get("url", "")
        mime_type, image_bytes = parse_data_url_to_bytes(image_url)
        if not image_bytes:
            continue
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        part = apply_gemini_media_resolution(part, use_media_resolution)
        parts.append(part)
    return parts


def get_api_model_name(model: str) -> str:
    return model


def get_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 OPENAI_API_KEY")
    return api_key


def get_openai_base_url() -> str | None:
    return os.environ.get("OPENAI_BASE_URL") or None


def get_gemini_base_url() -> str | None:
    return os.environ.get("GEMINI_BASE_URL") or None


def create_model_client(model: str):
    api_key = get_openai_api_key()

    if "gemini" in model:
        http_options = {"api_version": "v1alpha"}
        gemini_base_url = get_gemini_base_url()
        if gemini_base_url:
            http_options["base_url"] = gemini_base_url
        return genai.Client(http_options=http_options, api_key=api_key)

    client_kwargs = {"api_key": api_key}
    openai_base_url = get_openai_base_url()
    if openai_base_url:
        client_kwargs["base_url"] = openai_base_url
    return OpenAI(**client_kwargs)


def load_generation_params(config_path: str, model: str):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    defaults = config.get("default", {})
    model_overrides = config.get("models", {}).get(model, {})
    generation_params = {**defaults, **model_overrides}

    required_keys = ("temperature", "top_p", "max_tokens")
    missing_keys = [key for key in required_keys if key not in generation_params]
    if missing_keys:
        raise ValueError(f"配置文件缺少必要字段: {missing_keys}")

    generation_params["max_tokens"] = int(generation_params["max_tokens"])
    generation_params["temperature"] = float(generation_params["temperature"])
    generation_params["top_p"] = float(generation_params["top_p"])
    if "top_k" in generation_params and generation_params["top_k"] is not None:
        generation_params["top_k"] = int(generation_params["top_k"])
    return generation_params


def extract_web_page_code_block(raw_content: str) -> str:
    match = re.search(r"<web_page_code(?:\s+[^>]*)?>\s*(.*?)\s*</web_page_code>", raw_content, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def post_process_raw_content(raw_content: str) -> str:
    code = extract_web_page_code_block(raw_content)
    if code:
        return code
    return raw_content.strip()


def is_valid_html_output(code: str) -> bool:
    """Check whether the output looks like valid HTML."""
    if not code or len(code) < 10:
        return False
    code_lower = code.lower()
    # The output must include the basic HTML structure
    if "<html" not in code_lower and "<!doctype" not in code_lower:
        return False
    if "<body" not in code_lower and "<head" not in code_lower:
        return False
    return True


def infer_raw_content(
    client,
    model: str,
    frames,
    generation_params: dict,
    image_resource_content=None,
    video_path: str = None,
    video_fps: int = 2,
) -> str:
    image_resource_content = image_resource_content or []
    api_model = get_api_model_name(model)

    if "kimi" in model:
        if not video_path:
            raise ValueError("kimi 系列模型需要提供 video_path")
        suffix = os.path.splitext(video_path)[1].lower()
        mime_type = VIDEO_MIME_TYPES.get(suffix, "video/mp4")
        video_base64 = encode_video_to_base64(video_path)
        user_content = [{"type": "text", "text": REPLICATE_VIDEO_PROMPT}]
        user_content.append({"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{video_base64}"}})
        if image_resource_content:
            user_content.extend(image_resource_content)
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            max_tokens=generation_params["max_tokens"],
            temperature=generation_params["temperature"],
            top_p=generation_params["top_p"],
            stream=True,
        )
        raw_content = ""
        for chunk in response:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    raw_content += delta_content
                    print(delta_content, end="", flush=True)
        return raw_content

    if "qwen" in model.lower() or "seed" in model.lower() or "ernie" in model.lower() or "glm" in model.lower():
        print(f"使用 qwen/seed/ernie/glm 系列模型（原生视频输入）: {model}")
        if not video_path:
            raise ValueError("qwen/seed/ernie/glm 系列模型（原生视频输入）需要提供 video_path")
        suffix = os.path.splitext(video_path)[1].lower()
        mime_type = VIDEO_MIME_TYPES.get(suffix, "video/mp4")
        video_base64 = encode_video_to_base64(video_path)
        user_content = [
            {"type": "text", "text": REPLICATE_VIDEO_PROMPT},
            {
                "type": "video_url",
                "video_url": {"url": f"data:{mime_type};base64,{video_base64}"},
                "fps": int(video_fps),
            },
        ]
        if image_resource_content:
            user_content.extend(image_resource_content)
        response = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=generation_params["max_tokens"],
            temperature=generation_params["temperature"],
            top_p=generation_params["top_p"],
            stream=True,
        )
        raw_content = ""
        for chunk in response:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    raw_content += delta_content
                    print(delta_content, end="", flush=True)
        return raw_content

    if "gemini" in model:
        if not video_path:
            raise ValueError("gemini 系列模型需要提供 video_path")
        video_suffix = os.path.splitext(video_path)[1].lower()
        video_mime_type = VIDEO_MIME_TYPES.get(video_suffix, "video/mp4")
        use_media_resolution = "gemini-3" in model.lower()
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        if len(video_bytes) > 20 * 1024 * 1024:
            print(f"⚠️  视频大小 {len(video_bytes) / 1024 / 1024:.2f}MB，Gemini inline video 可能失败")

        video_part = types.Part(inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type))
        video_part = apply_gemini_media_resolution(video_part, use_media_resolution)
        contents = [
            types.Part.from_text(text=REPLICATE_VIDEO_PROMPT),
            video_part,
            *build_gemini_image_parts(image_resource_content, use_media_resolution=use_media_resolution),
        ]

        config_kwargs = {
            "max_output_tokens": generation_params["max_tokens"],
            "temperature": generation_params["temperature"],
            "top_p": generation_params["top_p"],
        }
        if "top_k" in generation_params:
            config_kwargs["top_k"] = generation_params["top_k"]

        response = client.models.generate_content_stream(
            model=api_model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        raw_content = ""
        for chunk in response:
            delta_content = getattr(chunk, "text", "") or ""
            if delta_content:
                raw_content += delta_content
                print(delta_content, end="", flush=True)
        return raw_content

    if "gpt" in model.lower():
        # GPT models do not support native video input, so use sampled frames.
        print(f"使用 GPT 系列模型（抽帧输入）: {model}")
        user_content = [{"type": "text", "text": REPLICATE_VIDEO_PROMPT}]
        # Add sampled video frames as images
        for frame in frames:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(frame)}"}
            })
        # Add extra image resources
        if image_resource_content:
            user_content.extend(image_resource_content)

        # gpt-5.2 uses reasoning_effort and does not support temperature/top_p.
        if model == "gpt-5.2":
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=generation_params["max_tokens"],
                reasoning_effort="medium",
                # stream=True,
            )
            raw_content = response.choices[0].message.content
            return raw_content
        else:
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=generation_params["max_tokens"],
                temperature=generation_params["temperature"],
                top_p=generation_params["top_p"],
                stream=True,
            )
        raw_content = ""
        for chunk in response:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    raw_content += delta_content
                    print(delta_content, end="", flush=True)
        return raw_content

    if "claude" in model.lower():
        # Claude models do not support native video input, so use sampled frames through the OpenAI-compatible API.
        print(f"使用 Claude 系列模型（抽帧输入）: {model}")
        user_content = [{"type": "text", "text": REPLICATE_VIDEO_CLAUDE_PROMPT}]

        # Claude accepts at most 100 images, so downsample uniformly if needed.
        extra_image_count = sum(1 for item in (image_resource_content or []) if item.get("type") == "image_url")
        max_frames_for_claude = 100 - extra_image_count
        if len(frames) > max_frames_for_claude:
            step = len(frames) / max_frames_for_claude
            frames = [frames[int(i * step)] for i in range(max_frames_for_claude)]
            print(f"  帧数超限，均匀采样到 {len(frames)} 帧")

        # Compute the total image count to decide the resize limit.
        # Claude allows up to 8000px for <=20 images and 2000px otherwise.
        total_image_count = len(frames) + extra_image_count
        max_size = 2000 if total_image_count > 20 else 8000
        print(f"  图片数量: {total_image_count} (帧={len(frames)}, 额外={extra_image_count}), max_size={max_size}")

        # Add sampled video frames as images
        for frame in frames:
            resized_frame = resize_frame_if_needed(frame, max_size=max_size)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(resized_frame)}"}
            })
        # Add extra image resources, resizing them as well
        for item in image_resource_content or []:
            if item.get("type") == "text":
                user_content.append(item)
            elif item.get("type") == "image_url":
                image_url = (item.get("image_url") or {}).get("url", "")
                resized_url = resize_base64_image_if_needed(image_url, max_size=max_size)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": resized_url}
                })

        response = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=generation_params["max_tokens"],
            temperature=generation_params["temperature"],
            # Claude (Bedrock) does not support setting temperature and top_p together.
            stream=True,
        )
        raw_content = ""
        for chunk in response:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    raw_content += delta_content
                    print(delta_content, end="", flush=True)
        return raw_content

    # content = [{"type": "text", "text": "Analyze the video and output the web page code."}]
    # if image_resource_content:
    #     content.extend(image_resource_content)
    # for frame in frames:
    #     content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(frame)}"}})

    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": REPLICATE_VIDEO_PROMPT},
    #         {"role": "user", "content": content},
    #     ],
    #     max_tokens=generation_params["max_tokens"],
    #     temperature=generation_params["temperature"],
    #     top_p=generation_params["top_p"],
    #     stream=True,
    # )

    # raw_content = ""
    # for chunk in response:
    #     if chunk.choices:
    #         delta_content = chunk.choices[0].delta.content
    #         if delta_content:
    #             raw_content += delta_content
    #             print(delta_content, end="", flush=True)
    # return raw_content
    raise ValueError("Invalid model")


def validate_input_path(input_path: str) -> str:
    if not input_path:
        raise ValueError("未提供输入路径")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    if os.path.isdir(input_path):
        raise IsADirectoryError(f"路径是目录，请传入视频文件或 JSON 文件: {input_path}")
    if not (is_video_file(input_path) or is_json_file(input_path)):
        raise ValueError(f"仅支持视频文件或 JSON 文件输入: {input_path}")
    return input_path


def find_video_in_sample_dir(sample_dir: str, sample_id: str) -> str:
    all_files = [os.path.join(sample_dir, name) for name in os.listdir(sample_dir)]
    video_files = [path for path in all_files if os.path.isfile(path) and is_video_file(path)]
    if not video_files:
        return ""

    def score(video_path: str):
        basename = os.path.basename(video_path).lower()
        return (
            0 if "_recorded" in basename else 1,
            0 if basename.startswith(f"{sample_id}_".lower()) else 1,
            basename,
        )

    video_files.sort(key=score)
    return video_files[0]


def collect_input_videos(input_path: str, limit: int = 0):
    if is_video_file(input_path):
        return [{"sample_id": "single", "video_path": input_path, "sample_dir": "", "image_urls": []}]

    with open(input_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, dict):
        raise ValueError("JSON 输入必须是对象，且 key 为子目录 ID")

    dataset_root = os.path.dirname(os.path.abspath(input_path))
    videos = []
    keys = sorted(data.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))
    if limit and limit > 0:
        keys = keys[:limit]
    for sample_id in keys:
        sample_dir = os.path.join(dataset_root, str(sample_id))
        if not os.path.isdir(sample_dir):
            print(f"⚠️  跳过 {sample_id}: 子目录不存在 {sample_dir}")
            continue
        video_path = find_video_in_sample_dir(sample_dir, str(sample_id))
        if not video_path:
            print(f"⚠️  跳过 {sample_id}: 未找到视频文件")
            continue
        sample_meta = data.get(sample_id, {})
        image_urls = sample_meta.get("image_urls", []) if isinstance(sample_meta, dict) else []
        videos.append(
            {
                "sample_id": str(sample_id),
                "video_path": video_path,
                "sample_dir": sample_dir,
                "image_urls": image_urls,
            }
        )

    if not videos:
        raise RuntimeError("JSON 中没有匹配到可用视频")
    return videos


def is_completed_output(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def run(args):
    input_path = validate_input_path(args.video)
    video_items = collect_input_videos(input_path, limit=args.limit)
    is_batch = len(video_items) > 1
    model_lower = args.model.lower()
    use_native_video_input = ("kimi" in model_lower) or ("gemini" in model_lower) or ("qwen" in model_lower) or ("seed" in model_lower) or ("ernie" in model_lower) or ("glm" in model_lower)

    generation_params = load_generation_params(args.config, args.model)
    top_k_display = generation_params.get("top_k", "N/A")
    print(
        "🧩 生成参数:"
        f" temperature={generation_params['temperature']},"
        f" top_p={generation_params['top_p']},"
        f" top_k={top_k_display},"
        f" max_tokens={generation_params['max_tokens']}"
    )

    output_dir = ""
    if is_batch:
        if args.output:
            output_dir = args.output
        else:
            model_name = args.model.replace("/", "_")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), f"formal_exp_outputs_{model_name}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 批量模式输出目录: {output_dir}")

    if output_dir:
        log_root = output_dir
    elif args.output:
        log_root = os.path.dirname(os.path.abspath(args.output)) or os.getcwd()
    else:
        log_root = os.path.dirname(os.path.abspath(input_path))
    logs_dir = os.path.join(log_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_log_path = os.path.join(logs_dir, f"run_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    run_log_payload = {
        "input_path": os.path.abspath(input_path),
        "output_dir": os.path.abspath(output_dir) if output_dir else "",
        "output_target": os.path.abspath(args.output) if args.output else "",
        "model": args.model,
        "generation_params": {
            "temperature": generation_params["temperature"],
            "top_p": generation_params["top_p"],
            "top_k": generation_params.get("top_k"),
            "max_tokens": generation_params["max_tokens"],
        },
        "video_params": {
            "fps": args.fps,
            "video_fps": args.video_fps,
            "max_frames": args.max_frames,
        },
        "sample_count": len(video_items),
        "resume": bool(args.resume),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_log_path, "w", encoding="utf-8") as log_file:
        json.dump(run_log_payload, log_file, ensure_ascii=False, indent=2)
    print(f"📝 运行参数日志已写入: {run_log_path}")

    def process_item(item):
        sample_id = item["sample_id"]
        video_path = item["video_path"]
        sample_dir = item["sample_dir"]
        image_urls = item["image_urls"]
        print(f"\n🎬 处理样本 {sample_id}: {video_path}")
        print(
            "⚙️ 推理参数:"
            f" temperature={generation_params['temperature']},"
            f" top_p={generation_params['top_p']},"
            f" top_k={generation_params.get('top_k', 'N/A')},"
            f" max_tokens={generation_params['max_tokens']},"
            f" video_fps={args.video_fps}"
        )
        image_resource_content = build_image_resource_content(sample_id, sample_dir, image_urls)

        if args.print_prompt_example:
            preview_text = build_prompt_preview_text(image_resource_content, has_video=bool(video_path))
            attached_images = sum(1 for item in image_resource_content if item.get("type") == "image_url")
            print("\n===== Prompt Preview (Text Parts) =====\n")
            print(preview_text)
            print(
                f"\n===== Prompt Preview Meta =====\n"
                f"Attached image_url count: {attached_images}\n"
                f"Has video input: {bool(video_path)}"
            )
            if args.prompt_example_output:
                with open(args.prompt_example_output, "w", encoding="utf-8") as output_file:
                    output_file.write(preview_text)
                print(f"✅ Prompt 文本已写入: {args.prompt_example_output}")
            return {"preview_only": True, "sample_id": sample_id}

        frames = []
        if not use_native_video_input:
            model_lower = args.model.lower()
            if "gpt" in model_lower or "claude" in model_lower:
                # GPT/Claude models: uniformly sample 32 frames plus k extra images.
                frames = extract_uniform_frames(video_path, num_frames=32)
            else:
                # Other models: sample frames according to fps.
                frames = extract_by_fps_fast(video_path, args.fps)
                if len(frames) > args.max_frames:
                    step = len(frames) / args.max_frames
                    frames = [frames[int(i * step)] for i in range(args.max_frames)]
            if not frames:
                print("⚠️  跳过: 无法抽帧")
                return {"sample_id": sample_id, "skipped": True}

        client = create_model_client(args.model)

        # Retry loop
        max_retries = getattr(args, 'max_retries', DEFAULT_MAX_RETRIES)
        retry_delay = getattr(args, 'retry_delay', DEFAULT_RETRY_DELAY)
        code = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                raw_content = infer_raw_content(
                    client,
                    args.model,
                    frames,
                    generation_params,
                    image_resource_content=image_resource_content,
                    video_path=video_path,
                    video_fps=args.video_fps,
                )
                code = post_process_raw_content(raw_content)

                # Check whether the output is valid
                if is_valid_html_output(code):
                    if attempt > 1:
                        print(f"\n✅ 第 {attempt} 次重试成功")
                    break
                else:
                    last_error = f"输出格式无效 (长度={len(code)}, 缺少 HTML 结构)"
                    print(f"\n⚠️  第 {attempt}/{max_retries} 次尝试: {last_error}")
                    if attempt < max_retries:
                        print(f"   {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)

            except Exception as e:
                last_error = str(e)
                print(f"\n⚠️  第 {attempt}/{max_retries} 次尝试失败: {last_error}")
                # Print the full error details
                if hasattr(e, 'response'):
                    print(f"   完整响应: {e.response}")
                if hasattr(e, 'body'):
                    print(f"   响应体: {e.body}")
                print(f"   Traceback:\n{traceback.format_exc()}")
                if attempt < max_retries:
                    print(f"   {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)

        # All retries failed
        if not code or not is_valid_html_output(code):
            error_msg = f"所有 {max_retries} 次尝试均失败: {last_error}"
            print(f"\n❌ {error_msg}")
            raise RuntimeError(error_msg)

        if is_batch:
            output_path = os.path.join(output_dir, f"{sample_id}.html")
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(code)
            print(f"\n✅ 已写入: {output_path}")
            return {"sample_id": sample_id, "output_path": output_path}

        if args.output:
            with open(args.output, "w", encoding="utf-8") as output_file:
                output_file.write(code)
            print(f"\n✅ 已写入: {args.output}")
            return {"sample_id": sample_id, "output_path": args.output}
        else:
            print("\n\n===== Extracted Front-End Code =====\n")
            print(code)
            return {"sample_id": sample_id, "output_path": ""}

    if not is_batch:
        if args.resume and args.output and is_completed_output(args.output):
            print(f"⏭️  已存在结果，跳过单样本: {args.output}")
            return
        process_item(video_items[0])
        return

    pending_items = video_items
    if args.resume:
        pending_items = []
        skipped_count = 0
        for item in video_items:
            sample_output = os.path.join(output_dir, f"{item['sample_id']}.html")
            if is_completed_output(sample_output):
                skipped_count += 1
                continue
            pending_items.append(item)
        print(f"♻️  断点续跑: 已跳过 {skipped_count} 条，待处理 {len(pending_items)} 条")

    if not pending_items:
        print("✅ 全部样本已存在输出，无需继续推理")
        return

    worker_count = max(1, int(args.workers))
    print(f"🚀 并发推理 workers={worker_count}, 样本数={len(pending_items)}")
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_sample = {executor.submit(process_item, item): item["sample_id"] for item in pending_items}
        future_iter = concurrent.futures.as_completed(future_to_sample)
        if tqdm is not None:
            future_iter = tqdm(future_iter, total=len(future_to_sample), desc="formal_exp")
        for future in future_iter:
            sample_id = future_to_sample[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((sample_id, str(exc)))
                print(f"\n❌ 样本 {sample_id} 失败: {exc}")

    if failures:
        print("\n⚠️ 失败样本汇总:")
        for sample_id, message in failures:
            print(f"- {sample_id}: {message}")
def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入视频文件路径，或包含样本索引的 JSON 文件路径")
    parser.add_argument("--model", default="gemini-3-flash-native", help="使用的模型")
    parser.add_argument("--config", default=DEFAULT_GENERATION_CONFIG_PATH, help="模型生成参数配置 JSON 路径")
    parser.add_argument("--fps", type=int, default=1, help="抽帧帧率")
    parser.add_argument("--video_fps", type=int, default=2, help="视频输入采样 fps（Qwen video_url）")
    parser.add_argument("--max_frames", type=int, default=100, help="最大帧数")
    parser.add_argument("--workers", type=int, default=1, help="并发 worker 数（仅批量模式有效）")
    parser.add_argument("--limit", type=int, default=0, help="JSON 模式下仅处理前 N 条（0 表示不限制）")
    parser.add_argument("--print_prompt_example", action="store_true", help="打印最终 prompt 示例并退出（不请求模型）")
    parser.add_argument("--prompt_example_output", help="将 prompt 示例文本写入文件")
    parser.add_argument("--output", help="输出前端代码的文件路径")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="断点续跑，跳过已存在输出文件")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="关闭断点续跑，强制重跑全部样本")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"模型输出格式无效时的最大重试次数（默认 {DEFAULT_MAX_RETRIES}）")
    parser.add_argument("--retry_delay", type=int, default=DEFAULT_RETRY_DELAY, help=f"重试间隔秒数（默认 {DEFAULT_RETRY_DELAY}）")
    return parser


if __name__ == "__main__":
    try:
        args = build_arg_parser().parse_args()
        run(args)
    except Exception as exc:
        print(f"❌ 运行失败: {exc}")
        sys.exit(1)
