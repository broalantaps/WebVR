import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from webvr_eval.capture_video import has_video_for_html, record_html_page


def _process_task(task):
    html_path, args_dict = task

    class Args:
        pass

    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)

    try:
        ok = record_html_page(str(html_path), args)
    except Exception as exc:
        return ("fail", html_path.stem, str(exc))
    if ok:
        return ("success", html_path.stem, "")
    return ("fail", html_path.stem, "record_html_page returned False")


def _iter_pending_html_files(html_dir: Path, skip_existing: bool) -> list[Path]:
    html_files = sorted(html_dir.glob("*.html"), key=lambda path: path.stem)
    if not skip_existing:
        return html_files
    return [path for path in html_files if not has_video_for_html(str(path))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record all HTML files in a directory.")
    parser.add_argument("--html_dir", type=str, required=True, help="Directory containing idx.html outputs")
    parser.add_argument("--workers", type=int, default=4, help="Number of recording workers")
    parser.add_argument("--test_hover", action="store_true", help="Run basic hover interactions during recording")
    parser.add_argument("--show_mouse", action="store_true", help="Overlay a mouse indicator")
    parser.add_argument("--hover_pause_ms", type=int, default=800, help="How long each hover stays active")
    parser.add_argument(
        "--interaction_scroll_step_px",
        type=int,
        default=120,
        help="Scroll delta used while testing hover interactions",
    )
    parser.add_argument(
        "--interaction_scroll_pause_ms",
        type=int,
        default=30,
        help="Pause after interaction-induced scrolls",
    )
    parser.add_argument("--record_format", choices=["webm", "mp4"], default="webm", help="Output video format")
    parser.add_argument("--record_fps", type=int, default=30, help="FPS used when converting to mp4")
    parser.add_argument("--video_crf", type=int, default=18, help="CRF used when converting to mp4")
    parser.add_argument("--viewport_width", type=int, default=1280, help="Browser viewport width")
    parser.add_argument("--viewport_height", type=int, default=720, help="Browser viewport height")
    parser.add_argument("--device_scale_factor", type=float, default=2.0, help="Browser device scale factor")
    parser.add_argument("--scroll_steps", type=int, default=200, help="Number of scroll steps across the page")
    parser.add_argument("--scroll_pause_ms", type=int, default=100, help="Pause between scroll steps")
    parser.add_argument("--start_wait_ms", type=int, default=3000, help="Initial wait after navigation")
    parser.add_argument("--headed", action="store_true", help="Show the browser window while recording")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip idx files with recorded videos")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of html files to process")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.test_hover:
        args.show_mouse = True

    html_dir = Path(args.html_dir)
    if not html_dir.exists():
        print(f"HTML directory does not exist: {html_dir}")
        sys.exit(1)

    pending_files = _iter_pending_html_files(html_dir, args.skip_existing)
    if args.limit > 0:
        pending_files = pending_files[: args.limit]

    if not pending_files:
        print("No HTML files require recording.")
        return

    args_dict = vars(args).copy()
    worker_count = min(args.workers, len(pending_files))
    tasks = [(path, args_dict) for path in pending_files]

    success_count = 0
    failure_count = 0

    with Pool(processes=worker_count) as pool:
        results = pool.imap_unordered(_process_task, tasks)
        iterator = results
        if tqdm is not None:
            iterator = tqdm(results, total=len(tasks), desc="record", unit="html")
        for status, _, _ in iterator:
            if status == "success":
                success_count += 1
            else:
                failure_count += 1

    print(
        f"recording finished: success={success_count} "
        f"failed={failure_count} total={len(tasks)}"
    )


if __name__ == "__main__":
    main()
