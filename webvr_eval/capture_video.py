import argparse
import os
import shutil
import subprocess
from pathlib import Path


def add_mouse_indicator(page):
    page.evaluate(
        """
        () => {
            if (document.getElementById('webvr-mouse-indicator')) {
                return;
            }

            const indicator = document.createElement('div');
            indicator.id = 'webvr-mouse-indicator';
            indicator.style.cssText = `
                position: fixed;
                width: 18px;
                height: 18px;
                border: 2px solid #ff3b30;
                border-radius: 999px;
                pointer-events: none;
                z-index: 2147483647;
                transform: translate(-50%, -50%);
                box-shadow: 0 0 0 2px rgba(255, 59, 48, 0.2);
            `;
            document.body.appendChild(indicator);

            document.addEventListener('mousemove', (event) => {
                indicator.style.left = `${event.clientX}px`;
                indicator.style.top = `${event.clientY}px`;
            }, { passive: true });
        }
        """
    )


def auto_hover_elements(
    page,
    hover_pause_ms: int,
    interaction_scroll_step_px: int,
    interaction_scroll_pause_ms: int,
):
    page.evaluate(
        """
        () => {
            let index = 0;
            const selector = 'a, button, [role="button"], .btn, .card, img, summary';
            for (const node of document.querySelectorAll(selector)) {
                const rect = node.getBoundingClientRect();
                const style = window.getComputedStyle(node);
                if (rect.width < 8 || rect.height < 8) continue;
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                if (!node.id) {
                    node.setAttribute('data-webvr-hover-id', `webvr-hover-${index++}`);
                } else {
                    node.setAttribute('data-webvr-hover-id', node.id);
                }
            }
        }
        """
    )

    hover_ids = page.evaluate(
        """
        () => Array.from(document.querySelectorAll('[data-webvr-hover-id]'))
            .map((node) => node.getAttribute('data-webvr-hover-id'))
        """
    )

    for hover_id in hover_ids:
        locator = page.locator(f'[data-webvr-hover-id="{hover_id}"]').first
        if locator.count() == 0:
            continue
        try:
            locator.scroll_into_view_if_needed(timeout=2000)
            box = locator.bounding_box()
            if not box:
                continue

            target_y = max(box["y"] - 120, 0)
            page.evaluate("(y) => window.scrollTo(0, y)", target_y)
            page.wait_for_timeout(interaction_scroll_pause_ms)

            center_x = box["x"] + box["width"] / 2
            center_y = box["y"] + min(box["height"] / 2, 24)
            page.mouse.move(center_x, center_y)
            page.wait_for_timeout(hover_pause_ms)

            if interaction_scroll_step_px > 0:
                page.mouse.wheel(0, interaction_scroll_step_px)
                page.wait_for_timeout(interaction_scroll_pause_ms)
                page.mouse.wheel(0, -interaction_scroll_step_px)
                page.wait_for_timeout(interaction_scroll_pause_ms)
        except Exception:
            continue


def _convert_webm_to_mp4(src_path: str, dst_path: str, fps: int, crf: int) -> bool:
    if not shutil.which("ffmpeg"):
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-r",
        str(fps),
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return True


def has_video_for_html(html_path: str, video_extensions: tuple[str, ...] = (".webm", ".mp4")) -> bool:
    html_dir = os.path.dirname(html_path)
    html_name = os.path.splitext(os.path.basename(html_path))[0]
    for file_name in os.listdir(html_dir):
        if not file_name.startswith(f"{html_name}_recorded"):
            continue
        if file_name.lower().endswith(video_extensions):
            return True
    return False


def record_html_page(html_path: str, args) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "playwright is required for recording. Install it with "
            "'pip install playwright' and then run 'playwright install chromium'."
        ) from exc

    html_path_obj = Path(html_path).resolve()
    if not html_path_obj.exists():
        raise FileNotFoundError(f"HTML file does not exist: {html_path_obj}")

    output_dir = html_path_obj.parent
    html_name = html_path_obj.stem

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            args=["--font-render-hinting=none"],
        )
        context = browser.new_context(
            viewport={"width": args.viewport_width, "height": args.viewport_height},
            device_scale_factor=args.device_scale_factor,
            record_video_dir=str(output_dir),
            record_video_size={"width": args.viewport_width, "height": args.viewport_height},
        )
        context.set_default_timeout(90000)
        context.set_default_navigation_timeout(90000)

        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.dismiss())

        page_loaded = False
        for wait_until, timeout in (
            ("networkidle", 60000),
            ("domcontentloaded", 30000),
            ("commit", 30000),
        ):
            try:
                page.goto(html_path_obj.as_uri(), wait_until=wait_until, timeout=timeout)
                page_loaded = True
                break
            except Exception:
                continue

        if not page_loaded:
            page.close()
            context.close()
            browser.close()
            return False

        page.wait_for_timeout(args.start_wait_ms)
        page.wait_for_timeout(1000)

        if args.show_mouse:
            add_mouse_indicator(page)

        max_scroll = page.evaluate("() => Math.max(document.documentElement.scrollHeight - window.innerHeight, 0)")
        if max_scroll > 0:
            for step in range(args.scroll_steps + 1):
                scroll_position = int(max_scroll * step / args.scroll_steps)
                page.evaluate("(y) => window.scrollTo(0, y)", scroll_position)
                page.wait_for_timeout(args.scroll_pause_ms)
            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(400)

        if args.test_hover:
            auto_hover_elements(
                page,
                hover_pause_ms=args.hover_pause_ms,
                interaction_scroll_step_px=args.interaction_scroll_step_px,
                interaction_scroll_pause_ms=args.interaction_scroll_pause_ms,
            )

        video_storage = page.video.path() if page.video else None
        page.close()
        context.close()
        browser.close()

    if not video_storage:
        return False

    if args.record_format == "mp4":
        final_video = output_dir / f"{html_name}_recorded.mp4"
        converted = _convert_webm_to_mp4(
            video_storage,
            str(final_video),
            fps=args.record_fps,
            crf=args.video_crf,
        )
        if not converted:
            return False
        os.remove(video_storage)
        return True

    final_video = output_dir / f"{html_name}_recorded.webm"
    shutil.move(video_storage, final_video)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a local HTML page into a webpage video.")
    parser.add_argument("--html_path", type=str, required=True, help="Path to a local HTML file")
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.test_hover:
        args.show_mouse = True
    ok = record_html_page(args.html_path, args)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
