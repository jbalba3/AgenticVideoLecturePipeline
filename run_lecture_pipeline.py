#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lecture_agents.config import (
    DEFAULT_PDF_NAME,
    PROJECTS_DIR,
    REPO_ROOT,
    STYLE_JSON_PATH,
    resolve_pdf_path,
)
from lecture_agents.pdf_raster import rasterize_pdf_to_pngs
from lecture_agents.style_agent import run_style_agent


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Agentic pipeline: PDF -> slide PNGs -> JSON agents -> TTS -> one lecture .mp4"
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help=f"Path to slide PDF (default: repo root / {DEFAULT_PDF_NAME})",
    )
    parser.add_argument(
        "--skip-style",
        action="store_true",
        help="Skip style.json regeneration (expects style.json to already exist).",
    )
    parser.add_argument(
        "--transcript-url",
        default=None,
        help="Override transcript URL for style.json generation.",
    )
    parser.add_argument(
        "--transcript-path",
        default=None,
        help="Local transcript .txt path (skips download; still produces style.json).",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project folder under projects/ (created if missing). If omitted, uses project_YYYYMMDD_HHMMSS.",
    )
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    if not pdf_path.is_file():
        _die(
            f"Missing PDF at: {pdf_path}\n"
            f"Place `{DEFAULT_PDF_NAME}` in the repository root (or pass `--pdf`)."
        )

    if args.skip_style:
        if not STYLE_JSON_PATH.is_file():
            _die("`--skip-style` was set but style.json is missing at repo root.")
    else:
        tp = Path(args.transcript_path).expanduser() if args.transcript_path else None
        run_style_agent(
            transcript_url=args.transcript_url,
            transcript_path=tp,
        )
        print(f"[ok] Wrote {STYLE_JSON_PATH}")

    if args.project_dir:
        proj = Path(args.project_dir)
        if not proj.is_absolute():
            proj = PROJECTS_DIR / proj
        proj.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        proj = PROJECTS_DIR / f"project_{stamp}"
        proj.mkdir(parents=True, exist_ok=True)

    slide_dir = proj / "slide_images"
    audio_dir = proj / "audio"
    slide_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Project directory: {proj}")

    slide_paths = rasterize_pdf_to_pngs(pdf_path, slide_dir)
    print(f"[ok] Rasterized {len(slide_paths)} slides -> {slide_dir}")

    slide_json = proj / "slide_description.json"
    premise_json = proj / "premise.json"
    arc_json = proj / "arc.json"
    narr_json = proj / "slide_description_narration.json"

    from lecture_agents.arc_agent import run_arc_agent
    from lecture_agents.narration_agent import run_narration_agent
    from lecture_agents.premise_agent import run_premise_agent
    from lecture_agents.slide_description_agent import run_slide_description_agent
    from lecture_agents.tts_step import run_tts_for_slides
    from lecture_agents.video_assembly import assemble_lecture_video

    run_slide_description_agent(
        slide_paths=slide_paths,
        out_json=slide_json,
        pdf_basename=pdf_path.name,
    )
    print(f"[ok] Wrote {slide_json}")

    run_premise_agent(slide_description_json=slide_json, out_json=premise_json)
    print(f"[ok] Wrote {premise_json}")

    run_arc_agent(
        premise_json=premise_json,
        slide_description_json=slide_json,
        out_json=arc_json,
    )
    print(f"[ok] Wrote {arc_json}")

    run_narration_agent(
        slide_paths=slide_paths,
        slide_description_json=slide_json,
        style_json=STYLE_JSON_PATH,
        premise_json=premise_json,
        arc_json=arc_json,
        out_json=narr_json,
    )
    print(f"[ok] Wrote {narr_json}")

    run_tts_for_slides(narration_json=narr_json, audio_dir=audio_dir)
    print(f"[ok] Wrote per-slide MP3s -> {audio_dir}")

    out_mp4 = pdf_path.parent / f"{pdf_path.stem}.mp4"
    final = assemble_lecture_video(
        slide_images_dir=slide_dir,
        audio_dir=audio_dir,
        pdf_path=pdf_path,
        out_mp4=out_mp4,
    )
    print(f"[ok] Wrote final video: {final}")


if __name__ == "__main__":
    main()
