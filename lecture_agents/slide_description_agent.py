from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.genai import types

from .config import gemini_model_agents
from .gemini_client import generate_json_from_multimodal, load_png_part


SYS = """You describe a single lecture slide image for downstream lecture-writing agents.
Return strict JSON: {"slide_index": <int>, "filename": "<png filename>", "description": "<string>"}.
Be specific about visible text, diagrams, bullets, and how this slide relates to prior slides when prior context exists."""


def run_slide_description_agent(*, slide_paths: list[Path], out_json: Path, pdf_basename: str) -> Path:
    descriptions: list[dict[str, Any]] = []
    for idx, png in enumerate(slide_paths, start=1):
        prev = descriptions.copy()
        prev_text = json.dumps(prev, indent=2, ensure_ascii=False) if prev else "[]"
        user_text = f"""PDF basename: {pdf_basename}
Current slide_index: {idx}
Current slide filename: {png.name}

All previous slide descriptions (JSON array, oldest first):
{prev_text}

Describe ONLY the current slide image."""
        parts = [
            types.Part.from_text(text=user_text),
            load_png_part(png),
        ]
        obj = generate_json_from_multimodal(
            model=gemini_model_agents(),
            system_instruction=SYS,
            parts=parts,
            temperature=0.25,
        )
        if not isinstance(obj, dict):
            raise RuntimeError(f"Slide {idx}: model did not return a JSON object.")
        if int(obj.get("slide_index", -1)) != idx:
            obj["slide_index"] = idx
        if not obj.get("filename"):
            obj["filename"] = png.name
        if not obj.get("description"):
            raise RuntimeError(f"Slide {idx}: missing description.")
        descriptions.append(
            {
                "slide_index": idx,
                "filename": obj["filename"],
                "description": obj["description"],
            }
        )

    payload = {"pdf_basename": pdf_basename, "slides": descriptions}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json
