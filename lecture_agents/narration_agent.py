from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.genai import types

from .config import gemini_model_agents
from .gemini_client import generate_json_from_multimodal, load_png_part


SYS = """You write spoken lecture narration for a slide deck video.
Return strict JSON: {"slide_index": <int>, "filename": "<png filename>", "description": "<copy or tighten prior description>", "narration": "<spoken script>"}.

Rules:
- Match the instructor style profile in style.json (tone, pacing, fillers policy per imitation_notes).
- Stay consistent with premise.json and arc.json.
- Do not read bullet points robotically; sound like a live lecture.
- If slide_index is 1 (title slide), begin with a natural self-introduction: use the instructor name shown on the slide if visible; otherwise say you are the instructor for this module. Then give a short overview of the lecture topic and why it matters.
- Keep each narration appropriate in length for a slide (typically ~20-90 seconds when read aloud), unless the slide is clearly a simple divider.
"""


def run_narration_agent(
    *,
    slide_paths: list[Path],
    slide_description_json: Path,
    style_json: Path,
    premise_json: Path,
    arc_json: Path,
    out_json: Path,
) -> Path:
    style = style_json.read_text(encoding="utf-8")
    premise = premise_json.read_text(encoding="utf-8")
    arc = arc_json.read_text(encoding="utf-8")
    deck = json.loads(slide_description_json.read_text(encoding="utf-8"))
    slides_meta: list[dict[str, Any]] = deck.get("slides") or []
    if len(slides_meta) != len(slide_paths):
        raise RuntimeError("slide_description.json slide count does not match rasterized slide images.")

    results: list[dict[str, Any]] = []
    for idx, png in enumerate(slide_paths, start=1):
        meta = slides_meta[idx - 1]
        prior = results.copy()
        prior_text = json.dumps(prior, indent=2, ensure_ascii=False) if prior else "[]"
        user_text = f"""style.json:
{style}

premise.json:
{premise}

arc.json:
{arc}

Entire slide_description.json (for grounding; do not ignore prior-slide context):
{json.dumps(deck, indent=2, ensure_ascii=False)}

Prior slide narrations (JSON array, oldest first). Empty on first slide:
{prior_text}

Now generate narration for ONLY slide_index {idx}.
Current slide filename: {png.name}
Associated description from slide_description.json for this slide:
{json.dumps(meta, indent=2, ensure_ascii=False)}
"""
        parts = [types.Part.from_text(text=user_text), load_png_part(png)]
        obj = generate_json_from_multimodal(
            model=gemini_model_agents(),
            system_instruction=SYS,
            parts=parts,
            temperature=0.35,
        )
        if not isinstance(obj, dict):
            raise RuntimeError(f"Slide {idx}: model did not return a JSON object.")
        if int(obj.get("slide_index", -1)) != idx:
            obj["slide_index"] = idx
        if not obj.get("filename"):
            obj["filename"] = png.name
        if not obj.get("description"):
            obj["description"] = meta.get("description", "")
        narration = (obj.get("narration") or "").strip()
        if not narration:
            raise RuntimeError(f"Slide {idx}: missing narration.")
        results.append(
            {
                "slide_index": idx,
                "filename": obj["filename"],
                "description": obj["description"],
                "narration": narration,
            }
        )

    out = {"slides": results}
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json
