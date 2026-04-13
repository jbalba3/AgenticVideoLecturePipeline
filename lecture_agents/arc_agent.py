from __future__ import annotations

import json
from pathlib import Path

from .config import gemini_model_agents
from .gemini_client import generate_json_from_prompt


SYS = """You design a coherent lecture arc for voiceover narration.
Use premise.json and slide_description.json. Return JSON including:
- flow_summary: string
- acts: array of objects with fields: name, slide_range (string like "1-4"), purpose, beats (array of strings)
- idea_progression: array of strings (how concepts build)
- transitions: array of strings (how to move between major sections)
Keep it consistent with the premise and slide content."""


def run_arc_agent(*, premise_json: Path, slide_description_json: Path, out_json: Path) -> Path:
    premise = premise_json.read_text(encoding="utf-8")
    slides = slide_description_json.read_text(encoding="utf-8")
    user = f"premise.json:\n{premise}\n\nslide_description.json:\n{slides}\n"
    arc = generate_json_from_prompt(
        model=gemini_model_agents(),
        system_instruction=SYS,
        user_text=user,
        temperature=0.25,
    )
    if not isinstance(arc, dict):
        raise RuntimeError("Arc agent must return a JSON object.")
    out_json.write_text(json.dumps(arc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json
