from __future__ import annotations

import json
from pathlib import Path

from .config import gemini_model_agents
from .gemini_client import generate_json_from_prompt


SYS = """You write a structured lecture premise grounded ONLY in the provided slide descriptions JSON.
Return JSON with keys you choose, but MUST include: thesis, scope, learning_objectives (array of strings),
audience, key_takeaways (array). Keep it consistent with the deck."""


def run_premise_agent(*, slide_description_json: Path, out_json: Path) -> Path:
    deck = slide_description_json.read_text(encoding="utf-8")
    user = f"slide_description.json contents:\n\n{deck}\n"
    premise = generate_json_from_prompt(
        model=gemini_model_agents(),
        system_instruction=SYS,
        user_text=user,
        temperature=0.25,
    )
    if not isinstance(premise, dict):
        raise RuntimeError("Premise agent must return a JSON object.")
    out_json.write_text(json.dumps(premise, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json
