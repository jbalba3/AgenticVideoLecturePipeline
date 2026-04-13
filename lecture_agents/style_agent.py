from __future__ import annotations

import json
from pathlib import Path

from .config import STYLE_JSON_PATH, STYLE_TRANSCRIPT_URL, gemini_model_agents, google_api_key
from .gemini_client import generate_json_from_prompt
from .transcript_fetch import fetch_style_transcript, load_transcript_from_path, strip_transcript_boilerplate


STYLE_SYSTEM = """You analyze instructor lecture transcripts and output a compact JSON style profile
for a narration voice-over agent. Ground every claim in the transcript; do not invent biographical facts
not supported by the text (you may infer tone/patterns)."""


def build_style_profile(*, transcript_text: str) -> dict:
    if not google_api_key():
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required for the style agent.")

    user = f"""Transcript (may include minor header lines):\n\n{transcript_text}\n\n
Return JSON with these keys (add subfields if helpful):
- tone: string summary
- pacing: string (e.g., bursts vs steady; where they speed up)
- register: string (formal/informal mix)
- fillers_and_discourse_markers: array of strings (e.g., "like", "um", "okay")
- audience_address: string (how they talk to students)
- humor_and_asides: string
- explanation_moves: array of strings (how they frame intuition, examples, warnings)
- rhetorical_questions: boolean or string notes
- signposting: string (how they transition topics)
- emphasis_patterns: string (repetition, "right?", analogies)
- taboo_or_avoid: array of strings (things narration should avoid)
- imitation_notes: string (practical instructions for a VO writer to mimic this instructor)
"""
    data = generate_json_from_prompt(
        model=gemini_model_agents(),
        system_instruction=STYLE_SYSTEM,
        user_text=user,
        temperature=0.25,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Style agent must return a JSON object.")
    data["source_transcript_url"] = STYLE_TRANSCRIPT_URL
    return data


def run_style_agent(
    *,
    out_path: Path | None = None,
    transcript_url: str | None = None,
    transcript_path: Path | None = None,
) -> Path:
    out = out_path or STYLE_JSON_PATH
    if transcript_path:
        raw = load_transcript_from_path(transcript_path)
    else:
        raw = fetch_style_transcript(transcript_url)
    cleaned = strip_transcript_boilerplate(raw)
    profile = build_style_profile(transcript_text=cleaned[:800_000])
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
