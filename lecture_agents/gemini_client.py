from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import google_api_key


def _client() -> genai.Client:
    key = google_api_key()
    if not key:
        raise RuntimeError(
            "Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY (or REACT_APP_GEMINI_API_KEY in .env)."
        )
    return genai.Client(api_key=key)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def generate_json_from_prompt(
    *,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float = 0.35,
) -> Any:
    client = _client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=temperature,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
        config=cfg,
    )
    if not getattr(resp, "candidates", None):
        pf = getattr(resp, "prompt_feedback", None)
        raise RuntimeError(f"Model returned no candidates (prompt_feedback={pf!r}).")
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Empty model response when JSON was expected.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_json_fence(raw))


def generate_json_from_multimodal(
    *,
    model: str,
    system_instruction: str,
    parts: list[types.Part],
    temperature: float = 0.35,
) -> Any:
    client = _client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=temperature,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=cfg,
    )
    if not getattr(resp, "candidates", None):
        pf = getattr(resp, "prompt_feedback", None)
        raise RuntimeError(f"Model returned no candidates (prompt_feedback={pf!r}).")
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Empty model response when JSON was expected.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_json_fence(raw))


def load_png_part(path: Path) -> types.Part:
    data = path.read_bytes()
    return types.Part.from_bytes(data=data, mime_type="image/png")
