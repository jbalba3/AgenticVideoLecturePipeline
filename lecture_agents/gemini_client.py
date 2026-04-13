from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import google_api_key, llm_provider


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


def _gemini_resource_hint(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "429" in msg or "resource_exhausted" in msg or "credits" in msg or "billing" in msg:
        return (
            "\n\nGemini returned quota/billing error (429). Options:\n"
            "  • Add credits / billing: https://aistudio.google.com/\n"
            "  • Or use OpenAI for agent steps only: set LLM_PROVIDER=openai and OPENAI_API_KEY in .env\n"
            "  • For TTS without Gemini: set TTS_PROVIDER=edge (or elevenlabs)\n"
            "  • To skip regenerating style: use --skip-style if style.json is already good\n"
        )
    return ""


def _parts_text_and_first_png(parts: list[types.Part]) -> tuple[str, bytes | None]:
    texts: list[str] = []
    png: bytes | None = None
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            texts.append(t)
        inline = getattr(p, "inline_data", None)
        if inline is not None:
            mime = (getattr(inline, "mime_type", None) or "").lower()
            data = getattr(inline, "data", None)
            if data and mime.startswith("image/") and png is None:
                if isinstance(data, str):
                    import base64

                    data = base64.b64decode(data)
                png = data
    return "\n\n".join(texts), png


def generate_json_from_prompt(
    *,
    model: str,
    system_instruction: str,
    user_text: str,
    temperature: float = 0.35,
) -> Any:
    if llm_provider() == "openai":
        from .openai_client import generate_json_text

        return generate_json_text(
            system_instruction=system_instruction,
            user_text=user_text,
            temperature=temperature,
            model=None,
        )

    client = _client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=temperature,
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
            config=cfg,
        )
    except Exception as exc:
        raise RuntimeError(str(exc) + _gemini_resource_hint(exc)) from exc
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
    if llm_provider() == "openai":
        from .openai_client import generate_json_with_png

        user_text, png = _parts_text_and_first_png(parts)
        if not png:
            raise RuntimeError("OpenAI multimodal path requires an image/png part in `parts`.")
        return generate_json_with_png(
            system_instruction=system_instruction,
            user_text=user_text,
            png_bytes=png,
            temperature=temperature,
            model=None,
        )

    client = _client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=temperature,
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=cfg,
        )
    except Exception as exc:
        raise RuntimeError(str(exc) + _gemini_resource_hint(exc)) from exc
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
