from __future__ import annotations

import base64
import json
import re
from typing import Any

from openai import OpenAI

from .config import openai_api_key, openai_model_agents


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _client() -> OpenAI:
    key = openai_api_key()
    if not key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY (or REACT_APP_OPENAI_API_KEY) when LLM_PROVIDER=openai."
        )
    return OpenAI(api_key=key)


def generate_json_text(
    *,
    system_instruction: str,
    user_text: str,
    temperature: float = 0.35,
    model: str | None = None,
) -> Any:
    client = _client()
    m = model or openai_model_agents()
    resp = client.chat.completions.create(
        model=m,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("Empty OpenAI response when JSON was expected.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_json_fence(raw))


def generate_json_with_png(
    *,
    system_instruction: str,
    user_text: str,
    png_bytes: bytes,
    temperature: float = 0.35,
    model: str | None = None,
) -> Any:
    client = _client()
    m = model or openai_model_agents()
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    resp = client.chat.completions.create(
        model=m,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("Empty OpenAI response when JSON was expected.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_json_fence(raw))
