from __future__ import annotations

import re
from pathlib import Path

import requests

from .config import STYLE_TRANSCRIPT_URL


def fetch_style_transcript(url: str | None = None) -> str:
    u = url or STYLE_TRANSCRIPT_URL
    r = requests.get(u, timeout=120)
    r.raise_for_status()
    return r.text


def load_transcript_from_path(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_transcript_boilerplate(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith("Source URL:") or line.startswith("Title:"):
            continue
        if "[Auto-generated transcript" in line:
            continue
        out.append(line)
    body = "\n".join(out).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body
