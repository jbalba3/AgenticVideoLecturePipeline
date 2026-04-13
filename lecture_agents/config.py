from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STYLE_TRANSCRIPT_URL = (
    "https://zlisto.github.io/genAI_social_media/slides_pdf/"
    "MGT%20575%2001-02%20(SP26)_%20%20Generative%20AI%20and%20Social%20Media%20"
    "Lecture%2011%20Section%202_Captions_English%20(United%20States).txt"
)

STYLE_JSON_PATH = REPO_ROOT / "style.json"
PROJECTS_DIR = REPO_ROOT / "projects"
DEFAULT_PDF_NAME = "Lecture_17_AI_screenplays.pdf"


def resolve_pdf_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    return REPO_ROOT / DEFAULT_PDF_NAME


def google_api_key() -> str | None:
    for key in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_API_KEY",
        "REACT_APP_GEMINI_API_KEY",
    ):
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip()
    return None


def tts_provider() -> str:
    return (os.environ.get("TTS_PROVIDER") or "gemini").strip().lower()


def elevenlabs_api_key() -> str | None:
    v = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("REACT_APP_ELEVENLABS_API_KEY")
    return v.strip() if v else None


def elevenlabs_voice_id() -> str | None:
    v = os.environ.get("ELEVENLABS_VOICE_ID") or os.environ.get("REACT_APP_ELEVENLABS_VOICE_ID")
    return v.strip() if v else None


def gemini_model_agents() -> str:
    return os.environ.get("GEMINI_MODEL_AGENTS", "gemini-2.0-flash").strip()


def gemini_model_tts() -> str:
    return os.environ.get("GEMINI_MODEL_TTS", "gemini-2.5-flash-preview-tts").strip()


def llm_provider() -> str:
    """Agent LLM: gemini (default) or openai."""
    return (os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()


def openai_api_key() -> str | None:
    for key in ("OPENAI_API_KEY", "REACT_APP_OPENAI_API_KEY"):
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip()
    return None


def openai_model_agents() -> str:
    return os.environ.get("OPENAI_MODEL_AGENTS", "gpt-4o-mini").strip()
