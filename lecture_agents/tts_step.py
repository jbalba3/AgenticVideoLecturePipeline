from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import requests

from .config import (
    elevenlabs_api_key,
    elevenlabs_voice_id,
    gemini_model_tts,
    google_api_key,
    tts_provider,
)


def _chunk_text_for_tts(text: str, max_chars: int = 3500) -> list[str]:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return [t]
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for sentence in re.split(r"(?<=[.!?])\s+", t):
        if not sentence:
            continue
        if size + len(sentence) + 1 > max_chars and buf:
            parts.append(" ".join(buf).strip())
            buf = [sentence]
            size = len(sentence)
        else:
            buf.append(sentence)
            size += len(sentence) + 1
    if buf:
        parts.append(" ".join(buf).strip())
    return [p for p in parts if p]


def _pcm_wav_bytes(pcm: bytes, rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _wav_to_mp3_ffmpeg(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH (required to encode MP3).")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "4",
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _concat_wavs_to_mp3_ffmpeg(wav_paths: list[Path], mp3_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH (required to merge audio).")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        list_path = td_path / "files.txt"
        lines = "\n".join([f"file '{p.as_posix()}'" for p in wav_paths])
        list_path.write_text(lines, encoding="utf-8")
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "4",
            str(mp3_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _first_inline_audio_part(resp) -> bytes:
    if not getattr(resp, "candidates", None):
        pf = getattr(resp, "prompt_feedback", None)
        raise RuntimeError(f"Gemini TTS returned no candidates (prompt_feedback={pf!r}).")
    parts = resp.candidates[0].content.parts
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if not inline:
            continue
        data = inline.data
        if isinstance(data, str):
            data = base64.b64decode(data)
        if data:
            return data
    raise RuntimeError("Gemini TTS response missing inline audio data.")


def _bytes_to_wav_file(audio_bytes: bytes, out_wav: Path) -> None:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        out_wav.write_bytes(audio_bytes)
        return
    out_wav.write_bytes(_pcm_wav_bytes(audio_bytes))


def synthesize_gemini_tts(text: str, out_mp3: Path) -> None:
    from google import genai
    from google.genai import types

    key = google_api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini TTS.")

    client = genai.Client(api_key=key)
    chunks = _chunk_text_for_tts(text)
    wav_piece_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, ch in enumerate(chunks):
            prompt = (
                "Read the following lecture narration aloud in a natural classroom speaking style. "
                "Do not add meta commentary. Text:\n\n"
                + ch
            )
            resp = client.models.generate_content(
                model=gemini_model_tts(),
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.AUDIO],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                        )
                    ),
                ),
            )
            data = _first_inline_audio_part(resp)
            piece = td_path / f"chunk_{i:03d}.wav"
            _bytes_to_wav_file(data, piece)
            wav_piece_paths.append(piece)

        if len(wav_piece_paths) == 1:
            _wav_to_mp3_ffmpeg(wav_piece_paths[0], out_mp3)
        else:
            _concat_wavs_to_mp3_ffmpeg(wav_piece_paths, out_mp3)


def synthesize_elevenlabs(text: str, out_mp3: Path) -> None:
    key = elevenlabs_api_key()
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs.")
    voice = elevenlabs_voice_id() or "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {"xi-api-key": key, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    chunks = _chunk_text_for_tts(text, max_chars=4500)
    if len(chunks) == 1:
        body = {"text": chunks[0], "model_id": "eleven_multilingual_v2"}
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        out_mp3.write_bytes(r.content)
        return

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        piece_paths: list[Path] = []
        for i, ch in enumerate(chunks):
            body = {"text": ch, "model_id": "eleven_multilingual_v2"}
            r = requests.post(url, headers=headers, json=body, timeout=120)
            r.raise_for_status()
            p = td_path / f"part_{i:03d}.mp3"
            p.write_bytes(r.content)
            piece_paths.append(p)
        concat_mp3s_ffmpeg(piece_paths, out_mp3)


async def _edge_save(text: str, out_mp3: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
    await communicate.save(str(out_mp3))


def synthesize_edge_tts(text: str, out_mp3: Path) -> None:
    asyncio.run(_edge_save(text, out_mp3))


def concat_mp3s_ffmpeg(mp3_paths: list[Path], out_mp3: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH.")
    with tempfile.TemporaryDirectory() as td:
        list_path = Path(td) / "files.txt"
        list_path.write_text("\n".join([f"file '{p.as_posix()}'" for p in mp3_paths]), encoding="utf-8")
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "4",
            str(out_mp3),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_tts_for_slides(*, narration_json: Path, audio_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(narration_json.read_text(encoding="utf-8"))
    slides = data.get("slides") or []
    provider = tts_provider()

    for s in slides:
        idx = int(s["slide_index"])
        text = (s.get("narration") or "").strip()
        if not text:
            raise RuntimeError(f"Missing narration for slide {idx}")
        out = audio_dir / f"slide_{idx:03d}.mp3"
        if provider == "elevenlabs":
            synthesize_elevenlabs(text, out)
        elif provider == "edge":
            synthesize_edge_tts(text, out)
        else:
            try:
                synthesize_gemini_tts(text, out)
            except Exception as exc:
                print(
                    f"[tts] Gemini TTS failed for slide {idx} ({exc!r}); falling back to edge-tts.",
                    file=sys.stderr,
                )
                synthesize_edge_tts(text, out)
