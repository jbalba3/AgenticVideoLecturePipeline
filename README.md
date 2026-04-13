# Homework 7 — Agentic Video Lecture Pipeline

This repository implements a local, multi-stage pipeline that turns a PDF slide deck into one narrated `.mp4`: one still per slide, synchronized audio per slide, concatenated into a single file.

## What you need locally

- **Python 3.10+**
- **`ffmpeg` on your PATH** (used for MP3 encoding/merging and final MP4 mux/concat)
- **A Google Gemini API key** with access to vision-capable models (used for all “agent” steps and default TTS)
- **`Lecture_17_AI_screenplays.pdf` in the repository root** (the grader expects it here; do not commit generated media)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root (see `.env.example`). At minimum:

- `GOOGLE_API_KEY=...` (or `GEMINI_API_KEY=...`)

This project also loads `REACT_APP_GEMINI_API_KEY` if present (useful if you already store your key that way).

## Run the full pipeline

```bash
python run_lecture_pipeline.py
```

Optional flags:

- `--pdf path\to\deck.pdf` (defaults to `./Lecture_17_AI_screenplays.pdf`)
- `--skip-style` if you already regenerated `style.json` and want faster iteration
- `--transcript-path path\to\captions.txt` to build `style.json` from a local transcript instead of downloading it
- `--project-dir my_run_01` to write outputs under `projects/my_run_01/` (folder is created if missing)

Outputs:

- `style.json` at repo root (from the **Lecture 11 Section 2** captions URL in the assignment, unless overridden)
- `projects/<project>/slide_images/slide_001.png`, …
- `projects/<project>/slide_description.json`
- `projects/<project>/premise.json`
- `projects/<project>/arc.json`
- `projects/<project>/slide_description_narration.json`
- `projects/<project>/audio/slide_001.mp3`, …
- `Lecture_17_AI_screenplays.mp4` next to the PDF (same basename)

## TTS providers

Default is Gemini native TTS (`TTS_PROVIDER=gemini`).

- **ElevenLabs**: set `TTS_PROVIDER=elevenlabs`, plus `ELEVENLABS_API_KEY` and (optionally) `ELEVENLABS_VOICE_ID`.
- **Edge TTS (offline-friendly)**: set `TTS_PROVIDER=edge`.
- If Gemini TTS fails at runtime, the pipeline **falls back to Edge TTS** so you can still get an end-to-end `.mp4` while debugging keys/models.

## GitHub submission (Canvas)

**Submit this URL on Canvas:**

[https://github.com/jbalba3/AgenticVideoLecturePipeline](https://github.com/jbalba3/AgenticVideoLecturePipeline)

If you still need to connect your local folder to GitHub:

```bash
git remote add origin https://github.com/jbalba3/AgenticVideoLecturePipeline.git
git branch -M main
git push -u origin main
```

(If `origin` already exists, use `git remote set-url origin https://github.com/jbalba3/AgenticVideoLecturePipeline.git` instead.)

## Submission hygiene (Canvas)

Do **not** commit generated binaries:

- slide PNGs, MP3s, MP4s (they are ignored via `.gitignore`)
- your `.env` / API keys

Commit **code**, **`Lecture_17_AI_screenplays.pdf`** (required for grading), **`requirements.txt`**, **`README.md`**, **`style.json`**, and the **project JSON artifacts** you want the grader to inspect under `projects/…/`.
