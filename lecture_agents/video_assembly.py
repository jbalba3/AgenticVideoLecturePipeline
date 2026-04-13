from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (required for video assembly).")
    return exe


def mux_slide_segment(png: Path, mp3: Path, out_mp4: Path) -> None:
    """One still image + one mp3 -> mp4; duration tracks audio via -shortest."""
    ff = _ffmpeg()
    cmd = [
        ff,
        "-y",
        "-loop",
        "1",
        "-i",
        str(png),
        "-i",
        str(mp3),
        "-shortest",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def concat_segments(segment_paths: list[Path], out_mp4: Path) -> None:
    ff = _ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "concat.txt"
        lst.write_text("\n".join([f"file '{p.as_posix()}'" for p in segment_paths]), encoding="utf-8")
        cmd = [
            ff,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def assemble_lecture_video(
    *,
    slide_images_dir: Path,
    audio_dir: Path,
    pdf_path: Path,
    out_mp4: Path | None = None,
) -> Path:
    pngs = sorted(slide_images_dir.glob("slide_*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG slides found in {slide_images_dir}")

    segments_dir = slide_images_dir.parent / "_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    segment_files: list[Path] = []

    for png in pngs:
        stem = png.stem
        mp3 = audio_dir / f"{stem}.mp3"
        if not mp3.is_file():
            raise FileNotFoundError(f"Missing audio for {png.name}: expected {mp3}")
        seg = segments_dir / f"{stem}.mp4"
        mux_slide_segment(png, mp3, seg)
        segment_files.append(seg)

    final = out_mp4 or (slide_images_dir.parent / f"{pdf_path.stem}.mp4")
    concat_segments(segment_files, final)
    for p in segment_files:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        segments_dir.rmdir()
    except OSError:
        pass
    return final
