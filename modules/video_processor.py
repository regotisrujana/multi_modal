"""Intro video processing: extract audio + Whisper transcription."""

from __future__ import annotations

import os
from pathlib import Path

from modules.audio_processor import extract_audio_from_video, extract_audio_text


def extract_video_text(file_path: str | Path) -> dict:
    """Transcribe spoken content from candidate intro / interview videos."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    audio_tmp = None
    try:
        audio_tmp = extract_audio_from_video(path)
        audio_result = extract_audio_text(audio_tmp)
        full_text = audio_result.get("text", "")
    except Exception as exc:
        raise RuntimeError(f"Video processing failed: {exc}") from exc
    finally:
        if audio_tmp and os.path.exists(audio_tmp):
            try:
                os.remove(audio_tmp)
            except OSError:
                pass

    return {
        "text": full_text,
        "metadata": {
            "source": path.name,
            "type": "intro_video",
            "char_count": len(full_text),
        },
    }
