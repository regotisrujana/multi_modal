"""Audio transcription using OpenAI Whisper."""

from __future__ import annotations

import tempfile
from pathlib import Path

from utils.helpers import ensure_ffmpeg_available

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper

        # base is a balance for student laptops; override via WHISPER_MODEL env
        import os

        size = os.getenv("WHISPER_MODEL", "base")
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def extract_audio_text(file_path: str | Path) -> dict:
    """Transcribe voice introductions and interview recordings."""
    ensure_ffmpeg_available()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    audio_type = (
        "interview_recording"
        if "interview" in path.name.lower()
        else "voice_intro"
    )

    try:
        model = _get_whisper()
        result = model.transcribe(str(path), fp16=False)
        full_text = (result.get("text") or "").strip()
        language = result.get("language", "unknown")
    except Exception as exc:
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc

    return {
        "text": full_text,
        "metadata": {
            "source": path.name,
            "type": audio_type,
            "language": language,
            "char_count": len(full_text),
        },
    }


def extract_audio_from_video(video_path: str | Path) -> str:
    """Extract audio track from video to temp WAV for Whisper."""
    ensure_ffmpeg_available()
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip

    path = Path(video_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        clip = VideoFileClip(str(path))
        if clip.audio is None:
            clip.close()
            raise RuntimeError("Video has no audio track")
        clip.audio.write_audiofile(tmp.name, logger=None)
        clip.close()
        return tmp.name
    except Exception as exc:
        raise RuntimeError(f"Video audio extraction failed: {exc}") from exc
