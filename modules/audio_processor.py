"""Audio transcription using OpenAI Whisper."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

from utils.helpers import PROJECT_ROOT, ensure_ffmpeg_available

_whisper_model = None

WHISPER_CACHE_DIR = PROJECT_ROOT / ".cache" / "whisper"


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper

        # tiny keeps first-run downloads reasonable; override via WHISPER_MODEL env.
        size = os.getenv("WHISPER_MODEL", "tiny")
        cache_dir = Path(os.getenv("WHISPER_CACHE_DIR", str(WHISPER_CACHE_DIR)))
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        _whisper_model = whisper.load_model(size, download_root=str(cache_dir))
    return _whisper_model


def extract_audio_text(file_path: str | Path) -> dict:
    """Transcribe voice introductions and interview recordings."""
    ffmpeg = ensure_ffmpeg_available()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Audio transcription requires FFmpeg. "
            "Please install it or ensure imageio-ffmpeg is working."
        )

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
        # Use fp16=False for stability across various hardware
        result = model.transcribe(str(path.absolute()), fp16=False)
        full_text = (result.get("text") or "").strip()
        language = result.get("language", "unknown")
    except Exception as exc:
        # Check for common whisper errors
        error_msg = str(exc)
        if "ffmpeg" in error_msg.lower():
            raise RuntimeError(f"Whisper requires FFmpeg to process {path.suffix} files. Error: {exc}")
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
