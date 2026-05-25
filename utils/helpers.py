"""
Shared utilities: file detection, hashing, Groq LLM client, JSON parsing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "app" / ".env")

# Supported extensions mapped to internal types
EXTENSION_TYPE_MAP = {
    ".pdf": "resume_pdf",
    ".docx": "resume_docx",
    ".doc": "unsupported_legacy_word",
    ".ppt": "unsupported_legacy_presentation",
    ".pptx": "portfolio_ppt",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".gif": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".webm": "video",
    ".mkv": "video",
}

SUPPORTED_EXTENSIONS = sorted(
    ext for ext, file_type in EXTENSION_TYPE_MAP.items()
    if not file_type.startswith("unsupported_legacy")
)

UNSUPPORTED_FORMAT_MESSAGES = {
    "unsupported_legacy_word": (
        "Legacy .doc files are not supported. Save the file as .docx or PDF "
        "and upload it again."
    ),
    "unsupported_legacy_presentation": (
        "Legacy .ppt files are not supported. Save the file as .pptx or PDF "
        "and upload it again."
    ),
}

EXPORTS_DIR = PROJECT_ROOT / "exports"
ASSETS_DIR = PROJECT_ROOT / "assets"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def detect_file_type(filename: str) -> str:
    """Return internal file type from extension."""
    ext = Path(filename).suffix.lower()
    return EXTENSION_TYPE_MAP.get(ext, "unknown")


def content_hash(text: str) -> str:
    """SHA-256 hash for duplicate detection."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def safe_filename(name: str) -> str:
    """Sanitize filename for exports."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)[:120]


def truncate_text(text: str, max_chars: int = 120_000) -> str:
    """Limit context size for LLM calls."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for length ...]"


def require_extractable_text(result: dict[str, Any], filename: str) -> None:
    """Raise a user-facing error when extraction produced no usable content."""
    text = (result.get("text") or "").strip()
    if text:
        return
    raise ValueError(
        f"No extractable text was found in {filename}. Use a text-based file, "
        "a clearer scan/screenshot, or an audio/video file with audible speech."
    )


def parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from LLM response."""
    if not raw:
        return {}
    text = raw.strip()
    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_text": raw}


def get_groq_client():
    """Return configured Groq client."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to .env or environment variables. "
            "Get a key at https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def get_ffmpeg_executable() -> Optional[str]:
    """Find an FFmpeg executable from PATH or the imageio-ffmpeg package."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ensure_ffmpeg_available() -> Optional[str]:
    """Expose bundled FFmpeg to subprocess-based libraries when possible."""
    ffmpeg_path = get_ffmpeg_executable()
    if not ffmpeg_path:
        return None
    ffmpeg_dir = str(Path(ffmpeg_path).parent)
    path_parts = os.getenv("PATH", "").split(os.pathsep)
    if ffmpeg_dir not in path_parts:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.getenv("PATH", "")
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_path)
    return ffmpeg_path


def get_system_health() -> dict[str, Any]:
    """Return lightweight readiness checks for the local app environment."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    ffmpeg_path = get_ffmpeg_executable()
    return {
        "groq_api_key": {
            "ok": bool(api_key and api_key != "your_groq_api_key_here"),
            "detail": "Configured" if api_key else "Missing",
        },
        "ffmpeg": {
            "ok": ffmpeg_path is not None,
            "detail": ffmpeg_path or "Missing from PATH and imageio-ffmpeg",
        },
        "exports_dir": {
            "ok": EXPORTS_DIR.exists(),
            "detail": str(EXPORTS_DIR),
        },
        "assets_dir": {
            "ok": ASSETS_DIR.exists(),
            "detail": str(ASSETS_DIR),
        },
        "supported_extensions": {
            "ok": True,
            "detail": ", ".join(SUPPORTED_EXTENSIONS),
        },
    }


def groq_generate(prompt: str, system_hint: str = "") -> str:
    """Call Groq chat completions with optional system message."""
    try:
        client = get_groq_client()
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        messages = []
        if system_hint:
            messages.append({"role": "system", "content": system_hint})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=8192,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# Backward-compatible alias (modules may still reference this name)
gemini_generate = groq_generate


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def merge_contexts(parts: list[str]) -> str:
    """Merge extracted content from multiple files."""
    sections = []
    for i, part in enumerate(parts, 1):
        if part and part.strip():
            sections.append(f"--- Source Block {i} ---\n{part.strip()}")
    return "\n\n".join(sections)
