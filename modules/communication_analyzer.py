"""Communication analysis from voice/video transcripts."""

from __future__ import annotations

from utils.helpers import groq_context_limit, groq_generate, parse_json_response, truncate_text


def analyze_communication(transcript: str, has_audio: bool = True) -> dict:
    """Analyze speaking confidence, clarity, filler words, professional tone."""
    if not transcript or len(transcript.strip()) < 20:
        return {
            "speaking_confidence": 0,
            "communication_clarity": 0,
            "filler_word_frequency": "N/A — no transcript",
            "professional_tone": "No audio/video content provided",
            "communication_score": 0,
            "notes": "Upload voice intro or video for communication analysis.",
        }

    prompt = f"""Analyze communication from this transcript (voice/video intro).
Return ONLY valid JSON:
{{
  "speaking_confidence": 0-100,
  "communication_clarity": 0-100,
  "filler_word_frequency": "low|medium|high with examples",
  "professional_tone": "string assessment",
  "communication_score": 0-100,
  "notable_phrases": [],
  "improvement_tips": []
}}

Transcript:
{truncate_text(transcript, groq_context_limit(8000))}
"""
    raw = groq_generate(
        prompt,
        system_hint="Communication coach for hiring. JSON only.",
    )
    return parse_json_response(raw)
