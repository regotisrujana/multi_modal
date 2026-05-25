"""Entity extraction: companies, colleges, skills, certifications."""

from __future__ import annotations

import json

from utils.helpers import groq_context_limit, groq_generate, parse_json_response, truncate_text


def extract_entities(context: str) -> dict:
    """Extract structured entities from merged candidate context."""
    prompt = f"""Analyze this candidate profile text and extract entities.
Return ONLY valid JSON with keys:
- companies: list of strings
- colleges: list of strings
- skills: list of strings
- certifications: list of strings
- technologies: list of strings

Candidate text:
{truncate_text(context, groq_context_limit())}
"""
    raw = groq_generate(
        prompt,
        system_hint="You are an HR entity extraction assistant. Output JSON only.",
    )
    data = parse_json_response(raw)
    for key in ("companies", "colleges", "skills", "certifications", "technologies"):
        if key not in data or not isinstance(data[key], list):
            data[key] = data.get(key, []) if isinstance(data.get(key), list) else []
    return data
