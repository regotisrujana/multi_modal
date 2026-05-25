"""Skill extraction and smart candidate tags."""

from __future__ import annotations

from utils.helpers import groq_generate, parse_json_response, truncate_text


def extract_skills(context: str) -> dict:
    """Detect programming languages, frameworks, tools, certifications, soft skills."""
    prompt = f"""From this candidate profile, extract skills and tags.
Return ONLY valid JSON:
{{
  "programming_languages": [],
  "frameworks": [],
  "tools": [],
  "certifications": [],
  "soft_skills": [],
  "smart_tags": []
}}

smart_tags examples: Python, AI/ML, Leadership, ReactJS

Profile:
{truncate_text(context, 50000)}
"""
    raw = groq_generate(prompt, system_hint="HR skill analyst. JSON only.")
    data = parse_json_response(raw)
    defaults = {
        "programming_languages": [],
        "frameworks": [],
        "tools": [],
        "certifications": [],
        "soft_skills": [],
        "smart_tags": [],
    }
    for k, v in defaults.items():
        if k not in data or not isinstance(data[k], list):
            data[k] = v
    return data
