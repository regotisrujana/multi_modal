"""ATS friendliness checker."""

from __future__ import annotations

from utils.helpers import groq_context_limit, groq_generate, parse_json_response, truncate_text


def check_ats(context: str, job_keywords: str = "") -> dict:
    """Evaluate ATS compatibility and keyword match."""
    kw = job_keywords or "general software engineering roles"
    prompt = f"""Evaluate ATS (Applicant Tracking System) friendliness.
Target role keywords: {kw}

Return ONLY valid JSON:
{{
  "ats_score": 0-100,
  "is_ats_friendly": true/false,
  "keyword_matches": [],
  "missing_keywords": [],
  "format_warnings": [],
  "recommendations": []
}}

Resume/profile:
{truncate_text(context, groq_context_limit())}
"""
    raw = groq_generate(prompt, system_hint="ATS optimization expert. JSON only.")
    return parse_json_response(raw)
