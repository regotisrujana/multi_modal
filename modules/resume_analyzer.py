"""Resume analysis: quality, formatting, ATS, missing sections."""

from __future__ import annotations

from utils.helpers import groq_generate, parse_json_response, truncate_text


def analyze_resume(context: str) -> dict:
    """Analyze resume quality and structure."""
    prompt = f"""Analyze resume/portfolio text for HR recruiters.
Return ONLY valid JSON:
{{
  "resume_quality_score": 0-100,
  "formatting_notes": "string",
  "missing_sections": [],
  "keyword_optimization": "string",
  "ats_friendliness_score": 0-100,
  "ats_issues": [],
  "strengths": [],
  "weaknesses": [],
  "education_summary": "string",
  "experience_summary": "string",
  "project_overview": "string",
  "technical_profile": "string",
  "cross_file_correlations": []
}}

cross_file_correlations: note when projects/skills appear across multiple sources.

Text:
{truncate_text(context, 60000)}
"""
    raw = groq_generate(prompt, system_hint="Expert resume analyst. JSON only.")
    return parse_json_response(raw)


def generate_candidate_summary(context: str) -> dict:
    """High-level candidate summary."""
    prompt = f"""Create a candidate summary for recruiters.
Return ONLY valid JSON:
{{
  "summary": "2-3 paragraph overview",
  "education_summary": "string",
  "experience_summary": "string",
  "project_overview": "string",
  "technical_profile": "string"
}}

Profile:
{truncate_text(context, 60000)}
"""
    raw = groq_generate(prompt, system_hint="HR summarization assistant. JSON only.")
    return parse_json_response(raw)
