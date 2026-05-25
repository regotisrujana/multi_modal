"""Interview question generator."""

from __future__ import annotations

from utils.helpers import groq_generate, parse_json_response, truncate_text


def generate_interview_questions(context: str, role: str = "") -> dict:
    """Generate technical, HR, project, and scenario questions."""
    role_hint = role or "the candidate's likely roles based on profile"
    prompt = f"""Generate interview questions for role: {role_hint}
Return ONLY valid JSON:
{{
  "technical_questions": [],
  "hr_questions": [],
  "project_based_questions": [],
  "scenario_questions": []
}}

Candidate profile:
{truncate_text(context, 50000)}
"""
    raw = groq_generate(
        prompt,
        system_hint="Senior technical interviewer. JSON only.",
    )
    return parse_json_response(raw)
