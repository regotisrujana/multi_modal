"""Candidate scoring and HR recommendation classification."""

from __future__ import annotations

from utils.helpers import groq_context_limit, groq_generate, parse_json_response, truncate_text


def score_candidate(context: str, communication: dict | None = None) -> dict:
    """Generate technical, communication, resume, confidence scores and recommendation."""
    comm_note = ""
    if communication:
        comm_note = f"Communication analysis: {communication}"

    prompt = f"""Score this candidate for hiring.
Return ONLY valid JSON:
{{
  "technical_score": 0-100,
  "communication_score": 0-100,
  "resume_quality_score": 0-100,
  "confidence_score": 0-100,
  "overall_score": 0-100,
  "hiring_recommendation": "Strong Candidate|Moderate Candidate|Needs Improvement",
  "recommendation_rationale": "string",
  "strengths": [],
  "weaknesses": [],
  "improvement_suggestions": [],
  "suggested_roles": [],
  "portfolio_analysis": {{
    "project_quality": "string",
    "presentation_structure": "string",
    "innovation_indicators": []
  }},
  "timeline": {{
    "education": [],
    "internships": [],
    "projects": []
  }}
}}

{comm_note}

Profile:
{truncate_text(context, groq_context_limit())}
"""
    raw = groq_generate(prompt, system_hint="HR scoring analyst. JSON only.")
    data = parse_json_response(raw)
    rec = data.get("hiring_recommendation", "Moderate Candidate")
    if rec not in (
        "Strong Candidate",
        "Moderate Candidate",
        "Needs Improvement",
    ):
        score = float(data.get("overall_score", 50))
        if score >= 75:
            data["hiring_recommendation"] = "Strong Candidate"
        elif score >= 50:
            data["hiring_recommendation"] = "Moderate Candidate"
        else:
            data["hiring_recommendation"] = "Needs Improvement"
    return data
