"""Export HR reports: PDF and text formats."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.helpers import EXPORTS_DIR, safe_filename


def _format_section(title: str, content: Any) -> str:
    lines = [f"\n{'='*60}", title.upper(), "=" * 60]
    if isinstance(content, (dict, list)):
        lines.append(json.dumps(content, indent=2, default=str))
    else:
        lines.append(str(content))
    return "\n".join(lines)


def build_full_report_text(candidate_name: str, analysis: dict) -> str:
    """Plain-text hiring report."""
    parts = [
        "AI MULTIMODAL RECRUITMENT ANALYZER — HR EVALUATION REPORT",
        f"Candidate: {candidate_name}",
        f"Generated: {datetime.utcnow().isoformat()}Z",
    ]
    for key, label in [
        ("summary", "Candidate Summary"),
        ("skills", "Skills"),
        ("resume_analysis", "Resume Analysis"),
        ("communication", "Communication Analysis"),
        ("scores", "Candidate Scoring"),
        ("ats", "ATS Check"),
        ("interview_questions", "Interview Questions"),
        ("entities", "Entities"),
        ("tags", "Smart Tags"),
    ]:
        if key in analysis:
            parts.append(_format_section(label, analysis[key]))
    return "\n".join(parts)


def export_text_report(candidate_name: str, analysis: dict) -> Path:
    """Save .txt report to exports folder."""
    text = build_full_report_text(candidate_name, analysis)
    fname = safe_filename(f"{candidate_name}_hr_report.txt")
    path = EXPORTS_DIR / fname
    path.write_text(text, encoding="utf-8")
    return path


def export_interview_questions(candidate_name: str, questions: dict) -> Path:
    fname = safe_filename(f"{candidate_name}_interview_questions.txt")
    path = EXPORTS_DIR / fname
    path.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    return path


def export_pdf_report(candidate_name: str, analysis: dict) -> Path:
    """Generate PDF using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    fname = safe_filename(f"{candidate_name}_hr_report.pdf")
    path = EXPORTS_DIR / fname
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    def add_para(text: str, style_name: str = "Normal"):
        # Escape minimal XML chars for reportlab
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        for chunk in safe.split("\n"):
            if chunk.strip():
                story.append(Paragraph(chunk[:3000], styles[style_name]))
        story.append(Spacer(1, 8))

    add_para("AI Multimodal Recruitment Analyzer", "Title")
    add_para(f"Candidate: {candidate_name}", "Heading2")
    add_para(f"Generated: {datetime.utcnow().isoformat()}Z", "Normal")

    summary = analysis.get("summary") or analysis.get("scores", {})
    add_para("Executive Summary", "Heading2")
    add_para(json.dumps(summary, indent=2)[:8000], "Normal")

    if analysis.get("scores"):
        add_para("Scoring & Recommendation", "Heading2")
        add_para(json.dumps(analysis["scores"], indent=2)[:6000], "Normal")

    if analysis.get("communication"):
        add_para("Communication Analysis", "Heading2")
        add_para(json.dumps(analysis["communication"], indent=2)[:4000], "Normal")

    if analysis.get("interview_questions"):
        add_para("Interview Questions", "Heading2")
        add_para(json.dumps(analysis["interview_questions"], indent=2)[:6000], "Normal")

    doc.build(story)
    return path
