"""
Unified file ingestion and analysis pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from modules.audio_processor import extract_audio_text
from modules.communication_analyzer import analyze_communication
from modules.docx_processor import extract_docx_text
from modules.entity_extractor import extract_entities
from modules.image_processor import extract_image_text
from modules.interview_generator import generate_interview_questions
from modules.pdf_processor import extract_pdf_text
from modules.ppt_processor import extract_ppt_text
from modules.resume_analyzer import analyze_resume, generate_candidate_summary
from modules.skill_extractor import extract_skills
from modules.video_processor import extract_video_text
from modules.ats_checker import check_ats
from modules.candidate_scorer import score_candidate
from utils.helpers import (
    UNSUPPORTED_FORMAT_MESSAGES,
    detect_file_type,
    require_extractable_text,
    truncate_text,
)
from vectorstore import chroma_db


def extract_txt(file_path: Path) -> dict:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return {
        "text": text,
        "metadata": {
            "source": file_path.name,
            "type": "text",
            "char_count": len(text),
        },
    }


def process_file(uploaded_bytes: bytes, filename: str) -> dict[str, Any]:
    """Detect type, extract content, return text + metadata."""
    file_type = detect_file_type(filename)
    if file_type in UNSUPPORTED_FORMAT_MESSAGES:
        raise ValueError(UNSUPPORTED_FORMAT_MESSAGES[file_type])

    suffix = Path(filename).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = Path(tmp.name)

    try:
        if file_type == "resume_pdf":
            result = extract_pdf_text(tmp_path)
        elif file_type == "resume_docx":
            result = extract_docx_text(tmp_path)
        elif file_type == "portfolio_ppt":
            result = extract_ppt_text(tmp_path)
        elif file_type == "text":
            result = extract_txt(tmp_path)
        elif file_type == "image":
            result = extract_image_text(tmp_path)
        elif file_type == "audio":
            result = extract_audio_text(tmp_path)
        elif file_type == "video":
            result = extract_video_text(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        require_extractable_text(result, filename)
        result["file_type"] = file_type
        return result
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_full_analysis(
    merged_context: str,
    file_metas: list[dict],
    candidate_name: str,
    job_keywords: str = "",
) -> dict[str, Any]:
    """Run all Groq-powered analyzers on merged profile."""
    # Communication uses audio/video transcripts only
    comm_parts = []
    for m in file_metas:
        t = m.get("metadata", {}).get("type", "")
        if t in ("voice_intro", "intro_video", "interview_recording"):
            comm_parts.append(m.get("text", ""))
    comm_transcript = "\n".join(comm_parts)

    summary = generate_candidate_summary(merged_context)
    skills = extract_skills(merged_context)
    resume = analyze_resume(merged_context)
    communication = analyze_communication(comm_transcript, has_audio=bool(comm_transcript))
    ats = check_ats(merged_context, job_keywords)
    questions = generate_interview_questions(merged_context, role=job_keywords)
    scores = score_candidate(merged_context, communication)
    entities = extract_entities(merged_context)

    # Cross-file correlation hint in resume may be enriched
    tags = skills.get("smart_tags", [])
    analysis = {
        "candidate_name": candidate_name,
        "summary": summary,
        "skills": skills,
        "resume_analysis": resume,
        "communication": communication,
        "ats": ats,
        "interview_questions": questions,
        "scores": scores,
        "entities": entities,
        "tags": tags,
        "merged_context": truncate_text(merged_context, 100000),
    }
    return analysis


def persist_candidate(
    candidate_id: str,
    candidate_name: str,
    merged_context: str,
    analysis: dict[str, Any],
    file_metas: list[dict],
) -> None:
    """Store in ChromaDB for indexing and analytics."""
    file_types = [m.get("file_type", "unknown") for m in file_metas]
    scores = analysis.get("scores", {})
    chroma_db.add_candidate_record(
        candidate_id,
        merged_context,
        {
            "candidate": candidate_name,
            "type": "profile",
            "skills": analysis.get("tags", []),
            "experience": str(
                analysis.get("summary", {}).get("experience_summary", "")
            )[:200],
            "overall_score": float(scores.get("overall_score", 0)),
            "hiring_recommendation": scores.get(
                "hiring_recommendation", "Moderate Candidate"
            ),
            "file_types": file_types,
            "tags": analysis.get("tags", []),
            "entities": analysis.get("entities", {}),
            "technical_score": float(scores.get("technical_score", 0)),
            "communication_score": float(scores.get("communication_score", 0)),
        },
    )
