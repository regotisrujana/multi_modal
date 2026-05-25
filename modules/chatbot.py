"""
AI HR Assistant Chatbot.
Uses Groq directly with processed extracted context — NOT RAG retrieval.
"""

from __future__ import annotations

from utils.helpers import groq_context_limit, groq_generate, truncate_text


def chat_with_context(
    user_message: str,
    candidate_context: str,
    analysis_snapshot: str = "",
) -> str:
    """
    Answer HR questions using full candidate context passed directly to Groq.
    No retrieval chains or vector search for answers.
    """
    context_block = truncate_text(candidate_context, groq_context_limit())
    analysis_block = truncate_text(analysis_snapshot, groq_context_limit(4000))

    system = """You are an AI HR Assistant for a recruitment platform.
Answer questions about the candidate using ONLY the provided context and analysis.
Be professional, concise, and actionable. If information is missing, say so clearly.
Do not invent credentials or experience not supported by the context."""

    prompt = f"""{system}

=== CANDIDATE EXTRACTED CONTEXT ===
{context_block}

=== PRIOR ANALYSIS (if any) ===
{analysis_block}

=== USER QUESTION ===
{user_message}

Provide a helpful HR-focused answer."""

    return groq_generate(prompt)
