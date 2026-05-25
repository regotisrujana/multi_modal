"""
AI Multimodal Recruitment Analyzer — Streamlit HR Dashboard
Run: streamlit run main.py  (from app/ directory)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root on path for modules/, utils/, vectorstore/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    persist_candidate,
    process_file,
    run_full_analysis,
)
from modules.chatbot import chat_with_context
from modules.report_generator import (
    build_full_report_text,
    export_interview_questions,
    export_pdf_report,
    export_text_report,
)
from utils.helpers import (
    EXPORTS_DIR,
    SUPPORTED_EXTENSIONS,
    get_system_health,
    merge_contexts,
    timestamp_now,
)
from vectorstore import chroma_db

# ---------------------------------------------------------------------------
# Page config & session state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Multimodal Recruitment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 12px; color: white;
    }
    .rec-strong { color: #059669; font-weight: 700; }
    .rec-moderate { color: #d97706; font-weight: 700; }
    .rec-weak { color: #dc2626; font-weight: 700; }
    div[data-testid="stSidebar"] { background-color: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE = {
    "analysis": None,
    "merged_context": "",
    "file_metas": [],
    "candidate_name": "Candidate",
    "candidate_id": None,
    "chat_history": [],
    "job_keywords": "",
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val


def rec_class(rec: str) -> str:
    if "Strong" in rec:
        return "rec-strong"
    if "Needs" in rec:
        return "rec-weak"
    return "rec-moderate"


def render_score_cards(scores: dict):
    cols = st.columns(5)
    metrics = [
        ("Technical", scores.get("technical_score", 0)),
        ("Communication", scores.get("communication_score", 0)),
        ("Resume Quality", scores.get("resume_quality_score", 0)),
        ("Confidence", scores.get("confidence_score", 0)),
        ("Overall", scores.get("overall_score", 0)),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, f"{val}/100")
    rec = scores.get("hiring_recommendation", "Moderate Candidate")
    st.markdown(
        f'<p class="{rec_class(rec)}">Hiring Recommendation: {rec}</p>',
        unsafe_allow_html=True,
    )


def page_upload():
    st.header("📤 Upload Candidate Materials")
    st.caption(
        "PDF/DOCX resumes, PPT portfolios, images, certificates, "
        "voice intros, interview audio, videos, TXT, LinkedIn screenshots"
    )

    candidate_name = st.text_input(
        "Candidate name",
        value=st.session_state.candidate_name,
        placeholder="e.g. Jane Doe",
    )
    job_keywords = st.text_input(
        "Target role keywords (optional, for ATS)",
        value=st.session_state.job_keywords,
        placeholder="Python, React, data analyst",
    )

    uploaded = st.file_uploader(
        "Drag and drop files here",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    )

    if uploaded:
        if not isinstance(uploaded, list):
            uploaded = [uploaded]
        
        with st.status("Processing files...", expanded=True) as status:
            texts, metas = [], []
            errors = []
            for i, uf in enumerate(uploaded):
                status.write(f"Extracting content from: {uf.name}...")
                try:
                    data = process_file(uf.getvalue(), uf.name)
                    texts.append(data.get("text", ""))
                    metas.append(
                        {
                            "text": data.get("text", ""),
                            "metadata": data.get("metadata", {}),
                            "file_type": data.get("file_type"),
                            "filename": uf.name,
                        }
                    )
                except Exception as exc:
                    errors.append(f"**{uf.name}**: {exc}")

            if errors:
                status.update(label="Extraction complete with some errors", state="error", expanded=True)
                for e in errors:
                    st.error(e)
            else:
                status.update(label=f"Successfully extracted {len(metas)} file(s)", state="complete")

        if texts:
            merged = merge_contexts(texts)
            st.session_state.merged_context = merged
            st.session_state.file_metas = metas
            st.session_state.candidate_name = candidate_name or "Candidate"
            st.session_state.job_keywords = job_keywords

            # Duplicate detection
            dup_hash = chroma_db.check_duplicate(merged)
            similar = chroma_db.find_similar_candidates(merged)
            if dup_hash:
                st.warning(
                    f"Possible duplicate: existing record "
                    f"({dup_hash['metadata'].get('candidate', 'unknown')})"
                )
            elif similar:
                st.info(
                    "Similar candidates found: "
                    + ", ".join(
                        f"{s['metadata'].get('candidate', s['id'])} "
                        f"({s['similarity']:.0%})"
                        for s in similar[:3]
                    )
                )

            st.success(f"Extracted {len(metas)} file(s). Ready for AI analysis.")
            with st.expander("Preview merged context"):
                st.text_area("Context", merged[:8000], height=200)

            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing with Groq (may take 1–2 minutes)..."):
                    try:
                        analysis = run_full_analysis(
                            merged,
                            metas,
                            st.session_state.candidate_name,
                            job_keywords,
                        )
                        cid = chroma_db.new_candidate_id()
                        persist_candidate(
                            cid,
                            st.session_state.candidate_name,
                            merged,
                            analysis,
                            metas,
                        )
                        st.session_state.analysis = analysis
                        st.session_state.candidate_id = cid
                        st.success("Analysis complete and saved to database.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")


def page_dashboard():
    st.header("📊 Candidate Dashboard")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Upload files and run analysis to see the dashboard.")
        return

    scores = analysis.get("scores", {})
    render_score_cards(scores)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Candidate Summary")
        summary = analysis.get("summary", {})
        st.write(summary.get("summary", "—"))
        st.markdown(f"**Education:** {summary.get('education_summary', '—')}")
        st.markdown(f"**Experience:** {summary.get('experience_summary', '—')}")
        st.markdown(f"**Projects:** {summary.get('project_overview', '—')}")
        st.markdown(f"**Technical:** {summary.get('technical_profile', '—')}")

    with col2:
        st.subheader("Strengths & Weaknesses")
        st.write("**Strengths:**", scores.get("strengths", []))
        st.write("**Weaknesses:**", scores.get("weaknesses", []))
        st.write("**Suggested roles:**", scores.get("suggested_roles", []))
        st.write("**Improvements:**", scores.get("improvement_suggestions", []))

    # Skill chart
    skills = analysis.get("skills", {})
    skill_rows = []
    for cat, label in [
        ("programming_languages", "Languages"),
        ("frameworks", "Frameworks"),
        ("tools", "Tools"),
        ("soft_skills", "Soft Skills"),
    ]:
        for s in skills.get(cat, []):
            skill_rows.append({"skill": s, "category": label})
    if skill_rows:
        df = pd.DataFrame(skill_rows)
        fig = px.bar(
            df,
            x="skill",
            color="category",
            title="Extracted Skills",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    timeline = scores.get("timeline", {})
    if timeline:
        st.subheader("Candidate Timeline")
        tcols = st.columns(3)
        for col, key, title in zip(
            tcols,
            ["education", "internships", "projects"],
            ["Education", "Internships", "Projects"],
        ):
            col.markdown(f"**{title}**")
            for item in timeline.get(key, []):
                col.write(f"• {item}")

    # Portfolio
    pa = scores.get("portfolio_analysis", {})
    if pa:
        st.subheader("Portfolio Analysis")
        st.json(pa)

    # Tags
    tags = analysis.get("tags", [])
    if tags:
        st.markdown("**Smart Tags:** " + " · ".join(f"`{t}`" for t in tags))


def page_communication():
    st.header("🎙️ Communication Analysis")
    comm = (st.session_state.analysis or {}).get("communication", {})
    if not comm:
        st.info("No communication analysis yet.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", comm.get("speaking_confidence", 0))
    c2.metric("Clarity", comm.get("communication_clarity", 0))
    c3.metric("Score", comm.get("communication_score", 0))
    st.write(comm.get("professional_tone", ""))
    st.write("**Filler words:**", comm.get("filler_word_frequency", ""))
    st.write("**Tips:**", comm.get("improvement_tips", []))


def page_resume_ats():
    st.header("📄 Resume & ATS")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Run analysis first.")
        return
    resume = analysis.get("resume_analysis", {})
    ats = analysis.get("ats", {})
    st.subheader("Resume Insights")
    st.metric("Resume Quality", resume.get("resume_quality_score", "—"))
    st.write(resume.get("formatting_notes", ""))
    st.write("**Missing sections:**", resume.get("missing_sections", []))
    st.write("**Cross-file correlations:**", resume.get("cross_file_correlations", []))
    st.subheader("ATS Check")
    st.metric("ATS Score", ats.get("ats_score", "—"))
    st.write("ATS Friendly:", ats.get("is_ats_friendly", "—"))
    st.write("**Matches:**", ats.get("keyword_matches", []))
    st.write("**Missing:**", ats.get("missing_keywords", []))
    st.write("**Recommendations:**", ats.get("recommendations", []))


def page_interview():
    st.header("❓ Interview Questions")
    q = (st.session_state.analysis or {}).get("interview_questions", {})
    if not q:
        st.info("Run analysis to generate questions.")
        return
    for key, title in [
        ("technical_questions", "Technical"),
        ("hr_questions", "HR"),
        ("project_based_questions", "Project-based"),
        ("scenario_questions", "Scenario"),
    ]:
        st.subheader(title)
        for item in q.get(key, []):
            st.write(f"• {item}")


def page_chatbot():
    st.header("💬 AI HR Assistant")
    st.caption("Uses Groq with full extracted context — not RAG retrieval.")
    if not st.session_state.merged_context:
        st.warning("Upload and analyze a candidate first.")
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompts = [
        "Summarize this candidate",
        "What are the strongest skills?",
        "Generate interview questions",
        "What are the weaknesses?",
        "Is this candidate ATS friendly?",
        "Suggest suitable job roles",
    ]
    st.write("Quick prompts:")
    cols = st.columns(3)
    for i, p in enumerate(prompts):
        if cols[i % 3].button(p, key=f"qp_{i}"):
            st.session_state["_pending_prompt"] = p

    user_input = st.chat_input("Ask about this candidate...")
    pending = st.session_state.pop("_pending_prompt", None)
    question = pending or user_input

    if question:
        snapshot = json.dumps(st.session_state.analysis or {}, default=str)[:15000]
        with st.spinner("Thinking..."):
            answer = chat_with_context(
                question,
                st.session_state.merged_context,
                snapshot,
            )
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


def page_analytics():
    st.header("📈 Analytics & Database")
    stats = chroma_db.get_database_stats()
    st.metric("Total indexed candidates", stats["total_candidates"])

    candidates = chroma_db.list_all_candidates()
    if candidates:
        rows = []
        for c in candidates:
            m = c.get("metadata", {})
            rows.append(
                {
                    "id": c["id"][:8] + "...",
                    "name": m.get("candidate", "—"),
                    "overall": m.get("overall_score", 0),
                    "technical": m.get("technical_score", 0),
                    "communication": m.get("communication_score", 0),
                    "recommendation": m.get("hiring_recommendation", "—"),
                }
            )
        df = pd.DataFrame(rows)

        # Plotly scores
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Technical", x=df["name"], y=df["technical"]))
        fig.add_trace(go.Bar(name="Communication", x=df["name"], y=df["communication"]))
        fig.add_trace(go.Bar(name="Overall", x=df["name"], y=df["overall"]))
        fig.update_layout(barmode="group", title="Candidate Score Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Altair skill distribution from tags
        tag_counts = stats.get("tag_counts", {})
        if tag_counts:
            tdf = pd.DataFrame(
                [{"tag": k, "count": v} for k, v in tag_counts.items()]
            )
            chart = (
                alt.Chart(tdf)
                .mark_bar()
                .encode(x=alt.X("tag", sort="-y"), y="count")
                .properties(title="Skill Tag Distribution")
            )
            st.altair_chart(chart, use_container_width=True)

        # Matplotlib pie for recommendations
        rec_counts = df["recommendation"].value_counts()
        if len(rec_counts):
            fig2, ax = plt.subplots()
            ax.pie(rec_counts.values, labels=rec_counts.index, autopct="%1.0f%%")
            ax.set_title("Hiring Recommendations")
            st.pyplot(fig2)
            plt.close(fig2)

        st.dataframe(df, use_container_width=True)

        # Comparison cards
        st.subheader("Candidate Comparison")
        card_cols = st.columns(min(3, len(candidates)))
        for i, c in enumerate(candidates[:3]):
            m = c.get("metadata", {})
            with card_cols[i % 3]:
                st.markdown(f"### {m.get('candidate', 'Unknown')}")
                st.write(f"Overall: **{m.get('overall_score', 0)}**")
                st.write(m.get("hiring_recommendation", ""))
    else:
        st.info("No candidates in database yet.")

    st.subheader("Database actions")
    if st.button("🗑️ Reset database", type="secondary"):
        chroma_db.reset_database()
        st.session_state.analysis = None
        st.session_state.candidate_id = None
        st.success("Database reset.")
        st.rerun()


def page_export():
    st.header("📥 Export Reports")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Complete an analysis to export reports.")
        return

    name = st.session_state.candidate_name
    report_bundle = {
        "summary": analysis.get("summary"),
        "skills": analysis.get("skills"),
        "resume_analysis": analysis.get("resume_analysis"),
        "communication": analysis.get("communication"),
        "scores": analysis.get("scores"),
        "ats": analysis.get("ats"),
        "interview_questions": analysis.get("interview_questions"),
        "entities": analysis.get("entities"),
        "tags": analysis.get("tags"),
    }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export TXT report"):
            p = export_text_report(name, report_bundle)
            st.success(f"Saved: {p}")
        if st.button("Export PDF report"):
            try:
                p = export_pdf_report(name, report_bundle)
                st.success(f"Saved: {p}")
            except Exception as exc:
                st.error(str(exc))
        if st.button("Export interview questions"):
            p = export_interview_questions(
                name, analysis.get("interview_questions", {})
            )
            st.success(f"Saved: {p}")

    with col2:
        full_text = build_full_report_text(name, report_bundle)
        st.download_button(
            "Download full analysis (TXT)",
            full_text,
            file_name=f"{name}_analysis.txt",
            mime="text/plain",
        )
        comm_txt = json.dumps(analysis.get("communication", {}), indent=2)
        st.download_button(
            "Download communication analysis",
            comm_txt,
            file_name=f"{name}_communication.json",
            mime="application/json",
        )
        scores_txt = json.dumps(analysis.get("scores", {}), indent=2)
        st.download_button(
            "Download hiring recommendation",
            scores_txt,
            file_name=f"{name}_hiring.json",
            mime="application/json",
        )

    # List exports folder
    files = sorted(EXPORTS_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if files:
        st.subheader("Recent exports on disk")
        for f in files[:10]:
            with open(f, "rb") as fh:
                st.download_button(
                    f"Download {f.name}",
                    fh.read(),
                    file_name=f.name,
                    key=f"dl_{f.name}",
                )


def page_indexed():
    st.header("🗂️ Indexed Candidates")
    candidates = chroma_db.list_all_candidates()
    for c in candidates:
        m = c.get("metadata", {})
        with st.expander(f"{m.get('candidate', 'Unknown')} — {c['id'][:8]}..."):
            st.json(m)
            st.text(c.get("document", "")[:500] + "...")


def page_system_health():
    st.header("System Health")
    st.caption("Quick readiness checks for local demos and deployments.")
    health = get_system_health()

    rows = []
    for name, item in health.items():
        rows.append(
            {
                "check": name.replace("_", " ").title(),
                "status": "Ready" if item["ok"] else "Needs attention",
                "detail": item["detail"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not health["groq_api_key"]["ok"]:
        st.warning("Set GROQ_API_KEY in the project .env file before running AI analysis.")
    if not health["ffmpeg"]["ok"]:
        st.info("Install FFmpeg before testing audio or video uploads.")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Upload & Analyze": page_upload,
    "Dashboard": page_dashboard,
    "Communication": page_communication,
    "Resume & ATS": page_resume_ats,
    "Interview Questions": page_interview,
    "HR Chatbot": page_chatbot,
    "Analytics": page_analytics,
    "Export Reports": page_export,
    "Indexed Candidates": page_indexed,
    "System Health": page_system_health,
}

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/job-seeker.png", width=64)
    st.title("Recruitment AI")
    st.caption("Multimodal HR Analyzer")
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Session:** {st.session_state.candidate_name}")
    if st.session_state.candidate_id:
        st.caption(f"ID: {st.session_state.candidate_id[:8]}...")
    st.caption(timestamp_now())

st.markdown('<p class="main-header">🎯 AI Multimodal Recruitment Analyzer</p>', unsafe_allow_html=True)
st.caption("Intelligent candidate analysis for HR teams, recruiters, and interviewers")

PAGES[page]()
