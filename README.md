# AI Multimodal Recruitment Analyzer

Production-ready **multimodal HR recruitment platform** built with Streamlit, Groq, ChromaDB, Whisper, and EasyOCR. Analyze resumes, portfolios, voice intros, videos, images, and more — then generate scores, interview questions, and downloadable HR reports.

![Stack](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![Groq](https://img.shields.io/badge/Groq-LLM-orange)

## Features

- **10+ file types**: PDF, DOCX, PPT, TXT, images, certificates, audio, video, LinkedIn screenshots
- **AI insights**: summaries, skills, resume/ATS analysis, communication scoring, interview questions, hiring reports
- **ChromaDB**: persistent candidate index, duplicate detection, analytics (not RAG)
- **HR chatbot**: Groq + full extracted context (no retrieval chains)
- **Exports**: PDF/TXT reports, interview questions, JSON downloads
- **Dashboard**: Plotly, Altair, Matplotlib analytics

## Current status

The app is ready for local demos with text-based PDF, DOCX, PPTX, TXT,
image, audio, and video inputs. The upload flow intentionally rejects legacy
`.doc` and `.ppt` files because the Python parsers used by this project do not
reliably support those older binary formats; convert them to DOCX, PPTX, or PDF
before upload.

Use the **System Health** page in the sidebar before a demo. It checks whether
the Groq API key is configured, whether FFmpeg is available for audio/video
processing, and where local assets/exports are stored.

## Project structure

```
multimodal/
├── app/
│   ├── main.py              # Streamlit UI
│   ├── pipeline.py          # Ingestion + analysis orchestration
│   ├── requirements.txt
│   ├── README.md            # Detailed setup (this file duplicated below)
│   ├── .env.example
│   ├── render.yaml
│   └── .streamlit/config.toml
├── modules/                 # Processors & AI modules
├── vectorstore/             # ChromaDB (storage only)
├── utils/
├── assets/
├── exports/
└── chroma_data/             # Created at runtime
```

## Quick start (local)

### 1. Prerequisites

- **Python 3.11**
- **FFmpeg** (required for Whisper / video audio)
  - Windows: `winget install FFmpeg` or download from https://ffmpeg.org
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### 2. Clone & virtual environment

```bash
cd multimodal
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r app/requirements.txt
```

> First run downloads **sentence-transformers**, **EasyOCR**, and **Whisper** models (several GB). Allow time and disk space.

### 4. Configure Groq API

```bash
copy app\.env.example .env
# Edit .env and set:
# GROQ_API_KEY=your_key_from_https://console.groq.com/keys
```

### 5. Run the app

```bash
cd app
streamlit run main.py
```

Open **http://localhost:8501**

### 6. Run smoke tests

```bash
python -m unittest discover -s tests
python -m compileall app modules utils vectorstore tests
```

## Sample test files

Create small fixtures to test each pipeline branch:

| File | Purpose | Example content |
|------|---------|-----------------|
| `sample_resume.pdf` | PDF extraction | Export any 1-page resume PDF |
| `sample_resume.docx` | DOCX | Word resume with Experience & Skills sections |
| `sample_portfolio.pptx` | PPT | 3 slides: About Me, Projects, Skills |
| `sample_bio.txt` | TXT | Plain-text career summary |
| `sample_linkedin.png` | OCR | Screenshot of a LinkedIn profile |
| `sample_cert.jpg` | Certificate OCR | Photo/scan of a certificate |
| `sample_intro.mp3` | Voice intro | 30–60s self-introduction recording |
| `sample_intro.mp4` | Video intro | Short webcam intro with speech |
| `sample_interview.wav` | Interview audio | Mock Q&A recording |

**Tips**

- Use **clear speech** for audio/video tests (Whisper `base` model).
- Keep PDFs **text-based** (not scanned-only) for fastest PDF parsing; use images for scan tests.
- Name files descriptively (`jane_voice_intro.mp3`) so metadata types detect correctly.

Place samples in `assets/` (optional) and upload via the UI.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model (see [Groq models](https://console.groq.com/docs/models)) |
| `WHISPER_MODEL` | No | `base` | Whisper size: tiny, base, small, medium |

## Render deployment

1. Push the repo to GitHub.
2. On [Render](https://render.com), create a **Web Service**.
3. Set **Root Directory** to `app`.
4. **Build command:** `pip install -r requirements.txt`
5. **Start command:**
   ```bash
   streamlit run main.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
6. Add environment variable `GROQ_API_KEY`.
7. Set `PYTHON_VERSION` to `3.11.9`.
8. For free tier, set `WHISPER_MODEL=tiny` to reduce memory.

Or use the included `app/render.yaml` with Render Blueprint.

**Render limitations:** Free instances have limited RAM. EasyOCR, Whisper, and Torch may fail or be slow. For demos, prefer **PDF + TXT + images** on Render; run full multimodal tests locally.

## ChromaDB usage (not RAG)

ChromaDB stores:

- Embeddings (`all-MiniLM-L6-v2`) for duplicate/similarity checks
- Candidate metadata for dashboards
- Document snippets for the indexed candidates viewer

The **chatbot does not query ChromaDB for answers** — it passes extracted text directly to Groq.

## Reset database

Use **Analytics → Reset database** in the sidebar, or delete the `chroma_data/` folder locally.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GROQ_API_KEY not set` | Create `.env` in project root or `app/` |
| Whisper/ffmpeg errors | Install FFmpeg and restart terminal |
| EasyOCR slow/fails | First run downloads models; ensure disk space |
| Out of memory | Set `WHISPER_MODEL=tiny`, test fewer files at once |
| Render timeout | Use smaller files; upgrade plan or run locally |

## License

Educational / student project use. Verify compliance with [Groq API terms](https://groq.com/terms) before production use.

## Authors

Built as a modular, deployable student-capable HR AI platform.
