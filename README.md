# ResumeIQ — AI Resume Analyzer
### CA2 Hackathon Project | Generative AI (IT315E)

**Tech Stack:** FastAPI · spaCy · LangChain · Mistral 7B (Ollama) · sentence-transformers · FAISS · Plain HTML/CSS/JS

---

## Project Structure

```
resume-analyzer/
├── backend/
│   ├── main.py          ← FastAPI app (all routes)
│   ├── parser.py        ← PDF/text parser using PyMuPDF + spaCy NER
│   ├── scorer.py        ← LLM scoring using LangChain + Mistral 7B
│   ├── matcher.py       ← JD matching using sentence-transformers + cosine similarity
│   └── requirements.txt
└── frontend/
    ├── index.html       ← Upload page
    └── results.html     ← Results dashboard
```

---

## STEP 1 — Install Python

Make sure Python 3.10 or above is installed.

Open terminal in VS Code (`Ctrl+`` ` ``) and check:
```
python --version
```

---

## STEP 2 — Install Backend Dependencies

In VS Code terminal, navigate to the backend folder:
```
cd resume-analyzer/backend
```

Create a virtual environment (recommended):
```
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

Install all packages:
```
pip install -r requirements.txt
```

Download the spaCy English model:
```
python -m spacy download en_core_web_sm
```

---

## STEP 3 — Install Ollama + Mistral (for AI scoring)

> Skip this step if you just want to test with rule-based scoring — the app still works without Ollama.

1. Download Ollama from: https://ollama.com/download
2. Install it and then open a NEW terminal and run:
```
ollama pull mistral
```
3. Keep Ollama running in the background (it starts automatically after install).

---

## STEP 4 — Run the Backend

In VS Code terminal (with venv activated, inside `backend/` folder):
```
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Test it by opening: http://localhost:8000

---

## STEP 5 — Open the Frontend

Two ways to open the frontend in VS Code:

**Option A (recommended) — Live Server extension:**
1. Install the "Live Server" extension in VS Code
2. Right-click on `frontend/index.html`
3. Click "Open with Live Server"
4. It opens at http://127.0.0.1:5500

**Option B — Direct file:**
Simply double-click `frontend/index.html` to open in your browser.

---

## STEP 6 — Use the App

1. Open the frontend in your browser
2. Upload a PDF or TXT resume
3. Choose a mode:
   - **Score My Resume** — get overall score, section breakdown, strengths & improvements
   - **Match to Job** — paste a job description and get match % + skill gap analysis
4. Click Analyze Resume
5. View results on the dashboard

---

## API Endpoints (FastAPI)

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | /          | Health check                         |
| POST   | /parse     | Extract text + entities from resume  |
| POST   | /score     | Score resume with LLM                |
| POST   | /match     | Match resume against job description |

Auto-generated docs at: http://localhost:8000/docs

---

## Troubleshooting

**"Connection failed" error in browser:**
→ Backend is not running. Run `python main.py` in the backend folder.

**"Ollama not available" in scored_by field:**
→ Ollama is not installed or Mistral is not pulled. The app uses rule-based scoring as fallback — this is fine for demos.

**CORS error in browser:**
→ Already handled. If it persists, open `frontend/index.html` via Live Server (not file://).

**spaCy model not found:**
→ Run `python -m spacy download en_core_web_sm`

---

## Tools Disclosed (CA2 Requirement)

| Tool | Purpose |
|------|---------|
| FastAPI | Backend REST API framework |
| PyMuPDF (fitz) | PDF text extraction |
| spaCy (en_core_web_sm) | Named Entity Recognition for name extraction |
| LangChain | LLM orchestration and prompt chaining |
| Ollama + Mistral 7B | Open-source LLM for resume scoring |
| sentence-transformers (all-MiniLM-L6-v2) | Embedding generation for JD matching |
| NumPy | Cosine similarity computation |
| Plain HTML/CSS/JS | Frontend interface |
