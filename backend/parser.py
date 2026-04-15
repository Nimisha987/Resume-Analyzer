import re
import os

# ── PDF text extraction ─────────────────────────────────────────────────────
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()  #get file extension
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)  #loops through each page, extracts text,joins everything with new line
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Section splitter ────────────────────────────────────────────────────────
SECTION_HEADERS = {
    "experience": r"(work\s*experience|experience|employment|professional\s*experience)",
    "education":  r"(education|academic|qualification)",
    "skills":     r"(skills|technical\s*skills|core\s*competencies|technologies)",
    "projects":   r"(projects|key\s*projects|personal\s*projects)",
    "summary":    r"(summary|objective|profile|about\s*me)",
    "certifications": r"(certifications?|certificates?|courses?)",
}

def split_sections(text: str) -> dict:
    lines = text.splitlines()
    sections = {k: [] for k in SECTION_HEADERS}
    sections["other"] = []  #creates empty sections
    current = "other"
    for line in lines:
        stripped = line.strip()  #removes spaces, tabs, newlines
        matched = False
        for sec, pattern in SECTION_HEADERS.items():
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 60:
                current = sec
                matched = True
                break
        if not matched:
            sections[current].append(stripped)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


# ── Name extraction ─────────────────────────────────────────────────────────
def extract_name(text: str) -> str:
    try:
        import spacy #can detect names, places etc.
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:500])   #only first 500 characters used
        for ent in doc.ents:   #detected entities like person,organization,location
            if ent.label_ == "PERSON":
                return ent.text.strip()  #removes extra space and give name
    except Exception:
        pass
    # fallback: first non-empty line that looks like a name
    for line in text.splitlines():
        line = line.strip()
        if line and len(line.split()) in (2, 3) and line.replace(" ", "").isalpha():    #only letter allowed
            return line
    return "Unknown"


# ── Contact extraction ──────────────────────────────────────────────────────
def extract_contact(text: str) -> dict:
    email   = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.IGNORECASE)
    phone   = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    linkedin = re.search(r"linkedin\.com/in/[\w-]+", text, re.IGNORECASE)
    github   = re.search(r"github\.com/[\w-]+", text, re.IGNORECASE)
    return {
        "email":    email.group()    if email    else "",
        "phone":    phone.group()    if phone    else "",
        "linkedin": linkedin.group() if linkedin else "",
        "github":   github.group()   if github   else "",
    }


# ── Skills extraction ───────────────────────────────────────────────────────
KNOWN_SKILLS = [
    # Languages
    "python","java","javascript","typescript","c++","c#","go","rust","kotlin","swift","php","ruby","r","scala","matlab",
    # Web
    "html","css","react","angular","vue","node.js","express","django","flask","fastapi","spring","laravel",
    # Data / ML
    "machine learning","deep learning","nlp","computer vision","tensorflow","pytorch","keras","scikit-learn",
    "pandas","numpy","matplotlib","seaborn","opencv","transformers","hugging face","langchain","langgraph",
    "llm","rag","faiss","chromadb","spacy","nltk","bert","gpt","llama","mistral",
    # Data engineering
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","spark","hadoop","kafka","airflow",
    # Cloud / DevOps
    "aws","azure","gcp","docker","kubernetes","ci/cd","git","github","gitlab","jenkins","terraform","linux",
    # Other
    "rest api","graphql","microservices","agile","scrum","tableau","power bi","excel",
]

def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))  # preserve order, deduplicate


# ── Experience extraction ───────────────────────────────────────────────────
def extract_experience_years(text: str) -> int:
    years_pattern = re.findall(r"(\d{4})\s*[-–]\s*(\d{4}|present|current)", text, re.IGNORECASE)
    import datetime
    current_year = datetime.datetime.now().year
    total = 0
    for start, end in years_pattern:
        s = int(start)
        e = current_year if end.lower() in ("present", "current") else int(end)
        if 1980 <= s <= current_year and s <= e:
            total += (e - s)
    return min(total, 40)  # cap at 40


# ── Education extraction ────────────────────────────────────────────────────
DEGREES = ["phd","ph.d","doctorate","m.tech","mtech","m.e","mba","m.sc","msc","master","b.tech","btech","b.e","b.sc","bsc","bachelor","diploma","10th","12th"]

def extract_education(text: str) -> list:
    text_lower = text.lower()
    found = []
    for deg in DEGREES:
        if deg in text_lower:
            found.append(deg.upper())
    return list(dict.fromkeys(found))


# ── Main parse function ─────────────────────────────────────────────────────
def parse_resume(file_path: str) -> dict:
    raw_text = extract_text(file_path)
    sections = split_sections(raw_text)

    skills_text = sections.get("skills", "") + "\n" + sections.get("other", "")
    all_text    = raw_text

    return {
        "raw_text":        raw_text[:4000],   # truncate for LLM context
        "name":            extract_name(raw_text),
        "contact":         extract_contact(raw_text),
        "skills":          extract_skills(all_text),
        "education":       extract_education(all_text),
        "experience_years": extract_experience_years(all_text),
        "sections":        {k: v[:800] for k, v in sections.items()},
    }
