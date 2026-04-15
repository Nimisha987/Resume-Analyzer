"""
matcher.py — Job description matching using sentence-transformers + cosine similarity.
Falls back to keyword overlap if sentence-transformers is unavailable.
"""

import re


# ── Same known skills list as parser (kept in sync) ─────────────────────────
KNOWN_SKILLS = [
    "python","java","javascript","typescript","c++","c#","go","rust","kotlin","swift","php","ruby","r","scala",
    "html","css","react","angular","vue","node.js","express","django","flask","fastapi","spring",
    "machine learning","deep learning","nlp","computer vision","tensorflow","pytorch","keras","scikit-learn",
    "pandas","numpy","transformers","hugging face","langchain","llm","rag","faiss","chromadb","spacy","nltk",
    "bert","gpt","llama","mistral","sql","mysql","postgresql","mongodb","redis","elasticsearch",
    "spark","hadoop","kafka","aws","azure","gcp","docker","kubernetes","ci/cd","git","linux",
    "rest api","graphql","microservices","agile","scrum","tableau","power bi","excel",
]

def extract_jd_skills(jd_text: str) -> list:   #find skills mentioned in jd
    text_lower = jd_text.lower()   #Makes matching case-insensitive
    return [s for s in KNOWN_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", text_lower)]   #word boundary and handles special chars


# ── Embedding-based similarity ───────────────────────────────────────────────
def embedding_similarity(resume_text: str, jd_text: str) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([resume_text[:1500], jd_text[:1500]])   #converts both texts into numerical vectors and only first 1500 characters are used
        # cosine similarity
        a, b = embeddings[0], embeddings[1]  #a: resume vector, b:job description vector
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception as e:
        print(f"[matcher] Embedding model unavailable ({e}), using keyword fallback.")
        return None


def keyword_similarity(resume_skills: list, jd_skills: list) -> float:
    if not jd_skills:
        return 0.5
    matched = set(s.lower() for s in resume_skills) & set(s.lower() for s in jd_skills)
    return round(len(matched) / len(jd_skills), 4)


# ── Public API ───────────────────────────────────────────────────────────────
def match_job_description(parsed: dict, jd_text: str) -> dict:
    resume_skills = [s.lower() for s in parsed.get("skills", [])]  #get skills from resume
    jd_skills     = extract_jd_skills(jd_text)

    resume_text = (
        " ".join(parsed.get("skills", [])) + " " +  #combines skills,summary,experience and embedding model works on full text not just keywords.
        parsed.get("sections", {}).get("experience", "") + " " +  #if sections missing->return empty dict
        parsed.get("sections", {}).get("summary", "")
    )

    sim = embedding_similarity(resume_text, jd_text)   #compute similarity
    if sim is None:
        sim = keyword_similarity(resume_skills, jd_skills)
        method = "keyword overlap"
    else:
        method = "sentence-transformers (all-MiniLM-L6-v2)"  #uses semantic understanding

    match_percent = round(sim * 100, 1)  #converting to percentage  

    resume_set = set(resume_skills)   #skill comparison
    jd_set     = set(s.lower() for s in jd_skills)
    matched_skills  = sorted(resume_set & jd_set)  #common skills
    missing_skills  = sorted(jd_set - resume_set)   #what resume wants but resume doesn't have
    extra_skills    = sorted(resume_set - jd_set)   #skills not required by JD

    # Fit label
    if match_percent >= 75:   fit = "Excellent Fit"
    elif match_percent >= 55: fit = "Good Fit"
    elif match_percent >= 35: fit = "Partial Fit"
    else:                     fit = "Low Fit"

    return {
        "match_percent":   match_percent,
        "fit_label":       fit,
        "matched_skills":  matched_skills,
        "missing_skills":  missing_skills[:10],
        "extra_skills":    extra_skills[:10],
        "jd_skills":       jd_skills,
        "match_method":    method,
        "recommendation":  _recommend(match_percent, missing_skills),
    }


def _recommend(score: float, missing: list) -> str:
    if score >= 75:
        return "Strong match! Apply confidently and tailor your cover letter."
    elif score >= 55:
        miss_str = ", ".join(missing[:3]) if missing else "a few areas"
        return f"Good match. Strengthen your profile by adding: {miss_str}."
    elif score >= 35:
        miss_str = ", ".join(missing[:4]) if missing else "key skills"
        return f"Partial match. Consider upskilling in: {miss_str} before applying."
    else:
        return "Low match. Significant skill gaps found. Consider a different role or upskill first."
