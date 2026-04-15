"""
scorer.py — LangChain + Ollama (Mistral 7B) resume scoring.

Falls back to a rule-based scorer if Ollama is not running,
so the app still works during development without a GPU.
"""

import re, json

# ── Prompt template ─────────────────────────────────────────────────────────
SCORE_PROMPT = """You are an expert HR recruiter and resume reviewer.

Analyze the following resume data and return a JSON object ONLY (no extra text).

Resume Data:
- Name: {name}
- Skills: {skills}
- Education: {education}
- Experience (years): {experience_years}
- Summary section: {summary}
- Experience section: {experience}

Return this exact JSON structure:
{{
  "overall_score": <integer 0-100>,
  "section_scores": {{
    "skills": <integer 0-100>,
    "experience": <integer 0-100>,
    "education": <integer 0-100>,
    "presentation": <integer 0-100>
  }},
  "strengths": [<list of 3 strengths as strings>],
  "improvements": [<list of 3 improvement suggestions as strings>],
  "ats_keywords_missing": [<list of 3-5 important missing keywords>],
  "summary": "<2-sentence overall assessment>"
}}"""


# ── LLM call via LangChain + Ollama ────────────────────────────────────────
def llm_score(parsed: dict) -> dict | None:
    try:
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import PromptTemplate

        llm = OllamaLLM(model="phi3", temperature=0.1)
        prompt = PromptTemplate.from_template(SCORE_PROMPT)
        chain = prompt | llm

        response = chain.invoke({
            "name":             parsed.get("name", ""),
            "skills":           ", ".join(parsed.get("skills", [])),
            "education":        ", ".join(parsed.get("education", [])),
            "experience_years": parsed.get("experience_years", 0),
            "summary":          parsed.get("sections", {}).get("summary", "")[:500],
            "experience":       parsed.get("sections", {}).get("experience", "")[:800],
        })

        # strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
        return json.loads(cleaned)

    except Exception as e:
        print(f"[scorer] LLM unavailable ({e}), using rule-based fallback.")
        return None


# ── Rule-based fallback ─────────────────────────────────────────────────────
def rule_based_score(parsed: dict) -> dict:
    skills  = parsed.get("skills", [])
    edu     = parsed.get("education", [])
    exp_yrs = parsed.get("experience_years", 0)
    contact = parsed.get("contact", {})

    skill_score = min(100, len(skills) * 5)
    exp_score   = min(100, exp_yrs * 12)
    edu_score   = 80 if any(d in ["B.TECH","BTECH","B.E","BSC","BACHELOR"] for d in edu) else (
                  95 if any(d in ["M.TECH","MTECH","MBA","MASTER","PHD"] for d in edu) else 50)
    present_score = 60
    if contact.get("email"):    present_score += 10
    if contact.get("phone"):    present_score += 10
    if contact.get("linkedin"): present_score += 10
    if contact.get("github"):   present_score += 10
    present_score = min(100, present_score)

    overall = int(skill_score*0.35 + exp_score*0.30 + edu_score*0.20 + present_score*0.15)

    strengths, improvements, missing_kw = [], [], []

    if len(skills) >= 10: strengths.append(f"Strong skill set with {len(skills)} technical skills listed.")
    elif len(skills) >= 5: strengths.append("Good range of technical skills.")
    else: improvements.append("Add more technical skills relevant to your target role.")

    if exp_yrs >= 3: strengths.append(f"{exp_yrs} years of professional experience — solid track record.")
    elif exp_yrs >= 1: strengths.append("Has professional work experience.")
    else: improvements.append("Highlight internships, freelance work, or personal projects to show experience.")

    if any(d in edu for d in ["B.TECH","BTECH","M.TECH","MTECH","PHD","MBA"]):
        strengths.append("Relevant engineering/technical degree.")
    else:
        improvements.append("Consider adding your degree details clearly.")

    if not contact.get("linkedin"): improvements.append("Add your LinkedIn profile URL for recruiter visibility.")
    if not contact.get("github"):   improvements.append("Add your GitHub profile to showcase your projects.")

    if "docker" not in [s.lower() for s in skills]:    missing_kw.append("Docker")
    if "aws" not in [s.lower() for s in skills]:       missing_kw.append("AWS")
    if "rest api" not in [s.lower() for s in skills]:  missing_kw.append("REST API")
    if "agile" not in [s.lower() for s in skills]:     missing_kw.append("Agile")
    if "sql" not in [s.lower() for s in skills]:       missing_kw.append("SQL")

    # keep lists at 3 items max for clean UI
    strengths    = (strengths    + ["Clear and structured layout."])[:3]
    improvements = (improvements + ["Quantify achievements with numbers and metrics.",
                                    "Write a concise professional summary at the top."])[:3]

    return {
        "overall_score": overall,
        "section_scores": {
            "skills":       skill_score,
            "experience":   exp_score,
            "education":    edu_score,
            "presentation": present_score,
        },
        "strengths":            strengths,
        "improvements":         improvements,
        "ats_keywords_missing": missing_kw[:5],
        "summary": (
            f"{parsed.get('name','The candidate')} has {exp_yrs} year(s) of experience "
            f"with {len(skills)} identified technical skills. "
            f"Overall ATS score is {overall}/100."
        ),
        "scored_by": "rule-based (Ollama not available)",
    }


# ── Public API ───────────────────────────────────────────────────────────────
def score_resume(parsed: dict) -> dict:
    result = llm_score(parsed)
    if result is None:
        result = rule_based_score(parsed)
        result["scored_by"] = "rule-based (Ollama not available)"
    else:
        result["scored_by"] = "Mistral 7B via Ollama"

    # always attach parsed metadata
    result["name"]             = parsed.get("name", "")
    result["skills"]           = parsed.get("skills", [])
    result["education"]        = parsed.get("education", [])
    result["experience_years"] = parsed.get("experience_years", 0)
    result["contact"]          = parsed.get("contact", {})
    return result
