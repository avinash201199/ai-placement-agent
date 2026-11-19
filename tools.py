"""
tools.py - The "hands" of the agent.

Every tool here is plain Python that works on real data (your resume, the jobs CSV).
The LLM decides WHICH tool to call and WHEN, but the facts always come from here.
That is how we stop the agent from inventing fake jobs.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

JOBS_FILE = Path(__file__).parent / "data" / "jobs.csv"


# ---------- Tool 0: read the resume (runs before the agent starts) ----------

def read_resume_text(pdf_file) -> str:
    """Extract raw text from an uploaded resume PDF."""
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


# ---------- Helpers ----------

def _normalise(skill: str) -> str:
    return skill.strip().lower()


def load_jobs(path: Path = JOBS_FILE) -> list[dict]:
    """Load jobs from CSV. Skills are stored as 'python;sql;docker'."""
    with open(path, newline="", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))
    for job in jobs:
        job["skills"] = [_normalise(s) for s in job["skills"].split(";") if s.strip()]
    return jobs


# ---------- Tool 1: search jobs ----------

def search_jobs(skills: list[str], role_keywords: str = "", batch: str = "", top_k: int = 5) -> str:
    """Return the jobs that best match the candidate's skills, best first."""
    candidate = {_normalise(s) for s in skills}
    keywords = [k.lower() for k in role_keywords.split() if k]
    results = []

    for job in load_jobs():
        if batch and batch not in job["batch"]:
            continue  # wrong graduation year, skip it
        matched = [s for s in job["skills"] if s in candidate]
        score = round(100 * len(matched) / len(job["skills"]))
        if keywords and any(k in job["title"].lower().split() for k in keywords):
            score = min(100, score + 10)  # small boost when the role title matches
        results.append({
            "job_id": job["job_id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "match_score": score,
            "matched_skills": matched,
            "apply_link": job["apply_link"],
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    return json.dumps(results[:top_k])


# ---------- Tool 2: skill-gap report ----------

def skill_gap_report(skills: list[str], job_ids: list[str]) -> str:
    """Which skills are missing, and in how many of the shortlisted jobs?"""
    candidate = {_normalise(s) for s in skills}
    jobs = [j for j in load_jobs() if j["job_id"] in job_ids]
    missing = Counter(s for j in jobs for s in j["skills"] if s not in candidate)
    report = [{"skill": s, "missing_in_jobs": n, "out_of": len(jobs)} for s, n in missing.most_common()]
    return json.dumps(report)


# ---------- Tool 3: full details of one job ----------

def get_job_details(job_id: str) -> str:
    """Full details of one job, used to write a tailored referral message."""
    for job in load_jobs():
        if job["job_id"] == job_id:
            return json.dumps(job)
    return json.dumps({"error": f"No job with id {job_id}"})


# ---------- Tool schemas: how we describe the tools to the LLM ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search open jobs and rank them by how well they match the candidate's skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "role_keywords": {"type": "string", "description": "e.g. 'backend python intern'"},
                    "batch": {"type": "string", "description": "Graduation year, e.g. '2027'"},
                    "top_k": {"type": "integer"},
                },
                "required": ["skills"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_gap_report",
            "description": "Find skills the candidate is missing across the shortlisted jobs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "job_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["skills", "job_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": "Get full details of one job before writing a referral message for it.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
]

# Maps the name the LLM asks for -> the real Python function
TOOL_FUNCTIONS = {
    "search_jobs": search_jobs,
    "skill_gap_report": skill_gap_report,
    "get_job_details": get_job_details,
}
