"""
agent.py - The "brain" of the agent.

A chatbot answers once. An agent works in a loop:
    think -> pick a tool -> read the result -> think again -> ... -> final answer
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_FUNCTIONS, TOOLS

load_dotenv()

# Works with any OpenAI-compatible provider that has a free tier (Groq, Gemini, ...).
# Set these three values in your .env file.
MODEL = os.getenv("LLM_MODEL")
_client = None


def get_client() -> OpenAI:
    """Create the LLM client once, the first time we need it."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    return _client

MAX_STEPS = 6  # safety limit so the agent can never loop forever

SYSTEM_PROMPT = """You are an AI Placement Agent for Indian engineering students.
Your goal: help the student find the best-matching jobs and get ready to apply.

Steps:
1. Call search_jobs with the student's skills, preferred roles and batch.
2. Call skill_gap_report for the top jobs you found.
3. Call get_job_details for the single best job.
4. Write the final answer in Markdown with three sections:
   ## Top Matches - each job with match score, why it fits, and apply link
   ## Skill Gaps - top 3 skills to learn next and why
   ## Referral Message - a short, polite LinkedIn message for the best job

Rules: only mention jobs returned by the tools. Never invent companies, links or numbers."""


def extract_profile(resume_text: str) -> dict:
    """One LLM call that turns messy resume text into clean JSON."""
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Extract the candidate profile from this resume. Reply with JSON only, keys: "
                "name, batch (graduation year as string), skills (list of short lowercase "
                "skill names), preferred_roles (string), projects (list of short strings).")},
            {"role": "user", "content": resume_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def run_agent(profile: dict, on_step=print) -> str:
    """The agent loop. `on_step` lets the UI show each step live."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is my profile: {json.dumps(profile)}"},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = get_client().chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message

        # No tool requested -> the agent is done, this is the final answer
        if not message.tool_calls:
            return message.content

        messages.append(message.model_dump(exclude_none=True))  # remember what the agent asked for
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            on_step(f"Step {step}: calling `{call.function.name}`")
            result = TOOL_FUNCTIONS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Stopped after the step limit. Try again with a clearer resume."


# ---------- Demo mode: no API key needed ----------
# Runs the SAME tools in a fixed order, so you can try the app before getting a key.
# The real agent above lets the LLM choose the steps and write the report.

KNOWN_SKILLS = ["python", "sql", "git", "fastapi", "django", "java", "javascript", "react", "html",
                "css", "docker", "aws", "dsa", "pandas", "machine learning", "llm", "node.js"]


def demo_extract_profile(resume_text: str) -> dict:
    """Simple keyword-based profile extraction (no LLM)."""
    import re
    text = resume_text.lower()
    batch = re.search(r"20\d\d", resume_text)
    return {
        "name": resume_text.splitlines()[0].strip() if resume_text else "Student",
        "batch": batch.group(0) if batch else "",
        "skills": [s for s in KNOWN_SKILLS if re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", text)],
        "preferred_roles": "python ai",
    }


def run_demo_agent(profile: dict, on_step=print) -> str:
    """Calls the real tools in a fixed order and fills a simple report template."""
    from tools import get_job_details, search_jobs, skill_gap_report

    on_step("Step 1: calling `search_jobs`")
    jobs = json.loads(search_jobs(profile["skills"], profile["preferred_roles"], profile["batch"], top_k=5))
    on_step("Step 2: calling `skill_gap_report`")
    gaps = json.loads(skill_gap_report(profile["skills"], [j["job_id"] for j in jobs]))
    on_step("Step 3: calling `get_job_details`")
    best = json.loads(get_job_details(jobs[0]["job_id"]))

    lines = ["## Top Matches"]
    for j in jobs[:3]:
        lines.append(f"- **{j['title']}**, {j['company']} ({j['location']}): **{j['match_score']}% match**. "
                     f"You have: {', '.join(j['matched_skills'])}. [Apply]({j['apply_link']})")
    lines.append("\n## Skill Gaps")
    for g in gaps[:3]:
        lines.append(f"- **{g['skill']}**: needed in {g['missing_in_jobs']} of your top {g['out_of']} jobs")
    lines.append("\n## Referral Message")
    lines.append(f"> Hi! I'm a {profile['batch']} batch student skilled in {', '.join(profile['skills'][:4])}. "
                 f"I'm applying for the {best['title']} role at {best['company']} and would be grateful "
                 f"for a referral. Happy to share my resume and projects. Thank you!")
    return "\n".join(lines)
