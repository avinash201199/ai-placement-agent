# 🤖 AI Placement Agent

An AI agent that reads your resume, finds matching jobs, shows your skill gaps and writes a referral message. Built in one weekend with free tools.

> Made for Let's Code students. More free placement resources: https://www.lets-code.co.in/

## What it does

1. **Reads your resume** (PDF) and turns it into a clean profile (skills, batch, roles).
2. **Searches jobs** and ranks them by match score.
3. **Finds skill gaps**, e.g. "Docker is missing in 2 of your top 3 jobs".
4. **Writes a referral message** for the best job.

## Why it's an agent, not a chatbot

A chatbot answers once. This agent works in a **loop**: it thinks, picks a tool, reads the result and decides the next step until the job is done (with a 6-step safety limit).

```
Resume PDF ──► extract_profile (LLM)
                    │
                    ▼
            ┌── Agent loop (LLM) ◄──┐
            │   decides next step   │
            ▼                       │
   search_jobs / skill_gap_report / get_job_details  (plain Python on real data)
            │                       │
            └──── tool result ──────┘
                    │
                    ▼
     Final report: Top Matches · Skill Gaps · Referral Message
```

**No fake jobs:** all job facts come from Python tools reading `data/jobs.csv`. The LLM only decides what to call and writes the explanation.

## Project structure

| File | Role |
|---|---|
| `tools.py` | The agent's "hands": resume reader, job search, skill gaps, job details |
| `agent.py` | The agent's "brain": profile extraction + the agent loop |
| `app.py` | Streamlit UI |
| `data/jobs.csv` | Sample jobs (replace with real ones) |
| `data/sample_resume.pdf` | Sample resume to try it out |
| `tests/` | Tests with a fake LLM, no API key needed |

## Run it locally

```bash
git clone https://github.com/avinash201199/ai-placement-agent.git
cd ai-placement-agent
pip install -r requirements.txt
cp .env.example .env        # add your free API key + model name
streamlit run app.py
```

**No API key yet?** The app runs in **demo mode**: it calls the same tools in a fixed order so you can see the flow. Add a key to let the AI agent plan the steps and write the report itself.

Works with any **OpenAI-compatible** provider with a free tier, such as Groq or Google Gemini. Pick a model that supports **tool calling** from your provider's docs.

Run tests: `pip install pytest && pytest -q`

## Deploy free on Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**.
2. Upload these files (or push with git).
3. Settings → **Variables and secrets** → add `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
4. Your live link is ready. Put it on your resume.

## Use real jobs

`data/jobs.csv` contains **sample** companies. Replace it with real openings from the [Let's Code Job Board](https://www.lets-code.co.in/jobs/). Keep the same columns, and separate skills with `;`.

## Ideas to extend it

- Fetch live jobs from an API instead of a CSV
- Add a cover letter tool
- Save results and track applications
- Add memory so it remembers past searches

## Resume bullet

> Built an AI agent that parses resumes, matches candidates to job listings using LLM tool calling, and generates skill-gap reports and referral drafts. Deployed on Hugging Face Spaces.

## Learn more (free)

- [AI Engineer Roadmap](https://www.lets-code.co.in/articles/AIEngineerRoadmap/)
- [LLMs Roadmap](https://www.lets-code.co.in/articles/llmsroadmap/)
- [Agentic AI Interview Questions](https://www.lets-code.co.in/interview/agentic-ai-interview-questions/)
- [GenAI, LLM & RAG Interview Questions](https://www.lets-code.co.in/interview/generative-ai-interview-questions/)
- [Cold Email & Referral Templates](https://www.lets-code.co.in/articles/coldemailtemplates/)
- [Final Year Projects](https://www.lets-code.co.in/articles/finalyearprojects/)
