"""Run with:  pytest -q   (no API key needed, the LLM is faked)"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from tools import get_job_details, read_resume_text, search_jobs, skill_gap_report  # noqa: E402

SKILLS = ["Python", "SQL", "Git", "FastAPI"]


def test_search_jobs_ranks_best_match_first():
    jobs = json.loads(search_jobs(SKILLS, batch="2027"))
    assert jobs[0]["job_id"] == "J010"  # Python Developer Intern: 4 of 5 skills
    assert jobs[0]["match_score"] == 80
    assert all(j["match_score"] >= jobs[i + 1]["match_score"] for i, j in enumerate(jobs[:-1]))


def test_batch_filter_excludes_other_years():
    ids = [j["job_id"] for j in json.loads(search_jobs(SKILLS, batch="2027", top_k=20))]
    assert "J009" not in ids  # 2026-only job


def test_skill_gap_counts_missing_skills():
    gaps = json.loads(skill_gap_report(SKILLS, ["J010", "J006"]))
    docker = next(g for g in gaps if g["skill"] == "docker")
    assert docker["missing_in_jobs"] == 2 and docker["out_of"] == 2


def test_unknown_job_returns_error():
    assert "error" in json.loads(get_job_details("NOPE"))


def test_sample_resume_is_readable():
    text = read_resume_text(Path(__file__).resolve().parent.parent / "data" / "sample_resume.pdf")
    assert "Python" in text


# ---------- fake LLM to test the agent loop ----------

def _tool_call(call_id, name, args):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, type="function", function=fn)


class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls]}


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


def test_agent_loop_calls_tools_then_answers(monkeypatch):
    script = [
        FakeMessage(content=None, tool_calls=[_tool_call("1", "search_jobs", {"skills": SKILLS, "batch": "2027"})]),
        FakeMessage(content=None, tool_calls=[_tool_call("2", "skill_gap_report", {"skills": SKILLS, "job_ids": ["J010"]})]),
        FakeMessage(content="## Top Matches\nJ010", tool_calls=None),
    ]
    fake = FakeClient(script)
    monkeypatch.setattr(agent, "get_client", lambda: fake)
    steps = []
    answer = agent.run_agent({"skills": SKILLS}, on_step=steps.append)
    assert answer.startswith("## Top Matches")
    assert len(steps) == 2


def test_agent_stops_at_step_limit(monkeypatch):
    loop_forever = [FakeMessage(content=None, tool_calls=[_tool_call(str(i), "get_job_details", {"job_id": "J001"})])
                    for i in range(agent.MAX_STEPS)]
    fake = FakeClient(loop_forever)
    monkeypatch.setattr(agent, "get_client", lambda: fake)
    assert "step limit" in agent.run_agent({"skills": SKILLS}, on_step=lambda s: None)


def test_demo_mode_report_uses_real_tool_results():
    text = read_resume_text(Path(__file__).resolve().parent.parent / "data" / "sample_resume.pdf")
    profile = agent.demo_extract_profile(text)
    assert profile["batch"] == "2027" and "fastapi" in profile["skills"]
    report = agent.run_demo_agent(profile, on_step=lambda s: None)
    assert "Python Developer Intern" in report and "docker" in report
