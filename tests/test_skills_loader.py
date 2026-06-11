from __future__ import annotations

from pathlib import Path

from travel_agent.skills.loader import build_skill_tools, load_skills, skills_prompt_section


def test_load_bundled_skills():
    root = Path(__file__).resolve().parents[1] / ".storyline" / "skills"
    skills = load_skills(root)
    ids = {s.skill_id for s in skills}
    assert "full_trip_planner" in ids
    assert "rainy_day_alternative" in ids
    assert "structured_planner" in ids


def test_build_skill_tools():
    root = Path(__file__).resolve().parents[1] / ".storyline" / "skills"
    skills = load_skills(root)
    tools = build_skill_tools(skills)
    names = {t.name for t in tools}
    assert "skill_full_trip_planner" in names
    section = skills_prompt_section(skills)
    assert "skill_full_trip_planner" in section
