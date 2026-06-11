"""从 ``.storyline/skills/*/SKILL.md`` 动态加载技能为 LangChain 工具。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SKILLS_DIR = PROJECT_ROOT / ".storyline" / "skills"


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    title: str
    description: str
    body: str
    path: Path

    @property
    def tool_name(self) -> str:
        return f"skill_{self.skill_id}"


def load_skills(skills_dir: Path | str | None = None) -> list[SkillSpec]:
    root = Path(skills_dir) if skills_dir else DEFAULT_SKILLS_DIR
    if not root.exists():
        return []
    specs: list[SkillSpec] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        spec = _parse_skill(skill_md)
        if spec:
            specs.append(spec)
    return specs


def build_skill_tools(skills: list[SkillSpec]) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    for spec in skills:

        def _make_invoke(skill: SkillSpec = spec):
            def invoke_skill() -> str:
                return json.dumps(
                    {
                        "skill_id": skill.skill_id,
                        "title": skill.title,
                        "instructions": skill.body,
                        "summary": f"已加载技能「{skill.title}」，请按其流程编排工具。",
                    },
                    ensure_ascii=False,
                )

            invoke_skill.__name__ = skill.tool_name
            invoke_skill.__doc__ = skill.description or f"加载并执行技能：{skill.title}"
            return invoke_skill

        tools.append(StructuredTool.from_function(_make_invoke()))
    return tools


def skills_prompt_section(skills: list[SkillSpec]) -> str:
    if not skills:
        return ""
    lines = ["可用声明式技能（调用 skill_<id> 获取详细流程）："]
    for spec in skills:
        lines.append(f"- {spec.tool_name}：{spec.title} — {spec.description}")
    return "\n".join(lines)


def _parse_skill(path: Path) -> SkillSpec | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    skill_id = path.parent.name
    title = skill_id
    description = ""
    body = text

    front_matter = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if front_matter:
        meta_block, body = front_matter.group(1), front_matter.group(2).strip()
        for line in meta_block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip(), value.strip()
                if key == "title":
                    title = value
                elif key == "description":
                    description = value
    else:
        first_line = body.splitlines()[0] if body else ""
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip()

    if not description:
        description = body.splitlines()[0][:120] if body else title

    return SkillSpec(
        skill_id=skill_id,
        title=title,
        description=description,
        body=body,
        path=path,
    )
