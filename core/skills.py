"""Agent Skills 渐进式披露（book §2.5——三层结构的 L1+L2 落地）。

- L1 目录常驻：所有 skill 的 name+description 渲染成 ~300 token 的清单，
  进入**稳定前缀**（目录不随对话变化，缓存友好）。
- L2 正文按需：模型通过 load_skill 工具调用时，正文作为工具结果注入轨迹——
  注入一次后永久驻留（emit-once，§2.5.3 图 2-13），不重复付费。
- L3 子文档：本实现未用（SKILL.md 均为单文件，触发条件见 BOOK_MAP）。

目录由代码在构建期渲染（绝不让 LLM 统计）；skill 内容来自仓库 skills/ 目录
的 SKILL.md（frontmatter: name/description），Git 版本化、可 diff、可回滚。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKILLS_DIR = ROOT / "skills"

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class Skill:
    name: str
    description: str
    content: str


class SkillRegistry:
    def __init__(self, skills_dir: Path | str = DEFAULT_SKILLS_DIR) -> None:
        self.skills: dict[str, Skill] = {}
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            return
        for path in sorted(skills_dir.glob("*/SKILL.md")):
            raw = path.read_text(encoding="utf-8")
            head, _, body = (
                raw.partition("---\n")[2].partition("---\n")
                if raw.startswith("---")
                else ("", "", raw)
            )
            meta: dict[str, str] = {}
            for line in head.splitlines():
                key, _, value = line.partition(":")
                if key.strip():
                    meta[key.strip()] = value.strip()
            name = meta.get("name", path.parent.name)
            self.skills[name] = Skill(
                name=name,
                description=meta.get("description", ""),
                content=body.strip(),
            )

    # -- L1: 常驻目录（进入稳定前缀） -----------------------------------------

    def listing_block(self) -> str:
        """渲染常驻 skill 清单（无正文——正文按需 load_skill）。"""
        if not self.skills:
            return ""
        lines = ["Available skills (use the load_skill tool to load full instructions):"]
        for skill in self.skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        lines.append(
            "Load a skill BEFORE handling a matching request; its instructions "
            "then apply for the rest of the conversation."
        )
        return "\n".join(lines)

    # -- L2: 正文按需加载 ------------------------------------------------------

    def load(self, name: str) -> str | None:
        skill = self.skills.get(name)
        return skill.content if skill else None

    def valid_names(self) -> list[str]:
        return list(self.skills)
