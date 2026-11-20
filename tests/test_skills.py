"""Skills 渐进式披露测试（book §2.5：L1 目录常驻 / L2 正文按需 emit-once）。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agents.support.graph import build_support_agent
from core.retrieval import Chunk, FakeHashEmbedder, HybridRetriever
from core.skills import SkillRegistry
from tools.business_tools import BusinessToolbox

REAL_ID = "e481f51cbdc54678b7cc49136f2d6af7"

pytestmark = pytest.mark.skipif(
    not Path("data/processed/business.db").exists(),
    reason="business DB not built (run scripts/build_business_db.py)",
)


@pytest.fixture(scope="module")
def toolbox(tmp_path_factory):
    db = tmp_path_factory.mktemp("biz") / "business.db"
    src = sqlite3.connect("data/processed/business.db")
    dst = sqlite3.connect(str(db))
    src.backup(dst)
    src.close()
    box = BusinessToolbox(db)
    yield box
    box.close()


@pytest.fixture(scope="module")
def retriever():
    r = HybridRetriever(FakeHashEmbedder())
    r.add(
        [
            Chunk(
                id="kb-1",
                text="Unused items may be returned within 30 days for a full refund.",
                doc_id="pol",
                source_url="x",
                source_name="policy",
                license="t",
                fetched_at="2026",
            ),
        ]
    )
    return r


class CapturingModel:
    def __init__(self, script: list[AIMessage]) -> None:
        self._it = iter(script)
        self.seen_inputs: list[list] = []

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.seen_inputs.append(list(messages))
        return next(self._it)


def test_skill_registry_listing_and_load() -> None:
    reg = SkillRegistry()
    assert "refund-policy" in reg.valid_names()
    assert "ticket-triage" in reg.valid_names()
    listing = reg.listing_block()
    assert "refund-policy" in listing and "load_skill" in listing
    assert "复述客户的核心诉求" not in listing, "L1 目录不得包含正文（渐进式披露）"
    content = reg.load("refund-policy")
    assert "处理流程" in content and "NEVER" in content


def test_load_skill_tool_through_graph(toolbox, retriever) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "load_skill", "args": {"name": "refund-policy"}, "id": "c1"}],
        ),
        AIMessage(content="Following the refund-policy skill, I confirmed the issue."),
    ]
    model = CapturingModel(script)
    agent = build_support_agent(model, retriever, toolbox, skill_registry=SkillRegistry())
    state = agent.invoke(
        {"messages": [("user", "I want a refund, item is broken")]}, config={"recursion_limit": 25}
    )
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", "") == "tool"]
    payload = json.loads(tool_msgs[0].content)
    assert payload["status"] == "ok"
    assert "instructions" in payload["data"]
    # L1 常驻清单到达稳定前缀
    assert "refund-policy" in str(model.seen_inputs[0][0].content)


def test_load_skill_unknown_name_errors(toolbox, retriever) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "load_skill", "args": {"name": "nonexistent"}, "id": "c1"}],
        ),
        AIMessage(content="Sorry, I do not have that capability."),
    ]
    agent = build_support_agent(
        CapturingModel(script), retriever, toolbox, skill_registry=SkillRegistry()
    )
    state = agent.invoke({"messages": [("user", "do it")]}, config={"recursion_limit": 25})
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", "") == "tool"]
    assert json.loads(tool_msgs[0].content)["error_code"] == "UNKNOWN_SKILL"
