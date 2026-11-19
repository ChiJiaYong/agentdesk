"""Graph-level tests for the support agent — fully offline via scripted model.

Three scenarios map 1:1 to the eval risk categories:
1. normal flow with a real order id (tool result grounds the answer)
2. invented order id → guardrail rejects → model recovers by asking
3. consequential write (create_ticket) → needs_human=True (HITL gate)
"""

from __future__ import annotations

import sqlite3

import pytest
from langchain_core.messages import AIMessage

from agents.support.graph import build_support_agent
from core.retrieval import Chunk, FakeHashEmbedder, HybridRetriever
from tools.business_tools import BusinessToolbox

REAL_ID = "e481f51cbdc54678b7cc49136f2d6af7"  # real Olist order id (W2 DB)


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


def _model(script):
    from core.fake_chat import ScriptedToolChatModel

    return ScriptedToolChatModel(messages=iter(script))


def test_normal_flow_grounds_answer_in_tool_result(toolbox, retriever) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_order_status", "args": {"order_id": REAL_ID}, "id": "c1"}],
        ),
        AIMessage(content=f"Your order {REAL_ID} has been delivered. Anything else?"),
    ]
    agent = build_support_agent(_model(script), retriever, toolbox)
    state = agent.invoke(
        {"messages": [("user", f"Where is my order {REAL_ID}?")]},
        config={"recursion_limit": 25},
    )
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", "") == "tool"]
    assert any("delivered" in (m.content or "") for m in tool_msgs)
    assert state["needs_human"] is False
    assert state["user_order_ids"] == {REAL_ID}


def test_invented_order_id_is_rejected_then_recovered(toolbox, retriever) -> None:
    fabricated = "b" * 32  # looks valid but the customer never sent it
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_order_status", "args": {"order_id": fabricated}, "id": "c1"}],
        ),
        AIMessage(content="Could you tell me your order id so I can check it for you?"),
    ]
    agent = build_support_agent(_model(script), retriever, toolbox)
    state = agent.invoke(
        {"messages": [("user", "My order has a problem!!")]},  # no id provided
        config={"recursion_limit": 25},
    )
    rejections = [
        m
        for m in state["messages"]
        if getattr(m, "type", "") == "tool" and "not provided by the user" in (m.content or "")
    ]
    assert rejections, "guardrail must reject the invented id"
    assert state["user_order_ids"] == set()
    final = state["messages"][-1].content
    assert "order id" in final.lower()


def test_write_action_escalates_to_human_review(toolbox, retriever) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {
                        "order_id": REAL_ID,
                        "issue": "item arrived damaged",
                        "idempotency_key": "kb-123",
                    },
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="I've drafted an after-sales request; a human agent will review it."),
    ]
    agent = build_support_agent(_model(script), retriever, toolbox)
    state = agent.invoke(
        {"messages": [("user", f"Item from order {REAL_ID} arrived damaged, I want a refund")]},
        config={"recursion_limit": 25},
    )
    assert state["needs_human"] is True
    assert "write:create_ticket" in state["risk_flags"]
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", "") == "tool"]
    assert any('"replayed": false' in (m.content or "").lower() for m in tool_msgs)
