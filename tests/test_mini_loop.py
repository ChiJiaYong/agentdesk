"""Offline tests for the hand-rolled loop: tool execution, termination, budget."""

from __future__ import annotations

from core.mini_loop import FakeChatClient, run_agent_loop


def test_tool_call_then_final_answer() -> None:
    client = FakeChatClient(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "get_order_status", "arguments": {"order_id": "ORD-1001"}}
                ],
            },
            {"content": "shipped, ETA 2026-11-25", "tool_calls": []},
        ]
    )
    result = run_agent_loop(client, "Where is my order ORD-1001?")

    assert not result.get("aborted")
    assert result["steps"] == 2
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = result["messages"][3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert "shipped" in tool_msg["content"]


def test_unknown_tool_returns_error_not_crash() -> None:
    client = FakeChatClient(
        [
            {"content": None, "tool_calls": [{"id": "c1", "name": "nope", "arguments": {}}]},
            {"content": "handled", "tool_calls": []},
        ]
    )
    result = run_agent_loop(client, "do something")
    tool_msg = result["messages"][3]
    assert tool_msg["content"].startswith("error: unknown tool")


def test_step_budget_aborts() -> None:
    # A model that never stops calling tools must hit the budget (guardrail).
    endless = {"content": None, "tool_calls": [{"id": "x", "name": "nope", "arguments": {}}]}
    client = FakeChatClient([dict(endless) for _ in range(20)])
    result = run_agent_loop(client, "loop forever", max_steps=4)
    assert result["aborted"] is True
    assert result["steps"] == 4


def test_tool_exception_is_caught() -> None:
    def boom(**_: str) -> str:
        raise ValueError("boom")

    client = FakeChatClient(
        [
            {"content": None, "tool_calls": [{"id": "c1", "name": "boom", "arguments": {}}]},
            {"content": "ok", "tool_calls": []},
        ]
    )
    result = run_agent_loop(client, "go", tools={"boom": boom})
    assert "boom" in result["messages"][3]["content"]
