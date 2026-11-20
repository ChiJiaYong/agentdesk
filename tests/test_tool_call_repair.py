"""Tool-call repair tests (book §5.1.4 — Harness 规范化响应侧格式漂移)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from core.tool_call_repair import repair_tool_calls, set_valid_tools

set_valid_tools(["get_order_status", "create_ticket"])


def test_repairs_double_brace_json_content() -> None:
    # llama-server --jinja + Qwen2.5 observed failure mode
    msg = AIMessage(
        content='{{"name": "get_order_status", '
        '"arguments": {"order_id": "e481f51cbdc54678b7cc49136f2d6af7"}}}'
    )
    out = repair_tool_calls(msg)
    assert out.tool_calls, "must promote text-JSON to a native tool_call"
    assert out.tool_calls[0]["name"] == "get_order_status"
    assert out.content == ""


def test_repairs_fenced_json_content() -> None:
    msg = AIMessage(
        content='```json\n{"name": "create_ticket", "arguments": '
        '{"order_id": "a", "issue": "b", "idempotency_key": "k"}}\n```'
    )
    out = repair_tool_calls(msg)
    assert out.tool_calls and out.tool_calls[0]["name"] == "create_ticket"


def test_rejects_unknown_tool_name() -> None:
    msg = AIMessage(content='{"name": "delete_everything", "arguments": {}}')
    out = repair_tool_calls(msg)
    assert not out.tool_calls, "unknown tool names must NOT be promoted"


def test_plain_content_untouched() -> None:
    msg = AIMessage(content="Your order has been delivered.")
    out = repair_tool_calls(msg)
    assert out.content == "Your order has been delivered."
    assert not out.tool_calls
    assert out.content == msg.content


def test_native_tool_calls_untouched() -> None:
    msg = AIMessage(
        content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "x"}, "id": "1"}]
    )
    out = repair_tool_calls(msg)
    assert out.tool_calls[0]["name"] == "get_order_status"


def test_repaired_flag_in_metadata() -> None:
    msg = AIMessage(content='{"name": "get_order_status", "arguments": {"order_id": "x"}}')
    out = repair_tool_calls(msg)
    assert out.additional_kwargs.get("repaired") is True
