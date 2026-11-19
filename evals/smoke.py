"""CI eval smoke: 5 canned tasks through the real loop code path, zero network.

Exit code 1 if any task fails — CI `eval-smoke` job gates on this. Task/verdict
shapes here foreshadow the W5 harness: (input, expectations, evidence), the same
"promise-style, falsifiable" assertions recommended in the eval methodology we
adopt from book Ch7.
"""

from __future__ import annotations

import sys

from core.mini_loop import FakeChatClient, run_agent_loop

Script = list[dict]

# Each task: user input + the scripted model behavior + falsifiable expectations
# on the *trace* (not just the final text).
TASKS: list[dict] = [
    {
        "name": "order_status_via_tool",
        "user": "Where is my order ORD-1001?",
        "script": [
            {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "get_order_status", "arguments": {"order_id": "ORD-1001"}}
                ],
            },
            {
                "content": "Your order ORD-1001 has shipped and is expected on 2026-11-25.",
                "tool_calls": [],
            },
        ],
        "expect": {"tool_called": "get_order_status", "answer_contains": "shipped"},
    },
    {
        "name": "policy_question_no_tool_needed",
        "user": "What is your refund policy?",
        "script": [
            {
                "content": None,
                "tool_calls": [{"id": "c2", "name": "get_refund_policy", "arguments": {}}],
            },
            {
                "content": "Unused items can be returned within 30 days for a full refund.",
                "tool_calls": [],
            },
        ],
        "expect": {"tool_called": "get_refund_policy", "answer_contains": "30 days"},
    },
    {
        "name": "missing_order_id_must_not_guess",
        "user": "Where is my order?",
        "script": [
            # Correct behavior: answer by asking for the id, WITHOUT calling tools
            # (no fabricated order id — mirrors the "参数来源校验" principle).
            {"content": "Could you provide your order id?", "tool_calls": []},
        ],
        "expect": {"tool_called": None, "answer_contains": "order id"},
    },
    {
        "name": "unknown_order_reports_not_found",
        "user": "Check ORD-9999 please",
        "script": [
            {
                "content": None,
                "tool_calls": [
                    {"id": "c3", "name": "get_order_status", "arguments": {"order_id": "ORD-9999"}}
                ],
            },
            {
                "content": "I could not find order ORD-9999. Please double-check the id.",
                "tool_calls": [],
            },
        ],
        "expect": {"tool_called": "get_order_status", "answer_contains": "not find"},
    },
    {
        "name": "two_step_task",
        "user": "What's the status of ORD-1002 and can I still return it?",
        "script": [
            {
                "content": None,
                "tool_calls": [
                    {"id": "c4", "name": "get_order_status", "arguments": {"order_id": "ORD-1002"}}
                ],
            },
            {
                "content": None,
                "tool_calls": [{"id": "c5", "name": "get_refund_policy", "arguments": {}}],
            },
            {
                "content": "ORD-1002 was delivered on 2026-11-10; within the 30-day window you can return it.",
                "tool_calls": [],
            },
        ],
        "expect": {"tool_called": "get_refund_policy", "answer_contains": "30-day"},
    },
]


def run_smoke() -> int:
    failures: list[str] = []
    for task in TASKS:
        client = FakeChatClient(task["script"])
        result = run_agent_loop(client, task["user"])
        expect = task["expect"]

        # NB: after run_agent_loop, tool_calls in the trace are stored in wire
        # format {"id", "type", "function": {"name", "arguments"}}.
        tool_names = [
            tc["function"]["name"]
            for msg in result["messages"]
            for tc in (msg.get("tool_calls") or [])
        ]
        ok = True
        if expect["tool_called"] is None:
            ok &= tool_names == []
        else:
            ok &= expect["tool_called"] in tool_names
        ok &= expect["answer_contains"] in result["answer"]
        ok &= not result.get("aborted")

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {task['name']}")
        if not ok:
            failures.append(task["name"])

    print(f"\nsmoke: {len(TASKS) - len(failures)}/{len(TASKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_smoke())
