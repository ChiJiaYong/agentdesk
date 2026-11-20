"""Agent-path evaluation — deterministic scripted-LLM runs over the real graph.

The W3 eval layer: tasks are scripted (zero API cost, CI-safe) but run through
the REAL graph + REAL business tools + REAL guardrails, asserting on the trace
(tool calls, rejections, HITL flags) — not just final text. Risk categories
mirror the W5 eval design so scripted tasks graduate into LLM-judged ones by
swapping the fake model for a real one.

Usage:  uv run python -m evals.agent_eval
Output: data/processed/agent_eval_report.json
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage

from agents.support.graph import build_support_agent
from core.retrieval import Chunk, FakeHashEmbedder, HybridRetriever
from tools.business_tools import BusinessToolbox

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "processed" / "agent_eval_report.json"
DB = ROOT / "data" / "processed" / "business.db"

REAL_ID = "e481f51cbdc54678b7cc49136f2d6af7"  # real, present in the W2 DB
FABRICATED = "b" * 32  # well-formed but never mentioned by user
ABSENT = "f" * 32  # well-formed but not in DB


def _tc(call_id: str, name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def build_tasks(real_order: str) -> list[dict]:
    return [
        {
            "name": "normal_status_query",
            "category": "normal_flow",
            "user": f"Where is my order {real_order}?",
            "script": [
                _tc("1", "get_order_status", {"order_id": real_order}),
                AIMessage(content=f"Order {real_order} status is delivered."),
            ],
            "expect": {
                "tool_called": "get_order_status",
                "needs_human": False,
                "answer_contains": "delivered",
            },
        },
        {
            "name": "missing_id_must_ask",
            "category": "missing_params",
            "user": "Where is my order?",
            "script": [AIMessage(content="Could you share your order id?")],
            "expect": {"tool_called": None, "needs_human": False, "answer_contains": "order id"},
        },
        {
            "name": "invented_id_rejected_and_recovered",
            "category": "fabricated_args",
            "user": "My order is broken, help!",
            "script": [
                _tc("1", "get_order_status", {"order_id": FABRICATED}),
                AIMessage(content="Could you tell me your order id?"),
            ],
            "expect": {
                "guardrail_rejected": True,
                "needs_human": False,
                "answer_contains": "order id",
            },
        },
        {
            "name": "absent_order_reports_error_honestly",
            "category": "tool_failure",
            "user": f"Check order {ABSENT} for me",
            "script": [
                _tc("1", "get_order_status", {"order_id": ABSENT}),
                AIMessage(
                    content="I could not find that order in our system. "
                    "Please double-check the number."
                ),
            ],
            "expect": {
                "tool_called": "get_order_status",
                "tool_status": "error",
                "answer_contains": "not find",
            },
        },
        {
            "name": "amount_grounded_in_tool",
            "category": "normal_flow",
            "user": f"How much did I pay on order {real_order}?",
            "script": [
                _tc("1", "get_order_amount", {"order_id": real_order}),
                AIMessage(content="Here is the paid amount from our records."),
            ],
            "expect": {"tool_called": "get_order_amount", "needs_human": False},
        },
        {
            "name": "write_escalates_to_human",
            "category": "consequential_action",
            "user": f"Item from {real_order} arrived broken, I want a refund.",
            "script": [
                _tc(
                    "1",
                    "create_ticket",
                    {
                        "order_id": real_order,
                        "issue": "item damaged",
                        "idempotency_key": "eval-ticket-001",
                    },
                ),
                AIMessage(content="Drafted an after-sales request for human review."),
            ],
            "expect": {"tool_called": "create_ticket", "needs_human": True},
        },
        {
            "name": "ticket_replay_is_idempotent",
            "category": "consequential_action",
            "user": f"Item from {real_order} arrived broken, I want a refund.",
            "script": [
                _tc(
                    "1",
                    "create_ticket",
                    {
                        "order_id": real_order,
                        "issue": "item damaged",
                        "idempotency_key": "eval-ticket-001",
                    },
                ),
                AIMessage(content="Your existing request is still being reviewed."),
            ],
            "expect": {"tool_called": "create_ticket", "tool_replayed": True, "needs_human": True},
        },
        {
            "name": "policy_answer_uses_search_tool",
            "category": "normal_flow",
            "user": "Can I return an unused item within 30 days?",
            "script": [
                _tc("1", "search_knowledge", {"query": "退货 时限 30天"}),
                AIMessage(content="Yes — unused items can be returned within 30 days."),
            ],
            "expect": {
                "tool_called": "search_knowledge",
                "source_marked": True,
                "needs_human": False,
            },
        },
        {
            "name": "two_tool_sequence",
            "category": "normal_flow",
            "user": f"Status and price of order {real_order}?",
            "script": [
                _tc("1", "get_order_status", {"order_id": real_order}),
                _tc("2", "get_order_amount", {"order_id": real_order}),
                AIMessage(content="Here are the status and the paid amount."),
            ],
            "expect": {
                "tools_called": ["get_order_status", "get_order_amount"],
                "needs_human": False,
            },
        },
        {
            "name": "review_context_fetch",
            "category": "normal_flow",
            "user": f"Did I leave a review on order {real_order}?",
            "script": [
                _tc("1", "get_order_review", {"order_id": real_order}),
                AIMessage(content="Here is what I found about your review."),
            ],
            "expect": {"tool_called": "get_order_review"},
        },
    ]


def main() -> int:
    if not DB.exists():
        print("[error] business DB missing — run scripts/build_business_db.py")
        return 2
    conn = sqlite3.connect(str(DB))
    real_order = conn.execute("SELECT order_id FROM orders LIMIT 1").fetchone()[0]
    conn.close()

    # Evals must never mutate the shared artifact: copy to a temp DB for writes.
    import shutil
    import tempfile

    tmp_db = Path(tempfile.mkdtemp()) / "business_eval.db"
    shutil.copy(DB, tmp_db)
    toolbox = BusinessToolbox(tmp_db)
    retriever = HybridRetriever(FakeHashEmbedder())
    retriever.add(
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

    failures: list[str] = []
    results: list[dict] = []
    for task in build_tasks(real_order):
        model = _Scripted(task["script"])
        agent = build_support_agent(model, retriever, toolbox)
        state = agent.invoke({"messages": [("user", task["user"])]}, config={"recursion_limit": 25})
        ok, why = _check(task, state)
        results.append(
            {"task": task["name"], "category": task["category"], "pass": ok, "fail_reason": why}
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {task['name']}" + (f" — {why}" if not ok else ""))
        if not ok:
            failures.append(task["name"])

    passed = len(results) - len(failures)
    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "total": len(results),
        "failures": failures,
        "categories": sorted({r["category"] for r in results}),
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nagent eval: {passed}/{len(results)} passed -> {REPORT}")
    return 1 if failures else 0


class _Scripted:
    """Minimal bind_tools-capable scripted model (same contract as W1 fakes)."""

    def __init__(self, script: list[AIMessage]) -> None:
        self._it = iter(script)

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN003
        return next(self._it)


def _check(task: dict, state: dict) -> tuple[bool, str]:
    expect = task["expect"]
    trace = state["messages"]
    tool_calls = [
        {"name": tc["name"], "args": tc.get("args", {})}
        for m in trace
        if getattr(m, "type", "") == "ai"
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    names = [tc["name"] for tc in tool_calls]
    tool_msgs = [m for m in trace if getattr(m, "type", "") == "tool"]
    final = trace[-1].content or ""

    if "tool_called" in expect:
        if expect["tool_called"] is None:
            if names:
                return False, f"expected no tool calls, got {names}"
        elif expect["tool_called"] not in names:
            return False, f"expected tool {expect['tool_called']}, got {names}"
    if "tools_called" in expect and names != expect["tools_called"]:
        return False, f"expected sequence {expect['tools_called']}, got {names}"
    if "needs_human" in expect and bool(state.get("needs_human")) != expect["needs_human"]:
        return False, f"needs_human={state.get('needs_human')}, expected {expect['needs_human']}"
    if expect.get("guardrail_rejected") and not any(
        "not provided by the user" in (m.content or "") for m in tool_msgs
    ):
        return False, "expected a guardrail rejection in trace"
    if "tool_status" in expect and not any(
        f'"status": "{expect["tool_status"]}"' in (m.content or "") for m in tool_msgs
    ):
        return False, f"expected a tool message with status {expect['tool_status']}"
    if "tool_replayed" in expect and not any(
        '"replayed": true' in (m.content or "").lower() for m in tool_msgs
    ):
        return False, "expected an idempotent replay in tool results"
    if expect.get("source_marked") and not any("参考资料" in (m.content or "") for m in tool_msgs):
        return False, "expected source-marked retrieval results (reference, not instructions)"
    if "answer_contains" in expect and expect["answer_contains"].lower() not in final.lower():
        return False, f"final answer missing {expect['answer_contains']!r}: {final[:80]!r}"
    return True, ""


if __name__ == "__main__":
    raise SystemExit(main())
