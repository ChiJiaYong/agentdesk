"""LangGraph/LangChain 1.x quickstart — same task as core/mini_loop.py, framework-orchestrated.

Comparison notes: docs/notes/mini-agent-loop.md

| Hand-rolled (core/mini_loop.py)      | create_agent (this file)                    |
| ------------------------------------ | ------------------------------------------- |
| while-loop over messages             | StateGraph: nodes are functions, typed state |
| step budget as a plain counter       | recursion_limit + graph-level guards         |
| no persistence                       | checkpointers (threads, resume) for free     |
| one file, everything visible         | graph structure inspectable / exportable     |

Decision record: docs/adr/0001-use-langgraph-for-orchestration.md — the hand-rolled
loop stays as the comprehension reference; product agents build on LangChain 1.x
`create_agent` (successor of LangGraph's prebuilt ReAct agent; book v2.0 §1.2.4.4
positions LangChain/LangGraph for "工作流+自主，复杂多步骤工作流").

Runs fully offline with a scripted fake chat model.
Usage:  uv run python -m examples.langgraph_quickstart
"""

from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from core.fake_chat import ScriptedToolChatModel


def get_order_status(order_id: str) -> str:
    """Query the current status of ONE order by its exact order id.

    Use ONLY an order id the user explicitly provided; never guess one.
    """
    orders = {"ORD-1001": "shipped, ETA 2026-11-25", "ORD-1002": "delivered 2026-11-10"}
    return orders.get(order_id, f"order {order_id} not found")


def main() -> None:
    # Same script as the hand-rolled demo, so both traces are directly comparable:
    # turn 1 = tool call, turn 2 = final answer.
    scripted = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order_status", "args": {"order_id": "ORD-1001"}, "id": "call_1"}
                ],
            ),
            AIMessage(content="Your order ORD-1001 has shipped and is expected on 2026-11-25."),
        ]
    )
    model = ScriptedToolChatModel(messages=scripted)

    agent = create_agent(
        model,
        tools=[get_order_status],
        system_prompt="You are a concise e-commerce customer-support assistant.",
    )
    result = agent.invoke(
        {"messages": [("user", "Where is my order ORD-1001?")]},
        config={"recursion_limit": 10},
    )

    for msg in result["messages"]:
        print(f"[{msg.type}] {getattr(msg, 'content', '')!r}")
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            print(f"    tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
