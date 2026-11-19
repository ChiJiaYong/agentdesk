"""create_agent (LangChain 1.x) runs offline via the scripted tool-capable model."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from core.fake_chat import ScriptedToolChatModel


def get_order_status(order_id: str) -> str:
    """Query the current status of ONE order by its exact order id."""
    orders = {"ORD-1001": "shipped, ETA 2026-11-25"}
    return orders.get(order_id, f"order {order_id} not found")


def test_react_agent_completes_with_tool_call() -> None:
    scripted = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order_status", "args": {"order_id": "ORD-1001"}, "id": "c1"}
                ],
            ),
            AIMessage(content="Your order ORD-1001 has shipped."),
        ]
    )
    agent = create_agent(
        ScriptedToolChatModel(messages=scripted),
        tools=[get_order_status],
        system_prompt="You are a concise e-commerce customer-support assistant.",
    )

    result = agent.invoke(
        {"messages": [("user", "Where is my order ORD-1001?")]},
        config={"recursion_limit": 10},
    )

    types = [m.type for m in result["messages"]]
    assert "tool" in types, f"expected a tool message in trace, got {types}"
    final = result["messages"][-1].content
    assert "shipped" in final
