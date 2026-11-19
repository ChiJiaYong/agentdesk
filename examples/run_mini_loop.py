"""Demo entry: run the hand-rolled loop offline with the scripted fake client.

Usage:  uv run python examples/run_mini_loop.py
Swap FakeChatClient for an OpenAI-compatible client (W3) by setting API keys.
"""

from __future__ import annotations

import json

from core.mini_loop import FakeChatClient, run_agent_loop

REPLIES = [
    # Turn 1: model asks for the order status via a tool call
    {
        "content": None,
        "tool_calls": [
            {"id": "call_1", "name": "get_order_status", "arguments": {"order_id": "ORD-1001"}}
        ],
    },
    # Turn 2: model answers in plain text — loop stops
    {
        "content": "Your order ORD-1001 has shipped and is expected on 2026-11-25.",
        "tool_calls": [],
    },
]


def main() -> None:
    client = FakeChatClient(REPLIES)
    result = run_agent_loop(client, "Where is my order ORD-1001?")
    print(json.dumps(result["messages"], indent=2, ensure_ascii=False))
    print("\n=== final answer ===")
    print(result["answer"])
    print(f"steps used: {result['steps']}")


if __name__ == "__main__":
    main()
