"""Minimal hand-rolled agent loop — the W1 comprehension exercise.

Reference: 《深入理解AI Agent》v2.0, §2.2.3–2.2.4 ("带工具调用的多轮交互" and
"用代码实现Agent的核心循环").

Key ideas implemented here (each maps to a book concept):

1. Context = a list of messages with 4 roles (system/user/assistant/tool) — §2.2.1.
2. The model decides *which* tool to call; our code *executes* it and appends the
   result as a ``tool`` message — the model never executes anything itself (§2.2.3).
3. The loop ends when the assistant replies without tool calls, or when the
   step budget is exhausted — a minimal guardrail against runaway loops (§1.2.5).
4. The LLM is injected via ``ChatClient`` so tests (and the CI eval smoke) can run
   fully offline with a scripted fake client.

This file is intentionally small (~120 lines) and has no dependencies beyond the
stdlib: every production feature we add in later weeks (LangGraph orchestration,
checkpointing, human-in-the-loop) should be explainable as "what I would have had
to build by hand if this function kept growing".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

# --------------------------------------------------------------------------- #
# Tool definitions: OpenAI function-calling JSON Schema format.
# Book anchor: §2.4.6 工具定义的设计 — description quality directly drives
# tool-selection accuracy, so descriptions state the precondition explicitly.
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Query the current status of ONE order by its exact order id. "
                "Use ONLY an order id the user explicitly provided; never guess one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id exactly as provided by the user.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_policy",
            "description": "Return the refund & return policy text. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Toy business data — replaced in W3 by the real SQLite business DB (Olist subset).
_ORDERS: dict[str, str] = {
    "ORD-1001": "shipped, ETA 2026-11-25",
    "ORD-1002": "delivered 2026-11-10",
}
_REFUND_POLICY = (
    "Unused items may be returned within 30 days of delivery for a full refund. "
    "Refunds are issued to the original payment method within 5 business days."
)

TOOL_IMPLS: dict[str, Callable[..., str]] = {
    "get_order_status": lambda order_id: _ORDERS.get(order_id, f"order {order_id} not found"),
    "get_refund_policy": lambda: _REFUND_POLICY,
}


# --------------------------------------------------------------------------- #
# Chat client protocol + offline fake client
# --------------------------------------------------------------------------- #

Message = dict[str, Any]  # {"role": ..., "content": ...} OpenAI chat format


class ChatClient(Protocol):
    """Minimal chat interface; mirrors what an OpenAI-compatible SDK exposes."""

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return {"content": str|None, "tool_calls": [ {name, arguments(dict), id} ]}."""
        ...


class FakeChatClient:
    """Scripted offline client: returns queued replies in order.

    Lets tests and the CI eval smoke run the *real loop code path* with zero
    network. The loop cannot tell this apart from a real provider — that is the
    point of the Protocol seam.
    """

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append(messages)
        if not self._replies:
            raise AssertionError("FakeChatClient ran out of scripted replies (loop did not stop?)")
        return self._replies.pop(0)


# --------------------------------------------------------------------------- #
# The loop itself
# --------------------------------------------------------------------------- #


def run_agent_loop(
    client: ChatClient,
    user_message: str,
    system_prompt: str = "You are a concise e-commerce customer-support assistant.",
    tools: dict[str, Callable[..., str]] | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    max_steps: int = 8,
) -> dict[str, Any]:
    """Run one ReAct interaction to completion.

    Returns the full message trace plus a summary; the trace is the seed of the
    Trace/observability module built in W4 (every production run gets persisted
    under ``runs/``).
    """
    tools = tools if tools is not None else TOOL_IMPLS
    tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS

    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for step in range(1, max_steps + 1):
        response = client.chat(messages, tool_schemas)

        assistant_msg: Message = {"role": "assistant", "content": response.get("content")}
        if response.get("tool_calls"):
            # Keep the raw assistant turn in the context (book §2.2.3): the next
            # tool result must pair with the call that produced it.
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                    },
                }
                for tc in response["tool_calls"]
            ]
        messages.append(assistant_msg)

        if not response.get("tool_calls"):
            return {"answer": response.get("content") or "", "messages": messages, "steps": step}

        for tc in response["tool_calls"]:
            impl = tools.get(tc["name"])
            if impl is None:
                result = f"error: unknown tool {tc['name']!r}"
            else:
                try:
                    result = str(impl(**tc["arguments"]))
                except Exception as exc:  # noqa: BLE001 — toy tools; W3 adds three-state results
                    result = f"error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # Step budget exhausted — fail loudly instead of looping forever (§1.2.5).
    return {
        "answer": "[aborted] step budget exhausted",
        "messages": messages,
        "steps": max_steps,
        "aborted": True,
    }
