"""Deterministic rule-based demo model — the zero-key demo tier.

Why: the personal-site demo must chat WITHOUT any API key (no key exposure,
no cost, no rate limits). This model implements the same BaseChatModel +
bind_tools contract, with routing rules that exercise the real graph:

  - order id present + "status/where/status/delivered" → get_order_status
  - order id + money words                             → get_order_amount
  - refund/return words  (id required)                 → create_ticket
  - else                                               → ask for what's missing

When DEEPSEEK_API_KEY is configured, service/main.py swaps this for a real
provider — the graph, tools, guardrails and traces are untouched.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

ORDER_RX = re.compile(r"\b[0-9a-f]{32}\b")
MONEY_RX = re.compile(r"how much|paid|price|amount|多少钱|金额", re.I)
REFUND_RX = re.compile(r"refund|return|broken|damaged|退款|退货|坏了|破损", re.I)
STATUS_RX = re.compile(r"status|where|shipped|delivered|物流|到哪|状态|进度", re.I)


class DemoSupportModel(GenericFakeChatModel):
    """Scripted-style model whose 'script' is generated from the input rules."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> DemoSupportModel:  # noqa: ANN401
        return self

    def _rules_reply(self, messages: list) -> AIMessage:
        # Termination: after tool results come back, produce a final answer
        # instead of re-issuing the same call (loop guard — a stateless rules
        # model would otherwise spin until recursion_limit).
        # NB: the graph appends an <agent_status> HumanMessage at the END of the
        # input (book §2.6) — skip it when finding the real last turn message.
        i = len(messages) - 1
        while i >= 0 and str(getattr(messages[i], "content", "")).startswith("<agent_status>"):
            i -= 1
        last = messages[i] if i >= 0 else None
        if getattr(last, "type", "") == "tool":
            return self._summarize_tool_result(str(getattr(last, "content", "")))

        user_text = " ".join(
            str(getattr(m, "content", ""))
            for m in messages
            if getattr(m, "type", "") == "human"
            and not str(getattr(m, "content", "")).startswith("<agent_status>")
        )
        order_ids = ORDER_RX.findall(user_text)
        order_id = order_ids[-1] if order_ids else None

        if order_id and REFUND_RX.search(user_text):
            issue = (
                "refund requested" if re.search(r"refund|退款", user_text, re.I) else "item damaged"
            )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "order_id": order_id,
                            "issue": issue,
                            "idempotency_key": f"demo-{uuid.uuid5(uuid.NAMESPACE_URL, order_id + issue)}",
                        },
                        "id": "demo-call-ticket",
                    }
                ],
            )
        if order_id and MONEY_RX.search(user_text):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order_amount",
                        "args": {"order_id": order_id},
                        "id": "demo-call-amount",
                    }
                ],
            )
        if order_id and STATUS_RX.search(user_text):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order_status",
                        "args": {"order_id": order_id},
                        "id": "demo-call-status",
                    }
                ],
            )
        if order_id:
            return AIMessage(
                content=(
                    "I can check the status, the paid amount, or open an after-sales request "
                    "for this order. What would you like to do?"
                )
            )
        if REFUND_RX.search(user_text):
            return AIMessage(
                content=(
                    "Sorry to hear that! Could you share your order id (it looks like a "
                    "32-character code from your order page) so I can look into it?"
                )
            )
        return AIMessage(
            content=(
                "Hi! I can help with order status, payment amounts and after-sales requests. "
                "Please share your order id and what you need."
            )
        )

    def _summarize_tool_result(self, content: str) -> AIMessage:
        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return AIMessage(content="Let me know if there's anything else I can help with.")
        if data.get("status") == "unknown":
            return AIMessage(
                content=(
                    "Our system is temporarily unavailable — I can't confirm the result right now. "
                    "Please try again shortly."
                )
            )
        if data.get("status") == "error":
            return AIMessage(
                content=(
                    "I couldn't find that in our system. Please double-check the details and try again."
                )
            )
        if data.get("tool") == "create_ticket":
            tid = (data.get("data") or {}).get("ticket_id")
            return AIMessage(
                content=(
                    f"I've drafted an after-sales request (ticket #{tid}) — a human agent will "
                    "review it before anything is executed."
                )
            )
        return AIMessage(
            content=("Here's what our records show — let me know if you need anything else.")
        )

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:  # noqa: ANN401
        return self._rules_reply(list(messages))
