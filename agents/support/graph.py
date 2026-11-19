"""Support Agent graph — the W3 core deliverable.

Flow (one customer turn):

    load_user_ids → retrieve_kb → plan ⇄ validate ⇄ execute → review_gate → END

Design rules:
- **Guardrails are graph nodes, not prompt text** (book §1.2.5 护栏): the
  `validate` node rejects any tool call whose order_id did not come from the
  user / previous tool output, with a typed error the model can recover from.
- **Write actions route to HITL**: `review_gate` flags create_ticket runs as
  needs_human — W4's workbench renders these drafts for approval.
- **Three-state results** are preserved end-to-end; the prompt explicitly
  forbids asserting success on unknown states.
- The LLM is injected (LangChain BaseChatModel) so the whole graph runs
  offline with ScriptedToolChatModel — tests and CI need no API key.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from core.guardrails import extract_order_ids, validate_tool_args
from tools.business_tools import BusinessToolbox

SYSTEM_PROMPT = """You are the customer-support assistant for a cross-border
e-commerce store. Rules you must never break:

1. Only state amounts, dates and statuses that a tool returned. If a tool
   returned status=unknown, say the system is temporarily unavailable — never
   invent or guess an outcome.
2. Never use an order id that the customer did not provide. If you need one,
   ask for it.
3. Creating an after-sales ticket is a consequential action: confirm the issue
   with the customer in your final answer and note that the request will be
   reviewed by a human agent before execution.
4. Answer refund/return policy questions using the provided knowledge-base
   excerpts and cite which excerpt you used.
"""

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Query real order status (delivered/shipped/canceled/...) by exact order id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Exact order id from the customer.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_amount",
            "description": "Query the real paid amount and payment types for an order. Use this before mentioning any money amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Exact order id from the customer.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_review",
            "description": "Fetch the customer's review score/comment for an order (complaint context).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Exact order id from the customer.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create an after-sales ticket (CONSEQUENTIAL: goes to human review). Requires an idempotency_key you generate as uuid5 of order_id+issue summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Exact order id from the customer.",
                    },
                    "issue": {
                        "type": "string",
                        "description": "One-line issue summary from the customer's words.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Stable key; same key never creates a second ticket.",
                    },
                },
                "required": ["order_id", "issue", "idempotency_key"],
            },
        },
    },
]

WRITE_TOOLS = {"create_ticket"}


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_order_ids: set[str]
    retrieved: list[str]
    needs_human: bool
    risk_flags: list[str]


def build_support_agent(model: Any, retriever: Any, toolbox: BusinessToolbox):
    """Compile the support-agent graph. `model` must support bind_tools()."""

    def load_user_ids(state: SupportState) -> dict:
        user_texts = [m.content for m in state["messages"] if getattr(m, "type", "") == "human"]
        return {"user_order_ids": extract_order_ids(user_texts)}

    def retrieve_kb(state: SupportState) -> dict:
        last = state["messages"][-1]
        query = getattr(last, "content", "") or ""
        if not isinstance(query, str) or not query.strip():
            return {"retrieved": []}
        hits = retriever.retrieve(query, k=3, mode="hybrid")
        excerpts = [f"[{h.chunk.source_name}#{h.chunk.id}] {h.chunk.text}" for h in hits]
        return {"retrieved": excerpts}

    def plan(state: SupportState) -> dict:
        ctx = ""
        if state.get("retrieved"):
            ctx = "\n\nKnowledge-base excerpts:\n" + "\n---\n".join(state["retrieved"])
        known = ", ".join(sorted(state.get("user_order_ids") or [])) or "(none yet)"
        sys = SystemMessage(
            content=f"{SYSTEM_PROMPT}\n\nOrder ids provided by the customer: {known}{ctx}"
        )
        response = model.bind_tools(TOOL_SCHEMAS).invoke([sys, *state["messages"]])
        return {"messages": [response]}

    def validate(state: SupportState) -> dict:
        """Guardrail node: filter tool calls; rejections become ToolMessages the
        model sees on its next plan() turn (it can then ask the user)."""
        last = state["messages"][-1]
        allowed: list[dict] = []
        rejected: list[ToolMessage] = []
        for call in getattr(last, "tool_calls", None) or []:
            ok, reason = validate_tool_args(
                call["name"], call["args"], state.get("user_order_ids") or set()
            )
            if ok:
                allowed.append(call)
            else:
                rejected.append(ToolMessage(content=reason, tool_call_id=call["id"]))
        out: list[AnyMessage] = []
        if allowed:
            out.append(AIMessage(content=last.content or "", tool_calls=allowed))
        out.extend(rejected)
        return {"messages": out}

    def execute(state: SupportState) -> dict:
        last = state["messages"][-1]
        results: list[AnyMessage] = []
        risk_flags: list[str] = []
        for call in getattr(last, "tool_calls", None) or []:
            fn = getattr(toolbox, call["name"], None)
            if fn is None:
                results.append(
                    ToolMessage(
                        content=json.dumps({"status": "error", "error_code": "UNKNOWN_TOOL"}),
                        tool_call_id=call["id"],
                    )
                )
                continue
            result = fn(**call["args"])
            results.append(ToolMessage(content=result.for_llm(), tool_call_id=call["id"]))
            if call["name"] in WRITE_TOOLS:
                risk_flags.append(f"write:{call['name']}")
        return {"messages": results, "risk_flags": risk_flags}

    def review_gate(state: SupportState) -> dict:
        needs = bool(state.get("risk_flags"))
        return {"needs_human": needs}

    def route_after_validate(state: SupportState) -> str:
        """Execute the surviving calls; a full rejection goes back to plan with
        the rejection visible (recursion_limit bounds pathological loops)."""
        last = state["messages"][-1]
        return "execute" if getattr(last, "tool_calls", None) else "plan"

    def route_after_plan(state: SupportState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "validate"
        return "review_gate"

    g = StateGraph(SupportState)
    g.add_node("load_user_ids", load_user_ids)
    g.add_node("retrieve_kb", retrieve_kb)
    g.add_node("plan", plan)
    g.add_node("validate", validate)
    g.add_node("execute", execute)
    g.add_node("review_gate", review_gate)

    g.add_edge(START, "load_user_ids")
    g.add_edge("load_user_ids", "retrieve_kb")
    g.add_edge("retrieve_kb", "plan")
    g.add_conditional_edges(
        "plan", route_after_plan, {"validate": "validate", "review_gate": "review_gate"}
    )
    g.add_conditional_edges(
        "validate", route_after_validate, {"execute": "execute", "plan": "plan"}
    )
    g.add_edge("execute", "plan")  # loop back with tool results
    g.add_edge("review_gate", END)

    return g.compile()
