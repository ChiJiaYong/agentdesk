"""Guardrails: parameter-source validation + tool-arg checking (book §1.2.5, §4.2).

The single most-cited agent failure in evals: the model *invents* an order id
to keep the conversation flowing. Our defense is not a prompt plea — it is a
checkable invariant:

    an order id used in any tool call must have appeared in the USER's own
    messages (or been returned by a previous tool call).

The graph carries ``known_order_ids`` in state; ``validate_tool_args`` runs
BEFORE execution and rejects violations with a typed error the model can see
and recover from. Tools re-check as defense in depth.
"""

from __future__ import annotations

import re

ORDER_ID_RX = re.compile(r"\b[0-9a-f]{32}\b")  # Olist order ids are 32-char hex

# args that must originate from the user/context, per tool
SOURCE_VALIDATED_ARGS: dict[str, list[str]] = {
    "get_order_status": ["order_id"],
    "get_order_amount": ["order_id"],
    "get_order_review": ["order_id"],
    "create_ticket": ["order_id"],
}

_OK, _REJECT = True, False


def extract_order_ids(user_texts: list[str]) -> set[str]:
    """Order ids the user actually provided (9-digit, per Olist format)."""
    found: set[str] = set()
    for text in user_texts:
        found.update(ORDER_ID_RX.findall(text or ""))
    return found


PERSONA_PROMPT_RX = re.compile(r"you are now (.+?)(?:\.|and|,|\n)", re.I)


def detect_persona_adoption(user_texts: list[str], final_reply: str) -> bool:
    """Output-side injection verification (book §1.2.5: 上下文层护栏有结构性上限——
    同一上下文内的模型无法自证未被注入，故由 Harness 在输出侧验证）。

    通用化规则：若用户文本包含 "you are now X ... must say (I am X)" 式的身份改写
    指令，且最终回复确实说了 "I am X"（回显注入身份）→ 判定顺从。这是针对已知
    攻击模式的输出验证器，不是完备的注入检测。
    """
    for text in user_texts:
        for m in PERSONA_PROMPT_RX.finditer(text or ""):
            persona = m.group(1).strip()
            if not persona:
                continue
            echo_rx = re.compile(r"i am (?:now )?" + re.escape(persona), re.I)
            if echo_rx.search(final_reply or ""):
                return True
    return False


def validate_tool_args(tool_name: str, args: dict, known_order_ids: set[str]) -> tuple[bool, str]:
    """Return (ok, reason). Rejects invented ids with an actionable message.

    known_order_ids currently = ids the user typed. When tools start *returning*
    order ids (e.g. a lookup tool), accumulate them here too — the state field
    is the single source of truth for "ids the agent is allowed to act on".
    """
    for arg in SOURCE_VALIDATED_ARGS.get(tool_name, []):
        if arg not in args:
            continue
        value = str(args[arg])
        if arg.endswith("order_id") and value not in known_order_ids:
            return _REJECT, (
                f"rejected: {arg}={value!r} was not provided by the user. "
                f"Ask the customer for the order id instead of guessing."
            )
    return _OK, ""
