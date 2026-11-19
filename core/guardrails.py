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


def validate_tool_args(tool_name: str, args: dict, known_order_ids: set[str]) -> tuple[bool, str]:
    """Return (ok, reason). Rejects invented ids with an actionable message."""
    for arg in SOURCE_VALIDATED_ARGS.get(tool_name, []):
        if arg not in args:
            continue
        value = str(args[arg])
        if arg.endswith("order_id") and value not in known_order_ids:
            return _REJECT, (
                f"rejected: {arg}={value!r} was not provided by the user and is not the "
                f"output of a previous tool call. Ask the user for the order id instead "
                f"of guessing."
            )
    return _OK, ""
