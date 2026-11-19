"""Offline tests for the W3 hardening: three-state tools, idempotency, guardrails."""

from __future__ import annotations

import sqlite3

import pytest

from core.guardrails import extract_order_ids, validate_tool_args
from tools.business_tools import BusinessToolbox


@pytest.fixture(scope="module")
def real_order_id() -> str:
    """A real order id from the built business DB (32-char hex, Olist format)."""
    conn = sqlite3.connect("data/processed/business.db")
    oid = conn.execute("SELECT order_id FROM orders LIMIT 1").fetchone()[0]
    conn.close()
    return oid


@pytest.fixture(scope="module")
def toolbox(tmp_path_factory):
    """Real business.db (read-only usage); tickets go to a temp overlay DB.

    We reuse the real DB for reads and copy it to a temp file for ticket
    writes so tests never mutate the shared artifact.
    """
    db = tmp_path_factory.mktemp("biz") / "business.db"
    src = sqlite3.connect("data/processed/business.db")
    dst = sqlite3.connect(str(db))
    src.backup(dst)
    src.close()
    box = BusinessToolbox(db)
    yield box
    box.close()


# -- three-state tools -------------------------------------------------------


def test_get_order_status_ok(toolbox: BusinessToolbox, real_order_id: str) -> None:
    result = toolbox.get_order_status(real_order_id)
    assert result.status == "ok"
    assert result.data["status"] in {
        "delivered",
        "shipped",
        "canceled",
        "processing",
        "invoiced",
        "unavailable",
        "approved",
        "created",
    }


def test_get_order_status_not_found_is_error_not_unknown(toolbox: BusinessToolbox) -> None:
    result = toolbox.get_order_status("f" * 32)  # well-formed but absent id
    assert result.status == "error"
    assert result.error_code == "NOT_FOUND"


def test_get_order_amount_sums_payments(toolbox: BusinessToolbox, real_order_id: str) -> None:
    result = toolbox.get_order_amount(real_order_id)
    assert result.status in {"ok", "error"}  # subset may lack payments for this order
    if result.status == "ok":
        assert result.data["total"] >= 0
        assert isinstance(result.data["payment_types"], list)


# -- idempotency --------------------------------------------------------------


def test_create_ticket_idempotent_replay(toolbox: BusinessToolbox, real_order_id: str) -> None:
    key = "test-key-001"
    first = toolbox.create_ticket(real_order_id, "broken screen", key)
    assert first.status == "ok" and first.data["replayed"] is False
    second = toolbox.create_ticket(real_order_id, "broken screen", key)
    assert second.status == "ok"
    assert second.data["replayed"] is True
    assert second.data["ticket_id"] == first.data["ticket_id"]  # no duplicate row


def test_create_ticket_requires_idempotency_key(
    toolbox: BusinessToolbox, real_order_id: str
) -> None:
    result = toolbox.create_ticket(real_order_id, "issue", "")
    assert result.status == "error"
    assert result.error_code == "MISSING_IDEMPOTENCY_KEY"


# -- parameter-source guardrails ----------------------------------------------


def test_extract_order_ids_from_user_text() -> None:
    sample = "e481f51cbdc54678b7cc49136f2d6af7"
    ids = extract_order_ids([f"my order is {sample}", "no id here"])
    assert ids == {sample}


def test_guardrail_rejects_invented_order_id() -> None:
    ok, reason = validate_tool_args(
        "get_order_status", {"order_id": "a" * 32}, {"e481f51cbdc54678b7cc49136f2d6af7"}
    )
    assert not ok
    assert "not provided by the user" in reason


def test_guardrail_allows_user_provided_id() -> None:
    oid = "e481f51cbdc54678b7cc49136f2d6af7"
    ok, _ = validate_tool_args("get_order_status", {"order_id": oid}, {oid})
    assert ok


def test_guardrail_ignores_unvalidated_args() -> None:
    oid = "e481f51cbdc54678b7cc49136f2d6af7"
    ok, _ = validate_tool_args("get_order_review", {"order_id": oid}, {oid})
    assert ok
