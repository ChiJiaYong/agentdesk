"""Business tools over the real Olist SQLite DB — with production hardening.

Three design rules (each maps to a book/interview anchor):

1. **Three-state results** (成功/失败/未知): a timeout or transport error is
   NOT a failure — it is "unknown", and the agent must never report an unknown
   as success. Mirrors the eval principle "未知态不得产生肯定性回复".
2. **Idempotency**: side-effectful tools (create_ticket) take a client-supplied
   idempotency key with a UNIQUE constraint; replays return the original row
   instead of creating duplicates.
3. **Parameter-source validation** lives in core/guardrails.py — tools declare
   which args must come from the user, and the graph enforces it *before*
   execution (defense in depth: tools re-check too).

All data is real (Olist subset) — see data/README.md for provenance.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "processed" / "business.db"

OK, ERROR, UNKNOWN = "ok", "error", "unknown"

_TICKETS_DDL = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL,
    issue TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class ToolResult:
    status: str  # ok | error | unknown
    tool: str
    data: dict | None = None
    error_code: str | None = None
    message: str = ""

    def for_llm(self) -> str:
        """Compact JSON for the tool message (what the model actually sees)."""
        return json.dumps(asdict(self), ensure_ascii=False)


class BusinessToolbox:
    """SQLite-backed tools; one connection per instance (single-process demo)."""

    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"business DB missing at {db_path} — run scripts/build_business_db.py first"
            )
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_TICKETS_DDL)
        self.conn.commit()

    # -- read tools -----------------------------------------------------------

    def get_order_status(self, order_id: str) -> ToolResult:
        """Order status + timestamps. Real Olist statuses (delivered/shipped/...)."""
        try:
            row = self.conn.execute(
                "SELECT order_id, order_status, order_purchase_timestamp,"
                " order_delivered_customer_date, order_estimated_delivery_date"
                " FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            return ToolResult(UNKNOWN, "get_order_status", error_code="DB_ERROR", message=str(exc))
        if row is None:
            return ToolResult(
                ERROR,
                "get_order_status",
                error_code="NOT_FOUND",
                message=f"order {order_id} not found",
            )
        data = {
            "order_id": row["order_id"],
            "status": row["order_status"],
            "purchased_at": row["order_purchase_timestamp"],
            "delivered_at": row["order_delivered_customer_date"],
            "estimated_delivery": row["order_estimated_delivery_date"],
        }
        return ToolResult(OK, "get_order_status", data=data)

    def get_order_amount(self, order_id: str) -> ToolResult:
        """Payment total — the source of truth for any refund/amount statement."""
        try:
            row = self.conn.execute(
                "SELECT order_id, SUM(payment_value) AS total, GROUP_CONCAT(payment_type) AS types"
                " FROM order_payments WHERE order_id = ? GROUP BY order_id",
                (order_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            return ToolResult(UNKNOWN, "get_order_amount", error_code="DB_ERROR", message=str(exc))
        if row is None:
            return ToolResult(
                ERROR,
                "get_order_amount",
                error_code="NOT_FOUND",
                message=f"no payment records for order {order_id}",
            )
        return ToolResult(
            OK,
            "get_order_amount",
            data={
                "order_id": row["order_id"],
                "total": round(row["total"], 2),
                "payment_types": (row["types"] or "").split(","),
            },
        )

    def get_order_review(self, order_id: str) -> ToolResult:
        """Review score/comment — context for complaint handling."""
        try:
            rows = self.conn.execute(
                "SELECT review_score, review_comment_message FROM order_reviews WHERE order_id = ?",
                (order_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            return ToolResult(UNKNOWN, "get_order_review", error_code="DB_ERROR", message=str(exc))
        if not rows:
            return ToolResult(
                ERROR,
                "get_order_review",
                error_code="NO_REVIEW",
                message=f"order {order_id} has no review",
            )
        return ToolResult(
            OK,
            "get_order_review",
            data={
                "order_id": order_id,
                "scores": [r["review_score"] for r in rows],
                "comments": [
                    r["review_comment_message"] for r in rows if r["review_comment_message"]
                ],
            },
        )

    # -- write tool (idempotent) ----------------------------------------------

    def create_ticket(self, order_id: str, issue: str, idempotency_key: str) -> ToolResult:
        """Create an after-sales ticket. Replaying the same idempotency key
        returns the ORIGINAL ticket instead of creating a duplicate row —
        the caller cannot tell a replay from a fresh success, which is exactly
        the contract we want for retry-after-timeout.
        """
        if not idempotency_key:
            return ToolResult(
                ERROR,
                "create_ticket",
                error_code="MISSING_IDEMPOTENCY_KEY",
                message="idempotency_key is required for side-effectful tools",
            )
        try:
            existing = self.conn.execute(
                "SELECT ticket_id, status, created_at FROM tickets WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return ToolResult(
                    OK,
                    "create_ticket",
                    data={
                        "ticket_id": existing["ticket_id"],
                        "status": existing["status"],
                        "created_at": existing["created_at"],
                        "replayed": True,
                    },
                )
            cur = self.conn.execute(
                "INSERT INTO tickets (idempotency_key, order_id, issue) VALUES (?,?,?)",
                (idempotency_key, order_id, issue),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT ticket_id, status, created_at FROM tickets WHERE ticket_id = ?",
                (cur.lastrowid,),
            ).fetchone()
            return ToolResult(
                OK,
                "create_ticket",
                data={
                    "ticket_id": row["ticket_id"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "replayed": False,
                },
            )
        except sqlite3.IntegrityError:
            # lost race between SELECT and INSERT — treat as replay
            row = self.conn.execute(
                "SELECT ticket_id, status, created_at FROM tickets WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return ToolResult(
                OK,
                "create_ticket",
                data={
                    "ticket_id": row["ticket_id"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "replayed": True,
                },
            )
        except sqlite3.Error as exc:
            return ToolResult(UNKNOWN, "create_ticket", error_code="DB_ERROR", message=str(exc))

    def close(self) -> None:
        self.conn.close()
