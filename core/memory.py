"""Session + customer memory (book §3.1 记忆系统).

Two layers, one SQLite file (data/processed/memory.db):

- **Session memory**: full conversation messages per session_id.
- **Customer facts**: cross-session key/value facts (order ids discussed,
  issue categories, explicit "remember this" statements). Writes follow the
  书 §3.1 write-discrimination rules:

    * explicit user statement  → fact, confidence=explicit
    * stable pattern in turn   → fact, confidence=inferred
    * conflicting new value    → latest wins, old value kept with superseded_at
      (never silently deleted — conflict history is the interview answer to
      "记忆冲突怎么解决")

Facts are injected into the context AFTER the system prompt and BEFORE the
conversation (stable-prefix ordering, §2.3 KV-cache friendliness: facts change
rarely, so the cacheable prefix stays intact across turns).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "processed" / "memory.db"

_DDL = """
CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    run_id TEXT
);
CREATE INDEX idx_session ON session_messages(session_id);
CREATE TABLE IF NOT EXISTS customer_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_key TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    fact_value TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT 'inferred',
    created_at TEXT NOT NULL,
    superseded_at TEXT
);
CREATE INDEX idx_facts ON customer_facts(customer_key, fact_key);
"""


class MemoryService:
    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI sync endpoints run on a threadpool.
        # Single-process demo scale; per-request connections land with async W5.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_DDL)
        self.conn.commit()

    # -- session layer ---------------------------------------------------------

    def append_message(
        self, session_id: str, role: str, content: str, run_id: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO session_messages (session_id, role, content, created_at, run_id)"
            " VALUES (?,?,?,?,?)",
            (session_id, role, content, datetime.now(UTC).isoformat(), run_id),
        )
        self.conn.commit()

    def recent_messages(self, session_id: str, limit: int = 12) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, created_at FROM session_messages"
            " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            {"role": r["role"], "content": r["content"], "at": r["created_at"]}
            for r in reversed(rows)
        ]

    # -- fact layer ------------------------------------------------------------

    def upsert_fact(
        self, customer_key: str, fact_key: str, fact_value: str, confidence: str = "inferred"
    ) -> None:
        """Write a fact; supersede (not delete) any previous value for the key."""
        now = datetime.now(UTC).isoformat()
        cur = self.conn.execute(
            "SELECT id, fact_value FROM customer_facts"
            " WHERE customer_key = ? AND fact_key = ? AND superseded_at IS NULL",
            (customer_key, fact_key),
        ).fetchone()
        if cur and cur["fact_value"] == fact_value:
            return  # no-op: identical value, keep the cacheable prefix stable
        if cur:
            self.conn.execute(
                "UPDATE customer_facts SET superseded_at = ? WHERE id = ?", (now, cur["id"])
            )
        self.conn.execute(
            "INSERT INTO customer_facts (customer_key, fact_key, fact_value, confidence, created_at)"
            " VALUES (?,?,?,?,?)",
            (customer_key, fact_key, fact_value, confidence, now),
        )
        self.conn.commit()

    def get_facts(self, customer_key: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT fact_key, fact_value, confidence, created_at FROM customer_facts"
            " WHERE customer_key = ? AND superseded_at IS NULL ORDER BY id",
            (customer_key,),
        ).fetchall()
        return [dict(r) for r in rows]

    def fact_history(self, customer_key: str, fact_key: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT fact_value, confidence, created_at, superseded_at FROM customer_facts"
            " WHERE customer_key = ? AND fact_key = ? ORDER BY id",
            (customer_key, fact_key),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- extraction (rule-based W4; LLM-based upgrade lands with the API key) ---

    def learn_from_turn(self, customer_key: str, user_text: str, assistant_text: str) -> list[dict]:
        """Extract facts from one turn. Rule-based, conservative by design:
        only explicit/stable signals get written — '今天想吃辣'式短偏好不落库."""
        from core.guardrails import extract_order_ids

        learned: list[dict] = []
        for oid in extract_order_ids([user_text]):
            self.upsert_fact(
                customer_key, f"order:{oid}", "discussed_by_customer", confidence="explicit"
            )
            learned.append({"fact_key": f"order:{oid}", "fact_value": "discussed_by_customer"})
        lowered = user_text.lower()
        if any(w in lowered for w in ("refund", "退款", "退货")):
            self.upsert_fact(customer_key, "last_issue_category", "refund_return")
            learned.append({"fact_key": "last_issue_category", "fact_value": "refund_return"})
        elif any(w in lowered for w in ("logistics", "shipping", "物流", "快递", "配送")):
            self.upsert_fact(customer_key, "last_issue_category", "logistics")
            learned.append({"fact_key": "last_issue_category", "fact_value": "logistics"})
        if "remember" in lowered or "记住" in lowered:
            self.upsert_fact(
                customer_key,
                "explicit_request",
                json.dumps({"raw": user_text[:200]}, ensure_ascii=False),
                confidence="explicit",
            )
            learned.append({"fact_key": "explicit_request", "fact_value": "user asked to remember"})
        return learned

    def close(self) -> None:
        self.conn.close()
