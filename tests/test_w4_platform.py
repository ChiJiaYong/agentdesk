"""W4 platform tests: redaction, memory conflict handling, trace persistence,
FastAPI /chat + workbench flow (demo model tier, real business DB)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.memory import MemoryService
from core.redaction import contains_pii, redact, restore
from core.trace import RunTrace
from tools.business_tools import BusinessToolbox

requires_db = pytest.mark.skipif(
    not Path("data/processed/business.db").exists(),
    reason="business DB not built (run scripts/build_business_db.py)",
)

REAL_ID = "e481f51cbdc54678b7cc49136f2d6af7"


# -- redaction ----------------------------------------------------------------


def test_redact_phone_and_email_roundtrip() -> None:
    original = "Call me at 13812345678 or mail john.doe@example.com about order " + REAL_ID
    redacted, mapping = redact(original)
    assert "13812345678" not in redacted
    assert "john.doe@example.com" not in redacted
    assert REAL_ID in redacted, "order ids must survive (tools need them)"
    assert "[PHONE_1]" in redacted and "[EMAIL_1]" in redacted
    assert restore(redacted, mapping) == original
    assert contains_pii(original) and not contains_pii(redacted)


def test_redact_is_stable_across_calls() -> None:
    a, _ = redact("phone 13812345678")
    b, _ = redact("phone 13812345678")
    assert a == b


# -- memory --------------------------------------------------------------------


def test_memory_supersede_keeps_history(tmp_path) -> None:
    mem = MemoryService(tmp_path / "mem.db")
    mem.upsert_fact("cust-1", "last_issue_category", "logistics")
    mem.upsert_fact("cust-1", "last_issue_category", "refund_return")  # conflict
    facts = {f["fact_key"]: f["fact_value"] for f in mem.get_facts("cust-1")}
    assert facts["last_issue_category"] == "refund_return"  # latest wins
    history = mem.fact_history("cust-1", "last_issue_category")
    assert len(history) == 2 and history[0]["superseded_at"]  # old value kept, superseded
    mem.close()


def test_memory_identical_write_is_noop(tmp_path) -> None:
    mem = MemoryService(tmp_path / "mem.db")
    mem.upsert_fact("c", "k", "v")
    mem.upsert_fact("c", "k", "v")
    assert len(mem.fact_history("c", "k")) == 1  # stable prefix stays cacheable
    mem.close()


def test_learn_from_turn_rules(tmp_path) -> None:
    mem = MemoryService(tmp_path / "mem.db")
    learned = mem.learn_from_turn("c", f"My order {REAL_ID} arrived broken, I want a refund", "ok")
    keys = {item["fact_key"] for item in learned}
    assert f"order:{REAL_ID}" in keys and "last_issue_category" in keys
    mem.close()


# -- trace ---------------------------------------------------------------------


def test_trace_persists_and_summarizes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("core.trace.RUNS_DIR", tmp_path / "runs")
    t = RunTrace(session_id="s1")
    t.add_node_update(
        "plan",
        {
            "messages": [
                type(
                    "M",
                    (),
                    {"type": "ai", "content": "hi", "tool_calls": [], "usage_metadata": None},
                )()
            ]
        },
    )
    t.finish({"needs_human": False, "messages": []})
    path = t.save()
    assert path.exists() and "runs" in str(path)
    assert json_ok(path)


def json_ok(path: Path) -> bool:
    import json

    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except json.JSONDecodeError:
        return False


# -- FastAPI flow (demo tier) ---------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("service.main._state", {})
    import service.main as svc

    svc._state["retriever"] = _fake_retriever()
    svc._state["toolbox"] = _copy_db(tmp_path)
    svc._state["memory"] = MemoryService(tmp_path / "mem.db")
    svc._state["model"] = _demo_model()
    svc._state["model_tier"] = "demo"
    svc._state["agent"] = __import__(
        "agents.support.graph", fromlist=["build_support_agent"]
    ).build_support_agent(svc._state["model"], svc._state["retriever"], svc._state["toolbox"])
    monkeypatch.setattr("core.trace.RUNS_DIR", tmp_path / "runs")
    return TestClient(svc.app)


def _fake_retriever():
    from core.retrieval import Chunk, FakeHashEmbedder, HybridRetriever

    r = HybridRetriever(FakeHashEmbedder())
    r.add(
        [
            Chunk(
                id="kb-1",
                text="Unused items may be returned within 30 days for a full refund.",
                doc_id="pol",
                source_url="x",
                source_name="policy",
                license="t",
                fetched_at="2026",
            )
        ]
    )
    return r


def _copy_db(tmp_path) -> BusinessToolbox:
    import sqlite3

    from tools.business_tools import BusinessToolbox

    db = tmp_path / "business.db"
    src = sqlite3.connect("data/processed/business.db")
    dst = sqlite3.connect(str(db))
    src.backup(dst)
    src.close()
    return BusinessToolbox(db)


def _demo_model():
    from service.demo_model import DemoSupportModel

    return DemoSupportModel(messages=iter([]))


@requires_db
def test_chat_flow_creates_trace_and_workbench_item(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={
            "session_id": "t1",
            "customer_key": "c1",
            "message": f"I want a refund, item from {REAL_ID} is broken. Call me at 13812345678",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["needs_human"] is True
    assert data["model_tier"] == "demo"
    assert data["pii_redacted"] is True
    assert "13812345678" not in Path(data["trace_path"]).read_text(encoding="utf-8"), (
        "trace must store redacted text only"
    )

    queue = client.get("/workbench/queue").json()
    assert any(t["order_id"] == REAL_ID for t in queue)
    ticket_id = queue[0]["ticket_id"]
    d = client.post(f"/workbench/{ticket_id}/decide", json={"action": "approve"})
    assert d.status_code == 200
    remaining = client.get("/workbench/queue").json()
    assert all(t["ticket_id"] != ticket_id for t in remaining)


@requires_db
def test_chat_asks_for_missing_order_id(client: TestClient) -> None:
    r = client.post(
        "/chat", json={"session_id": "t2", "customer_key": "c2", "message": "Where is my order?"}
    )
    data = r.json()
    assert "order id" in data["reply"].lower()
    assert data["needs_human"] is False
