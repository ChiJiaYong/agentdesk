"""FastAPI service: chat endpoint + HITL workbench + trace access.

Pipeline per /chat turn (each stage is a W4 deliverable):

    redact(PII out) → memory.append → memory facts → traced_invoke(graph)
    → memory.learn_from_turn → restore(PII back, display only) → reply

Model tier auto-detection: DEEPSEEK_API_KEY present → real provider (W5);
otherwise the deterministic DemoSupportModel (zero-key demo, labeled as such
in the UI). Graph/tools/guardrails/traces are identical across tiers.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agents.support.graph import build_support_agent
from core.memory import MemoryService
from core.redaction import redact, restore
from core.retrieval import BGEEmbedder, Chunk, FakeHashEmbedder, HybridRetriever
from core.trace import traced_invoke
from service.demo_model import DemoSupportModel
from tools.business_tools import BusinessToolbox

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_state: dict = {}


def _build_retriever() -> HybridRetriever:
    """Dense channel when sentence-transformers is installed, else sparse-only."""
    kb = ROOT / "data" / "processed" / "kb_chunks.jsonl"
    if not kb.exists():
        return HybridRetriever(None)
    records = [
        json.loads(line) for line in kb.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    try:
        embedder = BGEEmbedder()
    except ImportError:
        embedder = FakeHashEmbedder()
    retriever = HybridRetriever(embedder)
    retriever.add([Chunk(**r) for r in records])
    return retriever


def _build_model() -> tuple[Any, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI  # optional W5 tier

        model = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.2,
        )
        return model, "deepseek"
    return DemoSupportModel(), "demo"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["retriever"] = _build_retriever()
    _state["toolbox"] = BusinessToolbox()
    _state["memory"] = MemoryService()
    _state["model"], _state["model_tier"] = _build_model()
    _state["agent"] = build_support_agent(_state["model"], _state["retriever"], _state["toolbox"])
    yield
    _state["toolbox"].close()
    _state["memory"].close()


app = FastAPI(title="AgentDesk", version="0.4.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = "default"
    customer_key: str = "anonymous"
    message: str


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_tier": _state.get("model_tier", "booting")}


@app.post("/chat")
def chat(req: ChatRequest) -> JSONResponse:
    if not req.message.strip():
        raise HTTPException(400, "empty message")

    redacted, pii_map = redact(req.message)
    memory: MemoryService = _state["memory"]
    memory.append_message(req.session_id, "human", redacted)

    facts = memory.get_facts(req.customer_key)
    if facts:
        block = "\n".join(f"- {f['fact_key']}: {f['fact_value']}" for f in facts)
        memory.append_message(req.session_id, "system", f"[Known customer facts]\n{block}")

    final, trace_path = traced_invoke(
        _state["agent"], {"messages": [("user", redacted)]}, session_id=req.session_id
    )
    reply = restore(str(final["messages"][-1].content), pii_map)

    memory.append_message(req.session_id, "ai", reply, run_id=trace_path.stem)
    memory.learn_from_turn(req.customer_key, req.message, reply)

    try:  # traces live under ROOT in prod; tests monkeypatch RUNS_DIR elsewhere
        trace_rel = str(trace_path.relative_to(ROOT))
    except ValueError:
        trace_rel = str(trace_path)

    return JSONResponse(
        {
            "reply": reply,
            "needs_human": final.get("needs_human", False),
            "model_tier": _state["model_tier"],
            "run_id": trace_path.stem,
            "trace_path": trace_rel,
            "pii_redacted": bool(pii_map),
        }
    )


@app.get("/workbench/queue")
def queue() -> list[dict]:
    rows = (
        _state["toolbox"]
        .conn.execute(
            "SELECT ticket_id, order_id, issue, status, created_at FROM tickets"
            " WHERE status = 'open' ORDER BY ticket_id DESC"
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


class Decision(BaseModel):
    action: str  # approve | reject
    note: str = ""


@app.post("/workbench/{ticket_id}/decide")
def decide(ticket_id: int, decision: Decision) -> dict:
    if decision.action not in {"approve", "reject"}:
        raise HTTPException(400, "action must be approve|reject")
    status = "approved" if decision.action == "approve" else "rejected"
    cur = _state["toolbox"].conn.execute(
        "UPDATE tickets SET status = ? WHERE ticket_id = ? AND status = 'open'",
        (status, ticket_id),
    )
    _state["toolbox"].conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "ticket not found or already decided")
    return {"ticket_id": ticket_id, "status": status}


@app.get("/trace/{date}/{run_id}")
def get_trace(date: str, run_id: str) -> FileResponse:
    path = ROOT / "runs" / date / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "trace not found")
    return FileResponse(path, media_type="application/json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html")
