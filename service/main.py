r"""FastAPI service: chat endpoint + HITL workbench + trace access + model tiers.

Pipeline per /chat turn (each stage is a W4 deliverable):

    redact(PII out) → memory.append → memory facts → traced_invoke(graph)
    → memory.learn_from_turn → restore(PII back, display only) → reply

Model tiers (service/model_tiers.py) — user-facing choices:
    demo  : zero-config rule-based model (labeled as demo in the UI)
    local : llama-server loading a local GGUF (e.g. D:\AI\models\llm\*.gguf),
            OpenAI-compatible, data never leaves the machine
    api   : user pastes an API key in the Settings window (memory only, never
            persisted); any OpenAI-compatible endpoint

Tier resolution: explicit UI choice (validated) > auto-probe (api key →
local health → demo). Graph/tools/guardrails/traces are identical across tiers.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agents.support.graph import build_support_agent
from core.memory import MemoryService
from core.redaction import redact, restore
from core.retrieval import BGEEmbedder, Chunk, FakeHashEmbedder, HybridRetriever
from core.trace import traced_invoke
from service import model_tiers
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


def _apply_tier(
    choice: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
) -> dict:
    tier = model_tiers.resolve_tier(choice)
    model, info = model_tiers.build_model(
        tier, api_key=api_key, base_url=base_url, model_name=model_name
    )
    _state["model"] = model
    _state["model_tier"] = tier
    _state["model_detail"] = info["detail"]
    _state["agent"] = build_support_agent(_state["model"], _state["retriever"], _state["toolbox"])
    return info


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["retriever"] = _build_retriever()
    _state["toolbox"] = BusinessToolbox()
    _state["memory"] = MemoryService()
    _state["tier_choice"] = "auto"
    _apply_tier("auto")
    yield
    _state["toolbox"].close()
    _state["memory"].close()


app = FastAPI(title="AgentDesk", version="0.5.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = "default"
    customer_key: str = "anonymous"
    message: str


class TierSettings(BaseModel):
    tier: str = "auto"  # auto | demo | local | api
    api_key: str | None = None
    base_url: str | None = None
    model_name: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_tier": _state.get("model_tier", "booting"),
        "model_detail": _state.get("model_detail", ""),
    }


@app.get("/settings")
def get_settings() -> dict:
    return {
        "tier_choice": _state.get("tier_choice", "auto"),
        "active_tier": _state.get("model_tier"),
        "model_detail": _state.get("model_detail"),
        "availability": {
            "api": model_tiers.api_key_available(),
            "local": model_tiers.local_server_healthy(),
            "demo": True,
        },
        "local_server_url": model_tiers.DEFAULT_LOCAL_BASE_URL,
    }


@app.post("/settings")
def update_settings(settings: TierSettings) -> dict:
    """使用者交互窗口：选层。api key 仅存内存（不落盘）。"""
    if settings.api_key:
        os.environ["DEEPSEEK_API_KEY"] = settings.api_key  # runtime only
    _state["tier_choice"] = settings.tier
    try:
        info = _apply_tier(
            settings.tier,
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"active_tier": _state["model_tier"], "detail": info["detail"]}


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
