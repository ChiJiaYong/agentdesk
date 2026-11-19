"""Trace & observability — persist one JSON per agent run (book §7.8).

W4 pragmatic version: node-by-node updates captured via ``graph.stream(...,
stream_mode="updates")``, plus latency, token usage (when the provider reports
it) and the final state. Files land in ``runs/YYYY-MM-DD/{run_id}.json`` — the
bad-case evidence chain for W5's failure attribution.

Design rule: traces store CONTENT SUMMARIES of LLM inputs (post-redaction)
but FULL tool results — tool IO is what failure attribution needs, and it
never contains provider chat text beyond what the flow already produced.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def new_run_id() -> str:
    return f"run_{datetime.now(UTC).strftime('%Y%m%d')}_{uuid.uuid4().hex[:10]}"


def _safe(obj: Any) -> Any:
    """JSON-safe conversion: sets → sorted lists, recurse containers."""
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj


def _summarize(message: Any, limit: int = 600) -> dict:
    m_type = getattr(message, "type", "?")
    content = getattr(message, "content", "") or ""
    out: dict[str, Any] = {"type": m_type, "content": str(content)[:limit]}
    calls = getattr(message, "tool_calls", None)
    if calls:
        out["tool_calls"] = [{"name": c["name"], "args": c.get("args", {})} for c in calls]
    tc_id = getattr(message, "tool_call_id", None)
    if tc_id:
        out["tool_call_id"] = tc_id
    usage = getattr(message, "usage_metadata", None)
    if usage:
        out["tokens"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")}
    return out


class RunTrace:
    """Collects node updates + timings for one agent invocation."""

    def __init__(self, run_id: str | None = None, session_id: str = "default") -> None:
        self.run_id = run_id or new_run_id()
        self.session_id = session_id
        self.started = time.perf_counter()
        self.steps: list[dict[str, Any]] = []
        self.final_state: dict[str, Any] | None = None
        self.error: str | None = None

    def add_node_update(self, node: str, update: dict[str, Any]) -> None:
        entry: dict[str, Any] = {
            "node": node,
            "latency_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "messages": [_summarize(m) for m in update.get("messages", [])],
        }
        for key in ("retrieved", "risk_flags", "needs_human", "user_order_ids"):
            if key in update:
                entry[key] = _safe(update[key])
        self.steps.append(entry)

    def finish(self, final_state: dict[str, Any]) -> None:
        self.final_state = {
            "needs_human": final_state.get("needs_human", False),
            "risk_flags": _safe(final_state.get("risk_flags", [])),
            "messages": [_summarize(m, limit=1200) for m in final_state.get("messages", [])],
        }
        self.total_ms = round((time.perf_counter() - self.started) * 1000, 1)

    def fail(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"

    def save(self) -> Path:
        day_dir = RUNS_DIR / datetime.now(UTC).strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{self.run_id}.json"
        payload = _safe(
            {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "recorded_at": datetime.now(UTC).isoformat(),
                "total_ms": getattr(self, "total_ms", None),
                "error": self.error,
                "steps": self.steps,
                "final": self.final_state,
            }
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def traced_invoke(agent: Any, payload: dict, session_id: str = "default") -> tuple[dict, Path]:
    """Invoke the graph node-by-node, persisting a trace; returns (final_state, trace_path)."""
    trace = RunTrace(session_id=session_id)
    final: dict[str, Any] = {}
    try:
        for update in agent.stream(payload, config={"recursion_limit": 25}, stream_mode="updates"):
            for node, change in update.items():
                trace.add_node_update(node, change or {})
                final.update(change or {})
        trace.finish(final)
    except Exception as exc:  # noqa: BLE001 — trace must survive agent failures
        trace.fail(exc)
        raise
    finally:
        path = trace.save()
    return final, path
