# AgentDesk

**A reusable LLM agent service platform for cross-border e-commerce** — one
shared foundation (hybrid retrieval, governed tools, model routing, eval &
observability), two scenario agents:

- **Support Agent**（对外）— customer-service agent over a **real 99k-order
  business database** (Olist), tool calls hardened with three-state results +
  idempotency + parameter-source guardrails, refund-type actions escalated to a
  **human-in-the-loop workbench**.
- **KBuddy**（对内）— knowledge assistant over **real ingested corpora**: three
  official CN consumer-protection regulations + 3k real customer Q&A (Amazon
  QA, UCSD), hybrid retrieval (BM25 + bge-small-zh + RRF), citation grounding.

> 面向跨境电商的 LLM Agent 服务平台：对外「客服工单 Agent」+ 对内「知识库问答助手」，共用一套底座（混合检索、工具治理、模型路由、评测与可观测）。

**Theory backbone**: all design decisions trace back to
《深入理解AI Agent：设计原理与工程实践》(李博杰, v2.0) — see `docs/adr/` for the
decision records (each: constraint → candidates → choice → cost → revival clause).

## Verified results (all reproducible, no fabricated numbers)

| Claim | Evidence |
|---|---|
| Hybrid retrieval hit@5 **0.926** vs 0.852 single-channel (+7.4pp) | [retrieval-v0.1](docs/eval-reports/retrieval-v0.1.md) · 3,052 real chunks · 27 golden queries · zero LLM cost |
| Agent-path eval **10/10** across 6 risk classes | [agent-eval-v0.1](docs/eval-reports/agent-eval-v0.1.md) · trace-level assertions |
| **Eval caught a real bug pre-merge** (duplicate tool execution; idempotency contained it) | regression task `two_tool_sequence` |
| 99,441 real orders · 3,052 provenance-tracked chunks | `data/manifest.json` (URL/license/sha256/date) |

## Architecture

```
                ┌────────────────────────────────────────────────┐
                │  shared foundation                             │
                │  hybrid retrieval · tool framework (3-state,   │
                │  idempotent) · PII redaction · memory (facts   │
                │  + supersede) · run traces · eval harness      │
                └────────────┬───────────────────┬───────────────┘
      external customers     │                   │    internal staff
              ┌──────────────▼─────┐   ┌─────────▼─────────────┐
              │ Support Agent      │   │ KBuddy                │
              │ LangGraph:         │   │ RAG + citations       │
              │ plan⇄validate⇄exec │   │ permission-aware (W5) │
              │ review_gate → HITL │   │                       │
              └────────────────────┘   └───────────────────────┘
                      │ tools: status/amount/review/ticket (SQLite)
                      │ guardrail node: param-source validation
                      │ demo tier (zero-key) ⇄ DeepSeek (key set)
```

One customer turn:
`redact → memory → load_user_ids → retrieve_kb → plan ⇄ validate ⇄ execute → review_gate → reply`
— every stage persists a step into `runs/YYYY-MM-DD/{run_id}.json`.

## Quick start

```bash
# Option A — Docker (first boot ingests the real datasets automatically):
docker compose up            # → http://localhost:8000 (workbench UI)

# Option B — local:
uv sync --extra dev --extra ml
uv run python scripts/download_data.py          # real datasets (Olist + Amazon QA)
uv run python scripts/fetch_regulations.py      # official regulation texts
uv run python -m scripts.build_business_db      # SQLite: 99k real orders
uv run python -m scripts.build_knowledge_base --qa-sample 3000
uv run uvicorn service.main:app --port 8000
```

Configure `.env` from `.env.example`. **No API key needed** to run everything
above — the deterministic demo tier exercises the full graph; set
`DEEPSEEK_API_KEY` to upgrade the generation tier in place.

```bash
uv run pytest -m "not slow"        # 31 tests, fully offline
uv run python -m evals.smoke       # CI eval gate (5 tasks, mock LLM)
uv run python -m evals.agent_eval  # 10 graph-level tasks, 6 risk classes
uv run python -m evals.retrieval_eval --modes bm25,dense,hybrid
```

## Engineering practices

- Protected `main` (ruleset: PR-only + CI status checks, no bypass) — all 8 PRs
  merged through the gate; CI caught 3 issues pre-merge
- Conventional Commits, SemVer tags, issue-driven development
- Data never committed: download scripts + `manifest.json` provenance
  (source URL / license / sha256 / date); evals run on temp copies and never
  mutate the shared artifact
- Design decisions as ADRs; every eval claim links to a reproducible command

## Project structure

```
core/      retrieval (hybrid) · guardrails · redaction · memory · trace · fakes
agents/    support (LangGraph graph) · kbuddy
tools/     business tools over SQLite (3-state results, idempotency)
service/   FastAPI (chat / workbench / trace) · demo model · web UI
evals/     smoke · agent_eval · retrieval_eval
scripts/   download_data · fetch_regulations · build_business_db · build_knowledge_base
docs/      adr/ · eval-reports/ · notes/ · PROGRESS.md
```

## License

MIT. Third-party dataset licenses: see [data/README.md](data/README.md).
