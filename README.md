# AgentDesk

**A reusable LLM agent service platform for cross-border e-commerce**, delivering
two scenario agents on one shared foundation:

- **Support Agent** (external): customer-service ticket agent for an independent
  e-commerce store — RAG over real after-sales policies, tool calls against a
  real 100k-order business database (Olist), refunds escalated to human approval.
- **KBuddy** (internal): knowledge assistant for employees/developers — hybrid
  retrieval (BM25 + dense + rerank) over a real company handbook and developer
  docs, with permission-aware retrieval and citation grounding.

> 面向跨境电商的 LLM Agent 服务平台：对外「客服工单 Agent」+ 对内「知识库问答助手」，共用一套底座（混合检索、工具框架、模型路由、评测与可观测）。

## Status

`W1 — project scaffolding & comprehension exercises` (see [docs/PROGRESS.md](docs/PROGRESS.md))

| Week | Milestone | Status |
|---|---|---|
| 1 | Repo init, CI, hand-rolled loop vs LangGraph, data scripts | 🚧 in progress |
| 2 | Real-data pipeline + hybrid retrieval + KBuddy v0 | ⬜ |
| 3 | Support Agent graph + tools + SQLite business DB | ⬜ |
| 4 | HITL workbench + memory + PII redaction + trace | ⬜ |
| 5 | Full eval harness (Pass^k, LLM-as-Judge, cost A/B) | ⬜ |
| 6 | Docker, docs, personal site, v1.0 | ⬜ |

## Architecture (target)

```
                    ┌────────────────────────────────────┐
                    │  shared foundation (core/)         │
                    │  hybrid retrieval · tool framework │
                    │  model routing · memory · redaction│
                    │  eval harness · trace · guardrails │
                    └──────────┬──────────────┬──────────┘
        external customers     │              │     internal staff
                    ┌──────────▼─────┐  ┌─────▼──────────────┐
                    │ Support Agent  │  │ KBuddy             │
                    │ tools + HITL   │  │ RAG + permissions  │
                    └────────────────┘  └────────────────────┘
```

## Quick start

```bash
git clone <repo-url> && cd agentdesk
uv sync --extra dev          # install
uv run pytest -m "not slow"  # offline test suite
uv run python -m evals.smoke # 5-task eval smoke (mock LLM, no keys needed)

# W1 demos (offline):
uv run python -m examples.run_mini_loop        # hand-rolled ReAct loop
uv run python -m examples.langgraph_quickstart # same task via create_agent (LangChain 1.x)

# Real data (see data/README.md):
uv run python scripts/download_data.py --check
```

Configure providers by copying `.env.example` to `.env` (DeepSeek / GLM /
optional local Ollama). **No API key is required for tests or CI** — all offline
paths use a scripted fake client through the same `ChatClient` protocol seam.

## Engineering practices

- Trunk-based git flow: protected `main`, feature branches, PR + CI gate,
  conventional commits, SemVer tags & changelogs
- Issue-driven development; every bug filed as an issue with repro + logs
- Data never committed: download scripts + `manifest.json` provenance
  (source URL / license / checksum / date)
- Design decisions recorded as ADRs in `docs/adr/`
- Eval-gated changes: agent-behavior PRs must attach eval results

## License

MIT. See [data/README.md](data/README.md) for third-party dataset licenses.
