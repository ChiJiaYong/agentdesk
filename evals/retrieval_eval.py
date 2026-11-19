"""Retrieval evaluation — hit@k / MRR across bm25 / dense / hybrid modes.

Golden set (two sources, both real-data grounded):
1. Regulation queries: hand-written natural-language questions; the gold chunk
   is resolved *dynamically* by article-text regex, so changing the chunking
   does not silently invalidate the golden set.
2. Amazon QA sample: real customer questions from the UCSD dataset; the gold
   chunk is the QA pair itself (the seller-verified answer).

Zero LLM cost — this is the ablation that justifies the hybrid design
(book §3.2.4) and becomes the W2 eval report artifact.

Usage:
    uv run python -m evals.retrieval_eval            # full (needs --extra ml)
    uv run python -m evals.retrieval_eval --modes bm25  # light run (no torch)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

from core.retrieval import Chunk, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data" / "processed" / "kb_chunks.jsonl"
REPORT_DIR = ROOT / "data" / "processed"

# (query, doc_id_prefix, article_regex) — gold = first chunk in that doc matching regex
REGULATION_QUERIES = [
    ("七天无理由退货 哪些商品不适用", "consumer_protection_law", r"第二十五条"),
    ("退货运费应该谁来承担", "consumer_protection_law", r"第二十[四五]条"),
    ("商家欺诈 退一赔三 最低五百", "consumer_protection_law", r"第五十五条"),
    ("网购商品有瑕疵 多久内可以退货", "consumer_protection_law", r"第二十[四六]条"),
    ("格式条款 什么情况下无效", "consumer_protection_law", r"第二十六条"),
    ("个人信息 经营者可以随便用吗", "consumer_protection_law", r"第二十九条"),
    ("三包有效期 从什么时候开始计算", "three_guarantees_regulation", r"第八条"),
    ("买了几天下雨进水 7日内性能故障能退货吗", "three_guarantees_regulation", r"第九条"),
    ("15日内出现性能故障 只能修吗", "three_guarantees_regulation", r"第十条"),
    ("修了两次还是坏 能换新机吗", "three_guarantees_regulation", r"第十一条"),
    ("什么情况不实行三包 只能收费修理", "three_guarantees_regulation", r"第十七条"),
    ("自动续费 厂家应该提前提醒吗", "consumer_protection_law_impl_regulation", r"第十条"),
    ("商家停业 预付款余额怎么办", "consumer_protection_law_impl_regulation", r"第二十[一二]条"),
    ("押金 能设置条件不退吗", "consumer_protection_law_impl_regulation", r"第二十条"),
    ("没同意就打推销电话 违规吗", "consumer_protection_law_impl_regulation", r"第二十[三四]条"),
    ("赠品有问题 能要求修吗", "consumer_protection_law_impl_regulation", r"第七条"),
]


def load_chunks() -> list[Chunk]:
    chunks = []
    for line in KB.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        chunks.append(
            Chunk(
                id=d["id"],
                text=d["text"],
                doc_id=d["doc_id"],
                source_url=d["source_url"],
                source_name=d["source_name"],
                license=d["license"],
                fetched_at=d["fetched_at"],
                position=d.get("position", 0),
                extra=d.get("extra", {}),
            )
        )
    return chunks


def resolve_gold(chunks: list[Chunk], doc_prefix: str, pattern: str) -> str | None:
    rx = re.compile(pattern)
    for c in chunks:
        if c.doc_id.startswith(doc_prefix) and rx.search(c.text):
            return c.id
    return None


def build_golden(chunks: list[Chunk], qa_sample: int) -> tuple[list[dict], list[str]]:
    golden, skipped = [], []
    for query, doc, rx in REGULATION_QUERIES:
        gold = resolve_gold(chunks, doc, rx)
        if gold:
            golden.append({"query": query, "gold": gold, "type": "regulation"})
        else:
            skipped.append(query)
    # real user questions as queries; gold = their own QA chunk
    qa = [c for c in chunks if c.doc_id == "amazon_qa_electronics" and len(c.text) > 60]
    rng = random.Random(42)
    for c in rng.sample(qa, min(qa_sample, len(qa))):
        question = c.text.split("A:", 1)[0].removeprefix("Q:").strip()
        if 8 <= len(question) <= 120:
            golden.append({"query": question, "gold": c.id, "type": "amazon_qa"})
    return golden, skipped


def mrr_and_hit(retriever: HybridRetriever, golden: list[dict], mode: str, k: int = 5) -> dict:
    hits, rr_sum = 0, 0.0
    for item in golden:
        results = retriever.retrieve(item["query"], k=k, mode=mode)
        rank = next((i + 1 for i, r in enumerate(results) if r.chunk.id == item["gold"]), None)
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
    n = max(len(golden), 1)
    return {"hit@5": round(hits / n, 4), "mrr": round(rr_sum / n, 4), "n_evaluated": len(golden)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", default="bm25,dense,hybrid")
    ap.add_argument("--qa-sample", type=int, default=15)
    args = ap.parse_args()

    if not KB.exists():
        print("[error] kb_chunks.jsonl missing — run scripts/build_knowledge_base.py first")
        return 2

    chunks = load_chunks()
    golden, skipped = build_golden(chunks, args.qa_sample)
    print(
        f"[info] corpus={len(chunks)} chunks, golden={len(golden)} "
        f"(skipped {len(skipped)} regulation queries without gold)"
    )

    results = {
        "built_at": datetime.now(UTC).isoformat(),
        "corpus": len(chunks),
        "golden": len(golden),
        "skipped_golden": skipped,
        "modes": {},
    }
    for mode in args.modes.split(","):
        needs_dense = mode in {"dense", "hybrid"}
        if needs_dense:
            try:
                from core.retrieval import BGEEmbedder
            except ImportError:
                print(f"[warn] sentence-transformers unavailable — skipping {mode}")
                continue
            retriever = HybridRetriever(BGEEmbedder())
        else:
            retriever = HybridRetriever(None)  # sparse-only instance
        retriever.add(chunks)
        results["modes"][mode] = mrr_and_hit(retriever, golden, mode)
        print(f"{mode:>8}: {results['modes'][mode]}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "retrieval_eval_report.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
