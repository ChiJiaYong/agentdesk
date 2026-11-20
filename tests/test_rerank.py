"""Rerank stage tests (§3.2.4 阶段三) — run where the reranker model is cached."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.retrieval import (
    Chunk,
    CrossEncoderReranker,
    FakeHashEmbedder,
    HybridRetriever,
)

_RERANKER_CACHED = (
    any(
        p.is_dir()
        for p in (Path.home() / ".cache" / "huggingface" / "hub").glob(
            "models--BAAI--bge-reranker-base*"
        )
    )
    if (Path.home() / ".cache" / "huggingface" / "hub").exists()
    else False
)

pytestmark = pytest.mark.skipif(not _RERANKER_CACHED, reason="bge-reranker-base not cached")


def _retriever(reranker):
    r = HybridRetriever(FakeHashEmbedder(), reranker=reranker)
    r.add(
        [
            Chunk(
                id="c1",
                text="Unused items may be returned within 30 days for a full refund.",
                doc_id="pol",
                source_url="x",
                source_name="policy",
                license="t",
                fetched_at="2026",
            ),
            Chunk(
                id="c2",
                text="Member points expire on December 31 every year.",
                doc_id="pol",
                source_url="x",
                source_name="policy",
                license="t",
                fetched_at="2026",
            ),
            Chunk(
                id="c3",
                text="Shipping takes 5 to 10 business days depending on the destination.",
                doc_id="pol",
                source_url="x",
                source_name="policy",
                license="t",
                fetched_at="2026",
            ),
            Chunk(
                id="c4",
                text="We support credit card, debit card and bank transfer payments.",
                doc_id="pol",
                source_url="x",
                source_name="policy",
                license="t",
                fetched_at="2026",
            ),
        ]
    )
    return r


def test_reranker_improves_ranking_on_semantic_gap() -> None:
    rr = CrossEncoderReranker()
    retriever = _retriever(rr)
    hits = retriever.retrieve("Can I get a full refund within a month?", k=2, mode="hybrid")
    assert hits[0].chunk.id == "c1", "reranker must push the semantically-correct chunk first"


def test_no_reranker_is_zero_behavior_change() -> None:
    retriever = _retriever(None)
    hits = retriever.retrieve("full refund within 30 days", k=2, mode="hybrid")
    assert hits and hits[0].chunk.id == "c1"  # pipeline unchanged when reranker is None
