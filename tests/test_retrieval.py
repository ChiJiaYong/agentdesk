"""Offline tests for the hybrid retrieval core (FakeHashEmbedder, no torch)."""

from __future__ import annotations

import json

import pytest

from core.retrieval import Chunk, FakeHashEmbedder, HybridRetriever, chunk_text, tokenize


def _mk(cid: str, text: str, doc: str = "d1") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        doc_id=doc,
        source_url="https://example.com",
        source_name="test",
        license="test",
        fetched_at="2026-11-19",
    )


CORPUS = [
    _mk("c1", "未使用商品可在签收后30天内申请全额退款，退款原路返回。"),
    _mk("c2", "订单发货后可在订单详情页查看物流单号和配送进度。"),
    _mk("c3", "电子商品拆封后不支持七天无理由退货。"),
    _mk("c4", "How do I return an item within 30 days of delivery for a full refund?"),
    _mk("c5", "会员积分每年12月31日过期清零，请及时兑换。"),
]


@pytest.fixture()
def retriever() -> HybridRetriever:
    r = HybridRetriever(FakeHashEmbedder())
    r.add(CORPUS)
    return r


def test_tokenize_cjk_vs_latin() -> None:
    zh = tokenize("七天无理由退货")
    assert len(zh) >= 3  # jieba segments CJK when available
    en = tokenize("Full REFUND within 30 days!")
    assert "refund" in en and "30" in en


def test_chunk_text_overlap_and_size() -> None:
    # Each paragraph is 300+ chars — longer than max_chars, forcing hard splits.
    long_text = "\n\n".join(f"段落{i}，" + "内容" * 150 for i in range(2))
    pieces = chunk_text(long_text, max_chars=200, overlap=40)
    assert all(len(p) <= 220 for p in pieces)
    assert len(pieces) >= 4  # 2 paragraphs × ≥2 hard splits each
    # overlap: consecutive pieces share a region
    assert pieces[0][-40:] in pieces[1] or pieces[1][:40] in pieces[0]


def test_bm25_mode_finds_keyword_doc(retriever: HybridRetriever) -> None:
    hits = retriever.retrieve("物流单号 在哪看", k=2, mode="bm25")
    assert hits[0].chunk.id == "c2"


def test_dense_mode_finds_paraphrase(retriever: HybridRetriever) -> None:
    hits = retriever.retrieve("怎么退款 全额", k=2, mode="dense")
    assert hits[0].chunk.id == "c1"


def test_hybrid_mode_fuses_both(retriever: HybridRetriever) -> None:
    hits = retriever.retrieve("30天 退款", k=5, mode="hybrid")
    top_ids = [h.chunk.id for h in hits]
    assert "c1" in top_ids and "c4" in top_ids  # zh + en paraphrases both surface
    assert hits[0].ranks["bm25"] is not None or hits[0].ranks["dense"] is not None


def test_save_load_roundtrip(tmp_path, retriever: HybridRetriever) -> None:
    retriever.save(tmp_path)
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["count"] == 5
    loaded = HybridRetriever.load(tmp_path, FakeHashEmbedder())
    hits = loaded.retrieve("退款", k=1, mode="hybrid")
    assert hits and hits[0].chunk.id in {"c1", "c3"}
