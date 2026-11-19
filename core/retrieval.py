"""Hybrid retrieval core — book §3.2 (Chunking / 稠密 / 稀疏 / 混合检索).

Design anchors:
- Chunking: paragraph-aware split with overlap, sized for bge-small-zh's 512-token
  context (§3.2.1).
- Sparse: BM25Okapi over CJK-aware tokens (jieba when available) (§3.2.3).
- Dense: injectable Embedder protocol; the real implementation is a local
  bge-small-zh (CPU) with the model-card query prefix (§3.2.2).
- Fusion: Reciprocal Rank Fusion over per-mode rankings (§3.2.4 混合检索) —
  rank-based fusion avoids score-scale mismatch between BM25 and cosine.
- Store: local files (chunks.jsonl + bm25 pickle + vectors .npy). See
  docs/adr/0004-local-hybrid-store-before-vector-db.md for why not Chroma yet.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

try:  # jieba is optional at import time so tests can run a regex tokenizer
    import jieba  # type: ignore

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    _HAS_JIEBA = False

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    source_url: str
    source_name: str
    license: str
    fetched_at: str
    position: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    ranks: dict  # per-mode ranks for explainability, e.g. {"bm25": 1, "dense": 3}


# --------------------------------------------------------------------------- #
# Tokenization (sparse channel)
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """CJK-aware tokenization: jieba for Chinese, regex for latin/digits."""
    if _HAS_JIEBA and _CJK.search(text):
        return [t for t in jieba.lcut(text.lower()) if t.strip()]
    return [t.lower() for t in _WORD.findall(text)]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def chunk_text(
    text: str,
    *,
    max_chars: int = 450,
    overlap: int = 80,
) -> list[str]:
    """Paragraph-aware split with character overlap (bge-small-zh ≈ 450 CJK chars/512 tok)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    pieces: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > max_chars:
            # hard-split long paragraphs with overlap
            step = max_chars - overlap
            for i in range(0, len(para), step):
                pieces.append(para[i : i + max_chars])
                if i + max_chars >= len(para):
                    break
            buf = ""
            continue
        if len(buf) + len(para) + 1 > max_chars and buf:
            pieces.append(buf)
            buf = buf[-overlap:] + "\n" + para if overlap else para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf.strip():
        pieces.append(buf)
    return pieces


# --------------------------------------------------------------------------- #
# Embedders (dense channel) — injected so tests/offline run without torch
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class FakeHashEmbedder:
    """Deterministic n-gram hashing embedder — offline tests & CI only.

    Uses zlib.crc32 (stable across processes) rather than hash(), whose
    PYTHONHASHSEED randomization would make rankings non-reproducible.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _encode(self, text: str) -> np.ndarray:
        import zlib

        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for i in range(len(toks) - 1):
            gram = f"{toks[i]}_{toks[i + 1]}"
            vec[zlib.crc32(gram.encode("utf-8")) % self.dim] += 1.0
        # CJK char unigrams (weight 0.5): robust to jieba segmentation variance —
        # e.g. query "退款" must still match a passage tokenized as "全额退款".
        for ch in text:
            if _CJK.match(ch):
                vec[zlib.crc32(ch.encode("utf-8")) % self.dim] += 0.5
        if not toks and not any(_CJK.match(ch) for ch in text):
            vec[0] = 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._encode(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text)


class BGEEmbedder:
    """Local bge-small-zh-v1.5 via sentence-transformers (CPU-friendly).

    The query prefix follows the model card — bge models are trained with this
    instruction for retrieval queries; skipping it measurably hurts recall.
    HF_ENDPOINT (e.g. https://hf-mirror.com) is honored via env for CN networks.
    """

    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name, device=device)
        # ST 5.x renamed the getter; fall back for older installs
        getter = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        self.dim = getter()

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([self.QUERY_PREFIX + text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0]


# --------------------------------------------------------------------------- #
# Hybrid retriever
# --------------------------------------------------------------------------- #


def _rrf_fuse(rankings: list[dict[str, int]], k: int) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for chunk_id, rank in ranking.items():  # rank starts at 1
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """BM25 + dense in one queryable object; per-mode ranks kept for debugging."""

    DENSE_CANDIDATES = 50  # per-mode candidate depth before RRF (§3.2.4)

    def __init__(self, embedder: Embedder | None = None, rrf_k: int = 60) -> None:
        # embedder=None → sparse-only instance (bm25 mode); dense/hybrid need one.
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25 = None
        self._tokens: list[list[str]] = []
        self._matrix: np.ndarray | None = None

    # -- build ----------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._by_id.update({c.id: c for c in chunks})
        self._tokens = [tokenize(c.text) for c in self.chunks]
        from rank_bm25 import BM25Okapi  # lazy: only needed at build time

        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        self._matrix = (
            self.embedder.encode_passages([c.text for c in self.chunks])
            if self.embedder is not None
            else None
        )

    # -- query ----------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Retrieved]:
        if not self.chunks:
            return []
        if mode == "bm25":
            ranking = self._bm25_ranking(query)
            fused = {cid: 1.0 / (self.rrf_k + r) for cid, r in ranking.items()}
            ranks = {"bm25": ranking}
        elif mode == "dense":
            ranking = self._dense_ranking(query)
            fused = {cid: 1.0 / (self.rrf_k + r) for cid, r in ranking.items()}
            ranks = {"dense": ranking}
        else:
            b, d = self._bm25_ranking(query), self._dense_ranking(query)
            fused = _rrf_fuse([b, d], self.rrf_k)
            ranks = {"bm25": b, "dense": d}
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Retrieved(
                chunk=self._by_id[cid], score=score, ranks={m: r.get(cid) for m, r in ranks.items()}
            )
            for cid, score in top
        ]

    def _bm25_ranking(self, query: str) -> dict[str, int]:
        scores = self._bm25.get_scores(tokenize(query)) if self._bm25 else []
        order = np.argsort(scores)[::-1][: self.DENSE_CANDIDATES]
        return {self.chunks[i].id: rank + 1 for rank, i in enumerate(order) if scores[i] > 0}

    def _dense_ranking(self, query: str) -> dict[str, int]:
        if self.embedder is None or self._matrix is None:
            return {}
        qv = self.embedder.encode_query(query)
        sims = self._matrix @ qv
        order = np.argsort(sims)[::-1][: self.DENSE_CANDIDATES]
        return {self.chunks[i].id: rank + 1 for rank, i in enumerate(order) if sims[i] > 0}

    # -- persistence ----------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "chunks.jsonl").write_text(
            "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in self.chunks),
            encoding="utf-8",
        )
        with (dir_path / "bm25.pkl").open("wb") as f:
            pickle.dump(self._bm25, f)
        if self._matrix is not None:
            np.save(dir_path / "vectors.npy", self._matrix)
        (dir_path / "meta.json").write_text(
            json.dumps(
                {"count": len(self.chunks), "dim": self.embedder.dim, "rrf_k": self.rrf_k},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path, embedder: Embedder | None = None) -> HybridRetriever:
        dir_path = Path(dir_path)
        chunks = [
            Chunk(**json.loads(line))
            for line in (dir_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        inst = cls(embedder)
        inst.chunks = chunks
        inst._by_id = {c.id: c for c in chunks}
        with (dir_path / "bm25.pkl").open("rb") as f:
            inst._bm25 = pickle.load(f)  # noqa: S301 — own artifact, trusted
        vec_path = dir_path / "vectors.npy"
        inst._matrix = np.load(vec_path) if vec_path.exists() else None
        return inst
