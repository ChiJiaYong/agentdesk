"""Build the knowledge-base chunk corpus from real ingested sources.

Sources (all provenance-tracked, see data/README.md):
- data/raw/regulations/*.md            — official CN consumer-protection texts
- data/raw/amazon_qa/qa_electronics…   — real customer Q&A (sampled)
- data/raw/fastapi_docs (optional)     — developer docs (md)
- data/raw/gitlab_handbook (optional)  — HR/IT handbook (md)

Output: data/processed/kb_chunks.jsonl (Chunk JSON per line) + kb_stats.json.

Usage:
    uv run python scripts/build_knowledge_base.py [--qa-sample 3000]
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from core.retrieval import Chunk, chunk_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        head, _, body = text[3:].partition("---")
        meta = {}
        for line in head.strip().splitlines():
            key, _, value = line.partition(":")
            if key.strip():
                meta[key.strip()] = value.strip()
        return meta, body.strip()
    return {}, text.strip()


def from_regulations() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted((RAW / "regulations").glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        for i, piece in enumerate(chunk_text(body)):
            chunks.append(
                Chunk(
                    id=f"reg-{path.stem}-{i:03d}",
                    text=piece,
                    doc_id=path.stem,
                    source_url=meta.get("source_url", ""),
                    source_name=meta.get("source_name", path.stem),
                    license=meta.get("license", "官方公开政府文件"),
                    fetched_at=meta.get("fetched_at", ""),
                    position=i,
                )
            )
    return chunks


def from_amazon_qa(sample: int) -> list[Chunk]:
    path = RAW / "amazon_qa" / "qa_electronics.json.gz"
    if not path.exists():
        print("[warn] amazon_qa missing — skipping")
        return []
    source_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/qa/"
    license_ = "public academic dataset (ICDM 2016 / WWW 2016 citation)"
    chunks: list[Chunk] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no >= sample:
                break
            try:
                rec = (
                    json.loads(line.replace("'", '"'))
                    if line.startswith("{ ")
                    else json.loads(line)
                )
            except json.JSONDecodeError:
                # the UCSD release is python-repr formatted in places; be tolerant
                import ast

                try:
                    rec = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            q, a = (rec.get("question") or "").strip(), (rec.get("answer") or "").strip()
            if not q or not a:
                continue
            chunks.append(
                Chunk(
                    id=f"qa-{line_no:07d}",
                    text=f"Q: {q}\nA: {a}",
                    doc_id="amazon_qa_electronics",
                    source_url=source_url,
                    source_name="Amazon QA (McAuley Lab, UCSD)",
                    license=license_,
                    fetched_at=datetime.now(UTC).isoformat(),
                    position=line_no,
                    extra={"question_type": rec.get("questionType", "")},
                )
            )
    return chunks


def from_markdown_dir(
    dir_path: Path,
    prefix: str,
    base_url: str,
    source_name: str,
    license_: str,
    max_files: int = 200,
) -> list[Chunk]:
    if not dir_path.exists():
        print(f"[warn] {dir_path.name} missing — skipping")
        return []
    chunks: list[Chunk] = []
    files = sorted(dir_path.rglob("*.md"))[:max_files]
    for path in files:
        rel = path.relative_to(dir_path).as_posix()
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8", errors="ignore"))
        for i, piece in enumerate(chunk_text(body)):
            chunks.append(
                Chunk(
                    id=f"{prefix}-{abs(hash(rel)) % 10_000_000:07d}-{i:03d}",
                    text=piece,
                    doc_id=f"{prefix}:{rel}",
                    source_url=f"{base_url}/{rel}",
                    source_name=source_name,
                    license=license_,
                    fetched_at=datetime.now(UTC).isoformat(),
                    position=i,
                )
            )
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-sample", type=int, default=3000)
    args = ap.parse_args()

    chunks = from_regulations()
    reg_n = len(chunks)
    chunks += from_amazon_qa(args.qa_sample)
    qa_n = len(chunks) - reg_n
    chunks += from_markdown_dir(
        RAW / "fastapi_docs",
        "fastapi",
        "https://github.com/fastapi/fastapi/blob/master/docs/en/docs",
        "FastAPI official docs",
        "MIT",
    )
    fa_n = len(chunks) - reg_n - qa_n
    chunks += from_markdown_dir(
        RAW / "gitlab_handbook",
        "gitlab",
        "https://handbook.gitlab.com/handbook",
        "GitLab Handbook",
        "CC BY-SA 4.0",
    )
    gl_n = len(chunks) - reg_n - qa_n - fa_n

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "kb_chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "id": c.id,
                        "text": c.text,
                        "doc_id": c.doc_id,
                        "source_url": c.source_url,
                        "source_name": c.source_name,
                        "license": c.license,
                        "fetched_at": c.fetched_at,
                        "position": c.position,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats = {
        "total_chunks": len(chunks),
        "regulations": reg_n,
        "amazon_qa": qa_n,
        "fastapi_docs": fa_n,
        "gitlab_handbook": gl_n,
        "built_at": datetime.now(UTC).isoformat(),
        "avg_chars": round(sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
    }
    (OUT_DIR / "kb_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
