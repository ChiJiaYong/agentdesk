"""Data download script — real, licensed, provenance-tracked datasets.

Usage:
    uv run python scripts/download_data.py --check   # verify URLs/tokens only
    uv run python scripts/download_data.py           # fetch everything missing

Design rules (docs/adr/0003-data-not-in-git.md):
- raw data never enters git; downloads are recorded in data/manifest.json with
  size + sha256 + license + fetched date (reproducibility & compliance).
- failures are explicit: missing Kaggle token or unreachable source exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

# UCSD McAuley Lab Amazon QA (public direct download, no registration).
AMAZON_QA_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/qa"
AMAZON_QA_CANDIDATES = [
    f"{AMAZON_QA_BASE}/qa_electronics.json.gz",
    f"{AMAZON_QA_BASE}/qa_Electronics.json.gz",
]

OLIST_KAGGLE_SLUG = "olistbr/brazilian-ecommerce"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_url(candidates: list[str]) -> str | None:
    """HEAD-probe candidate URLs (file naming changed across dataset releases)."""
    for url in candidates:
        try:
            r = httpx.head(url, follow_redirects=True, timeout=30)
            if r.status_code == 200:
                return url
        except httpx.HTTPError:
            continue
    return None


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest


def fetch_amazon_qa(check_only: bool, manifest: dict) -> None:
    dest = DATA_RAW / "amazon_qa" / "qa_electronics.json.gz"
    if dest.exists():
        print(f"[skip] amazon_qa already present: {dest.name}")
        return
    url = resolve_url(AMAZON_QA_CANDIDATES)
    if url is None:
        print("[error] Amazon QA endpoint unreachable (tried all candidates)")
        sys.exit(2)
    print(f"[ok]   amazon_qa URL reachable: {url}")
    if check_only:
        return
    print("[dl]   downloading amazon_qa (Electronics, ~100+ MB) ...")
    download(url, dest)
    manifest["amazon_qa"] = {
        "source": url,
        "license": "public academic dataset (cite Wan & McAuley ICDM 2016; McAuley & Yang WWW 2016)",
        "fetched_at": datetime.now(UTC).isoformat(),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_of(dest),
    }
    print(f"[done] amazon_qa -> {dest}")


def fetch_olist(check_only: bool, manifest: dict) -> None:
    # Auth: kaggle CLI >=2.x natively reads KAGGLE_API_TOKEN (KGAT_ tokens) from
    # the environment; classic kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY also work.
    # We only check *presence* here — the token value never enters logs.
    has_token = (
        bool(os.getenv("KAGGLE_API_TOKEN"))
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    )
    if not has_token:
        print(
            "[warn] Kaggle auth not configured — skipping Olist.\n"
            "       Options: KAGGLE_API_TOKEN in .env (KGAT_ token, CLI>=2.x),\n"
            "       or kaggle.json in ~/.kaggle/, or KAGGLE_USERNAME/KAGGLE_KEY in .env."
        )
        return
    dest_dir = DATA_RAW / "olist"
    if dest_dir.exists() and any(dest_dir.glob("*.csv")):
        print("[skip] olist already present")
        return
    print("[dl]   downloading olist via kaggle CLI ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                OLIST_KAGGLE_SLUG,
                "-p",
                str(dest_dir),
                "--unzip",
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[error] kaggle CLI failed: {exc}. `uv tool install kaggle` first.")
        sys.exit(3)
    manifest["olist"] = {
        "source": f"https://www.kaggle.com/datasets/{OLIST_KAGGLE_SLUG}",
        "license": "CC BY-NC-SA 4.0",
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256_note": "per-file checksums added in build_business_db.py (W2)",
    }
    print(f"[done] olist -> {dest_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify reachability only")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")  # KAGGLE_API_TOKEN etc.; .env is gitignored
    manifest: dict = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    fetch_amazon_qa(args.check, manifest)
    fetch_olist(args.check, manifest)

    if not args.check and manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
