"""Fetch real Chinese consumer-protection regulation texts from official sites.

Produces markdown files under data/raw/regulations/ with a provenance header
(source URL, license note, fetched date). One-off ingest script, kept in the
repo for reproducibility (re-run = fresh fetch + checksum refresh).

Usage:  uv run python scripts/fetch_regulations.py
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "regulations"

# (filename, url, source_name, start_marker, end_marker)
DOCS = [
    (
        "consumer_protection_law.md",
        "https://www.samr.gov.cn/zfjcj/tzgg/art/2023/art_615af9ed6bcd4974bf853dd2e02bc663.html",
        "国家市场监督管理总局（转载全国人大文本）",
        "（1993年10月31日",
        "本法自1994年1月1日起施行",
    ),
    (
        "three_guarantees_regulation.md",
        "https://www.samr.gov.cn/zw/zfxxgk/zc/xzgfxwj/art/2019/art_f943e66de125437d92df701e1654c47c.html",
        "国家市场监督管理总局",
        "国经贸〔1995〕458号",
        "以本规定为准",
    ),
    (
        "consumer_protection_law_impl_regulation.md",
        "https://www.gov.cn/zhengce/zhengceku/202403/content_6940159.htm",
        "中国政府网（国务院政策文件库）",
        "中华人民共和国国务院令 第778号",
        "本条例自2024年7月1日起施行",
    ),
]


class TextExtractor(HTMLParser):
    """Minimal HTML → text: keep block-level breaks, drop scripts/styles."""

    BLOCK = {"p", "div", "br", "td", "tr", "li", "h1", "h2", "h3", "h4", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: dict) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def extract_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    text = unescape("".join(parser.parts))
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def slice_between(text: str, start: str, end: str) -> str:
    """Whitespace-insensitive marker slicing.

    Official pages insert unpredictable spaces/newlines inside key phrases
    (e.g. "国经贸〔1995〕 458号"), so match against a whitespace-stripped copy
    and map hit positions back to the original text.
    """
    compact = re.sub(r"\s+", "", text)
    c_start, c_end = re.sub(r"\s+", "", start), re.sub(r"\s+", "", end)
    s = compact.find(c_start)
    e = compact.find(c_end, s + 1) if s >= 0 else -1
    if s < 0 or e < 0:
        raise ValueError(f"markers not found: {start[:12]}... / {end[:12]}...")
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    return text[positions[s] : positions[e + len(c_end) - 1] + 1]


def main() -> int:
    parser_ = argparse.ArgumentParser(description=__doc__)
    parser_.add_argument("--check", action="store_true", help="probe reachability only")
    args = parser_.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for fname, url, source_name, start, end in DOCS:
        dest = OUT / fname
        if dest.exists():
            print(f"[skip] {fname} already present")
            continue
        try:
            resp = httpx.get(
                url,
                timeout=60,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgentDesk-data-ingest",
                },
            )
            resp.raise_for_status()
            if args.check:
                print(f"[ok]   {fname}: {url}")
                continue
            body = slice_between(extract_text(resp.text), start, end)
            header = (
                f"---\n"
                f"source_url: {url}\n"
                f"source_name: {source_name}\n"
                f"license: 官方公开政府文件\n"
                f"fetched_at: {datetime.now(UTC).isoformat()}\n"
                f"---\n\n"
            )
            dest.write_text(header + body + "\n", encoding="utf-8")
            print(f"[done] {fname}: {len(body)} chars")
        except Exception as exc:  # noqa: BLE001 — ingest script must report and continue
            failures += 1
            print(f"[error] {fname}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
