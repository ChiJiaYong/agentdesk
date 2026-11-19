"""PII redaction middleware (book §3.1.8 日志脱敏).

Customer messages may contain phone numbers / emails. Everything sent to a
third-party LLM API goes through ``redact`` first: PII is replaced with stable
placeholders (``[PHONE_1]``, ``[EMAIL_1]``…) and the mapping is kept locally so
the human workbench can render the original text. Order ids are NOT redacted —
they are business references required by tools (and are not contact PII).

Pure-regex W4 implementation (CPU, zero deps, fully offline). A NER upgrade
path (HanLP / compact-BERT) can replace ``PATTERNS`` without touching callers.
"""

from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"  # CN mobile
            r"|(?<!\d)\+?\d{1,3}[- ]?\d{3,4}[- ]?\d{4}(?!\d)"  # intl-ish loose
        ),
    ),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


def redact(text: str) -> tuple[str, dict[str, str]]:
    """Return (redacted_text, mapping placeholder -> original)."""
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    def _sub(match: re.Match[str]) -> str:
        kind = next(k for k, rx in PATTERNS if rx.pattern == match.re.pattern)
        counters[kind] = counters.get(kind, 0) + 1
        placeholder = f"[{kind}_{counters[kind]}]"
        mapping[placeholder] = match.group(0)
        return placeholder

    result = text
    for _, rx in PATTERNS:
        result = rx.sub(_sub, result)
    return result, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Render redacted text back to PII form — workbench display only."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def contains_pii(text: str) -> bool:
    return any(rx.search(text) for _, rx in PATTERNS)
