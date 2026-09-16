"""Shared text-normalization helpers for the escalation backend.

Single source of truth for cleaning escalation text so that:
- the FAISS index is built from cleaned cause/action text, and
- user queries are cleaned exactly the same way before embedding.

Previously this logic lived only in ``rag_service``, which meant the index
was built from RAW typo-laden text while queries were cleaned at search
time — a mismatch that hurt retrieval accuracy.
"""
from __future__ import annotations

import re
from typing import Any

# Common typo / abbreviation corrections applied to solutions and problem text.
_CORRECTIONS = {
    "recieved": "received",
    "recieve": "receive",
    "disaprch": "dispatch",
    "disapatch": "dispatch",
    "dispath": "dispatch",
    "worang": "wrong",
    "thki": "thickness",
    "reqd": "required",
    "req": "required",
    "obs": "observed",
    "observe": "observed",
    "visually": "visually observed",
    "premaching": "pre-machining",
    "pre-maching": "pre-machining",
    "maching": "machining",
    "cutomer": "customer",
    "cutomers": "customers",
    "thier": "their",
    "ammend": "amend",
    "ammendment": "amendment",
    "appliable": "applicable",
    "clernce": "clearance",
    "clernces": "clearances",
    "clearence": "clearance",
    "mili": "million",
    "deliver": "deliver",
    "improtant": "important",
    "notifcation": "notification",
    "seperation": "separation",
    "sep": "separation",
    "expidite": "expedite",
    "exsiting": "existing",
    "exsisting": "existing",
    "availkable": "available",
    "avialable": "available",
    "avabile": "available",
    "resloved": "resolved",
    "resolv": "resolve",
    "serch": "search",
    "whol": "whole",
    "tutne": "breaking",
    "puchha": "asked",
    "khali": "free",
    "kiye": "for",
    "karne": "to do",
    "baat": "thing",
    "ke": "of",
    "hai": "is",
    "hoga": "will be",
    "aata": "comes",
    "kitna": "how much",
    "lagne": "required",
    "mujhe": "I",
    "kilye": "for",
    "nahi": "not",
    "ata": "comes",
    "koi": "any",
    "chij": "item",
    "usme": "in it",
    "dhikhata": "shows",
    "job card": "job card",
    "mina": "mean",
    "handover": "hand over",
}


def _fix_punctuation(text: str) -> str:
    """Fix spacing around punctuation and sentence capitalization."""
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?=[A-Za-z0-9])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not parts:
        return ""

    cleaned_parts = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            part = part[0].upper() + part[1:] if part else part
        cleaned_parts.append(part)

    return " ".join(cleaned_parts).strip()


def summarize_text(text: Any, max_sentences: int = 1) -> str:
    """Return a short, readable summary for display."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        return cleaned

    if len(sentences) <= max_sentences:
        return sentences[0]

    return " ".join(sentences[:max_sentences]).strip()


def clean_text(text: Any) -> str:
    """Normalize, format, and clean a text field (cause/problem or solution).

    - Collapses runs of whitespace/newlines into single spaces.
    - Trims leading/trailing whitespace.
    - Fixes common typos and abbreviations.
    - Fixes spacing around punctuation and capitalizes sentences.
    - Returns the full text (no truncation, no ellipsis).
    """
    if text is None:
        return ""
    raw = str(text)
    normalized = " ".join(raw.split()).strip()
    if not normalized:
        return ""

    lowered = normalized
    for wrong, right in _CORRECTIONS.items():
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        lowered = pattern.sub(right, lowered)

    lowered = re.sub(r"\bhand over\b", "hand over", lowered)
    lowered = re.sub(r"\bcan not\b", "cannot", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\bdo not\b", "do not", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\bwill not\b", "will not", lowered, flags=re.IGNORECASE)

    return _fix_punctuation(lowered)


def tokenize(text: str) -> set[str]:
    """Lower-cased alphanumeric token set used by the keyword scorer."""
    return {token for token in re.findall(r"[a-zA-Z0-9]+", str(text).lower()) if token}
