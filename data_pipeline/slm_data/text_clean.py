"""Text normalisation for resumes / JDs / queries (PDF-to-text artefacts, unicode junk)."""
from __future__ import annotations

import hashlib
import re
import unicodedata

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "•": "-", "●": "-", "▪": "-", "■": "-", "‣": "-", "⁃": "-", "◦": "-",
    "": "-", "": "-", "": "-", "": "-", "➢": "-", "❖": "-", "·": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
}
_TRANS = str.maketrans({**_LIGATURES, **_PUNCT})
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINL = re.compile(r"\n{3,}")
_TRAIL = re.compile(r"[ \t]+\n")
_REPEAT_PUNCT = re.compile(r"([-_=.*~#])\1{4,}")


def clean_text(text: str | None, max_chars: int | None = None) -> str:
    if not text:
        return ""
    t = str(text).translate(_TRANS)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _CTRL.sub(" ", t)
    t = _REPEAT_PUNCT.sub(r"\1\1\1", t)
    t = _MULTISPACE.sub(" ", t)
    t = _TRAIL.sub("\n", t)
    t = _MULTINL.sub("\n\n", t)
    t = t.strip()
    if max_chars and len(t) > max_chars:
        cut = t.rfind("\n", 0, max_chars)
        t = t[: cut if cut > max_chars * 0.8 else max_chars]
    return t


def clean_inline(text: str | None, max_chars: int | None = None) -> str:
    """Single-line clean (queries, reasons, field values)."""
    t = clean_text(text).replace("\n", " ")
    t = _MULTISPACE.sub(" ", t).strip()
    if max_chars and len(t) > max_chars:
        t = t[: max_chars - 1].rstrip() + "…"
    return t


def quality_ok(text: str, min_alpha: float, min_ascii: float) -> bool:
    if not text:
        return False
    n = len(text)
    alpha = sum(c.isalpha() for c in text) / n
    if alpha < min_alpha * 0.8:  # alpha ratio of normal prose ~0.75-0.8
        return False
    if min_ascii > 0:
        ascii_ = sum(ord(c) < 128 for c in text) / n
        if ascii_ < min_ascii:
            return False
    return True


_NORM = re.compile(r"[^a-z0-9]+")


def content_hash(text: str) -> bytes:
    """Hash on a heavily normalised form so trivially different copies collide."""
    return hashlib.blake2b(_NORM.sub("", text.lower()).encode(), digest_size=12).digest()


def normalize_query(q: str | None) -> str:
    return clean_inline(q).strip().strip('"').strip()
