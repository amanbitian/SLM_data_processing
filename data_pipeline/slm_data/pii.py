"""PII redaction. The model never needs a candidate's contact details or name to judge
relevance or extract skills/education/experience, so they are replaced by placeholders.
This also stops the model from memorising and regurgitating personal data."""
from __future__ import annotations

import re
from typing import Iterable

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+\s?@\s?[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:linkedin\.com|github\.com|gitlab\.com|behance\.net|"
    r"dribbble\.com|medium\.com|kaggle\.com|leetcode\.com)/\S*",
    re.I,
)
# phone: +country, separators, 8-15 digits overall; avoid years / date ranges
PHONE = re.compile(r"(?<![\w/])(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,5}\)?[\s.\-]?){2,4}\d{2,5}(?![\w/])")


def _phone_sub(m: re.Match) -> str:
    s = m.group(0)
    digits = sum(c.isdigit() for c in s)
    if digits < 8 or digits > 15:
        return s
    # date ranges like 2019 - 2021 or 01/2020-03/2022 are not phones
    if re.fullmatch(r"\s*(19|20)\d{2}\s*[-\s]\s*(19|20)\d{2}\s*", s):
        return s
    return "[PHONE]"


def name_patterns(names: Iterable[str | None]) -> list[re.Pattern]:
    pats = []
    for n in names:
        if not n:
            continue
        for part in str(n).replace(",", " ").split():
            part = part.strip(".")
            if len(part) >= 3 and part.isalpha():
                pats.append(re.compile(rf"\b{re.escape(part)}\b", re.I))
    return pats


def redact(text: str, names: Iterable[str | None] = ()) -> str:
    if not text:
        return text
    t = EMAIL.sub("[EMAIL]", text)
    t = URL.sub("[URL]", t)
    t = PHONE.sub(_phone_sub, t)
    for p in name_patterns(names):
        t = p.sub("[NAME]", t)
    t = re.sub(r"(\[NAME\][\s.]*){2,}", "[NAME] ", t)
    return t
