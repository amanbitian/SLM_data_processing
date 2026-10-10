"""
Canonical extraction target + candidate profile block.

`parsed_resume_data` (OpenAI parse, the source of truth) is verbose and contains PII.
`canonical_target()` turns it into a compact, PII-free JSON the model learns to emit:

{"role":"Data Engineering","domains":["Data Engineering","Cloud Infrastructure"],
 "industry":"IT Software and Services","seniority":"Lead",
 "location":{"country":"IN","city":"Bengaluru"},"total_experience_months":84,
 "skills":[{"name":"Python","years":8},{"name":"Docker"}],
 "work_experience":[{"title":"Senior Data Engineer","company":"Acme","start":"2021-09","end":"present",
                     "months":null,"location":"Pune, India"}],
 "education":[{"level":"Bachelor","degree":"B.Tech","subject":"Electrical Engineering",
               "school":"BVC Engineering College","start":"2011-06","end":"2015-06","current":false}],
 "certifications":[{"name":"Azure AI Fundamentals","issuer":"Microsoft","year":2023}],
 "languages":["English"]}
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .text_clean import clean_inline

# ----------------------------------------------------------------------------- helpers
EDU_LEVELS = [
    ("post doctorate", "PostDoc"), ("postdoc", "PostDoc"), ("doctor", "PhD"), ("phd", "PhD"), ("ph.d", "PhD"),
    ("mba", "MBA"), ("master", "Master"), ("bachelor", "Bachelor"), ("associate", "Associate"),
    ("diploma", "Diploma"), ("secondary", "HighSchool"), ("high school", "HighSchool"),
    ("certificate", "Certificate"),
]


def norm_edu_level(level: Optional[str], degree: Optional[str] = None) -> Optional[str]:
    for src in (level, degree):
        if not src:
            continue
        low = str(src).lower()
        for key, val in EDU_LEVELS:
            if key in low:
                return val
    if degree:
        d = str(degree).lower().replace(".", "")
        if re.search(r"\b(btech|be|bsc|bs|ba|bcom|bca|bba|beng)\b", d):
            return "Bachelor"
        if re.search(r"\b(mtech|me|msc|ms|ma|mcom|mca|meng|mphil)\b", d):
            return "Master"
    return None


def _ym(d: Any) -> Optional[str]:
    """{'year':2021,'month':9} | '2021-09-01T..' -> '2021-09' / '2021'."""
    if not d:
        return None
    if isinstance(d, str):
        m = re.match(r"(\d{4})(?:-(\d{1,2}))?", d)
        if not m:
            return None
        return f"{m.group(1)}-{int(m.group(2)):02d}" if m.group(2) else m.group(1)
    if isinstance(d, dict):
        y = d.get("year")
        if not y:
            return None
        mth = d.get("month")
        return f"{int(y):04d}-{int(mth):02d}" if mth else f"{int(y):04d}"
    return None


def _months_between(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b or b == "present" or len(a) < 7 or len(b) < 7:
        return None
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    m = (yb - ya) * 12 + (mb - ma) + 1
    return m if 0 < m < 600 else None


def _s(v: Any, n: int = 120) -> Optional[str]:
    if v is None:
        return None
    t = clean_inline(str(v), n)
    return t or None


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f < 0:
        return None
    return int(f) if f.is_integer() else round(f, 1)


def _drop_empty(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


# ----------------------------------------------------------------------------- target
def canonical_target(p: dict, cfg: dict) -> dict:
    max_w = cfg.get("max_work_items", 12)
    max_e = cfg.get("max_education_items", 6)
    max_s = cfg.get("max_skills", 40)
    max_c = cfg.get("max_certifications", 8)
    with_desc = cfg.get("include_descriptions", False)

    # skills (prefer the yoe-annotated list; primary skills first)
    skills: list[dict] = []
    seen: set[str] = set()
    swy = p.get("skills_with_yoe") or []
    swy = sorted(swy, key=lambda s: (not (s or {}).get("is_primary"), -((s or {}).get("yoe") or 0)))
    for s in swy:
        name = _s((s or {}).get("name"), 60)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            item = {"name": name}
            y = _num((s or {}).get("yoe"))
            if y:
                item["years"] = y
            skills.append(item)
    for s in p.get("skills") or []:
        name = _s(s if isinstance(s, str) else (s or {}).get("name"), 60)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            skills.append({"name": name})
    skills = skills[:max_s]

    # work experience (most recent first)
    work = []
    for e in p.get("experiences") or []:
        e = e or {}
        start, end = _ym(e.get("starts_at")), _ym(e.get("ends_at"))
        if end is None and start:
            end = "present"
        item = {
            "title": _s(e.get("title"), 100),
            "company": _s(e.get("company"), 100),
            "start": start,
            "end": end,
            "months": _months_between(start, end),
            "location": _s(e.get("location"), 60),
        }
        if with_desc and e.get("description"):
            item["summary"] = _s(e["description"], 220)
        item = _drop_empty(item)
        if item.get("title") or item.get("company"):
            work.append(item)
    work.sort(key=lambda w: (w.get("start") or "0000"), reverse=True)
    work = work[:max_w]

    edu = []
    for e in p.get("education") or []:
        e = e or {}
        item = _drop_empty({
            "level": norm_edu_level(e.get("level"), e.get("degree_name")),
            "degree": _s(e.get("degree_name"), 100),
            "subject": _s(e.get("subject"), 80),
            "school": _s(e.get("school"), 100),
            "start": _ym(e.get("starts_at")),
            "end": _ym(e.get("ends_at")),
        })
        if e.get("is_current") is not None:
            item["current"] = bool(e.get("is_current"))
        if item.get("school") or item.get("degree"):
            edu.append(item)
    edu = edu[:max_e]

    certs = []
    for c in p.get("certifications") or []:
        c = c or {}
        item = _drop_empty({"name": _s(c.get("name"), 100), "issuer": _s(c.get("authority"), 60),
                            "year": _num(c.get("year"))})
        if item.get("name"):
            certs.append(item)
    certs = certs[:max_c]

    langs = []
    for l in p.get("languages") or []:
        n = _s(l.get("language_name") if isinstance(l, dict) else l, 30)
        if n and n not in langs:
            langs.append(n)

    domains = [d for d in (p.get("domains") or []) if d]
    return _drop_empty({
        "role": _s(p.get("role") or p.get("occupation"), 80),
        "domains": domains[:5],
        "industry": _s(p.get("industry"), 80),
        "seniority": _s(p.get("current_seniority"), 30),
        "location": _drop_empty({"country": _s(p.get("country"), 40), "city": _s(p.get("city"), 60)}),
        "total_experience_months": _num(p.get("total_experience_in_months")),
        "skills": skills,
        "work_experience": work,
        "education": edu,
        "certifications": certs,
        "languages": langs[:6],
    })


def taxonomy_target(r: dict) -> dict:
    """Convert a resume_extraction/output (gpt-5.x taxonomy) record into the canonical schema,
    keeping its curated domain / subDomain labels."""
    work = []
    for w in r.get("work_experience") or []:
        start, end = _ym(w.get("start_date")), _ym(w.get("end_date"))
        work.append(_drop_empty({
            "title": _s(w.get("position")), "company": _s(w.get("company")),
            "start": start, "end": end or ("present" if start else None),
            "months": w.get("durationInMonths"),
            "location": _s(", ".join(x for x in [(w.get("location") or {}).get("city"),
                                                (w.get("location") or {}).get("country")] if x)),
            "domain": _s(w.get("domain")), "subdomain": _s(w.get("subdomain")),
        }))
    edu = []
    for e in r.get("education") or []:
        edu.append(_drop_empty({
            "level": norm_edu_level(e.get("degree") or e.get("level"), e.get("degree")),
            "degree": _s(e.get("degree")), "subject": _s(e.get("subject") or e.get("field_of_study")),
            "school": _s(e.get("university") or e.get("school") or e.get("institution")),
            "start": _ym(e.get("start_date")), "end": _ym(e.get("end_date")),
        }))
    skills, seen = [], set()
    for w in r.get("work_experience") or []:
        for s in w.get("skills") or []:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                skills.append({"name": _s(s, 60)})
    return _drop_empty({
        "domains": [d for d in [r.get("domain")] if d],
        "subdomain": r.get("subDomain"),
        "location": _drop_empty({"country": r.get("location"), "region": r.get("geo")}),
        "skills": skills[:40], "work_experience": work, "education": edu,
    })


# ----------------------------------------------------------------------------- grounding
_TOK = re.compile(r"[a-z0-9+#.]{2,}")


def grounding_score(target: dict, resume_text: str) -> float:
    """Fraction of extracted companies + skills that are visibly supported by the resume text.
    Low scores flag hallucinated parses or a parse that belongs to a different resume."""
    low = resume_text.lower()
    toks = set(_TOK.findall(low))
    items = [w.get("company") for w in target.get("work_experience", []) if w.get("company")]
    items += [s["name"] for s in target.get("skills", [])][:25]
    if not items:
        return 0.0
    hit = 0
    for it in items:
        il = it.lower()
        if il in low:
            hit += 1
            continue
        parts = [t for t in _TOK.findall(il) if len(t) >= 3]
        if parts and sum(t in toks for t in parts) / len(parts) >= 0.5:
            hit += 1
    return hit / len(items)


# ----------------------------------------------------------------------------- profile block
def _yrs(months: Optional[int]) -> str:
    if not months:
        return ""
    y = months / 12
    return f"{y:.0f}yrs" if y >= 1 else f"{months}mos"


def profile_block_from_parse(p: dict, resume_text: str, max_resume_chars: int = 4000) -> str:
    """Candidate block in the same layout the production relevance judge receives
    (relevance-scoring.shared.ts buildCandidateBlock), minus the name."""
    loc = ", ".join(x for x in [p.get("city"), p.get("country_full_name") or p.get("country")] if x) or "—"
    months = _num(p.get("total_experience_in_months"))
    yoe = f"{months / 12:.0f}" if months else "—"
    skills = ", ".join(
        f"{s['name']} ({_num(s['yoe'])}yrs)" if _num(s.get("yoe")) else s["name"]
        for s in (p.get("skills_with_yoe") or []) if s and s.get("name")
    ) or ", ".join(s for s in (p.get("skills") or []) if isinstance(s, str)) or "—"
    work_lines = []
    for e in (p.get("experiences") or [])[:5]:
        st, en = _ym(e.get("starts_at")), _ym(e.get("ends_at")) or "present"
        line = f"{e.get('title') or '?'} at {e.get('company') or '?'} ({st or '?'} to {en})"
        if e.get("description"):
            line += f" — {clean_inline(e['description'], 300)}"
        work_lines.append(line[:500])
    edu_lines = []
    for e in p.get("education") or []:
        lvl = e.get("level") or e.get("degree_name") or ""
        status = "currently enrolled" if e.get("is_current") else "completed"
        edu_lines.append(f"{lvl} {e.get('subject') or e.get('degree_name') or ''} - {e.get('school') or ''} [{status}]".strip())
    certs = ", ".join(c.get("name") for c in (p.get("certifications") or []) if c and c.get("name")) or "—"
    langs = ", ".join((l.get("language_name") if isinstance(l, dict) else str(l))
                      for l in (p.get("languages") or []) if l) or "—"
    return (
        f"Role: {p.get('role') or p.get('occupation') or '—'}\n"
        f"Location: {loc}\n"
        f"Years of Experience: {yoe}\n"
        f"Skills: {skills}\n"
        f"Work Experience:\n" + ("\n".join(work_lines) or "  —") + "\n"
        f"Education:\n" + ("\n".join(edu_lines) or "  —") + "\n"
        f"Certifications: {certs}\n"
        f"Languages: {langs}\n"
        f"Resume:\n{resume_text[:max_resume_chars]}"
    )


def profile_block_from_eval_row(row: dict, resume_text: str, max_resume_chars: int = 4000) -> str:
    """Same layout, built from the flattened fields already present in eval_consolidated rows."""
    def c(k):
        v = row.get(k)
        if v is None:
            return None
        v = str(v).strip()
        return v if v and v.lower() not in ("nan", "none", "null") else None

    loc = ", ".join(x for x in [c("city"), c("country"), c("continent")] if x) or c("location") or "—"
    yoe = c("years_of_experience") or c("yoe") or "—"
    if yoe.endswith(".0"):
        yoe = yoe[:-2]
    skills = ", ".join(x for x in [c("skills"), c("additional_skills_v1")] if x) or "—"
    work = "\n".join(l[:500] for l in (c("work_experience") or "  —").split("\n")[:5])
    projects = "\n".join(l[:300] for l in (c("projects") or "  —").split("\n")[:3])
    return (
        f"Role: {c('role') or '—'}\n"
        f"Location: {loc}\n"
        f"Years of Experience: {yoe}\n"
        f"Availability: {c('availability') or '—'}\n"
        f"Skills: {skills}\n"
        f"Work Experience:\n{work}\n"
        f"Education:\n{c('education') or '  —'}\n"
        f"Projects:\n{projects}\n"
        f"Certifications: {c('certifications') or '—'}\n"
        f"Languages: {c('languages') or '—'}\n"
        f"Resume:\n{resume_text[:max_resume_chars]}"
    )
