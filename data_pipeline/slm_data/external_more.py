"""
External public text from training_data/external_data_1010 (Turn 45): pretraining text only, labels ignored.

  djinni_jobs          ~142K IT job descriptions (MIT)                 -> JD text
  djinni_candidate_profiles ~210K anonymised candidate profiles (MIT) -> profile text
  resume_job_fit (med2425) resumes + JDs, Qwen labels ignored          -> resume / JD text, de-duplicated
  wdc_jobposting       schema.org JobPosting from Common Crawl (N-Quads) -> JD text (+ structured fields)
  wikipedia_en / fineweb_edu / cosmopedia_v2  general text (Turn 51); Wikipedia articles whose title is a known
                       company / university / occupation (knowledge/kb_names.txt) go to the `knowledge` group
Each document carries a `group` (slm_data/groups.py); pack writes one stream per group and the trainer mixes them.

Every document is cleaned, PII-redacted (pattern level) and de-duplicated by content hash. External text has no
talent_id, so its train / val split is a content hash (see pack.py); none of it is evaluation data.
"""
from __future__ import annotations

import gzip
import html
import logging
import re
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator, Optional

from .config import Cfg
from .io_utils import ShardWriter, parquet_columns, parquet_num_row_groups, read_row_group
from .pii import EMAIL, redact
from .text_clean import clean_inline, clean_text, content_hash

log = logging.getLogger(__name__)

_TAGS = re.compile(r"<[^>]+>")


def first_glob(cfg, pattern) -> list:
    """A glob or a list of globs: the files of the FIRST pattern that matches anything (Turn 56: one source can
    live under its downloader folder name or an older manual folder name; both are listed in the config)."""
    for pat in ([pattern] if isinstance(pattern, str) else list(pattern or [])):
        files = [p for p in cfg.src_glob(pat) if not p.name.startswith("._")]
        if files:
            return files
    return []


def _html_to_text(s: str) -> str:
    s = re.sub(r"(?i)<\s*(br|/p|/li|/div|/h\d)\s*/?>", "\n", s or "")
    s = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", s)
    return html.unescape(_TAGS.sub(" ", s))


def _ascii_ratio(t: str) -> float:
    return sum(c.isascii() for c in t) / max(1, len(t))


# ----------------------------------------------------------------------------- parquet sources
_LIMIT_RG: Optional[int] = None      # sources.limit_row_groups (quick trials / first_run): first N row groups only


def _rows(files: list[Path], cols: list[str]) -> Iterator[dict]:
    for f in files:
        have = set(parquet_columns(f))
        use = [c for c in cols if c in have]
        n_rg = parquet_num_row_groups(f)
        for rg in range(min(n_rg, _LIMIT_RG) if _LIMIT_RG else n_rg):
            d = read_row_group(f, rg, use)
            n = len(d[use[0]]) if use else 0
            for i in range(n):
                yield {c: d[c][i] for c in use}


def djinni_jobs(files) -> Iterator[str]:
    for r in _rows(files, ["Position", "Long Description", "Company Name", "Exp Years", "Primary Keyword",
                           "Long Description_lang"]):
        if (r.get("Long Description_lang") or "en") != "en" or not r.get("Long Description"):
            continue
        head = [f"Job Title: {clean_inline(r.get('Position'))}"]
        if r.get("Company Name"):
            head.append(f"Company: {clean_inline(r['Company Name'])}")
        if r.get("Exp Years"):
            head.append(f"Experience: {clean_inline(str(r['Exp Years']))}")
        if r.get("Primary Keyword"):
            head.append(f"Primary skill: {clean_inline(r['Primary Keyword'])}")
        yield "\n".join(head) + "\n\n" + r["Long Description"]


def djinni_profiles(files) -> Iterator[str]:
    for r in _rows(files, ["Position", "Moreinfo", "Looking For", "Highlights", "Primary Keyword",
                           "Experience Years", "CV", "CV_lang"]):
        if (r.get("CV_lang") or "en") != "en":
            continue
        parts = [f"Position: {clean_inline(r.get('Position'))}"]
        if r.get("Experience Years") is not None:
            parts.append(f"Experience: {r['Experience Years']} years")
        if r.get("Primary Keyword"):
            parts.append(f"Primary skill: {clean_inline(r['Primary Keyword'])}")
        for k in ("Highlights", "CV", "Moreinfo", "Looking For"):
            if r.get(k):
                parts.append(f"{k}: {r[k]}" if k != "CV" else r[k])
        yield "\n".join(parts)


def resume_job_fit(files) -> Iterator[tuple[str, str]]:
    """med2425/resume-job-fit-merged-v1: resumes and JDs repeat across pairs -> caller de-duplicates."""
    for r in _rows(files, ["resume", "jd"]):
        if r.get("resume"):
            yield r["resume"], "resume_job_fit_resume"
        if r.get("jd"):
            yield r["jd"], "resume_job_fit_jd"


# ----------------------------------------------------------------------------- WDC N-Quads
_NQ = re.compile(r'^(\S+) <([^>]+)> (.+) <([^>]*)>\s*\.\s*$')
_ESC = re.compile(r'\\(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)')
_JD_TEXT_PROPS = ("description", "responsibilities", "qualifications", "skills", "experienceRequirements",
                  "educationRequirements", "jobBenefits")
_JD_FIELD_PROPS = ("title", "employmentType", "industry", "occupationalCategory", "datePosted",
                   "jobLocationType", "experienceRequirements", "educationRequirements", "skills")


def _unescape(s: str) -> str:
    def rep(m):
        x = m.group(1)
        if x[0] in "uU" and len(x) > 1:
            try:
                return chr(int(x[1:], 16))
            except ValueError:
                return x
        return {"n": "\n", "t": "\t", "r": "", '"': '"', "\\": "\\"}.get(x, x)
    return _ESC.sub(rep, s)


def _parse_obj(o: str):
    """-> (kind, value, lang): kind 'lit' | 'node' | 'iri'"""
    if o.startswith('"'):
        end = o.rfind('"')
        lang = o[end + 2:] if o[end + 1:end + 2] == "@" else None
        return "lit", _unescape(o[1:end]), lang
    if o.startswith("_:"):
        return "node", o, None
    return "iri", o.strip("<>"), None


def _local(p: str) -> str:
    return re.split(r"[/#]", p)[-1]


def _postings(page_triples: list) -> Iterator[tuple[str, dict]]:
    props: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    langs: dict[str, set] = defaultdict(set)
    jobs = []
    for s, p, (kind, val, lang) in page_triples:
        lp = _local(p)
        if lp == "type" and kind == "iri" and _local(val) == "JobPosting":
            jobs.append(s)
        props[s][lp].append((kind, val))
        if lang:
            langs[s].add(lang.lower())

    def lit(node, prop) -> list[str]:
        out = []
        for kind, val in props[node].get(prop, []):
            if kind == "lit":
                out.append(val)
            elif kind == "node":
                out += [v for k, v in props[val].get("name", []) if k == "lit"]
        return out

    for j in jobs:
        if langs[j] and not any(l.startswith("en") for l in langs[j]):
            continue
        title = clean_inline(" ".join(lit(j, "title")[:1]))
        org = ""
        for kind, val in props[j].get("hiringOrganization", []):
            org = clean_inline(" ".join(lit(val, "name")[:1])) if kind == "node" else clean_inline(val)
        loc = []
        for kind, val in props[j].get("jobLocation", []):
            for k2, addr in props[val].get("address", []) if kind == "node" else []:
                if k2 == "node":
                    for pr in ("addressLocality", "addressRegion", "addressCountry"):
                        loc += lit(addr, pr)[:1]
        body = []
        for pr in _JD_TEXT_PROPS:
            vals = lit(j, pr)
            if vals:
                txt = clean_text(_html_to_text(" ".join(vals)), 12000)
                if txt:
                    body.append(txt if pr == "description" else f"{pr[0].upper() + pr[1:]}: {txt}")
        if not body:
            continue
        head = [f"Job Title: {title}"] if title else []
        if org:
            head.append(f"Company: {org}")
        if loc:
            head.append(f"Location: {', '.join(dict.fromkeys(clean_inline(x) for x in loc if x))}")
        et = lit(j, "employmentType")
        if et:
            head.append(f"Employment type: {clean_inline(', '.join(et[:3]))}")
        fields = {pr: [clean_inline(_html_to_text(v), 300) for v in lit(j, pr)[:5]] for pr in _JD_FIELD_PROPS
                  if lit(j, pr)}
        if org:
            fields["hiringOrganization"] = org
        if loc:
            fields["location"] = loc[:3]
        yield "\n".join(head) + "\n\n" + "\n\n".join(body), fields


def _wdc_part(task) -> dict:
    path, out_dir, min_chars, max_docs = task
    st = Counter()
    seen = set()
    name = Path(path).name.split(".")[0]
    with ShardWriter(Path(out_dir), f"pretrain-wdc{name}") as w, gzip.open(path, "rt", encoding="utf-8",
                                                                          errors="replace") as f:
        cur, triples = None, []

        def flush():
            for text, fields in _postings(triples):
                st["postings"] += 1
                t = redact(clean_text(text, 20000))
                if len(t) < min_chars or _ascii_ratio(t) < 0.85:
                    st["too_short_or_non_en"] += 1
                    continue
                h = content_hash(t)
                if h in seen:
                    st["dup"] += 1
                    continue
                seen.add(h)
                w.write({"text": t, "source": "wdc_jobposting", "group": "jobs", "fields": fields})
                st["docs"] += 1
        for line in f:
            m = _NQ.match(line)
            if not m:
                st["bad_line"] += 1
                continue
            s, p, o, g = m.groups()
            if g != cur:
                if triples:
                    flush()
                cur, triples = g, []
                if max_docs and st["docs"] >= max_docs:
                    break
            triples.append((s, p, _parse_obj(o)))
        if triples:
            flush()
    return dict(st)


# ----------------------------------------------------------------------------- entry
def run(cfg: Cfg, pw: ShardWriter, st: Counter) -> None:
    ec = cfg["external"]
    src = cfg["sources"]["external"]
    root = cfg.out_root_path / "external"
    min_chars = int(ec.get("min_doc_chars", 200))
    seen: set = set()
    global _LIMIT_RG
    _LIMIT_RG = cfg["sources"].get("limit_row_groups")
    wdc_cap = ec.get("wdc_max_docs_per_part") or (200 * _LIMIT_RG if _LIMIT_RG else None)

    def emit(texts, source):
        from .groups import SOURCE_GROUP
        n = 0
        for t in texts:
            src = source
            if isinstance(t, tuple):           # (text, source) when one file mixes resumes and JDs
                t, src = t
            t = redact(clean_text(t, cfg["cleaning"]["max_resume_chars"]))
            if len(t) < min_chars:
                st[f"{source}_short"] += 1
                continue
            h = content_hash(t)
            if h in seen:
                st[f"{source}_dup"] += 1
                continue
            seen.add(h)
            pw.write({"text": t, "source": src, "group": SOURCE_GROUP[src]})   # KeyError = unmapped source
            n += 1
        st[source] += n
        log.info("external %s: %d docs", source, n)

    for key, fn, flag in (("djinni_jobs_glob", djinni_jobs, "use_djinni"),
                          ("djinni_profiles_glob", djinni_profiles, "use_djinni"),
                          ("resume_job_fit_glob", resume_job_fit, "use_resume_job_fit")):
        if ec.get(flag, True) and src.get(key):
            files = first_glob(cfg, src[key])
            if files:
                emit(fn(files), key.replace("_glob", ""))
    if ec.get("use_wdc", True) and src.get("wdc_glob"):
        parts = [p for p in cfg.src_glob(src["wdc_glob"]) if p.suffix == ".gz"]
        if parts:
            tasks = [(str(p), str(root), min_chars, wdc_cap) for p in parts]
            with get_context("spawn").Pool(min(cfg["workers"], len(tasks))) as pool:
                for r in pool.imap_unordered(_wdc_part, tasks):
                    for k, v in r.items():
                        st[f"wdc_{k}"] += v
            log.info("external wdc: %d docs from %d parts", st["wdc_docs"], len(parts))
    # general text (Turn 51): one worker per parquet file, each writing its own shards
    kb_names = cfg.out_root_path / "knowledge" / "kb_names.txt"
    gen = []
    for key, source, flag in (("wikipedia_glob", "wikipedia", "use_wikipedia"),
                              ("fineweb_glob", "fineweb_edu", "use_fineweb"),
                              ("cosmopedia_glob", "cosmopedia_v2", "use_cosmopedia")):
        if ec.get(flag, True) and src.get(key):
            gen += [(str(p), str(root), source, min_chars, _LIMIT_RG, ec.get("general_max_docs_per_file"),
                     str(kb_names) if source == "wikipedia" and kb_names.exists() else None)
                    for p in cfg.src_glob(src[key])]
    if gen:
        with get_context("spawn").Pool(min(cfg["workers"], len(gen))) as pool:
            for r in pool.imap_unordered(_general_file, gen):
                for k, v in r.items():
                    st[k] += v
        log.info("external general text: %s", {k: v for k, v in st.items() if k.startswith(("wikipedia", "fineweb", "cosmopedia"))})


def _general_file(task) -> dict:
    """one general-text parquet file -> shards external/pretrain-gen-<source>-<file>-*.jsonl.gz"""
    path, out_dir, source, min_chars, limit_rg, max_docs, kb_path = task
    global _LIMIT_RG
    _LIMIT_RG = limit_rg
    kb = set(Path(kb_path).read_text(encoding="utf-8").split("\n")) if kb_path else set()
    st = Counter()
    seen = set()
    stem = Path(path).stem
    with ShardWriter(Path(out_dir), f"pretrain-gen-{source}-{stem}") as w:
        for r in _rows([Path(path)], ["text", "title"]):
            if max_docs and st[f"{source}_docs"] >= max_docs:
                break
            t = clean_text(r.get("text"), 60000)
            if source == "wikipedia" and r.get("title"):
                t = f"{clean_inline(r['title'])}\n\n{t}"
            if len(t) < min_chars:
                st[f"{source}_short"] += 1
                continue
            h = content_hash(t)
            if h in seen:
                st[f"{source}_dup"] += 1
                continue
            seen.add(h)
            group = "general"
            if kb and source == "wikipedia" and _name_key(r.get("title") or "") in kb:
                group = "knowledge"
                st["wikipedia_kb_articles"] += 1
            # e-mails only: phone / name patterns would damage numbers and public names in encyclopaedic text
            w.write({"text": EMAIL.sub("[EMAIL]", t), "source": "wikipedia_kb" if group == "knowledge" else source,
                     "group": group})
            st[f"{source}_docs"] += 1
    return dict(st)


def _name_key(name: str) -> str:
    """normalised entity name (shared with build_knowledge): lower case, legal suffixes and punctuation removed"""
    s = re.sub(r"\(.*?\)", " ", (name or "").lower())
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\b(inc|incorporated|corp|corporation|co|ltd|limited|llc|plc|gmbh|ag|sa|the)\b", " ", s)
    return " ".join(s.split())
