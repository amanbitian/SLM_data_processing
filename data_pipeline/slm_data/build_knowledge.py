"""
Knowledge step (Turn 51): turns structured public data into (1) short knowledge text for pretraining (group
`knowledge`) and (2) entity tables for the KB / query understanding (ARCHITECTURE.md §19).

  O*NET 31.0       ~1K occupations: description, alternate / reported titles, core tasks, software, job zone
  ESCO (TalentCLEF training files)  occupation alternative titles + essential skills
  ROR              education / research organisations: names, acronyms, aliases, city, country, founding year
  OpenAlex         institutions: research volume (works, citations), acronyms, alternative names (joined on ROR)
  SEC EDGAR        US-listed companies: name, ticker, exchange
  Wikidata         companies / universities with an English Wikipedia article (industry, HQ, parent, founding)
  Skill taxonomy   cards for High / Medium-confidence skills (category, domain, parent, typical roles)

Outputs (under artifacts/data/knowledge/):
  pretrain-*.jsonl.gz   {"text", "source", "group": "knowledge"}
  kb_occupations.jsonl.gz, kb_institutions.jsonl.gz, kb_companies.jsonl.gz   structured entities
  kb_names.txt          normalised entity names (used by `external` to route Wikipedia articles to `knowledge`)
Missing sources are skipped with a warning. Nothing here is evaluation data; TalentCLEF test / validation files
are NOT read (kept for an external evaluation).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .config import Cfg
from .external_more import _name_key
from .io_utils import ShardWriter, write_json

log = logging.getLogger(__name__)

JOB_ZONE = {1: "little or no preparation", 2: "some preparation", 3: "medium preparation",
            4: "considerable preparation", 5: "extensive preparation"}


def _tsv(z: zipfile.ZipFile, name: str):
    full = next((n for n in z.namelist() if n.endswith("/" + name) or n == name), None)
    if full is None:
        return []
    with z.open(full) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t"))


def onet(path: Path, w, kb, names: set, st: Counter):
    z = zipfile.ZipFile(path)
    occ = {r["O*NET-SOC Code"]: r for r in _tsv(z, "Occupation Data.txt")}
    alt = defaultdict(list)
    for fname, col in (("Job Titles.txt", None), ("Alternate Titles.txt", "Alternate Title"),
                       ("Sample of Reported Titles.txt", "Reported Job Title")):
        for r in _tsv(z, fname):
            t = r.get(col) if col else (r.get("Title") or r.get("Job Title") or r.get("Alternate Title"))
            if t and len(alt[r["O*NET-SOC Code"]]) < 25:
                alt[r["O*NET-SOC Code"]].append(t.strip())
    tasks = defaultdict(list)
    for r in _tsv(z, "Task Statements.txt"):
        if r.get("Task Type") in ("Core", "") and len(tasks[r["O*NET-SOC Code"]]) < 8:
            tasks[r["O*NET-SOC Code"]].append(r["Task"].strip())
    soft = defaultdict(list)
    for fname in ("Software Skills.txt", "Technology Skills.txt"):
        for r in _tsv(z, fname):
            ex = r.get("Workplace Example") or r.get("Example") or r.get("Commodity Title")
            if ex and len(soft[r["O*NET-SOC Code"]]) < 15 and ex not in soft[r["O*NET-SOC Code"]]:
                soft[r["O*NET-SOC Code"]].append(ex.strip())
    zone = {r["O*NET-SOC Code"]: r.get("Job Zone") for r in _tsv(z, "Job Zones.txt")}
    for code, r in occ.items():
        title = r["Title"].strip()
        parts = [f"Occupation: {title} (O*NET-SOC {code}).", r.get("Description", "").strip()]
        if alt[code]:
            parts.append("Also called: " + ", ".join(dict.fromkeys(alt[code])) + ".")
        if tasks[code]:
            parts.append("Typical tasks: " + " ".join(f"- {t}" for t in tasks[code]))
        if soft[code]:
            parts.append("Software and technology used: " + ", ".join(soft[code]) + ".")
        zn = zone.get(code)
        if zn and zn.isdigit():
            parts.append(f"Job zone {zn}: {JOB_ZONE.get(int(zn), '')}.")
        w.write({"text": "\n".join(p for p in parts if p), "source": "onet", "group": "knowledge"})
        kb.write({"kind": "occupation", "source": "onet", "id": code, "name": title, "aliases": alt[code],
                  "tasks": tasks[code], "software": soft[code], "job_zone": zn})
        names.update(_name_key(x) for x in [title] + alt[code])
        st["onet_occupations"] += 1


def esco_talentclef(task_b_zip: Path, w, kb, names: set, st: Counter):
    """ESCO occupations from the TalentCLEF Task B TRAINING files: alternative titles + essential skills."""
    z = zipfile.ZipFile(task_b_zip)
    job_terms = json.loads(z.read(next(n for n in z.namelist() if n.endswith("training/jobid2terms.json"))))
    skill_terms = json.loads(z.read(next(n for n in z.namelist() if n.endswith("training/skillid2terms.json"))))
    ess = defaultdict(list)
    with z.open(next(n for n in z.namelist() if n.endswith("training/job2skill.tsv"))) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[2] == "essential" and len(ess[p[0]]) < 20 and skill_terms.get(p[1]):
                ess[p[0]].append(skill_terms[p[1]][0])
    for jid, terms in job_terms.items():
        if not terms:
            continue
        parts = [f"Occupation (ESCO): {terms[0]}."]
        if len(terms) > 1:
            parts.append("Other titles: " + ", ".join(terms[1:20]) + ".")
        if ess[jid]:
            parts.append("Essential skills: " + ", ".join(ess[jid]) + ".")
        w.write({"text": "\n".join(parts), "source": "esco_talentclef", "group": "knowledge"})
        kb.write({"kind": "occupation", "source": "esco", "id": jid, "name": terms[0], "aliases": terms[1:30],
                  "essential_skills": ess[jid]})
        names.update(_name_key(x) for x in terms[:10])
        st["esco_occupations"] += 1


def ror_openalex(ror_zip: Path, openalex: Path, w, kb, names: set, st: Counter, types=("education",)):
    oa = {}
    if openalex and openalex.exists():
        with gzip.open(openalex, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("ror"):        # keep only what the card uses (full records are large)
                    oa[r["ror"].rstrip("/").split("/")[-1]] = {"works_count": r.get("works_count"),
                                                               "cited_by_count": r.get("cited_by_count")}
    def vals(v, english_only=False):   # "en: Name A; hi: नाम; no_lang_code: ABC" -> ["Name A", "ABC"]
        out = []
        for x in (v or "").split(";"):
            x = x.strip()
            lang = None
            if ":" in x and x.split(":", 1)[0].strip().replace("_", "").isalnum() and len(x.split(":", 1)[0]) <= 12:
                lang, x = x.split(":", 1)[0].strip(), x.split(":", 1)[1].strip()
            if english_only and lang not in (None, "en", "no_lang_code"):
                continue
            if x:
                out.append(x)
        return out
    z = zipfile.ZipFile(ror_zip)
    cname = next(n for n in z.namelist() if n.endswith(".csv"))          # the CSV is ~6x smaller than the JSON
    with z.open(cname) as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
            rtypes = [t.strip() for t in (r.get("types") or "").split(";") if t.strip()]
            if r.get("status") != "active" or not set(rtypes) & set(types):
                continue
            name = (r.get("names.types.ror_display") or "").strip()
            if not name:
                continue
            acr = vals(r.get("names.types.acronym"))
            alias = [x for x in vals(r.get("names.types.alias"), True) + vals(r.get("names.types.label"), True)
                     if x != name]
            city = vals(r.get("locations.geonames_details.name"))[:1]
            country = vals(r.get("locations.geonames_details.country_name"))[:1]
            rid = r["id"].rstrip("/").split("/")[-1]
            o = oa.get(rid, {})
            kind = "/".join(rtypes)
            parts = [f"{name}" + (f" ({', '.join(acr)})" if acr else "") +
                     f" is {'an' if kind[:1] in 'aeiou' else 'a'} {kind} organisation"
                     + (f" in {', '.join(city + country)}." if city or country else ".")]
            if r.get("established"):
                parts.append(f"Founded in {r['established']}.")
            if alias:
                parts.append("Also known as: " + ", ".join(dict.fromkeys(alias[:10])) + ".")
            if o.get("works_count"):
                parts.append(f"Research output: about {o['works_count']:,} publications and "
                             f"{o.get('cited_by_count', 0):,} citations (OpenAlex).")
            w.write({"text": " ".join(parts), "source": "ror", "group": "knowledge"})
            kb.write({"kind": "institution", "source": "ror", "id": rid, "name": name, "acronyms": acr,
                      "aliases": alias[:20], "city": (city or [None])[0], "country": (country or [None])[0],
                      "established": r.get("established"), "types": rtypes, "works_count": o.get("works_count"),
                      "cited_by_count": o.get("cited_by_count")})
            names.update(_name_key(x) for x in [name] + alias[:5])
            st["ror_institutions"] += 1


def sec(path: Path, w, kb, names: set, st: Counter):
    d = json.loads(path.read_text())
    fields = d.get("fields") or []
    for row in d.get("data") or []:
        r = dict(zip(fields, row))
        if not r.get("name"):
            continue
        w.write({"text": f"{r['name']} is a company listed on {r.get('exchange') or 'a US exchange'} under the ticker "
                         f"{r.get('ticker')}.", "source": "sec", "group": "knowledge"})
        kb.write({"kind": "company", "source": "sec", "id": r.get("cik"), "name": r["name"], "ticker": r.get("ticker"),
                  "exchange": r.get("exchange")})
        names.add(_name_key(r["name"]))
        st["sec_companies"] += 1


def wikidata(folder: Path, w, kb, names: set, st: Counter, kb_inst=None):
    """companies_*.jsonl.gz / universities_*.jsonl.gz from the downloader (one row per item x optional value).
    Companies go to kb_companies, universities to kb_institutions (Turn 56; they all went to companies)."""
    for f in sorted(x for x in folder.glob("*.jsonl.gz") if not x.name.startswith("._")):
        kind = "company" if f.name.startswith("companies") else "university"
        items: dict[str, dict] = {}
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                it = items.setdefault(r["item"], {"name": r.get("itemLabel"), "alt": set(), "industry": set(),
                                                  "country": set(), "hq": set(), "parent": set(), "city": set()})
                for k_src, k_dst in (("alt", "alt"), ("industryLabel", "industry"), ("countryLabel", "country"),
                                     ("hqLabel", "hq"), ("parentLabel", "parent"), ("cityLabel", "city")):
                    if r.get(k_src):
                        it[k_dst].add(r[k_src])
                for k in ("inception", "employees", "students", "website"):
                    if r.get(k):
                        it[k] = r[k]
        for qid, it in items.items():
            name = it["name"]
            if not name or name.startswith("Q") and name[1:].isdigit():
                continue
            parts = [f"{name} is a {'company' if kind == 'company' else 'university'}"
                     + (f" in {', '.join(sorted(it['industry']))[:120]}" if it["industry"] else "")
                     + (f", based in {', '.join(sorted(it['hq'] or it['city']))}" if it["hq"] or it["city"] else "")
                     + (f" ({', '.join(sorted(it['country']))})" if it["country"] else "") + "."]
            if it.get("inception"):
                parts.append(f"Founded {it['inception'][:4]}.")
            if it["parent"]:
                parts.append("Parent organisation: " + ", ".join(x.rstrip(".") for x in sorted(it["parent"])) + ".")
            if it.get("employees"):
                parts.append(f"Employees: about {it['employees'].split('.')[0]}.")
            if it["alt"]:
                parts.append("Also known as: " + ", ".join(sorted(it["alt"])[:8]) + ".")
            w.write({"text": " ".join(parts), "source": "wikidata", "group": "knowledge"})
            (kb_inst if (kind == "university" and kb_inst is not None) else kb).write(
                     {"kind": "institution" if kind == "university" else kind, "source": "wikidata",
                      "id": qid.rsplit("/", 1)[-1], "name": name,
                      "aliases": sorted(it["alt"])[:20], "industry": sorted(it["industry"]),
                      "country": sorted(it["country"]), "hq": sorted(it["hq"] or it["city"])})
            names.update(_name_key(x) for x in [name] + sorted(it["alt"])[:5])
            st[f"wikidata_{kind}"] += 1


def skills(tx_path: Path, w, st: Counter):
    from .skill_taxonomy import SkillTaxonomy
    tx = SkillTaxonomy(tx_path)
    for sid in tx.entries:
        c = tx.card(sid, min_band="Medium")
        if c:
            w.write({"text": c, "source": "skill_taxonomy", "group": "knowledge"})
            st["skill_cards"] += 1


def run(cfg: Cfg) -> dict:
    src = cfg["sources"].get("knowledge") or {}
    out = cfg.out_root_path / "knowledge"
    out.mkdir(parents=True, exist_ok=True)
    st: Counter = Counter()
    names: set = set()
    p = lambda k: cfg.src(src[k]) if src.get(k) else None  # noqa: E731
    with ShardWriter(out, "pretrain") as w, ShardWriter(out, "kb_occupations") as ko, \
            ShardWriter(out, "kb_institutions") as ki, ShardWriter(out, "kb_companies") as kc:
        for key, fn, kbw in (("onet_zip", onet, ko), ("talentclef_task_b_zip", esco_talentclef, ko)):
            f = p(key)
            if f and f.exists():
                fn(f, w, kbw, names, st)
            else:
                log.warning("knowledge: %s missing (%s)", key, f)
        f = p("ror_zip_glob")
        rz = sorted(f.parent.glob(f.name)) if f else []
        if rz:
            ror_openalex(rz[-1], p("openalex_jsonl"), w, ki, names, st)
        else:
            log.warning("knowledge: ROR zip missing")
        f = p("sec_tickers_json")
        if f and f.exists():
            sec(f, w, kc, names, st)
        f = p("wikidata_dir")
        if f and f.exists() and any(f.glob("*.jsonl.gz")):
            wikidata(f, w, kc, names, st, kb_inst=ki)
        tx = cfg["sources"].get("skill_taxonomy_csv")
        if tx and cfg.src(tx).exists():
            skills(cfg.src(tx), w, st)
    names.discard("")
    (out / "kb_names.txt").write_text("\n".join(sorted(names)), encoding="utf-8")
    st["kb_names"] = len(names)
    write_json(cfg.out("reports", "knowledge_stats.json"), dict(st))
    log.info("knowledge done: %s", dict(st))
    return dict(st)
