"""Corrected skill taxonomy (turing_candidate_dataset/Skill_Taxonomy_Corrected.csv, 46,338 rows) — Turn 44.

Read-only: the CSV is loaded into memory and never written. Used for
  * skill linking:   a skill as written in a resume / parse / query  ->  one taxonomy entry
                     (canonical name, skill_id, parent, master, category, domain, roles)
  * de-duplication of extraction targets ("ReactJS", "React.js", "React" -> one skill)
  * knowledge cards: short text about each well-supported skill for pretraining / the KB (ARCHITECTURE §19)
  * later: query understanding ("React" => frontend) and checklist evidence.

The *_corrected columns win over the original ones. Rows with status `alias` point at their target
(`merge_into`). Every entry keeps its `confidence_band` (High 21,593 / Medium 13,784 / Low 10,961), so
callers can use Low-confidence hierarchy information only where a wrong parent does no harm.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Optional

BANDS = {"Low": 0, "Medium": 1, "High": 2}
# common abbreviations missing from the CSV (only added when the target skill exists in it).
# Ambiguous ones are left out on purpose: CV (resume vs computer vision), TF (TensorFlow vs Terraform),
# RN (React Native vs registered nurse), ES (Elasticsearch vs Spanish).
EXTRA_ALIASES = {
    "k8s": "Kubernetes", "ml": "Machine Learning", "dl": "Deep Learning", "ai": "Artificial Intelligence",
    "nlp": "Natural Language Processing", "postgres": "PostgreSQL", "psql": "PostgreSQL",
    "js": "JavaScript", "ts": "TypeScript", "golang": "Go", "py": "Python",
    "gcp": "Google Cloud Platform", "azure": "Microsoft Azure", "mongo": "MongoDB",
    "ci cd": "CI/CD", "cicd": "CI/CD", "llms": "LLM", "genai": "Generative AI",
}
_NONWORD = re.compile(r"[^a-z0-9+#.]+")


def skill_key(s: str) -> str:
    """'React.JS ' -> 'react.js' ; 'C++' stays 'c++' ; 'Node JS' -> 'node js'."""
    return _NONWORD.sub(" ", (s or "").lower()).strip(" .")


def compact_key(s: str) -> str:
    """second, looser key: 'react.js' / 'react js' / 'reactjs' -> 'reactjs'. Keeps + and # (c++, c#)."""
    return re.sub(r"[ .\-]", "", skill_key(s))


def _first(row: dict, *cols: str) -> str:
    for c in cols:
        v = (row.get(c) or "").strip()
        if v:
            return v
    return ""


def _split(v: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;|]", v or "") if x.strip()]


class SkillTaxonomy:
    def __init__(self, path: str | Path, min_band: str = "Low"):
        self.path = Path(path)
        self.min_band = BANDS[min_band]
        self.entries: dict[str, dict] = {}       # skill_id -> entry
        self.by_key: dict[str, str] = {}         # skill_key -> skill_id
        self.by_compact: dict[str, str] = {}     # compact_key -> skill_id (only when unambiguous)
        self.stats = {"rows": 0, "entries": 0, "aliases": 0, "keys": 0, "ambiguous_compact": 0, "below_band": 0}
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        csv.field_size_limit(10**9)
        aliases = []
        with open(self.path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.stats["rows"] += 1
                if BANDS.get(row.get("confidence_band", "Low"), 0) < self.min_band:
                    self.stats["below_band"] += 1
                    continue
                name = (row.get("suggested_name") or "").strip() or (row.get("canonical_name") or "").strip()
                if not name:
                    continue
                if (row.get("status") or "").strip() == "alias":
                    aliases.append((row.get("canonical_name", ""), (row.get("merge_into") or "").strip()))
                    continue
                sid = (row.get("skill_id") or "").strip() or f"new:{skill_key(name)}"
                e = {
                    "skill_id": sid,
                    "name": name,
                    "original_name": (row.get("canonical_name") or "").strip(),
                    "entity_type": _first(row, "entity_type_corrected", "entity_type_primary"),
                    "master": _first(row, "master_skill_corrected", "master_skill"),
                    "parent": _first(row, "parent_skill_corrected", "parent_skill"),
                    "category": _first(row, "category_corrected", "Skill_category"),
                    "industry": _first(row, "industry_corrected", "industry_level_0"),
                    "domain": _first(row, "primary_domain", "domain"),
                    "domain_group": (row.get("domain_group") or "").strip(),
                    "function": (row.get("function_corrected") or "").strip(),
                    "path": _first(row, "vertical_path_corrected", "vertical_path"),
                    "roles": _split(_first(row, "primary_roles_corrected", "primary_job_roles")),
                    "secondary_roles": _split(_first(row, "secondary_roles_corrected", "secondary_job_roles")),
                    "role_families": _split(row.get("role_families_corrected", "")),
                    "transferable": (row.get("transferable_skill") or "").strip().lower() in ("true", "yes", "1"),
                    "confidence": (row.get("confidence_band") or "Low").strip(),
                    "status": (row.get("status") or "").strip(),
                }
                self.entries[sid] = e
                for n in {name, e["original_name"]}:
                    self._index(n, sid)
        for alias, target in aliases:          # alias rows point at their merge target
            tid = self.by_key.get(skill_key(target))
            if tid:
                self._index(alias, tid)
                self.stats["aliases"] += 1
        for alias, target in EXTRA_ALIASES.items():
            tid = self.by_key.get(skill_key(target))
            if tid and skill_key(alias) not in self.by_key:
                self._index(alias, tid)
                self.stats["aliases"] += 1
        self.stats["entries"] = len(self.entries)
        self.stats["keys"] = len(self.by_key)

    def _index(self, name: str, sid: str):
        k = skill_key(name)
        if not k:
            return
        self.by_key.setdefault(k, sid)
        c = compact_key(name)
        if c in self.by_compact and self.by_compact[c] != sid:
            self.by_compact[c] = ""            # ambiguous -> never link by the loose key
            self.stats["ambiguous_compact"] += 1
        else:
            self.by_compact.setdefault(c, sid)

    # ------------------------------------------------------------------ lookups
    def link(self, name: str) -> Optional[dict]:
        """skill text -> taxonomy entry (exact normalised name first, then the loose key)."""
        sid = self.by_key.get(skill_key(name)) or self.by_compact.get(compact_key(name)) or ""
        return self.entries.get(sid) if sid else None

    def canonical(self, name: str) -> str:
        e = self.link(name)
        return e["name"] if e else name

    def dedupe(self, names: Iterable[str]) -> list[str]:
        """keep the first surface form of every distinct skill (synonyms collapse)."""
        seen, out = set(), []
        for n in names:
            e = self.link(n)
            key = e["skill_id"] if e else skill_key(n)
            if key and key not in seen:
                seen.add(key)
                out.append(n)
        return out

    def card(self, sid: str, min_band: str = "Medium") -> Optional[str]:
        """Short knowledge text about one skill (for pretraining / the KB). Hierarchy from Low-confidence rows
        is left out by default because a wrong parent would teach a wrong fact."""
        e = self.entries.get(sid)
        if not e or BANDS.get(e["confidence"], 0) < BANDS[min_band]:
            return None
        parts = [f"{e['name']} is a {e['entity_type'] or 'skill'}"]
        if e["category"]:
            parts[-1] += f" in {e['category']}"
        if e["domain"]:
            parts.append(f"It belongs to the {e['domain']} domain" + (f" ({e['function']})" if e["function"] else ""))
        if e["parent"] and e["parent"] != e["name"]:
            parts.append(f"Its parent skill is {e['parent']}")
        if e["roles"]:
            parts.append("Typical roles: " + ", ".join(e["roles"][:4]))
        if e["transferable"]:
            parts.append("It is a transferable skill")
        return ". ".join(parts) + "."

    def coverage(self, names: Iterable[str]) -> dict:
        n = linked = 0
        for x in names:
            n += 1
            linked += self.link(x) is not None
        return {"skills": n, "linked": linked, "rate": round(linked / n, 4) if n else None}
