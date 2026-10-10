"""
Split registry (Turn 45): ONE place that decides train / val / test for every candidate.

Rule: a talent_id lives in exactly one split, for every task (pretraining text, extraction, relevance,
tokenizer sample). Accounts that share an e-mail address are treated as one person and share the split.

How the split is chosen
  1. eval      : candidates of the evaluation assets (extraction gold, curated sets, fraud ids, the
                 JOB-110870 screening, the human gold set). Never used for training.
  2. test      : ~20,000 candidates drawn from the relevance pool across ALL queries
                 (hash-stratified, so every query keeps roughly the same share; small queries get a
                 minimum and both verdicts where possible). All their pairs, for every query, are test.
  3. val / dev : ~10,000 more relevance candidates the same way, split in two halves:
                 val (early stopping, temperature / threshold calibration) and dev (model decisions,
                 ablations, baselines) — the tools expect both. Plus a small hash slice (0.5%) of every
                 other candidate -> val (pretraining perplexity, extraction validation).
  4. excluded  : obvious test / QA accounts (configurable e-mail patterns).
  5. train     : everybody else (the default; not stored).
  Priority when one person qualifies twice: eval > test > val / dev > excluded > train.

Second axis — QUERIES (Turn 47): near-duplicate queries are merged into families (token Jaccard >= 0.6 or
character similarity >= 0.8 on a normalised form; manual overrides possible) and every FAMILY is assigned to
train / dev / test (~80 / 10 / 10 % of the pairs). A relevance pair's split combines both axes (pair_split):

  query family | candidate split          -> pair split
  test         | any (not eval/excluded)  -> test        (unseen query; `strict_test` also needs a test candidate,
                                                          otherwise -> test_qonly)
  dev          | train / val / dev        -> dev         (unseen query; locked-test candidates are left out)
  train        | train                    -> train
  train        | val                      -> val         (calibration / early stopping on seen queries)
  train        | dev                      -> dev_cand    (seen query, unseen candidate)
  train        | test                     -> test_cand   (seen query, the ~20K locked candidates)

So dev / test measure UNSEEN QUERIES, test_cand / dev_cand measure unseen candidates, and the audit asserts
zero query-family overlap between training and dev / test.

The registry is written to <out>/splits/registry.sqlite (+ registry_report.json) and read by every builder.
Assignments are deterministic (blake2b hash with a salt), so re-running gives the same split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .config import Cfg
from .io_utils import parquet_columns, parquet_num_row_groups, read_row_group, read_shards, write_json
from .splits import unit_hash

log = logging.getLogger(__name__)

PRIORITY = {"eval": 4, "test": 3, "val": 2, "dev": 2, "excluded": 1, "train": 0}
SPLITS = ("train", "val", "dev", "test", "eval", "excluded")


def _rid(x) -> Optional[int]:
    if x in (None, ""):
        return None
    m = re.match(r"\s*(\d+)", str(x))
    return int(m.group(1)) if m else None


def email_key(e: str) -> Optional[int]:
    """normalised e-mail -> 64-bit key (gmail dots and +tags ignored). The address itself is not stored."""
    s = (e or "").strip().lower()
    if "@" not in s:
        return None
    local, _, dom = s.partition("@")
    local = local.split("+")[0]
    if dom in ("gmail.com", "googlemail.com"):
        local, dom = local.replace(".", ""), "gmail.com"
    return int.from_bytes(hashlib.blake2b(f"{local}@{dom}".encode(), digest_size=8).digest(), "big", signed=True)


# ----------------------------------------------------------------------------- reading
def query_key(q: str) -> str:
    """normalised query text used for families and overlap checks"""
    return re.sub(r"[^a-z0-9+#$]+", " ", (q or "").lower()).strip()


class Registry:
    """Read side, used by the builders: split_of(talent_id), query_split(query), pair_split(query, talent_id)."""

    def __init__(self, path: Path, default_val_frac: float = 0.005, salt: str = "v1"):
        self.path = Path(path)
        db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self.assign = dict(db.execute("SELECT talent_id, split FROM assign"))
        meta = dict(db.execute("SELECT k, v FROM meta"))
        try:
            self.qmap = {k: (f, sp) for k, f, sp in db.execute("SELECT qkey, family, split FROM queries")}
        except sqlite3.OperationalError:       # registry built before Turn 47: no query axis
            self.qmap = {}
        db.close()
        self.strict_test = meta.get("strict_test", "false") == "true"
        self.default_val_frac = float(meta.get("default_val_frac", default_val_frac))
        self.salt = meta.get("salt", salt)
        self.version = meta.get("version", "?")

    def split_of(self, talent_id) -> str:
        tid = _rid(talent_id)
        if tid is None:
            return "train"
        sp = self.assign.get(tid)
        if sp:
            return sp
        return "val" if unit_hash(f"{self.salt}:pv:{tid}") < self.default_val_frac else "train"

    def held_out(self, talent_id) -> bool:
        return self.split_of(talent_id) in ("test", "eval", "excluded")

    def query_split(self, query: str) -> str:
        """train / dev / test of the query's family; a query unknown to the registry is new -> 'test' would be
        unsafe to train on, so unknown queries are treated as held out ('test')."""
        v = self.qmap.get(query_key(query))
        if v is None:
            return "train" if not self.qmap else "test"
        return v[1]

    def query_family(self, query: str) -> Optional[str]:
        v = self.qmap.get(query_key(query))
        return v[0] if v else None

    def pair_split(self, query: str, talent_id) -> Optional[str]:
        """split of one (query, candidate) pair; None = never used (eval / excluded / locked candidate in dev)."""
        cs = self.split_of(talent_id)
        if cs in ("eval", "excluded"):
            return None
        qs = self.query_split(query)
        if qs == "test":
            return "test" if (cs == "test" or not self.strict_test) else "test_qonly"
        if qs == "dev":
            return None if cs == "test" else "dev"
        return {"train": "train", "val": "val", "dev": "dev_cand", "test": "test_cand"}[cs]


def load_registry(cfg: Cfg) -> Optional[Registry]:
    import os
    root = Path(os.environ.get("SLM_OUT_ROOT", cfg["out_root"]))   # no mkdir: read side only
    p = root / "splits" / "registry.sqlite"
    return Registry(p) if p.exists() else None


# ----------------------------------------------------------------------------- building
def _eval_asset_ids(cfg: Cfg) -> dict[int, str]:
    """{talent_id: asset name} from every configured evaluation asset (csv / json / parquet)."""
    out: dict[int, str] = {}
    for a in cfg["splits"].get("eval_assets") or []:
        p = cfg.src(a["path"])
        files = sorted(p.parent.glob(p.name)) if any(ch in p.name for ch in "*?[") else [p]
        n0 = len(out)
        for f in files:
            if not f.exists():
                log.warning("eval asset missing: %s", f)
                continue
            cols = a.get("id_cols") or [a.get("id_col", "talent_id")]

            def add(rec_get):
                for c in cols:
                    t = _rid(rec_get(c))
                    if t is not None:
                        out.setdefault(t, a["name"])
                        return
            if f.suffix == ".csv":
                csv.field_size_limit(1 << 30)
                with open(f, encoding="utf-8-sig", errors="replace", newline="") as fh:
                    for r in csv.DictReader(fh):
                        add(r.get)
            elif f.suffix == ".json":
                data = json.load(open(f, encoding="utf-8"))
                for r in data if isinstance(data, list) else data.get("items", []):
                    add((r or {}).get)
            elif f.suffix == ".jsonl":
                for line in open(f, encoding="utf-8"):
                    if line.strip():
                        add(json.loads(line).get)
            elif f.suffix == ".parquet":
                have = set(parquet_columns(f))
                use = [c for c in cols if c in have]
                for rg in range(parquet_num_row_groups(f)):
                    d = read_row_group(f, rg, use)
                    for i in range(len(d[use[0]]) if use else 0):
                        add(lambda c: d[c][i] if c in d else None)
            elif f.suffix in (".xlsx", ".xls"):
                import pandas as pd
                for _, df in pd.read_excel(f, sheet_name=None).items():   # every sheet; first id column found
                    for r in df.to_dict("records"):
                        add(r.get)
        log.info("eval asset %s: %d candidates", a["name"], len(out) - n0)
    return out


def _relevance_pool(cfg: Cfg) -> tuple[dict[str, set], dict[int, dict]]:
    """query -> candidate ids, and per candidate the verdicts it received (from relevance/labels shards)."""
    by_q: dict[str, set] = defaultdict(set)
    verdicts: dict[int, Counter] = defaultdict(Counter)
    for r in read_shards(cfg.out_root_path / "relevance", "labels"):
        t = _rid(r.get("resume_id"))
        if t is None:
            continue
        by_q[r["query"]].add(t)
        verdicts[t][(r["query"], r["verdict"])] += 1
    return by_q, verdicts


def _pick_stratified(by_q: dict[str, set], eligible: set, target: int, salt: str, min_per_query: int,
                     labels: dict[int, Counter]) -> set:
    """~target candidates: every candidate with hash < f is picked (f = target / |eligible|), which gives each
    query about the same share; then each query is topped up to `min_per_query` and to both verdicts."""
    if not eligible or target <= 0:
        return set()
    f = min(1.0, target / len(eligible))
    h = {t: unit_hash(f"{salt}:{t}") for t in eligible}
    picked = {t for t, u in h.items() if u < f}
    for q, cands in by_q.items():
        cand = sorted((t for t in cands if t in eligible), key=h.get)
        if not cand:
            continue
        have = [t for t in cand if t in picked]
        need = min(len(cand), max(min_per_query, math.ceil(f * len(cand)))) - len(have)
        for t in cand:
            if need <= 0:
                break
            if t not in picked:
                picked.add(t)
                need -= 1
        for verdict in ("relevant", "not_relevant"):   # both classes per query if the query has both
            if not any(labels[t].get((q, verdict)) for t in cand if t in picked):
                t = next((t for t in cand if labels[t].get((q, verdict)) and t not in picked), None)
                if t is not None:
                    picked.add(t)
    return picked


def _email_links(cfg: Cfg, held: dict[int, str]) -> tuple[dict[int, str], Counter]:
    """Two light passes over (talent_id, e-mail) of every resume source: (1) e-mail keys of held-out people,
    (2) every other talent_id with one of those keys inherits the split. Also finds QA / test accounts."""
    pats = [re.compile(p, re.I) for p in cfg["splits"].get("exclude_email_patterns") or []]
    files = []
    for g in cfg["sources"]["resume_parquet_globs"]:
        files += cfg.src_glob(g)
    st = Counter()

    def rows():
        for f in files:
            cols = set(parquet_columns(f))
            ecol = next((c for c in ("email", "account_email") if c in cols), None)
            if ecol is None:
                continue
            n_rg = parquet_num_row_groups(f)
            lim = cfg["sources"].get("limit_row_groups")
            for rg in range(min(n_rg, lim) if lim else n_rg):
                d = read_row_group(f, rg, ["talent_id", ecol])
                yield from zip(d["talent_id"], d[ecol])

    key_split: dict[int, str] = {}
    extra: dict[int, str] = {}
    for tid, e in rows():                                  # pass 1
        tid = _rid(tid)
        if tid is None or not e:
            continue
        if pats and any(p.search(str(e)) for p in pats):
            extra[tid] = "excluded"
            st["excluded_accounts"] += 1
        sp = held.get(tid)
        k = email_key(str(e))
        if sp and k is not None and PRIORITY[sp] > PRIORITY.get(key_split.get(k, "train"), 0):
            key_split[k] = sp
    for tid, e in rows():                                  # pass 2
        tid = _rid(tid)
        k = email_key(str(e)) if e else None
        if tid is None or k is None or k not in key_split:
            continue
        sp = key_split[k]
        cur = held.get(tid) or extra.get(tid) or "train"
        if PRIORITY[sp] > PRIORITY[cur]:
            extra[tid] = sp
            st[f"linked_to_{sp}"] += 1
    return extra, st


def _query_families(queries: list[str], overrides: dict[str, str]) -> dict[str, str]:
    """query -> family id. Near-duplicates are merged (union-find): token Jaccard >= 0.6 or character
    similarity >= 0.8 on query_key(); `overrides` {query: family name} wins (manual review)."""
    import difflib
    keys = {q: query_key(q) for q in queries}
    toks = {q: set(k.split()) for q, k in keys.items()}
    par = {q: q for q in queries}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    qs = sorted(queries)
    # candidate pairs by PREFIX FILTERING (Turn 53, review D12; exact, not a heuristic): both rules need token
    # Jaccard >= 0.3, and two sets with Jaccard >= t must share a token among the first |A| - ceil(t|A|) + 1 of
    # each set's tokens in a fixed (rarest-first) order. Only those pairs are compared — instead of all
    # n^2/2 pairs with a SequenceMatcher each, which stops scaling past a few thousand queries.
    t_min = 0.3
    df = Counter(t for q in qs for t in toks[q])
    order_tok = lambda q: sorted(toks[q], key=lambda t: (df[t], t))  # noqa: E731
    index: dict[str, list[int]] = defaultdict(list)
    pos = {q: i for i, q in enumerate(qs)}
    pairs = set()
    for q in qs:
        ot = order_tok(q)
        plen = len(ot) - math.ceil(t_min * len(ot)) + 1 if ot else 0
        for t in ot[:plen]:
            for j_ in index[t]:
                pairs.add((j_, pos[q]))
            index[t].append(pos[q])
    for i, k in sorted(pairs):
        a, b = qs[i], qs[k]
        ta, tb = toks[a], toks[b]
        j = len(ta & tb) / max(1, len(ta | tb))
        if j >= 0.6 or (j >= 0.3 and difflib.SequenceMatcher(None, keys[a], keys[b]).ratio() >= 0.8):
            par[find(a)] = find(b)
    fam = {}
    for q in qs:
        root = find(q)
        fam[q] = "f" + hashlib.blake2b(keys[root].encode(), digest_size=5).hexdigest()
    for q, f in (overrides or {}).items():
        if q in fam:
            fam[q] = f
    return fam


def _assign_families(fam: dict[str, str], size: dict[str, int], salt: str, test_frac: float, dev_frac: float,
                     min_families: int, overrides: dict[str, str]) -> dict[str, str]:
    """family -> train / dev / test so that dev and test each hold ~their share of the PAIRS (families taken in
    a salted hash order; at least `min_families` each)."""
    fsize: Counter = Counter()
    for q, f in fam.items():
        fsize[f] += size.get(q, 0)
    total = sum(fsize.values()) or 1
    order = sorted(fsize, key=lambda f: unit_hash(f"{salt}:qf:{f}"))
    out, got = {}, Counter()
    for f in order:
        if got["test"] < test_frac * total or sum(1 for v in out.values() if v == "test") < min_families:
            sp = "test"
        elif got["dev"] < dev_frac * total or sum(1 for v in out.values() if v == "dev") < min_families:
            sp = "dev"
        else:
            sp = "train"
        out[f] = sp
        got[sp] += fsize[f]
    for q, sp in (overrides or {}).items():            # manual {query: split}
        if q in fam:
            out[fam[q]] = sp
    return out


def build(cfg: Cfg) -> dict:
    t0 = time.time()
    sc = cfg["splits"]
    salt = str(sc.get("salt", "v1"))
    out_dir = cfg.out_root_path / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    assign: dict[int, str] = {}
    reason: dict[int, str] = {}

    # 1) evaluation assets
    for t, name in _eval_asset_ids(cfg).items():
        assign[t], reason[t] = "eval", name

    # 2) + 3) test and val candidates from the relevance pool, across all queries
    by_q, labels = _relevance_pool(cfg)
    pool = set().union(*by_q.values()) if by_q else set()
    eligible = pool - set(assign)
    test = _pick_stratified(by_q, eligible, int(sc.get("test_candidates", 20000)), f"{salt}:test",
                            int(sc.get("min_test_per_query", 10)), labels)
    for t in test:
        assign[t], reason[t] = "test", "relevance_pool"
    val = _pick_stratified(by_q, eligible - test, int(sc.get("val_candidates", 10000)), f"{salt}:val",
                           int(sc.get("min_val_per_query", 5)), labels)
    dev_share = float(sc.get("dev_share_of_val", 0.5))
    for t in val:
        assign[t] = "dev" if unit_hash(f"{salt}:dev:{t}") < dev_share else "val"
        reason[t] = "relevance_pool"

    # 4) accounts sharing an e-mail with a held-out person inherit its split; QA accounts are excluded
    st = Counter()
    if sc.get("link_by_email", True):
        extra, st = _email_links(cfg, {t: s for t, s in assign.items()})
        for t, sp in extra.items():
            if PRIORITY[sp] > PRIORITY[assign.get(t, "train")]:
                assign[t], reason[t] = sp, f"email_link_{sp}" if sp != "excluded" else "test_account"

    # 5) query axis: near-duplicate queries -> families -> train / dev / test (Turn 47)
    def _load_overrides(key):
        f = sc.get(key)
        if not f:
            return {}
        p_ = cfg.src(f)
        if not p_.exists():
            log.warning("override file missing: %s", p_)
            return {}
        with open(p_, encoding="utf-8-sig", newline="") as fh:
            return {r["query"]: r["value"] for r in csv.DictReader(fh) if r.get("query") and r.get("value")}
    fam = _query_families(list(by_q), _load_overrides("query_family_overrides"))
    size = {q: len(c) for q, c in by_q.items()}
    fsplit = _assign_families(fam, size, salt, float(sc.get("test_query_frac", 0.10)),
                              float(sc.get("dev_query_frac", 0.10)), int(sc.get("min_query_families", 2)),
                              _load_overrides("query_split_overrides"))

    # write
    p = out_dir / "registry.sqlite"
    if p.exists():
        p.unlink()
    db = sqlite3.connect(str(p))
    db.execute("CREATE TABLE assign (talent_id INTEGER PRIMARY KEY, split TEXT, reason TEXT)")
    db.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    db.executemany("INSERT INTO assign VALUES (?,?,?)", [(t, s, reason.get(t, "")) for t, s in assign.items()])
    db.execute("CREATE TABLE queries (qkey TEXT PRIMARY KEY, query TEXT, family TEXT, split TEXT)")
    db.executemany("INSERT OR REPLACE INTO queries VALUES (?,?,?,?)",
                   [(query_key(q), q, fam[q], fsplit[fam[q]]) for q in by_q])
    meta = {"version": str(sc.get("version", "v1")), "salt": salt,
            "strict_test": "true" if sc.get("strict_test", False) else "false",
            "default_val_frac": str(sc.get("default_val_frac", 0.005)), "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    db.executemany("INSERT INTO meta VALUES (?,?)", list(meta.items()))
    db.commit()
    db.close()

    # report: per split counts, and per query how many candidates / pairs ended where
    per_split = Counter(assign.values())
    q_rep = {}
    for q, cands in by_q.items():
        c = Counter(assign.get(t, "train") for t in cands)
        q_rep[q[:120]] = {k: c.get(k, 0) for k in ("train", "val", "dev", "test", "eval", "excluded")}
    queries_without_test = [q for q, v in q_rep.items() if v["test"] == 0]
    for q in by_q:
        q_rep[q[:120]]["query_split"] = fsplit[fam[q]]
        q_rep[q[:120]]["family"] = fam[q]
    fam_rep = Counter(fsplit.values())
    pair_rep = Counter()
    for q, c in by_q.items():
        pair_rep[fsplit[fam[q]]] += len(c)
    with open(cfg.out("reports", "query_families.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "split", "pairs", "query"])     # review this file; fix with the override CSVs
        for q in sorted(by_q, key=lambda x: (fam[x], x)):
            w.writerow([fam[q], fsplit[fam[q]], size[q], q])
    report = {"meta": meta, "explicit_assignments": dict(per_split), "relevance_pool_candidates": len(pool),
              "queries": len(by_q), "queries_without_test_candidates": queries_without_test[:50],
              "query_families": {"families": len(set(fam.values())), "families_per_split": dict(fam_rep),
                                 "pairs_per_query_split": dict(pair_rep),
                                 "queries_per_split": dict(Counter(fsplit[fam[q]] for q in by_q))},
              "email_linking": dict(st), "per_query": q_rep, "seconds": round(time.time() - t0)}
    write_json(cfg.out("reports", "registry_report.json"), report)
    log.info("registry: %s (pool %d, %d queries, %d without test)", dict(per_split), len(pool), len(by_q),
             len(queries_without_test))
    return report
