"""
Stage B — query/JD ↔ candidate relevance pairs (the "later stage" data).

scan   : stream eval_consolidated parquet, collect every judge's verdict per unique
         (normalised query, resume_id), write consensus labels + the resume ids that will
         need their text joined from resume_data.
build  : sample/balance per query, split (unseen-query test set), join resume text /
         profile block, write relevance/pairs-*.jsonl.gz.

Label consensus: each judge column (gpt, bench, primary, gemini majority, ...) is a weighted
vote; repeated runs of the same judge are averaged first. Pairs whose judges disagree too
much are dropped — noisy labels hurt a small model more than fewer labels do.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional

from .config import Cfg
from .io_utils import ResumeStore, ShardWriter, parquet_num_row_groups, read_row_group, read_shards, write_json
from .pii import redact
from .resume_schema import profile_block_from_eval_row
from .splits import unit_hash
from .text_clean import clean_inline, clean_text, normalize_query

log = logging.getLogger(__name__)

CAND_COLS = ["role", "city", "country", "continent", "location", "years_of_experience", "yoe", "availability",
             "skills", "additional_skills_v1", "work_experience", "education", "projects", "certifications",
             "languages", "resume", "resume_plain_text", "name", "name_x", "name_y", "bench_name"]
BASE_COLS = ["query", "resume_id", "rid", "domain", "judge_date", "source_folder"]

_CFG: Optional[dict] = None


def _init(cfg):
    global _CFG
    _CFG = cfg


# ----------------------------------------------------------------------------- normalisers
_REL = {"relevant", "yes", "true", "1", "match", "fit", "good fit", "good_fit", "strong", "pass"}
_NOT = {"not_relevant", "not relevant", "irrelevant", "no", "false", "0", "no fit", "no_fit", "reject", "fail",
        "not-relevant", "nonrelevant"}


def norm_verdict(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in _REL:
        return "relevant"
    if s in _NOT:
        return "not_relevant"
    return None


def norm_score(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f < 0 or f > 10:  # nan / out of range
        return None
    return f


def norm_failed(v: Any) -> Optional[list[str]]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan",):
        return None
    if s.lower() in ("none", "[]", "null", "-"):
        return []
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return sorted({str(x).strip().lower() for x in j if str(x).strip()})
    except Exception:
        pass
    parts = [p.strip().lower().replace(" ", "_") for p in s.replace(";", ",").replace("|", ",").split(",")]
    return sorted({p for p in parts if p and p != "none"})


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


# ----------------------------------------------------------------------------- scan
GEMINI_RUNS = 12   # gemini_{verdict,score}_run_{0..11}: repeated runs of one judge -> score stability only


def _judge_cols(judges) -> set:
    cols = set(BASE_COLS + CAND_COLS)
    for j in judges:
        cols.update(v for k, v in j.items() if k in ("verdict", "score", "reason", "failed", "require_true"))
    for i in range(GEMINI_RUNS):
        cols.update({f"gemini_verdict_run_{i}", f"gemini_score_run_{i}"})
    return cols


def _row_judgments(get, judges) -> list[tuple]:
    """judgments of one source row: (judge, verdict, score, reason, failed). `get(col)` returns a cell or None.
    Repeated gemini runs are returned as judge 'gemini_run<i>' (weight 0: they feed score stability, not votes)."""
    out = []
    for j in judges:
        if j.get("require_true") and not _truthy(get(j["require_true"])):
            continue
        raw_v = get(j["verdict"])
        v = norm_verdict(raw_v)
        s = norm_score(get(j["score"])) if j.get("score") else None
        if v is None and s is not None and raw_v in (None, ""):
            v = "relevant" if s >= 5 else "not_relevant"
        if v is None:
            continue
        r = clean_inline(get(j["reason"]), 400) if j.get("reason") else ""
        f = norm_failed(get(j["failed"])) if j.get("failed") else None
        out.append((j["name"], v, s, r, tuple(f) if f is not None else None))
    for i in range(GEMINI_RUNS):
        v = norm_verdict(get(f"gemini_verdict_run_{i}"))
        s = norm_score(get(f"gemini_score_run_{i}"))
        if v is None and s is not None:
            v = "relevant" if s >= 5 else "not_relevant"
        if v is not None:
            out.append((f"gemini_run{i}", v, s, "", None))
    return out


def _scan_rg(task):
    path, rg = task
    judges = _CFG["relevance"]["judges"]
    data = read_row_group(Path(path), rg, sorted(_judge_cols(judges)))
    out_j: dict[tuple, set] = defaultdict(set)
    meta: dict[tuple, dict] = {}
    cands: dict[tuple, dict] = {}
    st = Counter()
    n = len(data["query"])
    for i in range(n):
        st["rows"] += 1
        q = normalize_query(data["query"][i])
        rid = _rid(data["resume_id"][i] or data["rid"][i])
        if not q or rid is None:
            st["no_key"] += 1
            continue
        key = (q, rid)
        js = _row_judgments(lambda c: data[c][i] if c in data else None, judges)
        out_j[key].update(js)
        if js:
            st["judged_rows"] += 1
        if key not in meta:
            meta[key] = {"domain": data["domain"][i], "date": (str(data["judge_date"][i])[:10]
                                                                if data["judge_date"][i] else None)}
        if key not in cands:
            c = {k: data[k][i] for k in CAND_COLS if data[k][i] not in (None, "")}
            if c.get("resume") or c.get("resume_plain_text") or c.get("skills") or c.get("work_experience"):
                cands[key] = c
    return {"j": {k: list(v) for k, v in out_j.items()}, "meta": meta, "cands": cands, "stats": st}


def _rid(x) -> Optional[int]:
    """'6722084', 6722084.0, '6722084__fraud_1' -> 6722084 ; junk -> None"""
    if x in (None, ""):
        return None
    m = re.match(r"\s*(\d+)", str(x))
    return int(m.group(1)) if m else None


_ID_COLS = ("resume_id", "talent_id", "rid", "candidate_id", "developer_id")


def _scan_csv(path: Path, judges, judgments: dict, meta: dict, cdb, st: Counter) -> None:
    """Judgments from one evaluation/ CSV (same column names as eval_consolidated; '#' comment lines skipped).
    Files without a query column, an id column or any judge column are skipped."""
    csv.field_size_limit(1 << 30)
    try:
        fh = open(path, encoding="utf-8-sig", errors="replace", newline="")
    except OSError:
        st["csv_unreadable"] += 1
        return
    with fh:
        rows = csv.reader(line for line in fh if not line.startswith("#"))
        try:
            header = [h.strip().lower().lstrip("\ufeff") for h in next(rows)]
        except StopIteration:
            return
        col = {h: i for i, h in enumerate(header) if h}
        idc = next((c for c in _ID_COLS if c in col), None)
        jcols = {v for j in judges for k, v in j.items() if k in ("verdict", "score")}
        if "query" not in col or idc is None or not (jcols & set(col)):
            st["csv_skipped_no_labels"] += 1
            return
        st["csv_used"] += 1
        for row in rows:
            get = lambda c: (row[col[c]] if c in col and col[c] < len(row) and row[col[c]] != "" else None)  # noqa: E731
            q, rid = normalize_query(get("query")), _rid(get(idc))
            if not q or rid is None:
                st["csv_no_key"] += 1
                continue
            js = _row_judgments(get, judges)
            if not js:
                continue
            key = (q, rid)
            judgments[key].update(js)
            st["csv_judged_rows"] += 1
            meta.setdefault(key, {"domain": get("domain"), "date": (get("judge_date") or "")[:10] or None})
            c = {k: get(k) for k in CAND_COLS if get(k) is not None}
            if c.get("resume") or c.get("resume_plain_text") or c.get("skills"):
                cdb.execute("INSERT OR IGNORE INTO c VALUES (?,?,?)", (q, rid, json.dumps(c, ensure_ascii=False)))


def score_stability(judgments: list[tuple], threshold: float = 5.0) -> dict:
    """Spread of every score this pair received (all judges, all repeated runs).
    verdict_flip: some judge gave BOTH verdicts across its runs (the same model changed its mind);
    boundary:     scores fall on both sides of the relevance threshold (min < 5 <= max);
    risky:        flip or boundary -> the label is uncertain; trained as a soft target, reported apart in eval."""
    scores = [s for _, _, s, _, _ in judgments if s is not None]
    per_judge: dict[str, set] = defaultdict(set)
    for name, v, *_ in judgments:
        per_judge[name.split("_run")[0] if name.startswith("gemini_run") else name].add(v)
    flip = any(len(vs) > 1 for vs in per_judge.values())
    out = {"n_scores": len(scores), "verdict_flip": flip}
    if scores:
        mean = sum(scores) / len(scores)
        var = sum((x - mean) ** 2 for x in scores) / len(scores)
        out.update(score_mean=round(mean, 3), score_min=min(scores), score_max=max(scores),
                   score_std=round(var ** 0.5, 3))
        out["boundary"] = min(scores) < threshold <= max(scores)
    else:
        out["boundary"] = False
    out["risky"] = bool(flip or out["boundary"])
    return out


def _aggregate(judgments: list[tuple], weights: dict[str, float], rcfg: dict) -> Optional[dict]:
    stab = score_stability(judgments)
    by_judge: dict[str, list] = defaultdict(list)
    for name, v, s, r, f in judgments:
        if name.startswith("gemini_run"):
            continue                    # repeated runs: stability only (gemini_maj already votes for them)
        by_judge[name].append((v, s, r, f))
    votes = Counter()
    per_judge = {}
    for name, items in by_judge.items():
        c = Counter(v for v, *_ in items)
        maj, cnt = c.most_common(1)[0]
        consistency = cnt / len(items)
        w = weights.get(name, 0.5) * consistency
        votes[maj] += w
        scores = [s for v, s, *_ in items if v == maj and s is not None]
        reason = next((r for v, s, r, f in items if v == maj and r), "")
        failed = next((list(f) for v, s, r, f in items if v == maj and f is not None), None)
        per_judge[name] = {"v": maj, "w": w, "s": sum(scores) / len(scores) if scores else None,
                           "r": reason, "f": failed}
    total = sum(votes.values())
    if total <= 0:
        return None
    verdict, wv = votes.most_common(1)[0]
    agreement = wv / total
    low_agree = agreement < rcfg["min_agreement"]
    policy = rcfg.get("risky_policy", "soft")      # soft: keep uncertain pairs as soft targets; drop: old rule
    if total < rcfg["min_weight"] or (low_agree and policy == "drop"):
        return {"drop": True, "agreement": agreement, "weight": total}
    stab["risky"] = bool(stab["risky"] or low_agree)
    agree = sorted([(n, p) for n, p in per_judge.items() if p["v"] == verdict], key=lambda x: -x[1]["w"])
    sc = [(p["s"], p["w"]) for _, p in agree if p["s"] is not None]
    score = sum(s * w for s, w in sc) / sum(w for _, w in sc) if sc else None
    if score is not None:  # keep score on the same side as the verdict (score>=5 <=> relevant)
        score = max(score, 5.0) if verdict == "relevant" else min(score, 4.0)
    reason = next((p["r"] for _, p in agree if p["r"]), "")
    failed = next((p["f"] for _, p in agree if p["f"] is not None), None)
    if verdict == "relevant" and failed:
        failed = None  # contradictory metadata — omit rather than teach noise
    return {"verdict": verdict, "score": score, "reason": reason, "failed": failed,
            "agreement": round(agreement, 3), "weight": round(total, 3), "judges": sorted(per_judge), **stab}


def scan(cfg: Cfg) -> dict:
    t0 = time.time()
    rcfg = cfg["relevance"]
    path = cfg.src(cfg["sources"]["relevance_parquet"])
    out_dir = cfg.out_root_path / "relevance"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rg = parquet_num_row_groups(path)
    if cfg["sources"].get("limit_row_groups"):
        n_rg = min(n_rg, cfg["sources"]["limit_row_groups"])
    tasks = [(str(path), rg) for rg in range(n_rg)]
    log.info("relevance scan: %s (%d row groups)", path.name, n_rg)

    cand_db_path = out_dir / "candidates.sqlite"
    if cand_db_path.exists():
        cand_db_path.unlink()
    cdb = sqlite3.connect(str(cand_db_path))
    cdb.execute("PRAGMA journal_mode=WAL")
    cdb.execute("PRAGMA synchronous=OFF")
    cdb.execute("CREATE TABLE c (q TEXT, rid INTEGER, payload TEXT, PRIMARY KEY (q, rid))")

    judgments: dict[tuple, set] = defaultdict(set)
    meta: dict[tuple, dict] = {}
    stats = Counter()
    with get_context("spawn").Pool(cfg["workers"], initializer=_init, initargs=(dict(cfg),)) as pool:
        for k, res in enumerate(pool.imap(_scan_rg, tasks, chunksize=1)):
            stats.update(res["stats"])
            for key, js in res["j"].items():
                judgments[key].update(map(tuple, js))
            for key, m in res["meta"].items():
                meta.setdefault(key, m)
            cdb.executemany("INSERT OR IGNORE INTO c VALUES (?,?,?)",
                            [(q, rid, json.dumps(c, ensure_ascii=False)) for (q, rid), c in res["cands"].items()])
            cdb.commit()
            if k % 20 == 0:
                log.info("[%d/%d] pairs=%d (%.0fs)", k, n_rg, len(meta), time.time() - t0)

    # extra judged pairs from the evaluation/ folder CSVs (identical judgments collapse in the set)
    for g in cfg["sources"].get("relevance_csv_globs") or []:
        files = cfg.src_glob(g)
        lim = cfg["sources"].get("limit_row_groups")
        if lim:                          # quick trials / first_run: a few files only (the CSVs total ~14 GB)
            files = files[: 3 * int(lim)]
        log.info("relevance csv scan: %s (%d files)", g, len(files))
        for f in files:
            _scan_csv(f, rcfg["judges"], judgments, meta, cdb, stats)
        cdb.commit()
    weights = {j["name"]: j["weight"] for j in rcfg["judges"]}
    have_inline = {(q, rid) for q, rid, p in cdb.execute("SELECT q, rid, payload FROM c")
                   if '"resume"' in p or '"resume_plain_text"' in p}
    needed_ids: set[int] = set()
    agg_stats = Counter()
    with ShardWriter(out_dir, "labels", shard_size=200_000) as w:
        for key, js in judgments.items():
            a = _aggregate(list(js), weights, rcfg)
            if a is None:
                continue
            if a.get("drop"):
                agg_stats["dropped_disagreement"] += 1
                continue
            agg_stats[a["verdict"]] += 1
            agg_stats["risky"] += a["risky"]
            agg_stats["verdict_flip"] += a["verdict_flip"]
            agg_stats["boundary"] += a["boundary"]
            q, rid = key
            w.write({"query": q, "resume_id": rid, **a, **meta.get(key, {})})
            if key not in have_inline:
                needed_ids.add(rid)
    cdb.close()
    (out_dir / "needed_resume_ids.txt").write_text("\n".join(map(str, sorted(needed_ids))))
    report = {"rows": dict(stats), "pairs_total": len(judgments), "labels": dict(agg_stats),
              "queries": len({q for q, _ in judgments}), "need_store_join": len(needed_ids),
              "seconds": round(time.time() - t0)}
    write_json(cfg.out("reports", "relevance_scan.json"), report)
    log.info("relevance scan done: %s", report)
    return report


# ----------------------------------------------------------------------------- build
def _priority(rec: dict) -> float:
    """Higher = keep first: confident labels, and boundary scores (3..6) which are the hard cases."""
    s = rec.get("score")
    hard = 1.0 if s is not None and 3 <= s <= 6 else 0.0
    return rec["agreement"] + 0.5 * hard + 0.1 * min(rec["weight"], 3)


def build(cfg: Cfg) -> dict:
    t0 = time.time()
    rcfg = cfg["relevance"]
    out_dir = cfg.out_root_path / "relevance"
    rng = random.Random(cfg["seed"])
    by_q: dict[str, list] = defaultdict(list)
    for rec in read_shards(out_dir, "labels"):
        by_q[rec["query"]].append(rec)
    if not by_q:
        raise RuntimeError("no relevance labels — run `relevance-scan` first")

    store_path = cfg.out_root_path / "resume_store.sqlite"
    store = ResumeStore(store_path, readonly=True) if store_path.exists() else None
    if store is None:
        log.warning("resume_store.sqlite missing — pairs without inline resume text will be dropped")
    cdb = sqlite3.connect(f"file:{(out_dir / 'candidates.sqlite').as_posix()}?mode=ro", uri=True)

    stats = Counter()
    mcc = int(rcfg.get("max_candidate_chars", 4000))
    from .registry import load_registry
    reg = load_registry(cfg)
    if reg is None:
        raise RuntimeError("no split registry — run `python -m slm_data registry` after relevance-scan")

    # Turn 53 (review D4): candidate payloads are fetched ONCE per query (one indexed range read on the (q, rid)
    # primary key) and store rows in batches, instead of one SELECT per pair in each of the two passes
    qcache: dict = {"q": None, "pay": {}, "store": {}}

    def load_query(q, rids):
        if qcache["q"] == q:
            return
        qcache["q"] = q
        qcache["pay"] = {int(rid): pl for rid, pl in cdb.execute("SELECT rid, payload FROM c WHERE q=?", (q,))}
        need = [int(x) for x in rids]
        qcache["store"] = store.get_many(need) if (rcfg.get("allow_store_join") and store is not None and need) \
            else {}

    def make_block(q, r):
        """-> (candidate block with placeholders, source, names) or None"""
        if qcache["q"] != q:
            load_query(q, [r["resume_id"]])
        pl = qcache["pay"].get(int(r["resume_id"]))
        cand = json.loads(pl) if pl else {}
        names = [cand.get(k) for k in ("name", "name_x", "name_y", "bench_name")]
        text = cand.get("resume") or cand.get("resume_plain_text")
        block, src = None, None
        if text:
            text = redact(clean_text(text, max(6000, mcc)), names)
            block = profile_block_from_eval_row(cand, text, mcc)
            src = "inline"
        elif rcfg.get("allow_store_join") and store is not None:
            s_ = qcache["store"].get(int(r["resume_id"]))
            if s_ is None and int(r["resume_id"]) not in qcache["store"]:
                s_ = store.get(r["resume_id"])
            if s_ and s_.get("text"):
                # store text may carry pseudonyms (pii_mode: pseudonymize): relevance inputs use
                # placeholders instead, so names cannot act as a proxy in scoring
                names = names + list((s_.get("meta") or {}).get("pseudo_names") or [])
                block = (profile_block_from_eval_row(cand, s_["text"], mcc) if cand.get("skills")
                         else s_.get("profile") or f"Resume:\n{s_['text'][:mcc]}")
                src = "store"
        return (redact(block, names), src, names) if block else None

    def plan(q, recs):
        """split = query axis x candidate axis (registry.pair_split, Turn 47): dev / test = UNSEEN query
        families; val / dev_cand / test_cand = held-out candidates on seen queries. Evaluation assets, QA
        accounts and locked-test candidates under dev queries are never written. Only TRAIN is balanced."""
        by_split: dict[str, list] = defaultdict(list)
        for r in recs:
            sp = reg.pair_split(q, r["resume_id"])
            if sp is None:
                stats["skip_" + reg.split_of(r["resume_id"])] += 1
                continue
            by_split[sp].append(r)
        pos = [r for r in by_split.get("train", []) if r["verdict"] == "relevant"]
        neg = [r for r in by_split.get("train", []) if r["verdict"] == "not_relevant"]
        for lst in (pos, neg):
            rng.shuffle(lst)
            lst.sort(key=_priority, reverse=True)
        ratio = rcfg["max_pos_neg_ratio"]
        if pos and neg:
            if len(pos) > ratio * len(neg):
                pos = pos[: int(ratio * len(neg))]
            elif len(neg) > ratio * len(pos):
                neg = neg[: int(ratio * len(pos))]
        keep = pos + neg
        keep.sort(key=_priority, reverse=True)
        keep = [(r, "train") for r in keep[: rcfg["max_per_query"]]]
        keep += [(r, sp) for sp in ("val", "dev", "dev_cand", "test", "test_cand", "test_qonly")
                 for r in by_split.get(sp, [])]
        return keep

    plans = {q: plan(q, recs) for q, recs in by_q.items()}
    # near-duplicate decontamination (Turn 47): a TRAINING pair whose candidate text is a near-copy of a
    # held-out candidate's text (same person under another id, re-uploads) is dropped. Pass 1 indexes the
    # held-out candidate blocks; pass 2 writes everything and checks each training block against the index.
    held_sp = {"val", "dev_cand", "test_cand"} | ({"test"} if reg.strict_test else set())
    held_blocks: dict = {}
    held_cache_max = int(rcfg.get("held_block_cache", 300000))
    idx = None
    if rcfg.get("near_dup_decontam", True):
        from .near_dup import MinHashIndex
        idx = MinHashIndex(float(rcfg.get("near_dup_threshold", 0.8)))
        for q, keep in plans.items():
            load_query(q, [r["resume_id"] for r, _ in keep])
            for r, split in keep:
                if split in held_sp:
                    b_ = make_block(q, r)
                    if b_:
                        idx.add(b_[0])
                        if len(held_blocks) < held_cache_max:      # reused by pass 2 (no second build)
                            held_blocks[(q, int(r["resume_id"]))] = b_
    with ShardWriter(out_dir, "pairs") as w:
        for q, keep in plans.items():
            load_query(q, [r["resume_id"] for r, _ in keep])
            for r, split in keep:
                made = held_blocks.pop((q, int(r["resume_id"])), None) or make_block(q, r)
                if not made:
                    stats["no_text"] += 1
                    continue
                block, src, names = made
                if idx is not None and split == "train" and idx.query(block):
                    stats["train_dropped_near_dup_of_heldout"] += 1
                    continue
                reason = redact(clean_inline(r.get("reason"), rcfg["reason_max_chars"]), names) or None
                w.write({"query": q, "date": r.get("date"), "candidate": block, "resume_id": r["resume_id"],
                         "verdict": r["verdict"], "score": r.get("score"), "failed": r.get("failed"),
                         "reason": reason, "agreement": r["agreement"], "split": split, "src": src,
                         "domain": r.get("domain"),
                         # label stability across judges / repeated runs (Turn 45)
                         "score_mean": r.get("score_mean"), "score_min": r.get("score_min"),
                         "score_max": r.get("score_max"), "score_std": r.get("score_std"),
                         "n_scores": r.get("n_scores"), "verdict_flip": r.get("verdict_flip", False),
                         "boundary": r.get("boundary", False), "risky": r.get("risky", False)})
                stats[f"{split}_{r['verdict']}"] += 1
                stats[f"{split}_risky"] += bool(r.get("risky"))
                stats[f"src_{src}"] += 1
    report = {"stats": dict(stats), "queries": len(by_q), "seconds": round(time.time() - t0)}
    write_json(cfg.out("reports", "relevance_build.json"), report)
    log.info("relevance build done: %s", report)
    return report
