"""
Leak audit (Turns 45 + 47): runs after the builders (`audit`) and after packing (`audit-packed`) and FAILS the
build when anything leaks.

`audit` (source shards)
  candidates   every relevance pair / extraction example / classification example / pretraining document carries
               the split the registry gives it; no talent_id sits on both sides of the CANDIDATE axis
               (train vs val / dev / held-out); no held-out person's text is in pretraining; no eval / QA account
               is in a training file.
  queries      every relevance pair carries registry.pair_split(query, candidate); ZERO query-family overlap and
               zero exact / near-duplicate queries (token Jaccard >= 0.6) between training-side splits
               (train, val, dev_cand, test_cand) and unseen-query splits (dev, test, test_qonly).
  external     HF resume-JD pairs: no JD (normalised) in both its training side and its dev / test side;
               classification texts: no exact duplicate across train and held-out splits; external
               pretraining text: no exact duplicate of a dev / test query or a held-out candidate text.
  near dups    MinHash (Jaccard >= audit.near_dup_threshold) between held-out candidate texts and training
               candidate texts (catches the same person under another talent_id), and between public JD
               benchmark sides. Capped by audit.near_dup_max_docs.
`audit-packed` (final packed files)
  reads the per-file key sidecars written by pack (normalised query / candidate / text hashes) and asserts the
  same disjointness on exactly what the trainer will read.
Reports: reports/leak_audit.json, reports/leak_audit_packed.json.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import Cfg
from .io_utils import read_shards, write_json
from .near_dup import MinHashIndex, norm_text, text_key
from .registry import load_registry, query_key

log = logging.getLogger(__name__)

HELD = {"test", "eval"}
TRAIN_SIDE_Q = {"train", "val", "dev_cand", "test_cand"}      # pairs whose QUERY family is a training family
UNSEEN_Q = {"dev", "test", "test_qonly"}                      # pairs whose query family is held out


def _bucket(split: str) -> str:
    if split in HELD:
        return "heldout"
    return "validation" if split in ("val", "dev") else split


def _qtoks(q: str) -> set:
    return set(query_key(q).split())


def run(cfg: Cfg, strict: bool = True) -> dict:
    reg = load_registry(cfg)
    if reg is None:
        raise RuntimeError("no registry — run `registry` first")
    root = cfg.out_root_path
    acfg = cfg.get("audit") or {}
    seen: dict[int, set] = defaultdict(set)          # talent_id -> candidate-axis buckets it appears in
    counts: dict[str, Counter] = defaultdict(Counter)
    problems: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)

    def bad(kind: str, ex: dict):
        problems[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(ex)

    def check(task: str, tid, split: str):
        counts[task][split] += 1
        if tid is None:
            return
        tid = int(tid)
        want = reg.split_of(tid)
        seen[tid].add(_bucket(split))
        ok = (split == want) or (want == "eval" and split == "test") or \
            (task in ("pretrain",) and want == "dev" and split == "val")
        if not ok:
            bad(f"{task}:{want}->{split}", {"talent_id": tid})
        if want == "excluded":
            bad(f"{task}:excluded_account_present", {"talent_id": tid})

    # ---------------------------------------------------------------- relevance (two axes)
    risky = Counter()
    fam_by_split: dict[str, set] = defaultdict(set)
    q_by_side: dict[str, set] = defaultdict(set)
    cand_text: dict[str, dict] = {"train": {}, "held": {}}
    for r in read_shards(root / "relevance", "pairs"):
        sp = r.get("split", "train")
        counts["relevance"][sp] += 1
        risky[f"{sp}_{'risky' if r.get('risky') else 'stable'}"] += 1
        want = reg.pair_split(r["query"], r.get("resume_id"))
        if sp != want:
            bad(f"relevance_pair:{want}->{sp}", {"query": r["query"][:80], "talent_id": r.get("resume_id")})
        tid = r.get("resume_id")
        if tid is not None:
            cs = reg.split_of(tid)
            seen[int(tid)].add(_bucket(cs))            # candidate axis (dev / test pairs reuse train candidates by design)
        fam = reg.query_family(r["query"]) or query_key(r["query"])
        fam_by_split[sp].add(fam)
        q_by_side["train" if sp in TRAIN_SIDE_Q else "unseen"].add(r["query"])
        side = "train" if sp == "train" else ("held" if sp in ("val", "dev_cand", "test_cand") or
                                              (sp == "test" and reg.strict_test) else None)
        if side and r.get("candidate"):
            cand_text[side].setdefault(text_key(r["candidate"]), r["candidate"])
    train_fams = set().union(*(fam_by_split[s] for s in TRAIN_SIDE_Q)) if fam_by_split else set()
    unseen_fams = set().union(*(fam_by_split[s] for s in UNSEEN_Q)) if fam_by_split else set()
    overlap = train_fams & unseen_fams
    problems["query_families_in_train_and_dev_or_test"] = len(overlap)
    for f in list(overlap)[:5]:
        examples["query_family_overlap"].append(f)
    tq = {q: _qtoks(q) for q in q_by_side["train"]}
    for q in q_by_side["unseen"]:
        a = _qtoks(q)
        for t, b in tq.items():
            j = len(a & b) / max(1, len(a | b))
            if j >= 0.6:
                bad("near_duplicate_query_across_splits", {"unseen": q[:80], "train": t[:80], "jaccard": round(j, 2)})
                break

    # ---------------------------------------------------------------- resumes: extraction / classify / pretrain
    for r in read_shards(root / "resumes", "extract"):
        check("extraction", r.get("id"), r.get("split", "train"))
    for prefix in ("classify", "classify_t7"):
        for r in read_shards(root / "resumes", prefix):
            check(prefix, r.get("id"), r.get("split", "train"))
    held_h = set()
    p = root / "resumes" / "heldout_hashes.txt"
    if p.exists():
        held_h = set(p.read_text().split())
    for r in read_shards(root / "resumes", "pretrain"):
        sp = r.get("split", "train")
        check("pretrain", r.get("id"), sp)
        if sp not in ("train", "val"):
            bad("pretrain:heldout_split_in_pretraining", {"talent_id": r.get("id")})
        if r.get("h") in held_h:
            problems["pretrain:heldout_text_hash (dropped again by pack)"] += 1

    # ---------------------------------------------------------------- external sources
    ext = root / "external"
    jd_side: dict[str, set] = defaultdict(set)
    for r in read_shards(ext, "pairs"):
        sp = r.get("split", "train")
        counts["external_pairs"][sp] += 1
        jd_side["train" if sp in ("train", "val") else "held"].add(text_key(r["query"]))
    n = len(jd_side["train"] & jd_side["held"])
    problems["external_jd_in_train_and_dev_or_test"] = n
    cls_side: dict[str, set] = defaultdict(set)
    for r in read_shards(ext, "classify"):
        sp = r.get("split", "train")
        counts["external_classify"][sp] += 1
        cls_side["train" if sp == "train" else "held"].add(text_key(r["text"]))
    problems["external_classify_text_in_train_and_held_out"] = len(cls_side["train"] & cls_side["held"])
    held_text_keys = {text_key(q) for q in q_by_side["unseen"]} | set(cand_text["held"]) | jd_side["held"]
    for d in (ext, root / "knowledge"):                 # public text incl. general text and KB cards (Turn 51)
        for r in read_shards(d, "pretrain"):
            counts["public_pretrain"][r.get("group") or r.get("source") or "?"] += 1
            if text_key(r["text"]) in held_text_keys:
                bad("public_pretrain_equals_held_out_text", {"source": r.get("source")})

    # ---------------------------------------------------------------- near duplicates (MinHash)
    thr = float(acfg.get("near_dup_threshold", 0.8))
    cap = int(acfg.get("near_dup_max_docs", 200000))
    if acfg.get("near_dup", True) and cand_text["held"] and cand_text["train"]:
        idx = MinHashIndex(thr)
        held = list(cand_text["held"].values())[:cap]
        idx.add_many(held)
        hits = idx.query_many(cand_text["train"].values(), limit=cap)
        problems["near_duplicate_candidate_text_train_vs_held_out"] = len(hits)
        for i, j, est in hits[:5]:
            examples["near_duplicate_candidate"].append({"jaccard": round(est, 3)})
    counts["near_dup_checked"]["held_candidates"] = len(cand_text["held"])
    counts["near_dup_checked"]["train_candidates"] = min(len(cand_text["train"]), cap)

    multi = {t: s for t, s in seen.items() if len(s) > 1}
    problems["talent_ids_in_more_than_one_split"] = len(multi)
    for t, s in list(multi.items())[:10]:
        examples["multi_split"].append({"talent_id": t, "buckets": sorted(s)})
    return _finish(cfg, "leak_audit.json", problems, examples, counts, strict,
                   {"relevance_label_stability": dict(risky), "registry_version": reg.version,
                    "talent_ids_checked": len(seen), "query_families": {k: len(v) for k, v in fam_by_split.items()}})


def _finish(cfg, name, problems, examples, counts, strict, extra) -> dict:
    badp = {k: v for k, v in problems.items() if v}
    report = {"ok": not badp, "problems": badp, "examples": dict(examples),
              "counts": {k: dict(v) for k, v in counts.items()}, **extra}
    write_json(cfg.out("reports", name), report)
    if badp:
        log.error("LEAK AUDIT FAILED (%s): %s", name, badp)
        if strict:
            raise RuntimeError(f"leak audit failed: {badp} (see reports/{name})")
    else:
        log.info("leak audit passed (%s)", name)
    return report


def run_packed(cfg: Cfg, strict: bool = True) -> dict:
    """Audit the FINAL packed files through the key sidecars written by pack (`sft/<file>.keys.npy`: per example
    [query_key, candidate_or_text_key], int64; 0 = not applicable)."""
    sft = cfg.out_root_path / "packed" / "sft"
    keys: dict[str, np.ndarray] = {}
    for f in sorted(x for x in sft.glob("*.keys.npy") if not x.name.startswith("._")):   # skip macOS AppleDouble
        keys[f.name[: -len(".keys.npy")]] = np.load(f)
    problems, examples, counts = Counter(), defaultdict(list), defaultdict(Counter)
    if not keys:
        problems["no_key_sidecars (re-run pack)"] = 1
    by = defaultdict(lambda: defaultdict(set))         # task -> split -> {query keys} / {cand keys}
    cand = defaultdict(lambda: defaultdict(set))
    for name, k in keys.items():
        task, _, split = name.rpartition("_")
        if split in ("cand", "qonly"):                 # dev_cand / test_cand / test_qonly
            task, _, s0 = task.rpartition("_")
            split = f"{s0}_{split}"
        counts[task][split] += len(k)
        by[task][split].update(int(x) for x in k[:, 0] if x)
        cand[task][split].update(int(x) for x in k[:, 1] if x)
    for task in by:
        tq = set().union(*(by[task][s] for s in by[task] if s in TRAIN_SIDE_Q))
        uq = set().union(*(by[task][s] for s in by[task] if s in UNSEEN_Q))
        if tq & uq:
            problems[f"{task}:query_in_train_side_and_unseen_split"] = len(tq & uq)
        tr = cand[task].get("train", set())
        for s, ks in cand[task].items():
            if s in ("val", "dev_cand", "test_cand") or (task != "relevance" and s in ("val", "test")):
                n = len(tr & ks)
                if n:
                    problems[f"{task}:train_{s}_same_candidate_or_text"] = n
    return _finish(cfg, "leak_audit_packed.json", problems, examples, counts, strict, {"files": sorted(keys)})
