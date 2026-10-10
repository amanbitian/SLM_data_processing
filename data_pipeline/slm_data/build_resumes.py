"""
Stage A — resume corpus.

Streams every row group of resume_data/*.parquet in parallel and produces:
  resumes/pretrain-*.jsonl.gz     cleaned + PII-redacted resume text (domain pretraining)
  resumes/extract-*.jsonl.gz      (resume, canonical target) pairs from parsed_resume_data
  resumes/gold-*.jsonl.gz         gpt-5.x taxonomy extractions (held-out gold eval)
  resumes/classify-*.jsonl.gz     resume -> curated Domain label (selected_candidate_per_domain_resume.csv)
  resume_store.sqlite             id -> cleaned text + relevance profile block (for joining)
  reports/resumes_stats.json
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

from .config import Cfg
from .registry import load_registry
from .io_utils import (ResumeStore, ShardWriter, parquet_num_row_groups, read_row_group, read_shards,
                       write_json)
from .pii import redact
from .pseudonymize import Pseudonymizer, audit_leaks, known_from_row
from .resume_schema import canonical_target, grounding_score, profile_block_from_parse, taxonomy_target
from .splits import split_of, unit_hash
from .text_clean import clean_text, content_hash, quality_ok

log = logging.getLogger(__name__)

# resume_data (Sep 2026 export) and turing_candidate_dataset (T7, Oct 2026) use different column names for the
# same facts; missing columns come back as None. The PII columns are read only to find and replace those values.
COLS = ["talent_id", "resume_plain", "parsed_resume_data", "resume_parse_status",
        "account_full_name", "name", "country", "years_of_experience",
        "email", "account_email", "phone_number", "account_phone", "linkedin_url", "github_url",
        "domain", "subdomain"]

_CFG: Optional[dict] = None
_NEEDED: Optional[set] = None
_PS: Optional[Pseudonymizer] = None
_TX = None
_REG = None


def _init(cfg: dict, needed: Optional[set]):
    global _CFG, _NEEDED, _PS, _TX, _REG
    _CFG, _NEEDED = cfg, needed
    _REG = load_registry(Cfg(cfg))                  # one split per talent_id for every task
    if cfg["cleaning"].get("pii_mode", "redact") == "pseudonymize":
        _PS = Pseudonymizer()                       # needs SLM_PSEUDO_SALT in the environment
    tx_path = cfg["sources"].get("skill_taxonomy_csv")
    if tx_path and cfg["extraction"].get("link_skills", True):
        from .skill_taxonomy import SkillTaxonomy
        p = Path(cfg["data_root"]) / tx_path
        _TX = SkillTaxonomy(p) if p.exists() else None


def _dedupe_skills(skills: list[dict]) -> list[dict]:
    """synonyms collapse to one skill ("ReactJS", "React.js", "React"): keep the first surface form."""
    if _TX is None:
        return skills
    seen, out = set(), []
    for sk in skills:
        e = _TX.link(sk.get("name", ""))
        key = e["skill_id"] if e else sk.get("name", "").lower()
        if key not in seen:
            seen.add(key)
            out.append(sk)
    return out


def _process_row_group(task: tuple[str, int]) -> dict:
    path, rg = task
    cc, ec = _CFG["cleaning"], _CFG["extraction"]
    out = {"items": [], "stats": Counter(), "path": path}
    try:
        data = read_row_group(Path(path), rg, COLS)
    except Exception as e:  # corrupted row group: skip, but report
        out["stats"]["rg_error"] += 1
        out["error"] = f"{path}:{rg}: {e}"
        return out
    st = out["stats"]
    for i in range(len(data["talent_id"])):
        st["rows"] += 1
        raw = data["resume_plain"][i]
        if not raw:
            st["no_text"] += 1
            continue
        tid = data["talent_id"][i]
        split = _REG.split_of(tid) if _REG is not None else "train"
        if split == "excluded":                     # QA / test accounts
            st["excluded_account"] += 1
            continue
        parsed = None
        pj = data["parsed_resume_data"][i]
        status = data["resume_parse_status"][i]
        if pj and (ec.get("require_parse_status") in (None, status)):
            try:
                parsed = json.loads(pj)
                if not isinstance(parsed, dict):
                    parsed = None
            except Exception:
                st["bad_parse_json"] += 1
        text = clean_text(raw, cc["max_resume_chars"])
        if len(text) < cc["min_resume_chars"]:
            st["too_short"] += 1
            continue
        if not quality_ok(text, cc["min_alpha_ratio"], cc["min_ascii_ratio"]):
            st["low_quality"] += 1
            continue
        h = content_hash(text)          # dedupe key from the ORIGINAL text (before any PII change)
        contact, meta = None, {}
        if _PS is not None:             # pii_mode: pseudonymize -> realistic fake values, verifiable codes
            row = {c: data[c][i] for c in COLS}
            known = known_from_row(row, parsed)
            country = data["country"][i] or (parsed or {}).get("country")
            text, contact, pst = _PS.pseudonymize(text, known, person_key=tid, country=country)
            for k, v in pst.items():
                st[f"pseudo_{k}"] += v
            leaks = audit_leaks(text, known)          # pseudonymiser recall: known real values still present
            if any(leaks.values()):
                st["docs_with_leak"] += 1
                for k, v in leaks.items():
                    st[f"leak_{k}"] += v
                if cc.get("drop_on_leak", True):
                    st["dropped_leak"] += 1
                    continue
            prof = _PS.profile(tid, country)
            meta["pseudo_names"] = [prof["first"], prof["middle"], prof["last"]]
        elif cc.get("redact_pii", True):
            names = [data["account_full_name"][i], data["name"][i]]
            if parsed:
                names += [parsed.get(k) for k in ("full_name", "first_name", "middle_name", "last_name")]
            text = redact(text, names)
        item = {"id": int(tid) if tid is not None else None, "h": h, "text": text, "meta": meta, "split": split}
        dom = data["domain"][i]
        if dom and str(dom).strip() not in ("", "Unclassified"):      # T7 domain label -> classification task
            item["domain"] = str(dom).strip()
        if parsed:
            st["with_parse"] += 1
            try:
                tgt = canonical_target(parsed, ec)
                if tgt.get("skills"):
                    tgt["skills"] = _dedupe_skills(tgt["skills"])
                g = grounding_score(tgt, text)
                if ec.get("include_contact", True) and contact:
                    # ground truth = the injected pseudonyms that occur in this text (other people's e-mails excluded)
                    tgt = {"contact": {k: v for k, v in contact.items() if k != "other_emails"}, **tgt}
                item["target"], item["grounding"] = tgt, round(g, 3)
                item["profile"] = profile_block_from_parse(
                    parsed, text, _CFG["relevance"].get("max_candidate_chars", 4000))
                item["meta"] = {**meta, "country": parsed.get("country"), "role": parsed.get("role")}
            except Exception:  # malformed parse: keep the text for pretraining only
                st["target_error"] += 1
                for k in ("target", "grounding", "profile"):
                    item.pop(k, None)
        elif _NEEDED is not None and item["id"] in _NEEDED:
            item["profile"] = None
        out["items"].append(item)
    return out


def _tasks(files: list[Path], limit: Optional[int] = None) -> list[tuple[str, int]]:
    tasks = []
    for f in files:
        n = parquet_num_row_groups(f)
        tasks += [(str(f), rg) for rg in range(min(n, limit) if limit else n)]
    return tasks


def build_classify(cfg: Cfg, out_dir: Path) -> dict:
    """Curated Domain labels (selected_candidate_per_domain_resume.csv) -> resume classification."""
    import pandas as pd
    stats = Counter()
    csv = cfg.src(cfg["sources"]["taxonomy_input_csv"])
    if not csv.exists():
        log.warning("taxonomy csv missing: %s", csv)
        return {}
    df = pd.read_csv(csv)
    reg = load_registry(cfg)
    with ShardWriter(out_dir, "classify") as cw:
        for r in df.itertuples(index=False):
            t = clean_text(getattr(r, "resume", ""), cfg["cleaning"]["max_resume_chars"])
            if len(t) < cfg["cleaning"]["min_resume_chars"]:
                continue
            t = redact(t) if cfg["cleaning"].get("redact_pii", True) else t
            rid = int(r.resume_id)
            dom = getattr(r, "Domain", None)
            if isinstance(dom, str) and dom.strip():
                sp = reg.split_of(rid) if reg is not None else split_of(f"cls:{rid}", val=0.05, test=0.10)
                if sp == "excluded":
                    continue
                cw.write({"id": rid, "text": t, "label": dom.strip(), "label_space": "domain",
                          "split": "test" if sp == "eval" else sp})
                stats["classify"] += 1
    return dict(stats)


def _gold_records(cfg: Cfg) -> list[tuple[str, dict]]:
    recs = []
    for rel in cfg["sources"]["taxonomy_extraction_jsons"]:
        p = cfg.src(rel)
        if p.exists():
            model = p.stem.replace("all_candidates", "").strip("_")
            recs += [(model, r) for r in json.load(open(p, encoding="utf-8")) if r.get("id") is not None]
    return recs


def build_gold(gold: list[tuple[str, dict]], out_dir: Path, store: ResumeStore) -> dict:
    """gpt-5.x taxonomy extractions joined to their resume text (by talent_id) — held-out gold eval."""
    stats = Counter()
    with ShardWriter(out_dir, "gold") as gw:
        for model, rec in gold:
            s = store.get(int(rec["id"]))
            if not s or not s.get("text"):
                stats["gold_missing_text"] += 1
                continue
            gw.write({"id": int(rec["id"]), "text": s["text"], "target": taxonomy_target(rec), "model": model,
                      "split": "test"})
            stats["gold"] += 1
    return dict(stats)


def run(cfg: Cfg) -> dict:
    t0 = time.time()
    out_dir = cfg.out_root_path / "resumes"
    files, primary_files = [], set()
    globs = cfg["sources"]["resume_parquet_globs"]
    for gi, g in enumerate(globs):
        fs = cfg.src_glob(g)
        files += fs
        if gi == 0 and len(globs) > 1:
            primary_files = {str(f) for f in fs}
    if not files:
        raise FileNotFoundError("no resume parquet files matched sources.resume_parquet_globs")
    if cfg["cleaning"].get("pii_mode", "redact") == "pseudonymize":
        Pseudonymizer()                 # fail fast when SLM_PSEUDO_SALT is missing
    # the FIRST glob is the newest export (turing_candidate_dataset): a talent_id found there is skipped in
    # the older exports, so each candidate is read once, from the freshest source
    primary_ids: Optional[set] = None
    if primary_files and cfg["sources"].get("prefer_first_source", True):
        primary_ids = set()
        for f in sorted(primary_files):
            for rg in range(parquet_num_row_groups(Path(f))):
                primary_ids.update(x for x in read_row_group(Path(f), rg, ["talent_id"])["talent_id"] if x is not None)
        log.info("primary export: %d talent_ids from %d files", len(primary_ids), len(primary_files))
    needed_path = cfg.out_root_path / "relevance" / "needed_resume_ids.txt"
    needed = None
    if needed_path.exists():
        needed = {int(x) for x in needed_path.read_text().split()}
        log.info("store will keep %d ids needed by relevance (+ every parsed resume)", len(needed))

    tasks = _tasks(files, cfg["sources"].get("limit_row_groups"))
    log.info("%d files, %d row groups, workers=%d", len(files), len(tasks), cfg["workers"])

    store_path = cfg.out_root_path / "resume_store.sqlite"
    if store_path.exists():
        store_path.unlink()
    store = ResumeStore(store_path)
    stats = Counter()
    seen_pre_h, seen_ext_id, seen_ext_h = set(), set(), set()
    cls_stats = build_classify(cfg, out_dir)
    gold = _gold_records(cfg)
    gold_ids = {int(r["id"]) for _, r in gold}
    needed = (needed or set()) | gold_ids
    errors = []
    store_buf = []
    ec = cfg["extraction"]
    held_h: set[str] = set()
    with ShardWriter(out_dir, "pretrain") as pw, ShardWriter(out_dir, "extract") as ew, \
            ShardWriter(out_dir, "classify_t7") as dw, ShardWriter(out_dir, "heldout") as hw:
        ctx = get_context("spawn")
        with ctx.Pool(cfg["workers"], initializer=_init, initargs=(dict(cfg), needed)) as pool:
            for k, res in enumerate(pool.imap(_process_row_group, tasks, chunksize=1)):
                stats.update(res["stats"])
                if "error" in res:
                    errors.append(res["error"])
                older = primary_ids is not None and res.get("path") not in primary_files
                for it in res["items"]:
                    if older and it["id"] in primary_ids:
                        stats["skipped_in_newer_export"] += 1
                        continue
                    sp = it.get("split", "train")
                    if sp in ("test", "eval"):
                        # held-out people: never in pretraining; their text hash also blocks copies elsewhere
                        held_h.add(it["h"].hex())
                        stats[f"pretrain_skip_{sp}"] += 1
                        # Turn 56: their (pseudonymised) text is kept for near-duplicate decontamination at
                        # pack time (MinHash against every training resume, ours and public) - never trained on
                        hw.write({"id": it["id"], "text": it["text"], "split": sp})
                    else:
                        dup_text = it["h"] in seen_pre_h
                        if not dup_text or not cfg["cleaning"].get("dedupe", True):
                            seen_pre_h.add(it["h"])
                            # dev people are validation too: their text goes to the pretraining VAL stream
                            pw.write({"id": it["id"], "text": it["text"], "source": "resume",
                                      "split": "val" if sp == "dev" else sp,
                                      "h": it["h"].hex()})
                            stats[f"pretrain_docs_{sp}"] += 1
                        else:
                            stats["dup_text"] += 1
                    if it.get("domain") and (sp != "train" or
                                             unit_hash(f"dom:{it['id']}") < float(ec.get("domain_train_frac", 0.3))):
                        dw.write({"id": it["id"], "text": it["text"], "label": it["domain"], "label_space": "domain_t7",
                                  "split": "test" if sp == "eval" else sp})
                        stats[f"classify_t7_{sp}"] += 1
                    if "target" in it:
                        if it["id"] in seen_ext_id or it["h"] in seen_ext_h:
                            stats["dup_extract"] += 1
                        elif it["grounding"] < ec["min_grounding"]:
                            stats["low_grounding"] += 1
                        elif not it["target"].get("work_experience") and not it["target"].get("education"):
                            stats["empty_target"] += 1
                        elif ec.get("max_examples") and stats["extract_examples"] >= ec["max_examples"]:
                            pass
                        else:
                            seen_ext_id.add(it["id"])
                            seen_ext_h.add(it["h"])
                            # gold candidates are evaluation assets -> test; everyone else follows the registry
                            split = "test" if (it["id"] in gold_ids or sp == "eval") else sp
                            ew.write({"id": it["id"], "text": it["text"], "target": it["target"],
                                      "grounding": it["grounding"], "split": split})
                            stats["extract_examples"] += 1
                    if it.get("id") is not None and ("target" in it or (needed and it["id"] in needed)):
                        store_buf.append((it["id"], it["text"], it.get("profile"), int("target" in it),
                                          json.dumps(it.get("meta") or {})))
                if len(store_buf) > 5000:
                    store.put_many(store_buf)
                    store_buf.clear()
                if k % 200 == 0:
                    log.info("[%d/%d rg] docs=%d extract=%d (%.0fs)", k, len(tasks), stats["pretrain_docs_train"] + stats["pretrain_docs_val"],
                             stats["extract_examples"], time.time() - t0)
    store.put_many(store_buf)
    (out_dir / "heldout_hashes.txt").write_text("\n".join(sorted(held_h)))
    stats["heldout_text_hashes"] = len(held_h)
    stats["store_rows"] = store.count()
    gold_stats = {**cls_stats, **build_gold(gold, out_dir, store)}
    store.close()
    report = {"stats": dict(stats), "gold": gold_stats, "errors": errors[:50], "seconds": round(time.time() - t0)}
    write_json(cfg.out("reports", "resumes_stats.json"), report)
    log.info("resumes done: %s", json.dumps(report["stats"]))
    return report
