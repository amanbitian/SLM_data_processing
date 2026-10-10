"""
CLI:  python -m slm_data <step> [--config configs/data_config.yaml] [--set key=value ...]

Steps (in dependency order — `all` runs them in this order):
  inventory        check every source exists, count rows
  relevance-scan   collect judge labels per (query, resume_id) + score stability; list resume ids to join
  registry         query families -> train/dev/test, and one split per talent_id (eval, ~20K test, ~10K val/dev)
  resumes          clean + pseudonymise resumes, build extraction targets, resume_store.sqlite
  relevance-build  join resume text; pair split = query family x candidate split (registry.pair_split)
  knowledge        KB text + entity tables from O*NET, ESCO (TalentCLEF), ROR, OpenAlex, SEC, the skill taxonomy
  external         public JD / resume / general text (incl. external_data_1010)
  audit            leak audit on the source shards: candidates, query families, external data, near duplicates
  tokenizer        train SentencePiece BPE (32k)
  pack             tokenize to memory-mapped .bin files  (--only pretrain|sft)
  audit-packed     leak audit on the FINAL packed files (key sidecars written by pack)
  preview          print a few formatted training examples
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import load_config

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "data_config.yaml"


def preview(cfg, n: int = 2):
    from slm_model import prompting as P
    from .io_utils import read_shards
    root = cfg.out_root_path
    for d, prefix in (("resumes", "extract"), ("relevance", "pairs"), ("external", "pairs")):
        for i, r in enumerate(read_shards(root / d, prefix)):
            if i >= n:
                break
            if prefix == "extract":
                s = P.extraction_prompt(r["text"][:600] + " …") + P.extraction_target(r["target"])
            else:
                s = P.relevance_prompt(r["query"], r["candidate"][:600] + " …", r.get("date")) + \
                    P.relevance_target(r["verdict"], r.get("score"), r.get("failed"), r.get("reason"))
            print(f"\n===== {d}/{prefix} #{i} (split={r.get('split')}) =====\n{s[:2500]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slm_data")
    ap.add_argument("step", choices=["inventory", "relevance-scan", "registry", "resumes", "relevance-build",
                                     "knowledge", "external", "audit", "tokenizer", "pack", "audit-packed", "preview", "all"])
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set workers=12")
    ap.add_argument("--only", choices=["pretrain", "sft"], default=None, help="pack: only one part")
    ap.add_argument("--force", action="store_true", help="`all`: re-run steps even if already completed")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = load_config(a.config, a.set)
    t0 = time.time()
    from . import audit, build_external, build_knowledge, build_relevance, build_resumes, inventory, pack, registry, tokenizer_train
    steps = {
        "inventory": lambda: inventory.run(cfg),
        "relevance-scan": lambda: build_relevance.scan(cfg),
        "registry": lambda: registry.build(cfg),
        "resumes": lambda: build_resumes.run(cfg),
        "relevance-build": lambda: build_relevance.build(cfg),
        "knowledge": lambda: build_knowledge.run(cfg),
        "external": lambda: build_external.run(cfg),
        "audit": lambda: audit.run(cfg),
        "audit-packed": lambda: audit.run_packed(cfg),
        "tokenizer": lambda: tokenizer_train.run(cfg),
        "pack": lambda: pack.run(cfg, a.only),
        "preview": lambda: preview(cfg),
    }
    order = ["inventory", "relevance-scan", "registry", "resumes", "relevance-build", "knowledge", "external", "audit",
             "tokenizer", "pack", "audit-packed"]
    # step-level resume: `all` skips steps already completed with the SAME config (marker file holds a
    # hash of the resolved config). A changed config, --force, or naming a step explicitly re-runs it.
    # Turn 53 (review D16): the marker key is content-addressed — config + pipeline CODE + source-file stats +
    # the key of the step before it. Editing a builder, replacing a source file, or re-running an upstream step
    # therefore re-runs every step after it (a config-only hash missed code and input changes).
    from .io_utils import parquet_cache_clear
    from .step_keys import step_keys
    keys = step_keys(cfg, order)
    log = logging.getLogger("slm_data")
    for s in (order if a.step == "all" else [a.step]):
        marker = cfg.out("reports", f"{s}.done")
        key = keys.get(s, "preview")
        if a.step == "all" and not a.force and marker.exists() and marker.read_text().split()[:1] == [key]:
            log.info("==== step: %s already done (key %s) -> skip (use --force to redo) ====", s, key)
            continue
        log.info("==== step: %s ====", s)
        if marker.exists():
            marker.unlink()  # a crash mid-step must not leave a stale "done"
        try:
            steps[s]()
        finally:
            parquet_cache_clear()   # no Parquet handle outlives its step (USB drive can be ejected)
        if s != "preview":
            marker.write_text(f"{key} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    logging.getLogger("slm_data").info("finished in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
