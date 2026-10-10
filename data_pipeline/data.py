#!/usr/bin/env python3
"""
data.py — prepares ALL training data in one command, then splits it by candidate (Turn 45).

    export SLM_PSEUDO_SALT='<long random secret>'      # fake-identity key, keep it out of the repo
    python data_pipeline/data.py                       # every step, skipping steps already done
    python data_pipeline/data.py registry audit        # only some steps
    python data_pipeline/data.py --force               # redo everything

What it does, in order
  1. inventory        check every source (turing_candidate_dataset, training_data, external_data_1010)
  2. relevance-scan   every judged (query, candidate) pair from eval_consolidated + evaluation/ CSVs, with
                      score_mean / score_min / score_max / score_std and a `risky` flag when the verdict changed
  3. registry         two axes: query FAMILIES (near-duplicates merged) -> train / dev / test, and one split per
                      talent_id (eval assets, ~20,000 test + ~10,000 val/dev candidates, QA accounts excluded,
                      everyone else train; shared-e-mail accounts follow)
  4. resumes          resume corpus (newest export first), pseudonymised PII, extraction targets, domain labels
  5. relevance-build  pair split = query family x candidate (dev / test = unseen queries; val / dev_cand /
                      test_cand = held-out candidates); train balanced; near-copies of held-out text dropped
  6. knowledge        KB text + entity tables: O*NET, ESCO (TalentCLEF training), ROR, OpenAlex, SEC, skill taxonomy
  7. external         public text: HF resume-JD, Kaggle, JDs, Djinni, resume-job-fit, WDC, Wikipedia, FineWeb-Edu,
                      Cosmopedia (each document tagged resume / jobs / knowledge / general)
  8. audit            fails on candidate / query-family / external / near-duplicate leaks
  9. tokenizer        trained on training people + public text only
 10. pack             memory-mapped files for the trainer (one file per task and split) + key sidecars
 11. audit-packed     the same leak checks on exactly what the trainer reads

Not used for training: the test and val candidates, the evaluation assets (extraction gold, curated sets,
fraud sets, JOB-110870 screening, human gold) and QA / test accounts. Everything is configured in
configs/data_config.yaml; the builders live in slm_data/.
(the training repository's training/slm_train/data.py is the trainer's LOADER for the packed files; this file PREPARES them.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "model")]

STEPS = ["inventory", "relevance-scan", "registry", "resumes", "relevance-build", "knowledge", "external", "audit",
         "tokenizer", "pack", "audit-packed"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help=f"steps to run (default: all): {' '.join(STEPS)}")
    ap.add_argument("--config", default=str(HERE / "configs" / "data_config.yaml"))
    ap.add_argument("--set", action="append", default=[], help="override a config key, e.g. --set workers=12")
    ap.add_argument("--force", action="store_true", help="re-run steps that are already done")
    a = ap.parse_args(argv)
    bad = [s for s in a.steps if s not in STEPS]
    if bad:
        ap.error(f"unknown step(s) {bad}; choose from {STEPS}")
    from slm_data.__main__ import main as slm_main
    base = ["--config", a.config] + sum((["--set", s] for s in a.set), [])
    if not a.steps:
        return slm_main(["all"] + base + (["--force"] if a.force else []))
    for s in a.steps:
        slm_main([s] + base)


if __name__ == "__main__":
    main()
