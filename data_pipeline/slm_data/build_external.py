"""
Stage C — public/external data.

  external/pretrain-*.jsonl.gz   job descriptions + Kaggle resumes (pretraining text)
  external/pairs-*.jsonl.gz      HF resume-JD-match pairs as relevance examples (JD = query)
  external/classify-*.jsonl.gz   Kaggle resume -> category, skills/qualification -> job role
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from .config import Cfg
from .io_utils import (ShardWriter, parquet_num_row_groups, read_csv_from_zip, read_row_group, write_json)
from .pii import redact
from .splits import split_of
from .near_dup import norm_text
from .text_clean import clean_inline, clean_text

log = logging.getLogger(__name__)

HF_RE = re.compile(r"job description\s*<<(?P<jd>.*?)>>\s*the resume:\s*<<(?P<cv>.*?)>>", re.S | re.I)
# (verdict, score, confidence). "Potential Fit" is genuinely borderline -> soft target P(relevant)=0.6
HF_LABELS = {"good fit": ("relevant", 8, 0.9), "potential fit": ("relevant", 5, 0.6), "no fit": ("not_relevant", 2, 0.9)}


def _hf_pairs(cfg: Cfg, w: ShardWriter, st: Counter):
    from .near_dup import text_key
    files = cfg.src_glob(cfg["sources"]["external"]["hf_jd_match_glob"])
    # decontamination (Turn 47): a JD that occurs in the benchmark's test file is never trained on
    test_jds = set()
    for p in files:
        if "test" in p.stem:
            for rg in range(parquet_num_row_groups(p)):
                for text in read_row_group(p, rg, ["text"])["text"]:
                    m = HF_RE.search(text or "")
                    if m:
                        test_jds.add(text_key(clean_text(m.group("jd"), 2500)))
    for p in files:
        split_name = "test" if "test" in p.stem else None
        for rg in range(parquet_num_row_groups(p)):
            d = read_row_group(p, rg, ["text", "label"])
            for text, label in zip(d["text"], d["label"]):
                m = HF_RE.search(text or "")
                lab = HF_LABELS.get(str(label).strip().lower())
                if not m or not lab:
                    st["hf_unparsed"] += 1
                    continue
                jd = clean_text(m.group("jd"), 2500)
                cv = redact(clean_text(m.group("cv"), cfg["cleaning"]["max_resume_chars"]))
                if len(cv) < 200 or len(jd) < 50:
                    st["hf_short"] += 1
                    continue
                if not split_name and text_key(jd) in test_jds:
                    st["hf_train_jd_in_test_dropped"] += 1
                    continue
                if split_name:  # the benchmark's own test file -> half unseen dev, half locked test (by JD)
                    from .splits import unit_hash
                    split = "dev" if unit_hash(f"hfq:{jd[:200]}") < 0.5 else "test"
                else:
                    split = split_of(f"hf:{jd[:200]}:{cv[:100]}", 0.02, 0.0)
                w.write({"query": jd, "date": None, "candidate": f"Resume:\n{cv}", "verdict": lab[0],
                         "score": lab[1], "failed": None, "reason": None, "agreement": lab[2], "split": split,
                         "src": "hf_jd_match", "domain": None})
                st[f"hf_{lab[0]}"] += 1


def run(cfg: Cfg) -> dict:
    ext = cfg["external"]
    src = cfg["sources"]["external"]
    out_dir = cfg.out_root_path / "external"
    st = Counter()
    with ShardWriter(out_dir, "pretrain") as pw, ShardWriter(out_dir, "pairs") as rw, \
            ShardWriter(out_dir, "classify") as cw:
        if ext.get("use_hf_jd_match"):
            _hf_pairs(cfg, rw, st)
        if ext.get("use_job_descriptions") and cfg.src(src["job_desc_zip"]).exists():
            df = read_csv_from_zip(cfg.src(src["job_desc_zip"]))
            for title, desc in df[["Job Title", "Job Description"]].itertuples(index=False, name=None):
                desc = clean_text(desc, 8000)
                if len(desc) > 100:
                    pw.write({"text": f"Job Title: {clean_inline(title)}\n{desc}", "source": "job_desc"})
                    st["job_desc"] += 1
        if ext.get("use_kaggle_resume") and cfg.src(src["kaggle_resume_zip"]).exists():
            df = read_csv_from_zip(cfg.src(src["kaggle_resume_zip"]), usecols=["ID", "Resume_str", "Category"])
            for r in df.itertuples(index=False):
                t = redact(clean_text(r.Resume_str, cfg["cleaning"]["max_resume_chars"]))
                if len(t) < cfg["cleaning"]["min_resume_chars"]:
                    continue
                pw.write({"text": t, "source": "kaggle_resume"})
                # split by TEXT (not id): duplicate resumes under different ids land in the same split
                cw.write({"id": int(r.ID), "text": t, "label": str(r.Category).strip(), "label_space": "category",
                          "split": split_of(f"kag:{norm_text(t)[:2000]}", 0.05, 0.05)})
                st["kaggle_resume"] += 1
        if ext.get("use_candidate_role") and cfg.src(src["candidate_role_csv"]).exists():
            import pandas as pd
            df = pd.read_csv(cfg.src(src["candidate_role_csv"]))
            for r in df.itertuples(index=False):
                t = (f"Skills: {r.skills}\nQualification: {r.qualification}\n"
                     f"Experience level: {r.experience_level}")
                cw.write({"id": int(r.candidate_id), "text": t, "label": str(r.job_role).strip(),
                          "label_space": "job_role", "split": split_of(f"role:{norm_text(t)}", 0.05, 0.05)})
                st["candidate_role"] += 1
        # external_data_1010: Djinni, resume-job-fit text, WDC JobPosting (pretraining text, Turn 45)
        from . import external_more
        external_more.run(cfg, pw, st)
    write_json(cfg.out("reports", "external_stats.json"), dict(st))
    log.info("external done: %s", dict(st))
    return dict(st)
