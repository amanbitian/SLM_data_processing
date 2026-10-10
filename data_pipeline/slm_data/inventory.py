"""Stage 0 — inventory of every configured source (exists? rows? columns?)."""
from __future__ import annotations

import logging

from .config import Cfg
from .io_utils import HAVE_PYARROW, parquet_columns, parquet_num_row_groups, parquet_num_rows, write_json

log = logging.getLogger(__name__)


def run(cfg: Cfg) -> dict:
    s = cfg["sources"]
    rep: dict = {"pyarrow": HAVE_PYARROW, "data_root": str(cfg.data_root_path), "sources": {}}

    def pq_info(p):
        return {"path": str(p), "rows": parquet_num_rows(p), "row_groups": parquet_num_row_groups(p),
                "columns": len(parquet_columns(p))}

    files = [f for g in s["resume_parquet_globs"] for f in cfg.src_glob(g)]
    rep["sources"]["resume_parquet"] = {"files": len(files), "rows": sum(parquet_num_rows(f) for f in files),
                                       "columns": parquet_columns(files[0]) if files else []}
    rp = cfg.src(s["relevance_parquet"])
    rep["sources"]["relevance_parquet"] = pq_info(rp) if rp.exists() else "MISSING"
    rep["sources"]["taxonomy_jsons"] = {p: cfg.src(p).exists() for p in s["taxonomy_extraction_jsons"]}
    rep["sources"]["taxonomy_input_csv"] = cfg.src(s["taxonomy_input_csv"]).exists()
    ext = s["external"]
    rep["sources"]["external"] = {
        "hf_jd_match": [str(p) for p in cfg.src_glob(ext["hf_jd_match_glob"])],
        "kaggle_resume_zip": cfg.src(ext["kaggle_resume_zip"]).exists(),
        "job_desc_zip": cfg.src(ext["job_desc_zip"]).exists(),
        "candidate_role_csv": cfg.src(ext["candidate_role_csv"]).exists(),
    }
    if not HAVE_PYARROW:
        log.warning("pyarrow not installed — using the slow pure-python parquet fallback. pip install pyarrow")
    write_json(cfg.out("reports", "inventory.json"), rep)
    log.info("inventory: resume rows=%s, relevance=%s", rep["sources"]["resume_parquet"]["rows"],
             rep["sources"]["relevance_parquet"])
    return rep
