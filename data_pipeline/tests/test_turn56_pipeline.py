"""Turn 56 (second review) data-side checks on invented data — no real data, no network, no sentencepiece.

  step keys   editing pack.py re-runs only pack / audit-packed; salt and eval assets are in the keys;
              AppleDouble ._*.py files are ignored
  pack        bottom-k validation is independent of source order; global dedupe across sources (all groups);
              near-copies of held-out text are dropped from resume training text; content digests; SFT written
              atomically (stale files disappear) with a query-FAMILY column
  groups      unknown sources fail loudly
  downloader  shard sampling spreads over the whole source; incomplete / empty Wikidata files not accepted
  eval        length-aware eval sharding never gives a GPU an example longer than its cap
Run:  python data_pipeline/tests/test_turn56_pipeline.py
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "data_pipeline"), str(ROOT / "model")]
# the eval-sharding check needs the training repository (slm_train.evaluate) next to this one
TRAIN_REPO = Path(os.environ.get("SLM_TRAINING_REPO") or (ROOT.parent / "SLM_training"))

from slm_data import pack as PK  # noqa: E402
from slm_data import step_keys as SK  # noqa: E402
from slm_data.config import Cfg  # noqa: E402
from slm_data.io_utils import ShardWriter  # noqa: E402


class FakeTok:
    bos_id, eos_id, vocab_size = 1, 2, 1000

    def _ids(self, t):
        return [3 + (hash(w) % 900) for w in t.split()][:4000]

    def encode_batch(self, texts, num_threads=1):
        return [self._ids(t) for t in texts]

    def encode_answer(self, t):
        return self._ids(t) + [2]

    def encode_relevance(self, q, c, budget, date=None):
        return self._ids(q + " " + c)[:budget]

    def encode_extraction(self, t, budget):
        return self._ids(t)[:budget]

    def encode_classify(self, t, space, budget):
        return self._ids(t)[:budget]


class SeqPool:
    def imap(self, fn, it):
        return map(fn, it)


def words(n, seed):
    r = random.Random(seed)
    return " ".join(f"w{r.randint(0, 5000)}" for _ in range(n))


def t_step_keys():
    d = Path(tempfile.mkdtemp())
    (d / "a.parquet").write_bytes(b"x" * 10)
    (d / "assets.csv").write_bytes(b"id\n1\n")
    cfg = {"data_root": str(d), "sources": {"g": "*.parquet"}, "splits": {"eval_assets": [{"path": "assets.csv"}]},
           "seed": 1}
    order = list(SK.STEP_DEPS)
    k1 = SK.step_keys(cfg, order)
    real = SK.code_hash
    SK.code_hash = lambda mods=None: "edited" if mods and "pack" in mods else real(mods)
    try:
        k2 = SK.step_keys(cfg, order)
    finally:
        SK.code_hash = real
    changed = {s for s in order if k1[s] != k2[s]}
    assert changed == {"pack", "audit-packed"}, changed
    (d / "assets.csv").write_bytes(b"id\n2\n")                 # eval asset edited (same size)
    k3 = SK.step_keys(cfg, order)
    assert k3["registry"] != k1["registry"] and k3["inventory"] != k1["inventory"]
    os.environ["SLM_PSEUDO_SALT"] = "salt-A"
    ka = SK.step_keys(cfg, order)
    os.environ["SLM_PSEUDO_SALT"] = "salt-B"
    kb = SK.step_keys(cfg, order)
    del os.environ["SLM_PSEUDO_SALT"]
    assert ka["resumes"] != kb["resumes"] and ka["registry"] == kb["registry"] and ka["pack"] != kb["pack"]
    assert "salt-A" not in json.dumps(ka)
    # AppleDouble files: hash a TEMP copy of the package with and without a ._ file (never write into the repo)
    import shutil
    tmp = Path(tempfile.mkdtemp()) / "slm_data"
    shutil.copytree(SK._PKG, tmp, ignore=shutil.ignore_patterns("__pycache__", "._*"))
    real_pkg = SK._PKG
    SK._PKG = tmp
    try:
        h0 = SK.code_hash()
        (tmp / "._zz.py").write_bytes(b"\x00\x05\x16\x07 apple double")
        assert SK.code_hash() == h0
    finally:
        SK._PKG = real_pkg
    print("step keys: per-step DAG, eval assets, salt fingerprint, AppleDouble ignored  OK")


def _cfg(root: Path, **pack) -> Cfg:
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    return Cfg({"out_root": str(out), "data_root": str(root), "workers": 1, "seed": 1,
                "pack": {"pretrain_val_tokens": 300, "chars_per_token_estimate": 4.0, "max_seq_len": 4096,
                         "max_answer_tokens": 4096, "think_frac": 0.0, **pack}})


def _write(dir_: Path, prefix: str, recs: list):
    dir_.mkdir(parents=True, exist_ok=True)
    with ShardWriter(dir_, prefix) as w:
        for r in recs:
            w.write(r)


def t_pack_pretrain():
    root = Path(tempfile.mkdtemp())
    cfg = _cfg(root)
    o = cfg.out_root_path
    held_text = words(300, 999)
    resumes = [{"id": i, "text": words(200, i), "source": "resume", "split": "train", "h": f"{i:032x}"}
               for i in range(40)]
    resumes.append({"id": 900, "text": held_text + " extra tail words", "source": "resume", "split": "train",
                    "h": "f" * 32})                                    # near-copy of a held-out resume
    resumes += [{"id": 100 + i, "text": words(50, 100 + i), "source": "resume", "split": "val"} for i in range(30)]
    _write(o / "resumes", "pretrain", resumes)
    _write(o / "resumes", "heldout", [{"id": 7, "text": held_text, "split": "test"}])
    jd = words(80, 5000)
    _write(o / "external", "pretrain", [{"text": jd, "source": "djinni_jobs"},
                                        {"text": jd.upper(), "source": "wdc_jobposting", "group": "jobs"}] +
           [{"text": words(60, 6000 + i), "source": "fineweb_edu"} for i in range(400)])
    (o / "knowledge").mkdir(parents=True, exist_ok=True)
    (o / "relevance").mkdir(parents=True, exist_ok=True)
    out = o / "packed"
    out.mkdir(parents=True, exist_ok=True)
    PK._TOK, PK._PCFG, PK._ND = FakeTok(), {"_heldout_index": str(PK._build_heldout_index(cfg, out))}, None
    rep = PK.pack_pretrain(cfg, SeqPool(), out)
    f = rep["filters"]
    assert f.get("dropped_duplicate_jobs") == 1, f                      # same JD in two sources (case differs)
    assert f.get("dropped_near_dup_of_heldout_resume") == 1, f
    assert rep["streams"]["resume"]["val_docs"] < 30                   # budget respected
    assert set(rep["digests"]) >= {"pretrain_resume_train.bin", "pretrain_jobs_train.bin"}
    assert rep["digests"]["pretrain_resume_train.bin"] == PK.file_digest(out / "pretrain_resume_train.bin")
    val1 = (out / "pretrain_resume_val.bin").read_bytes()
    # same documents, reversed source order -> identical validation stream (bottom-k by hash)
    _write(o / "resumes", "pretrain", list(reversed(resumes)))
    PK._ND = None
    PK.pack_pretrain(cfg, SeqPool(), out)
    assert (out / "pretrain_resume_val.bin").read_bytes() == val1, "validation selection depends on order"
    print(f"pack pretrain: cross-source dedupe, held-out near-dup drop, order-independent val, digests  OK  {f}")


def t_pack_sft():
    root = Path(tempfile.mkdtemp())
    cfg = _cfg(root)
    o = cfg.out_root_path
    pairs = [{"query": q, "candidate": words(40, i), "resume_id": i, "verdict": "relevant" if i % 2 else "not_relevant",
              "score": 7.0, "split": "train", "agreement": 1.0}
             for i, q in enumerate(["python developer remote"] * 4 + ["python developer  remote!"] * 4)]
    _write(o / "relevance", "pairs", pairs)
    for d in ("resumes", "external"):
        (o / d).mkdir(parents=True, exist_ok=True)
    out = o / "packed"
    (out / "sft").mkdir(parents=True, exist_ok=True)
    (out / "sft" / "classify_val.bin").write_bytes(b"stale")       # from an older build
    PK._TOK, PK._PCFG = FakeTok(), dict(cfg["pack"])
    fam = {"python developer remote": "fA", "python developer  remote!": "fA"}

    class Reg:
        def query_family(self, q):
            return fam.get(q)
    import slm_data.registry as R
    real = R.load_registry
    R.load_registry = lambda c: Reg()
    try:
        rep = PK.pack_sft(cfg, SeqPool(), out, np.uint16)
    finally:
        R.load_registry = real
    assert not (out / "sft" / "classify_val.bin").exists(), "stale SFT file survived"
    assert not (out / "sft.tmp").exists() and not (out / "sft.old").exists()
    lab = np.load(out / "sft" / "relevance_train.labels.npy")
    assert lab.shape[1] == 7
    assert len(set(lab[:, 3])) == 2 and len(set(lab[:, 6])) == 1   # 2 exact queries, 1 family
    assert "sft/relevance_train.bin" in rep["digests"]
    print("pack sft: atomic directory swap, family column, digests  OK")


def t_groups():
    from slm_data.groups import group_of
    assert group_of({"source": "djinni_jobs"}) == "jobs"
    assert group_of({"source": "anything", "group": "knowledge"}) == "knowledge"
    try:
        group_of({"source": "djini_jobs"})
        raise AssertionError("unknown source accepted")
    except ValueError:
        pass
    print("groups: unknown source fails loudly  OK")


def t_downloader():
    spec = importlib.util.spec_from_file_location("dl", ROOT / "data_pipeline" / "scripts" / "download_external_data.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    files = [(f"shard-{i:03d}.parquet", 1) for i in range(100)]
    first = [int(f[0][6:9]) for f in dl.spread(files)[:10]]
    assert max(first) > 30, first                                  # not the first shards by name
    assert dl.spread(files) == dl.spread(list(reversed(files)))    # deterministic
    d = Path(tempfile.mkdtemp())
    with gzip.open(d / "e.jsonl.gz", "wt") as f:
        pass
    assert dl._gz_rows(d / "e.jsonl.gz") == 0
    print("downloader: spread shard sampling, empty files detected  OK")


def t_eval_shard():
    from slm_train.evaluate import shard_rows   # needs torch at import of evaluate -> stub if missing
    rng = np.random.default_rng(0)
    L = np.concatenate([rng.integers(500, 8000, 300), rng.integers(9000, 65000, 30)])
    caps = [65536, 8192]
    parts = [shard_rows(L, L.astype(float), r, 2, caps) for r in range(2)]
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(L)))      # each example exactly once
    assert (L[parts[1]] <= 8192).all(), "a long example went to the small GPU"
    print(f"eval sharding: small GPU max length {L[parts[1]].max()} <= 8192, split {len(parts[0])}/{len(parts[1])}  OK")


if __name__ == "__main__":
    t_step_keys()
    t_pack_pretrain()
    t_pack_sft()
    t_groups()
    t_downloader()
    try:
        import torch  # noqa: F401
        if not (TRAIN_REPO / "training" / "slm_train").is_dir():
            raise ImportError("training repository not found")
        sys.path.append(str(TRAIN_REPO / "training"))
        t_eval_shard()
    except ImportError as e:
        print(f"eval sharding: skipped ({e}; covered by the training repository's training/scripts/selftest.py)")
    print("\nTURN 56 PIPELINE TESTS PASSED")
