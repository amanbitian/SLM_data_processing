"""Turn 53 (optimisation review) data-side checks on invented data — no real data, no network.

  D16  content-addressed step keys change with code / inputs / upstream keys
  D14  char-budgeted encode batches
  D15  compact _SftWriter writes the same files as before (idx / labels / keys)
  D11  tokenizer sampler reads beyond the first shards and keeps prefixes apart
  D12  prefix-filtered query families == brute force
Run:  PYTHONPATH=data_pipeline:model python data_pipeline/tests/test_turn53_pipeline.py
"""
from __future__ import annotations

import difflib
import gzip
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm_data import step_keys as SK  # noqa: E402
from slm_data.pack import _SftWriter, _batched  # noqa: E402
from slm_data.registry import _query_families, query_key  # noqa: E402


def t_step_keys():
    d = Path(tempfile.mkdtemp())
    (d / "a.parquet").write_bytes(b"x")
    cfg = {"data_root": str(d), "sources": {"g": "*.parquet", "workers": 4}, "seed": 1}
    order = ["inventory", "registry", "pack"]
    k1 = SK.step_keys(cfg, order)
    assert SK.step_keys(cfg, order) == k1, "keys not deterministic"
    (d / "a.parquet").write_bytes(b"xy")                       # source file changed -> every key changes
    k2 = SK.step_keys(cfg, order)
    assert all(k1[s] != k2[s] for s in order)
    cfg2 = dict(cfg, seed=2)                                   # config changed
    assert SK.step_keys(cfg2, order)["pack"] != k2["pack"]
    real = SK.code_hash
    SK.code_hash = lambda *a, **k: "different"                 # code changed
    try:
        assert SK.step_keys(cfg, order)["inventory"] != k2["inventory"]
    finally:
        SK.code_hash = real
    print("D16 step keys: config / code / input changes invalidate the chain  OK")


def t_batched():
    items = [("k", {"text": "x" * n}) for n in (10, 10, 500, 10, 10, 10)]
    out = list(_batched(iter(items), 64, 505, lambda kr: len(kr[1]["text"])))
    assert [len(b) for b in out] == [3, 3], out
    assert [len(b) for b in _batched(iter(range(130)), 64)] == [64, 64, 2]
    print("D14 char-budgeted batches  OK")


def t_writer():
    d = Path(tempfile.mkdtemp())
    w = _SftWriter(d, "relevance_train", np.uint16)
    rng = np.random.default_rng(0)
    ref_idx, ref_lab, ref_keys, off = [], [], [], 0
    for i in range(2500):                                      # crosses the 1024-row growth boundary twice
        n = int(rng.integers(3, 40))
        lab = (float(i % 2), float(rng.random()), 0.5, float(i % 7), -1.0, 0.0)
        w.add(rng.integers(0, 1000, n), 2, lab, (i, -i))
        ref_idx.append((off, n, 2))
        ref_lab.append(lab)
        ref_keys.append((i, -i))
        off += n
    st = w.close()
    assert st["examples"] == 2500 and st["tokens"] == off
    assert (np.load(d / "relevance_train.idx.npy") == np.asarray(ref_idx)).all()
    assert np.allclose(np.load(d / "relevance_train.labels.npy"), np.asarray(ref_lab, dtype=np.float32))
    assert (np.load(d / "relevance_train.keys.npy") == np.asarray(ref_keys)).all()
    w2 = _SftWriter(d, "classify_val", np.uint16)              # no labels -> no labels file
    w2.add([1, 2, 3], 1, None)
    w2.close()
    assert not (d / "classify_val.labels.npy").exists()
    print("D15 compact SFT writer == list-based output  OK")


def t_tokenizer_sampler():
    from slm_data import tokenizer_train as TT
    root = Path(tempfile.mkdtemp())
    ext = root / "external"
    ext.mkdir()

    def shard(name, texts):
        with gzip.open(ext / name, "wt", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t, "split": "train"}) + "\n")
    for s in range(10):                                        # 10 JD shards; the old sampler only saw shard 0
        shard(f"pretrain-{s:05d}.jsonl.gz", [f"jd{s}_{i}" for i in range(200)])
    shard("pretrain-gen-wikipedia-a-00000.jsonl.gz", [f"wiki{i}" for i in range(50)])

    class C(dict):
        out_root_path = root
    cfg = C(seed=0, tokenizer={"sample_docs": 1000, "sample_max_files": 64})
    for sub in ("resumes", "knowledge", "relevance"):
        (root / sub).mkdir()
    out = root / "sample.txt"
    TT._sample(cfg, out)
    lines = out.read_text().split("\n")
    jd_shards = {ln.split("_")[0] for ln in lines if ln.startswith("jd")}
    assert len(jd_shards) >= 8, jd_shards                      # spread over (almost) all shards
    n_wiki = sum(ln.startswith("wiki") for ln in lines)
    assert n_wiki <= 40, n_wiki                                # wikipedia only through its own (4%) entry
    print(f"D11 tokenizer sample spread over {len(jd_shards)}/10 shards, prefixes kept apart  OK")


def t_query_families():
    rng = random.Random(3)
    words = [f"w{i}" for i in range(30)] + ["python", "developer", "senior", "java"]
    qs = list({" ".join(rng.sample(words, rng.randint(2, 6))) for _ in range(300)})
    fam = _query_families(qs, {})
    keys = {q: query_key(q) for q in qs}
    toks = {q: set(k.split()) for q, k in keys.items()}
    par = {q: q for q in qs}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    s = sorted(qs)
    for i, a in enumerate(s):
        for b in s[i + 1:]:
            j = len(toks[a] & toks[b]) / max(1, len(toks[a] | toks[b]))
            if j >= 0.6 or (j >= 0.3 and difflib.SequenceMatcher(None, keys[a], keys[b]).ratio() >= 0.8):
                par[find(a)] = find(b)
    g1, g2 = {}, {}
    for q in s:
        g1.setdefault(find(q), set()).add(q)
        g2.setdefault(fam[q], set()).add(q)
    assert sorted(map(sorted, g1.values())) == sorted(map(sorted, g2.values()))
    print(f"D12 prefix-filtered query families == brute force ({len(g1)} families)  OK")


if __name__ == "__main__":
    t_step_keys()
    t_batched()
    t_writer()
    t_tokenizer_sampler()
    t_query_families()
    print("\nTURN 53 PIPELINE TESTS PASSED")
