"""Split registry + relevance build + leak audit on synthetic data (no real data, no parquet needed).
Run:  python data_pipeline/tests/test_registry_split.py"""
import collections
import csv
import json
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "data_pipeline")]

from slm_data import audit, build_relevance, registry  # noqa: E402
from slm_data.config import Cfg  # noqa: E402
from slm_data.io_utils import ShardWriter, read_shards  # noqa: E402


def make_world(tmp: Path):
    rng = random.Random(0)
    data_root, out_root = tmp / "data", tmp / "out"
    (out_root / "relevance").mkdir(parents=True)
    data_root.mkdir()
    vocab = [f"skill{i}" for i in range(400)]
    queries = [" ".join(rng.sample(vocab, 6)) + f" with {rng.randint(1, 9)}+ years" for _ in range(40)]
    queries[1] = queries[0] + " remote"              # a near-duplicate pair: must land in the same family
    # 40 queries, very different sizes (5 .. 3000 candidates), candidates shared across queries
    pairs = []
    for qi, q in enumerate(queries):
        n = [5, 30, 300, 3000][qi % 4]
        for c in rng.sample(range(1, 60000), n):
            pairs.append((q, c, "relevant" if rng.random() < 0.3 else "not_relevant"))
    with ShardWriter(out_root / "relevance", "labels") as w:
        for q, c, v in pairs:
            w.write({"query": q, "resume_id": c, "verdict": v, "score": 7 if v == "relevant" else 2,
                     "agreement": 0.9, "weight": 2.0, "risky": c % 17 == 0, "score_mean": 5.0, "score_min": 3, "score_max": 7})
    # candidate payloads for relevance-build (inline resume text)
    cdb = sqlite3.connect(str(out_root / "relevance" / "candidates.sqlite"))
    cdb.execute("CREATE TABLE c (q TEXT, rid INTEGER, payload TEXT, PRIMARY KEY (q, rid))")
    cdb.executemany("INSERT OR IGNORE INTO c VALUES (?,?,?)",
                    [(q, c, json.dumps({"resume": " ".join(random.Random(c).choices(vocab, k=120))}))
                     for q, c, _ in pairs])
    cdb.commit()
    cdb.close()
    # evaluation asset: 50 candidates, some of them inside the relevance pool
    eval_ids = sorted({c for _, c, _ in pairs})[:30] + list(range(70000, 70020))
    with open(data_root / "gold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resume_id", "x"])
        for t in eval_ids:
            w.writerow([t, "y"])
    cfg = Cfg({
        "data_root": str(data_root), "out_root": str(out_root), "workers": 1, "seed": 1,
        "sources": {"resume_parquet_globs": ["fake/*.parquet"]},
        "splits": {"salt": "t", "test_candidates": 2000, "min_test_per_query": 3, "val_candidates": 800,
                   "min_val_per_query": 2, "default_val_frac": 0.005, "link_by_email": True,
                   "exclude_email_patterns": ["@turing\\.com$"],
                   "eval_assets": [{"name": "gold", "path": "gold.csv", "id_col": "resume_id"}]},
        "relevance": {"max_candidate_chars": 4000, "max_pos_neg_ratio": 3.0, "max_per_query": 4000,
                      "allow_store_join": False, "reason_max_chars": 200},
        "cleaning": {"redact_pii": True},
    })
    return cfg, pairs, set(eval_ids)


def fake_email_rows(eval_ids, test_like_id):
    """talent_id/e-mail rows: 99001 shares an address with an eval person, 99002 with `test_like_id`,
    99003 is a QA account."""
    e0 = next(iter(eval_ids))
    rows = [(e0, "Same.Person@gmail.com"), (99001, "sameperson+jobs@gmail.com"),
            (test_like_id, "twin@yahoo.com"), (99002, "TWIN@yahoo.com"), (99003, "qa1@turing.com")]
    return {"talent_id": [r[0] for r in rows], "email": [r[1] for r in rows]}


def main():
    # score stability: flips / boundary / risky
    js = [("gpt", "relevant", 6.0, "", None), ("gpt", "not_relevant", 4.0, "", None), ("bench", "relevant", 7.0, "", None)]
    s = build_relevance.score_stability(js)
    assert s["verdict_flip"] and s["boundary"] and s["risky"] and s["score_min"] == 4.0 and s["score_max"] == 7.0, s
    s2 = build_relevance.score_stability([("gpt", "relevant", 8.0, "", None), ("bench", "relevant", 9.0, "", None)])
    assert not s2["risky"] and s2["score_mean"] == 8.5, s2

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg, pairs, eval_ids = make_world(tmp)

        # registry with a mocked resume source for e-mail linking (pass 1 uses assignments made before it)
        by_q, _ = registry._relevance_pool(cfg)
        pool = set().union(*by_q.values())
        registry.parquet_columns = lambda f: ["talent_id", "email"]
        registry.parquet_num_row_groups = lambda f: 1
        cfg_src_glob = lambda pattern: [Path("fake/a.parquet")]  # noqa: E731
        cfg.src_glob = cfg_src_glob
        holder = {}

        def rrg(f, rg, cols):
            return holder["rows"]
        registry.read_row_group = rrg
        # pick a candidate that will be test: build once without e-mail rows to learn one
        holder["rows"] = {"talent_id": [], "email": []}
        rep0 = registry.build(cfg)
        reg0 = registry.Registry(cfg.out_root_path / "splits" / "registry.sqlite")
        test_like = next(t for t in sorted(pool) if reg0.split_of(t) == "test")
        holder["rows"] = fake_email_rows(eval_ids, test_like)
        rep = registry.build(cfg)
        reg = registry.Registry(cfg.out_root_path / "splits" / "registry.sqlite")

        n = collections.Counter(reg.assign.values())
        print("explicit:", dict(n), "| e-mail linking:", rep["email_linking"])
        assert 1800 <= n["test"] <= 2600, n                  # ~2000 test candidates (+ per-query top-ups)
        assert not rep["queries_without_test_candidates"], rep["queries_without_test_candidates"]
        for q, v in rep["per_query"].items():                # every query keeps test candidates
            assert v["test"] >= 1, (q, v)
        assert all(reg.split_of(t) == "eval" for t in eval_ids)
        assert reg.split_of(99001) == "eval"                  # same person as an eval candidate (gmail dots / +tag)
        assert reg.split_of(99002) == "test"                  # same e-mail as a test candidate
        assert reg.split_of(99003) == "excluded"              # QA account
        # both verdicts present among the test candidates of the big queries
        lab = collections.defaultdict(set)
        for q, c, v in pairs:
            if reg.split_of(c) == "test":
                lab[q].add(v)
        assert sum(len(v) == 2 for v in lab.values()) >= 30, {q: v for q, v in lab.items()}
        # deterministic
        rep_again = registry.build(cfg)
        assert registry.Registry(cfg.out_root_path / "splits" / "registry.sqlite").assign == reg.assign

        # query axis: families split train / dev / test; the near-duplicate pair shares a family
        assert reg.query_family(pairs_q0 := sorted({q for q, _, _ in pairs})[0]) is not None
        q0 = next(q for q, _, _ in pairs if q.endswith(" remote"))
        q0_base = q0[: -len(" remote")]
        assert reg.query_family(q0) == reg.query_family(q0_base), "near-duplicate queries split apart"
        qsplit = collections.Counter(reg.query_split(q) for q in {q for q, _, _ in pairs})
        print("queries per split:", dict(qsplit), "| report:", rep["query_families"])
        assert qsplit["dev"] >= 1 and qsplit["test"] >= 1 and qsplit["train"] >= 20

        # plant a near-copy of a held-out candidate's resume under a TRAINING candidate -> must be dropped
        cdb = sqlite3.connect(str(cfg.out_root_path / "relevance" / "candidates.sqlite"))
        q_pl, held_c, train_c = None, None, None
        for q in sorted({q for q, _, _ in pairs}):
            cs = [c for qq, c, _ in pairs if qq == q]
            h = [c for c in cs if reg.pair_split(q, c) == "test_cand"]
            t = [c for c in cs if reg.pair_split(q, c) == "train"]
            if h and t:
                q_pl, held_c, train_c = q, h[0], t[0]
                break
        txt = json.loads(cdb.execute("SELECT payload FROM c WHERE q=? AND rid=?", (q_pl, held_c)).fetchone()[0])["resume"]
        for c in [c for qq, c, _ in pairs if qq == q_pl and reg.pair_split(q_pl, c) == "train"]:
            cdb.execute("UPDATE c SET payload=? WHERE q=? AND rid=?", (json.dumps({"resume": txt + f" extra {c}"}), q_pl, c))
        cdb.commit()
        cdb.close()

        # relevance build: split = pair_split(query, candidate); eval candidates never written
        rep_b = build_relevance.build(cfg)
        assert rep_b["stats"].get("train_dropped_near_dup_of_heldout", 0) >= 1, rep_b["stats"]
        print("near-duplicate decontamination dropped:", rep_b["stats"]["train_dropped_near_dup_of_heldout"])
        cnt = collections.Counter()
        fam = collections.defaultdict(set)
        for r in read_shards(cfg.out_root_path / "relevance", "pairs"):
            cnt[r["split"]] += 1
            assert r["split"] == reg.pair_split(r["query"], r["resume_id"]), r
            assert "risky" in r and "score_min" in r
            assert r["resume_id"] not in eval_ids
            fam["train_side" if r["split"] in ("train", "val", "dev_cand", "test_cand") else "unseen"].add(
                reg.query_family(r["query"]))
            if r["split"] in ("dev", "test"):
                assert reg.query_split(r["query"]) == r["split"]
            if r["split"] == "dev":
                assert reg.split_of(r["resume_id"]) != "test"     # locked candidates stay out of dev
        assert not (fam["train_side"] & fam["unseen"]), "query family overlap"
        print("pairs per split:", dict(cnt))
        assert cnt["train"] and cnt["val"] and cnt["dev"] and cnt["test"] and cnt["test_cand"]

        # leak audit passes on the clean build ...
        rep_a = audit.run(cfg)
        assert rep_a["ok"], rep_a["problems"]
        # ... fails when a test candidate is planted in the training pairs ...
        rows = list(read_shards(cfg.out_root_path / "relevance", "pairs"))
        victim = next(r for r in rows if r["split"] == "test_cand")
        with ShardWriter(cfg.out_root_path / "relevance", "pairs") as w:
            for r in rows:
                w.write(r)
            w.write({**victim, "split": "train"})
        try:
            audit.run(cfg)
            raise AssertionError("audit should have failed")
        except RuntimeError as e:
            assert "relevance_pair:test_cand->train" in str(e), e
            print("planted candidate leak caught:", str(e)[:120])
        # ... and when an unseen-query pair is planted in train (query-family overlap)
        qvictim = next(r for r in rows if r["split"] == "dev")
        with ShardWriter(cfg.out_root_path / "relevance", "pairs") as w:
            for r in rows:
                w.write(r)
            w.write({**qvictim, "split": "train"})
        try:
            audit.run(cfg)
            raise AssertionError("audit should have failed")
        except RuntimeError as e:
            assert "query_families_in_train_and_dev_or_test" in str(e), e
            print("planted query leak caught:", str(e)[:120])

        # packed-file audit through the key sidecars
        import numpy as np
        from slm_data.near_dup import text_key
        sft = cfg.out_root_path / "packed" / "sft"
        sft.mkdir(parents=True)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["split"]].append((text_key(r["query"]), text_key(r["candidate"])))
        for sp, k in by.items():
            np.save(sft / f"relevance_{sp}.keys.npy", np.asarray(k, dtype=np.int64))
        assert audit.run_packed(cfg)["ok"]
        np.save(sft / "relevance_train.keys.npy", np.asarray(by["train"] + by["dev"][:3], dtype=np.int64))
        try:
            audit.run_packed(cfg)
            raise AssertionError("packed audit should have failed")
        except RuntimeError as e:
            assert "query_in_train_side_and_unseen_split" in str(e), e
            print("packed query leak caught:", str(e)[:120])
    print("\nREGISTRY / SPLIT / AUDIT TESTS PASSED")


if __name__ == "__main__":
    main()
