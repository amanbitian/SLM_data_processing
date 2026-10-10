"""
Stage E — tokenize everything into flat binary files the trainer memory-maps.

packed/
  pretrain_<group>_{train,val}.bin   uint16 token stream per group (resume / jobs / knowledge / general), docs as <s> ... </s>
  sft/{task}_{split}.bin             uint16 concatenated examples
  sft/{task}_{split}.idx.npy         int64 [N,3] = (offset, length, prompt_len)  -> loss on [prompt_len:]
  sft/{task}_{split}.labels.npy      float32 [N,7] = (is_relevant, score or -1, P_judges(relevant), query_group,
                                     P_teacher(relevant) or -1, risky 0/1, query_family_group)
                                     (relevance only; blend chosen at training time)
  meta.json
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .config import Cfg
from .io_utils import read_shards, write_json
from .near_dup import text_key
from .splits import pair_key, unit_hash

log = logging.getLogger(__name__)

_TOK = None
_PCFG: Optional[dict] = None


def _init(model_file: str, pcfg: dict):
    global _TOK, _PCFG
    from slm_model.tokenizer import SLMTokenizer
    _TOK, _PCFG = SLMTokenizer(model_file), pcfg


def _dtype(vocab: int):
    return np.uint16 if vocab < 65535 else np.uint32


# ----------------------------------------------------------------------------- pretrain
_ND = None          # held-out MinHash index, loaded once per worker (Turn 56)


def _nd_index():
    global _ND
    if _ND is None:
        p = (_PCFG or {}).get("_heldout_index")
        if p and Path(p).exists():
            import pickle
            with open(p, "rb") as f:
                _ND = pickle.load(f)
        else:
            _ND = False
    return _ND or None


def _enc_pretrain(task) -> tuple:
    """(texts, check_near_dup) -> (token array, documents dropped as near-copies of held-out text)."""
    texts, check = task if isinstance(task, tuple) else (task, False)
    dropped = 0
    if check:
        idx = _nd_index()
        if idx is not None:
            keep = [t for t in texts if not idx.query(t[:20000])]
            dropped = len(texts) - len(keep)
            texts = keep
    ids = _TOK.encode_batch(texts, num_threads=1) if texts else []
    out = []
    for x in ids:
        out.append(_TOK.bos_id)
        out.extend(x)
        out.append(_TOK.eos_id)
    return np.asarray(out, dtype=_dtype(_TOK.vocab_size)), dropped


def _batched(it: Iterator, n: int, max_chars: int = 0, size=None):
    """Batches of up to n items, cut earlier once they hold max_chars characters (Turn 53, review D14: a fixed
    item count gave one worker 64 short classification texts and another 64 long resumes)."""
    buf, chars = [], 0
    for x in it:
        buf.append(x)
        if max_chars and size is not None:
            chars += size(x)
        if len(buf) >= n or (max_chars and chars >= max_chars):
            yield buf
            buf, chars = [], 0
    if buf:
        yield buf


def _rec_chars(kr) -> int:
    _, r = kr
    return sum(len(v) for v in r.values() if isinstance(v, str))


def _heldout_hashes(cfg: Cfg) -> set:
    p = cfg.out_root_path / "resumes" / "heldout_hashes.txt"
    return set(p.read_text().split()) if p.exists() else set()


class _HashedFile:
    """Binary output file that hashes what it writes (content digest for meta.json, Turn 56)."""

    def __init__(self, path: Path):
        import hashlib
        self.path, self.f, self.h = path, open(path, "wb"), hashlib.blake2b(digest_size=16)

    def write(self, b: bytes) -> None:
        self.f.write(b)
        self.h.update(b)

    def close(self) -> str:
        self.f.close()
        return self.h.hexdigest()


def file_digest(path: Path) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class _GlobalDedupe:
    """Exact (normalised-text) duplicate filter across ALL sources of the configured groups, backed by an
    on-disk SQLite table instead of a Python set (15M+ documents would cost GBs of RAM; Turn 56)."""

    def __init__(self, path: Path):
        import sqlite3
        path.unlink(missing_ok=True)
        self.path, self.db = path, sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("CREATE TABLE k (h INTEGER PRIMARY KEY) WITHOUT ROWID")
        self.n = 0

    def seen(self, text: str) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO k VALUES (?)", (text_key(text[:20000]),))
        self.n += 1
        if self.n % 500000 == 0:
            self.db.commit()
        return cur.rowcount == 0

    def close(self) -> None:
        self.db.close()
        self.path.unlink(missing_ok=True)


def _pretrain_docs(cfg: Cfg, stats: Counter) -> Iterator[tuple[str, Optional[str], str]]:
    """(text, split, group) — split comes from the registry for our resumes, None for public text; group from
    slm_data/groups.py. A document whose text hash belongs to a held-out (test / eval) person is never trained on,
    even when it arrives under another talent_id (re-uploads, duplicate accounts)."""
    from .groups import group_of
    root = cfg.out_root_path
    held = _heldout_hashes(cfg)
    # global exact de-duplication (Turn 53 for jobs / knowledge; Turn 56: all groups, disk-backed): builders
    # de-duplicate per source / per WDC part, so the same posting in two WDC parts, a Djinni JD reposted on the
    # web, a Kaggle resume also in resume-job-fit, or a Wikipedia passage copied into FineWeb survived
    dd_groups = set(cfg["pack"].get("global_dedupe_groups", ["resume", "jobs", "knowledge", "general"]) or [])
    dd = _GlobalDedupe(cfg.out_root_path / "packed" / ".dedupe.sqlite") if dd_groups else None
    try:
        for d in ("resumes", "knowledge", "external"):
            for rec in read_shards(root / d, "pretrain"):
                if rec.get("h") and rec["h"] in held:
                    stats["dropped_heldout_text_hash"] += 1
                    continue
                g = group_of(rec)
                if dd is not None and g in dd_groups and dd.seen(rec["text"]):
                    stats[f"dropped_duplicate_{g}"] += 1
                    continue
                yield rec["text"], rec.get("split"), g
    finally:
        if dd is not None:
            dd.close()
    log.info("pretrain filters: %s", dict(stats))


def _build_heldout_index(cfg: Cfg, out: Path) -> Optional[Path]:
    """MinHash index of every held-out text a training document must not near-copy (Turn 56): the resumes of
    test / eval people (resumes/heldout-*) and the held-out relevance candidate blocks."""
    from .near_dup import MinHashIndex
    if not cfg["pack"].get("heldout_near_dup", True):
        return None
    root = cfg.out_root_path
    idx = MinHashIndex(float(cfg["pack"].get("heldout_near_dup_threshold", 0.8)))
    n = 0
    for r in read_shards(root / "resumes", "heldout"):
        idx.add(r["text"][:20000])
        n += 1
    for r in read_shards(root / "relevance", "pairs"):
        if r.get("split") in ("val", "dev_cand", "test_cand", "test") and r.get("candidate"):
            idx.add(r["candidate"][:20000])
            n += 1
    if not n:
        log.warning("no held-out texts for near-duplicate decontamination (resumes/heldout-* missing: re-run "
                    "the resumes step)")
        return None
    import pickle
    p = out / ".heldout_minhash.pkl"
    with open(p, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("held-out near-duplicate index: %d texts", n)
    return p


def pack_pretrain(cfg: Cfg, pool, out: Path) -> dict:
    """One token stream per text group (Turn 51): pretrain_<group>_{train,val}.bin. The trainer mixes the groups
    with explicit weights (pack.pretrain_mix / a task's `mix`), so the share of each group is a training choice,
    not an accident of how much text each source has.

    Turn 56:
      - validation documents are the LOWEST stable hashes per group within the budget (bottom-k), so the val
        stream samples every source of the group instead of whichever source happened to be read first;
      - training documents of pack.heldout_near_dup_groups (default: resume) that are MinHash near-copies of
        held-out text are dropped (checked in the encode workers);
      - every output file's content digest is returned (meta.json -> the trainer's data fingerprint)."""
    import heapq
    from .groups import GROUPS
    vbudget = cfg["pack"]["pretrain_val_tokens"]                 # per group
    files = {(g, sp): _HashedFile(out / f"pretrain_{g}_{sp}.bin") for g in GROUPS for sp in ("train", "val")}
    for old in (out / "pretrain_train.bin", out / "pretrain_val.bin"):   # the pre-Turn-51 single stream
        old.unlink(missing_ok=True)
    nd_groups = set(cfg["pack"].get("heldout_near_dup_groups", ["resume"]) or [])
    ntok, docs, st = Counter(), Counter(), Counter()
    t0 = time.time()
    pending: dict[tuple, list] = defaultdict(list)

    def flush(force=False):
        for key, batches in list(pending.items()):
            if not batches or (not force and len(batches) < max(1, cfg["workers"] * 2)):
                continue
            check = key[1] == "train" and key[0] in nd_groups
            for arr, dropped in pool.imap(_enc_pretrain, [(b, check) for b in batches]):
                files[key].write(arr.tobytes())
                ntok[key] += arr.size
                if dropped:
                    st[f"dropped_near_dup_of_heldout_{key[0]}"] += dropped
            pending[key] = []

    buf: dict[tuple, list] = defaultdict(list)
    bchars: Counter = Counter()
    cpt = float(cfg["pack"].get("chars_per_token_estimate", 4.0))
    max_chars = int(cfg["pack"].get("batch_max_chars", 2_000_000))
    val_heap: dict[str, list] = defaultdict(list)   # per group: max-heap on the hash: (-h, est_tokens, text)
    val_tok: Counter = Counter()
    for text, sp, g in _pretrain_docs(cfg, st):
        if sp is None:
            sp = "val" if unit_hash(text[:256]) < 0.004 else "train"
        if sp not in ("train", "val"):
            continue
        if sp == "val":
            # bottom-k by a stable hash within the token budget (estimated at queue time): independent of the
            # order sources are read. Val candidates that miss the cut are dropped, never moved into train.
            hv = unit_hash("val:" + text[:512])
            est = len(text) / cpt
            heapq.heappush(val_heap[g], (-hv, est, text))
            val_tok[g] += est
            while val_tok[g] > vbudget and len(val_heap[g]) > 1:
                _, e, _ = heapq.heappop(val_heap[g])
                val_tok[g] -= e
            continue
        key = (g, sp)
        buf[key].append(text)
        bchars[key] += len(text)
        docs[key] += 1
        if len(buf[key]) >= 256 or bchars[key] >= max_chars:   # char budget too (review D14)
            pending[key].append(buf[key])
            buf[key] = []
            bchars[key] = 0
            flush()
        if sum(docs.values()) % 200000 == 0:
            log.info("pretrain: %d docs, %s (%.0fs)", sum(docs.values()),
                     {f"{k[0]}_{k[1]}": round(v / 1e6, 1) for k, v in ntok.items()}, time.time() - t0)
    for g, heap in val_heap.items():                     # val docs in hash order (deterministic)
        texts = [t for _, _, t in sorted(heap, key=lambda x: -x[0])]
        docs[(g, "val")] += len(texts)
        for i in range(0, len(texts), 256):
            pending[(g, "val")].append(texts[i:i + 256])
    for key, b in buf.items():
        if b:
            pending[key].append(b)
    flush(force=True)
    digests = {}
    for (g, sp), f in files.items():
        dg = f.close()
        pth = out / f"pretrain_{g}_{sp}.bin"
        if pth.exists() and pth.stat().st_size == 0:     # drop empty streams
            pth.unlink()
        else:
            digests[pth.name] = dg
    streams = {g: {"train_tokens": int(ntok[(g, "train")]), "val_tokens": int(ntok[(g, "val")]),
                   "train_docs": int(docs[(g, "train")]), "val_docs": int(docs[(g, "val")])} for g in GROUPS}
    return {"streams": streams, "mix": dict(cfg["pack"].get("pretrain_mix") or {}), "filters": dict(st),
            "digests": digests,
            "train_tokens": int(sum(v["train_tokens"] for v in streams.values())),
            "val_tokens": int(sum(v["val_tokens"] for v in streams.values()))}


# ----------------------------------------------------------------------------- sft
def soft_relevance_p(r: dict, teacher_weight: float = 0.5) -> float:
    """Soft target P(relevant) for the verdict token.
    - judge-ensemble agreement a (vote share of the majority verdict, 0.5..1) -> a or 1-a
    - if a teacher model scored this pair (`p_teacher`, from slm_baselines.teacher_label), blend:
      p = (1-w)*p_judges + w*p_teacher   (verdict-level knowledge distillation; vocabularies differ, so
      token-level KL with e.g. Qwen is impossible — this is the distillable signal)"""
    a = min(1.0, max(0.5, float(r.get("agreement") or 1.0)))
    p = a if r["verdict"] == "relevant" else 1.0 - a
    if r.get("p_teacher") is not None and teacher_weight > 0:
        p = (1.0 - teacher_weight) * p + teacher_weight * float(r["p_teacher"])
    return min(1.0, max(0.0, p))


def load_teacher(relevance_dir: Path) -> dict:
    """{pair_key: {"p_teacher": float, "think": str?}} from relevance/teacher-*.jsonl.gz.
    Keyed per PAIR (query + resume_id + candidate text): (query, resume_id) collides for external pairs
    without a resume_id. Old teacher files without `pair_key` are skipped with a warning (re-run labelling)."""
    out, old = {}, 0
    for p in sorted(Path(relevance_dir).glob("teacher-*.jsonl.gz")):
        import gzip as _gz
        with _gz.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if not t.get("pair_key"):
                    old += 1
                    continue
                out[t["pair_key"]] = {k: t[k] for k in ("p_teacher", "think") if t.get(k) is not None}
    if old:
        log.warning("teacher labels: %d rows without pair_key ignored (written by an older teacher_label; re-run it)", old)
    return out


def _think_text(r: dict) -> str | None:
    """Reasoning for think-mode targets: an explicit teacher chain-of-thought (`think` field, e.g.
    from a later re-judging pass) wins; otherwise the judge's evidence summary + failed constraints."""
    if r.get("think"):
        return str(r["think"])
    if not r.get("reason"):
        return None
    t = r["reason"]
    if r.get("failed"):
        t += f" Failed constraints: {', '.join(r['failed'])}."
    return t


def _enc_sft(batch: list[tuple[str, dict]]) -> list:
    """-> list of (status, payload, split, file_task). One relevance record can yield two examples:
    the fast format (always) and a think-format variant (written to relevance_think_*)."""
    from slm_model import prompting as P
    L = _PCFG["max_seq_len"]
    think_frac = float(_PCFG.get("think_frac", 0.0))
    out = []
    for kind, r in batch:
        split = r.get("split", "train")
        try:
            if kind in ("extract", "gold"):
                keys = (0, text_key(r["text"]))
                ans = _TOK.encode_answer(P.extraction_target(r["target"]))
                if len(ans) > _PCFG["max_answer_tokens"]:
                    out.append(("drop_long_answer", None, split, kind))
                    continue
                prompt = _TOK.encode_extraction(r["text"], L - len(ans))
                label = None
            elif kind == "relevance":
                keys = (text_key(r["query"]), text_key(r["candidate"]))
                # col 2 = judge-agreement soft target, col 4 = teacher P(relevant) (-1 if none). They are kept
                # SEPARATE so the blend is a training-time knob (task `teacher_weight`) -> KD is ablatable
                # without re-packing.
                is_rel = r["verdict"] == "relevant"
                p_judge = soft_relevance_p(r, 0.0)
                p_teacher = float(r["p_teacher"]) if r.get("p_teacher") is not None else -1.0
                # col 3 = query group id (exact in float32 < 2^24) -> query-level bootstrap CIs in eval
                qgroup = float(int(unit_hash(f"q:{r['query']}") * (1 << 24)))
                # col 6 (Turn 56) = query FAMILY group (registry near-duplicate family; exact query when the
                # registry has none, e.g. external pairs) -> family-level bootstrap CIs; col 3 stays the exact
                # query (ranking batches)
                fgroup = float(r["_family_group"]) if r.get("_family_group") is not None else qgroup
                # col 5 = risky (label unstable across judges / runs) -> stable-vs-risky slices in eval
                label = (1.0 if is_rel else 0.0, float(r["score"]) if r.get("score") is not None else -1.0,
                         p_judge, qgroup, p_teacher, 1.0 if r.get("risky") else 0.0, fgroup)
                think = _think_text(r)
                if think and unit_hash(f"think:{r['query'][:200]}|{r.get('resume_id')}") < think_frac:
                    t_ans = _TOK.encode_answer(P.relevance_target(r["verdict"], r.get("score"), r.get("failed"),
                                                                  None, think=think))
                    t_prompt = _TOK.encode_relevance(r["query"], r["candidate"], L - len(t_ans), r.get("date"))
                    out.append(("ok", (t_prompt + t_ans, len(t_prompt), label, keys), split, "relevance_think"))
                ans = _TOK.encode_answer(P.relevance_target(r["verdict"], r.get("score"), r.get("failed"),
                                                            r.get("reason")))
                prompt = _TOK.encode_relevance(r["query"], r["candidate"], L - len(ans), r.get("date"))
            elif kind == "classify":
                keys = (0, text_key(r["text"]))
                ans = _TOK.encode_answer(r["label"])
                prompt = _TOK.encode_classify(r["text"], r["label_space"], L - len(ans))
                label = None
            else:
                raise ValueError(kind)
        except (ValueError, KeyError, TypeError):
            out.append(("drop_error", None, split, kind))
            continue
        out.append(("ok", (prompt + ans, len(prompt), label, keys), split, kind))
    return out


class _SftWriter:
    def __init__(self, out: Path, name: str, dtype):
        self.bin = _HashedFile(out / f"{name}.bin")
        self.name, self.out, self.dtype = name, out, dtype
        # Turn 53 (review D15): growable numpy arrays instead of Python lists of tuples (~100+ bytes per example
        # as tuples vs 24 + 16 bytes here; matters for millions of relevance / classification examples)
        self.idx = np.zeros((1024, 3), dtype=np.int64)
        self.keys = np.zeros((1024, 2), dtype=np.int64)   # (query key, candidate / text key) -> audit-packed
        self.labels = None                                # float32 [n, k], allocated at the first label
        self.n = self.n_lab = 0
        self.off = 0

    @staticmethod
    def _grow(a: np.ndarray, n: int) -> np.ndarray:
        if n < len(a):
            return a
        b = np.zeros((max(n + 1, 2 * len(a)),) + a.shape[1:], dtype=a.dtype)
        b[:len(a)] = a
        return b

    def add(self, ids, plen, label, keys=(0, 0)):
        a = np.asarray(ids, dtype=self.dtype)
        self.bin.write(a.tobytes())
        self.idx = self._grow(self.idx, self.n)
        self.keys = self._grow(self.keys, self.n)
        self.idx[self.n] = (self.off, a.size, plen)
        self.keys[self.n] = keys
        self.n += 1
        self.off += a.size
        if label is not None:
            lab = np.asarray(label, dtype=np.float32).ravel()
            if self.labels is None:
                self.labels = np.zeros((1024, lab.size), dtype=np.float32)
            self.labels = self._grow(self.labels, self.n_lab)
            self.labels[self.n_lab] = lab
            self.n_lab += 1

    def close(self) -> dict:
        self.digest = self.bin.close()
        idx = self.idx[:self.n]
        np.save(self.out / f"{self.name}.idx.npy", idx)
        if self.n_lab:
            np.save(self.out / f"{self.name}.labels.npy", self.labels[:self.n_lab])
        np.save(self.out / f"{self.name}.keys.npy", self.keys[:self.n])
        lens = idx[:, 1] if self.n else np.zeros(1)
        return {"examples": int(self.n), "tokens": int(self.off), "bin_digest": self.digest, "mean_len": float(lens.mean()),
                "p95_len": float(np.percentile(lens, 95))}


def _sft_sources(cfg: Cfg):
    root = cfg.out_root_path
    yield "extract", root / "resumes", "extract"
    yield "gold", root / "resumes", "gold"
    yield "relevance", root / "relevance", "pairs"
    yield "relevance", root / "external", "pairs"
    yield "classify", root / "resumes", "classify"
    yield "classify", root / "resumes", "classify_t7"
    yield "classify", root / "external", "classify"


def pack_sft(cfg: Cfg, pool, out: Path, dtype) -> dict:
    """Turn 56: written into packed/sft.tmp and swapped in only when complete, so files of task / split
    combinations that no longer exist (or a crash mid-pack) can never leave stale files next to fresh ones."""
    import shutil
    from .registry import load_registry, query_key
    final, sft = out / "sft", out / "sft.tmp"
    shutil.rmtree(sft, ignore_errors=True)
    sft.mkdir(parents=True)
    reg = load_registry(cfg)
    fam_cache: dict = {}

    def family_group(q: str) -> Optional[float]:
        if reg is None:
            return None
        if q not in fam_cache:
            f = reg.query_family(q)
            fam_cache[q] = float(int(unit_hash(f"fam:{f}") * (1 << 24))) if f else None
        return fam_cache[q]

    writers: dict[str, _SftWriter] = {}
    st = Counter()
    teacher = load_teacher(cfg.out_root_path / "relevance")
    if teacher:
        log.info("teacher labels: %d relevance pairs get p_teacher / teacher reasoning", len(teacher))
    for kind, d, prefix in _sft_sources(cfg):
        recs = (r if kind != "relevance" else {**r, **teacher.get(pair_key(r), {}),
                                                   "_family_group": family_group(r["query"])}
                for r in read_shards(d, prefix))
        batches = _batched(((kind, r) for r in recs), 64, int(cfg["pack"].get("sft_batch_max_chars", 400_000)),
                           _rec_chars)
        for res in pool.imap(_enc_sft, batches):
            for status, payload, split, file_task in res:
                st[f"{file_task}_{status}"] += 1
                if status != "ok":
                    continue
                name = f"{file_task}_{split}"
                if name not in writers:
                    writers[name] = _SftWriter(sft, name, dtype)
                writers[name].add(*payload)
        log.info("sft %s/%s done: %s", d.name, prefix, {k: v for k, v in st.items() if k.startswith(kind)})
    files = {n: w.close() for n, w in writers.items()}
    digests = {f"sft/{p.name}": file_digest(p) for p in sorted(sft.iterdir())
               if p.is_file() and p.suffix == ".npy"}
    digests.update({f"sft/{n}.bin": v["bin_digest"] for n, v in files.items()})
    old = out / "sft.old"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    sft.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    return {"files": files, "status": dict(st), "digests": digests}


def run(cfg: Cfg, only: Optional[str] = None) -> dict:
    out = cfg.out_root_path / "packed"
    out.mkdir(parents=True, exist_ok=True)
    model_file = cfg.out_root_path / "tokenizer" / "slm_sp.model"
    if not model_file.exists():
        raise FileNotFoundError("train the tokenizer first (python -m slm_data tokenizer)")
    from slm_model.tokenizer import SLMTokenizer
    tok = SLMTokenizer(model_file)
    dtype = _dtype(tok.vocab_size)
    report = {"tokenizer": str(model_file), "vocab_size": tok.vocab_size, "dtype": np.dtype(dtype).name,
              "max_seq_len": cfg["pack"]["max_seq_len"], "special": tok.special, "bos_id": tok.bos_id,
              "eos_id": tok.eos_id, "pad_id": tok.pad_id}
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        report = {**old, **report}
    pcfg = dict(cfg["pack"])
    if only in (None, "pretrain"):
        nd = _build_heldout_index(cfg, out)
        pcfg["_heldout_index"] = str(nd) if nd else None
    with get_context("spawn").Pool(cfg["workers"], initializer=_init,
                                   initargs=(str(model_file), pcfg)) as pool:
        if only in (None, "pretrain"):
            report["pretrain"] = pack_pretrain(cfg, pool, out)
        if only in (None, "sft"):
            report["sft"] = pack_sft(cfg, pool, out, dtype)
    import shutil
    shutil.copy(model_file, out / "slm_sp.model")  # packed/ is self-contained: copy it to the GPU box
    # content digests of every packed file (Turn 56): the trainer's data fingerprint hashes meta.json, so a
    # same-size re-pack with different content can no longer resume an old data cursor
    # the prompt / tokenizer format contract with the training repository (slm_model/format_version.py)
    from slm_model.format_version import format_digest
    report["format_digest"] = format_digest()
    report["digests"] = {**(report.get("digests") or {}),
                         **((report.get("pretrain") or {}).get("digests") or {}),
                         **((report.get("sft") or {}).get("digests") or {}),
                         "slm_sp.model": file_digest(out / "slm_sp.model")}
    (out / ".heldout_minhash.pkl").unlink(missing_ok=True)
    write_json(meta_path, report)
    write_json(cfg.out("reports", "pack.json"), report)
    log.info("pack done -> %s", out)
    return report
