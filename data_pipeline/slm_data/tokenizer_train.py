"""Train the SentencePiece BPE tokenizer on a sample of every text type the model sees."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .config import Cfg
from .io_utils import read_shards, write_json

log = logging.getLogger(__name__)


def _sample(cfg: Cfg, path: Path) -> int:
    rng = random.Random(cfg["seed"])
    budget = cfg["tokenizer"]["sample_docs"]
    root = cfg.out_root_path
    # (dir, prefix, field, share of budget)
    plan = [
        (root / "resumes", "pretrain", "text", 0.45),
        (root / "external", "pretrain", "text", 0.06),                    # JDs (first shards: Djinni, Kaggle, ...)
        (root / "external", "pretrain-wdcpart", "text", 0.02),            # WDC job postings (all parts)
        (root / "external", "pretrain-gen-wikipedia", "text", 0.04),      # general text (Turn 51)
        (root / "external", "pretrain-gen-fineweb_edu", "text", 0.04),
        (root / "external", "pretrain-gen-cosmopedia_v2", "text", 0.01),
        (root / "knowledge", "pretrain", "text", 0.03),
        (root / "resumes", "extract", "target", 0.20),
        (root / "relevance", "pairs", "query", 0.05),
        (root / "relevance", "pairs", "reason", 0.05),
        (root / "relevance", "pairs", "candidate", 0.05),
    ]
    n = 0
    max_files = int(cfg["tokenizer"].get("sample_max_files", 64))
    with open(path, "w", encoding="utf-8") as f:
        for d, prefix, field, share in plan:
            want = int(budget * share)
            got = 0
            # Turn 53 (review D11): sample across the WHOLE source, not its first shards. The old sampler kept
            # each record with p=0.35 and stopped at the quota, so it only ever saw the first ~quota/0.35 records
            # (first export files, first WDC part, first Djinni shards). Now: shard files of this prefix (minus
            # the files of longer prefixes in the same plan, e.g. external/pretrain-gen-*) in random order, up to
            # sample_max_files of them, and an unbiased reservoir sample of quota records from each.
            longer = [p_ for d_, p_, _, _ in plan if d_ == d and p_ != prefix and p_.startswith(prefix)]
            files = [x for x in sorted(Path(d).glob(f"{prefix}*.jsonl.gz"))
                     if not any(x.name.startswith(lp) for lp in longer)]
            rng.shuffle(files)
            files = files[:max_files]
            quota = -(-want // max(1, len(files)))          # ceil
            for fp in files:
                if got >= want:
                    break
                res, seen_n = [], 0
                for rec in _read_one(fp):
                    if rec.get("split", "train") != "train":     # tokenizer sees training people only (Turn 45)
                        continue
                    v = rec.get(field)
                    if not v:
                        continue
                    seen_n += 1
                    if len(res) < quota:
                        res.append(v)
                    else:
                        j = rng.randrange(seen_n)
                        if j < quota:
                            res[j] = v
                for v in res[: want - got]:
                    if not isinstance(v, str):
                        v = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                    for line in v.split("\n"):
                        line = line.strip()
                        if line:
                            f.write(line[:8000] + "\n")
                    got += 1
            n += got
            log.info("tokenizer sample: %-10s %-26s %-9s %d docs from %d files", d.name, prefix, field, got,
                     len(files))
    return n


def _read_one(fp: Path):
    import gzip
    with gzip.open(fp, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def run(cfg: Cfg) -> dict:
    import sentencepiece as spm
    from slm_model.prompting import SPECIAL_TOKENS
    from slm_model.tokenizer import PII_TOKENS, SLMTokenizer

    tcfg = cfg["tokenizer"]
    out = cfg.out_root_path / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / "tokenizer_sample.txt"
    n_docs = _sample(cfg, sample_path)
    prefix = out / "slm_sp"
    spm.SentencePieceTrainer.train(
        input=str(sample_path),
        model_prefix=str(prefix),
        vocab_size=tcfg["vocab_size"],
        model_type=tcfg["model_type"],
        character_coverage=tcfg["character_coverage"],
        byte_fallback=True,             # never <unk>; every byte representable
        split_digits=True,              # better number handling (years, YOE, scores)
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",  # text is already normalised by the pipeline
        add_dummy_prefix=False,         # segment-wise encoding == whole-string encoding
        user_defined_symbols=SPECIAL_TOKENS + PII_TOKENS,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        input_sentence_size=tcfg["sample_docs"] * 20,
        shuffle_input_sentence=True,
        max_sentence_length=tcfg["max_sentence_length"],
        num_threads=cfg["workers"],
        train_extremely_large_corpus=False,
    )
    tok = SLMTokenizer(f"{prefix}.model")
    tok.save_meta(out / "tokenizer_meta.json")
    probe = "Senior Python developer, 5+ yrs FastAPI, Bengaluru. {\"skills\":[{\"name\":\"Python\",\"years\":8}]}"
    ids = tok.encode(probe)
    report = {"docs": n_docs, "vocab": tok.vocab_size, "probe_tokens": len(ids), "probe_chars": len(probe),
              "roundtrip_ok": tok.decode(ids) == probe, "special": tok.special}
    write_json(cfg.out("reports", "tokenizer.json"), report)
    log.info("tokenizer: %s", report)
    return report
