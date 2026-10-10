"""
Content-addressed, per-step keys for `python -m slm_data all` / `data.py` step skipping (Turn 53, reworked
Turn 56 after review).

key(step) = sha1( step's code, step's config sections, step's raw inputs, salt fingerprint (PII steps),
                  keys of the steps it reads from )

  code      the step's module plus everything it imports inside slm_data / slm_model (found by parsing the
            imports), so editing pack.py re-runs pack and audit-packed only, not inventory / registry / resumes
  config    only the config sections the step reads (+ global keys data_root / out_root / seed)
  inputs    for steps that read raw sources: every file referenced by `sources` and by `splits` (evaluation
            assets, query-family / split override CSVs): path, size, mtime and a content sample (first, middle
            and last 64 KiB), so a same-size edit is caught too. AppleDouble `._*` files are ignored
  salt      a one-way fingerprint of SLM_PSEUDO_SALT for the steps that pseudonymise (resumes and later) ->
            changing the salt re-runs them instead of keeping the old fake identities
  upstream  the keys of the steps whose outputs this step reads (a DAG, not the linear order)
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_MODEL = _PKG.parents[1] / "model" / "slm_model"

# step -> (entry modules, config sections, reads raw sources?, uses the PII salt?, upstream steps)
STEP_DEPS: dict[str, tuple] = {
    "inventory":       (["inventory"], ["sources"], True, False, []),
    "relevance-scan":  (["build_relevance"], ["sources", "relevance", "cleaning"], True, False, []),
    "registry":        (["registry"], ["sources", "splits"], True, False, ["relevance-scan"]),
    "resumes":         (["build_resumes"], ["sources", "cleaning", "extraction", "splits"], True, True,
                        ["registry"]),
    "relevance-build": (["build_relevance"], ["relevance", "cleaning", "splits"], False, True,
                        ["relevance-scan", "registry", "resumes"]),
    "knowledge":       (["build_knowledge"], ["sources", "external"], True, False, []),
    "external":        (["build_external", "external_more"], ["sources", "external", "cleaning"], True, False,
                        ["knowledge"]),
    "audit":           (["audit"], ["audit", "splits"], False, False,
                        ["registry", "resumes", "relevance-build", "knowledge", "external"]),
    "tokenizer":       (["tokenizer_train"], ["tokenizer"], False, False,
                        ["resumes", "relevance-build", "knowledge", "external"]),
    "pack":            (["pack"], ["pack", "relevance"], False, False, ["tokenizer", "audit", "registry"]),
    "audit-packed":    (["audit"], ["audit"], False, False, ["pack"]),
}
_SAMPLE = 64 * 1024


def _is_apple_double(p) -> bool:
    return Path(p).name.startswith("._")


def _module_file(name: str, pkg: str):
    base = _PKG if pkg == "slm_data" else _MODEL
    f = base / f"{name}.py"
    return f if f.exists() else None


@lru_cache(maxsize=None)
def _deps(path: str) -> frozenset:
    """Transitive slm_data / slm_model source files imported by `path` (relative and absolute imports)."""
    seen, todo = set(), [Path(path)]
    while todo:
        f = todo.pop()
        if f in seen or not f.exists() or _is_apple_double(f):
            continue
        seen.add(f)
        pkg = "slm_model" if f.parent == _MODEL else "slm_data"
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level and not mod:                       # from . import a, b
                    cands = [(a.name, pkg) for a in node.names]
                elif node.level:                                 # from .x import y
                    cands = [(mod.split(".")[0], pkg)]
                elif mod.startswith(("slm_data", "slm_model")):  # from slm_model.x import y / from slm_model import x
                    parts = mod.split(".")
                    cands = [(parts[1], parts[0])] if len(parts) > 1 else [(a.name, parts[0]) for a in node.names]
                else:
                    cands = []
                for name, p in cands:
                    mf = _module_file(name, p)
                    if mf:
                        todo.append(mf)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    if parts[0] in ("slm_data", "slm_model") and len(parts) > 1:
                        mf = _module_file(parts[1], parts[0])
                        if mf:
                            todo.append(mf)
    return frozenset(seen)


def code_hash(modules=None) -> str:
    """Hash of the given entry modules and everything they import (all of slm_data when modules is None)."""
    if modules is None:
        files = {f for f in _PKG.glob("*.py") if not _is_apple_double(f)}
        files |= {f for f in (_MODEL / "prompting.py", _MODEL / "tokenizer.py") if f.exists()}
    else:
        files = set()
        for m in modules:
            mf = _module_file(m, "slm_data")
            if mf:
                files |= _deps(str(mf))
        files |= {f for f in (_PKG / "config.py", _PKG / "io_utils.py") if f.exists()}
    h = hashlib.sha1()
    for f in sorted(files):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def _file_fp(p: str) -> tuple:
    """(path, size, mtime, sample digest): the first, middle and last 64 KiB are hashed."""
    st = os.stat(p)
    h = hashlib.blake2b(digest_size=8)
    try:
        with open(p, "rb") as f:
            h.update(f.read(_SAMPLE))
            if st.st_size > 3 * _SAMPLE:
                f.seek(st.st_size // 2)
                h.update(f.read(_SAMPLE))
                f.seek(st.st_size - _SAMPLE)
                h.update(f.read(_SAMPLE))
    except OSError:
        return (p, st.st_size, st.st_mtime_ns, "unreadable")
    return (p, st.st_size, st.st_mtime_ns, h.hexdigest())


def _input_stats_raw(spec: str, root: str, max_files: int) -> str:
    rows, n = [], 0
    for s in json.loads(spec):
        if "/" not in s and "*" not in s and "." not in s:
            continue                                   # a plain option value, not a path
        pat = s if os.path.isabs(s) else str(Path(root) / s)
        hits = sorted(glob.glob(pat, recursive=True)) if any(c in s for c in "*?[") else [pat]
        for p in hits:
            if n >= max_files:
                break
            if _is_apple_double(p):
                continue
            if not os.path.exists(p):
                rows.append((p, "missing"))
                continue
            if os.path.isdir(p):
                try:
                    for c in sorted(os.listdir(p))[:2000]:
                        cp = os.path.join(p, c)
                        if not _is_apple_double(cp) and os.path.isfile(cp):
                            rows.append(_file_fp(cp))
                            n += 1
                except OSError:
                    rows.append((p, "unreadable"))
            else:
                rows.append(_file_fp(p))
                n += 1
    return hashlib.sha1(json.dumps(rows, default=str).encode()).hexdigest()[:12]


def input_stats(cfg, max_files: int = 20000) -> str:
    """Fingerprint of every file the `sources` and `splits` sections point at (raw inputs + eval assets +
    override CSVs)."""
    root = os.environ.get("SLM_DATA_ROOT", cfg.get("data_root", "."))
    spec = sorted(set(_strings(cfg.get("sources") or {})) | set(_strings(cfg.get("splits") or {})))
    return _input_stats_raw(json.dumps(spec), str(root), max_files)


def salt_fingerprint() -> str:
    """One-way: says whether the salt changed, reveals nothing about it."""
    s = os.environ.get("SLM_PSEUDO_SALT") or ""
    return hashlib.sha256(b"slm-salt-fp:" + s.encode()).hexdigest()[:12] if s else "no-salt"


def step_keys(cfg, order: list[str]) -> dict[str, str]:
    glob_cfg = {k: cfg.get(k) for k in ("data_root", "out_root", "seed", "workers")}
    out: dict[str, str] = {}
    memo: dict[str, str] = {}

    def inputs() -> str:          # computed at most once per call (stats + samples of every source file)
        if "v" not in memo:
            memo["v"] = input_stats(cfg)
        return memo["v"]

    def key(s: str) -> str:
        if s in out:
            return out[s]
        mods, sections, raw, salted, ups = STEP_DEPS.get(s, ([], list(cfg.keys()), True, True, []))
        parts = {
            "step": s,
            "code": code_hash(mods) if mods else code_hash(),
            "config": json.dumps({k: cfg.get(k) for k in sections} | {"_": glob_cfg}, sort_keys=True, default=str),
            "inputs": inputs() if raw else "",
            "salt": salt_fingerprint() if salted else "",
            "up": [key(u) for u in ups],
        }
        out[s] = hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
        return out[s]

    for s in order:
        key(s)
    return out
