from __future__ import annotations

import gzip
import io
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:  # fast path
    import pyarrow.parquet as pq  # type: ignore
    HAVE_PYARROW = True
except Exception:  # pragma: no cover
    HAVE_PYARROW = False
    from . import _mini_parquet as _mp


# ----------------------------------------------------------------------------- parquet
# Turn 53 (review D1): open each Parquet file once per process (footer + schema parsed once) instead of twice per
# row group. Keyed by (path, size, mtime) so a rewritten file is re-opened. Small LRU; parquet_cache_clear()
# closes everything (called after every pipeline step so no handle keeps a USB drive busy).
from functools import lru_cache


def _stat_key(path) -> tuple:
    st = Path(path).stat()
    return str(path), st.st_size, st.st_mtime_ns


@lru_cache(maxsize=16)
def _pf(key: tuple):
    return pq.ParquetFile(key[0])


@lru_cache(maxsize=256)
def _columns(key: tuple) -> tuple:
    if HAVE_PYARROW:
        return tuple(_pf(key).schema_arrow.names)
    return tuple(_mp.columns(Path(key[0])))


def parquet_cache_clear() -> None:
    """Drop the cached handles (pyarrow closes a ParquetFile's file when it is garbage-collected)."""
    _pf.cache_clear()
    _columns.cache_clear()
    import gc
    gc.collect()


def parquet_columns(path: Path) -> list[str]:
    return list(_columns(_stat_key(path)))


def parquet_num_row_groups(path: Path) -> int:
    if HAVE_PYARROW:
        return _pf(_stat_key(path)).num_row_groups
    return _mp.num_row_groups(path)


def parquet_num_rows(path: Path) -> int:
    if HAVE_PYARROW:
        return _pf(_stat_key(path)).metadata.num_rows
    return _mp.num_rows(path)


def read_row_group(path: Path, rg: int, columns: list[str]) -> dict[str, list]:
    """Return {col: list}. Columns absent from the file are returned as all-None."""
    key = _stat_key(path)
    have = set(_columns(key))
    use = [c for c in columns if c in have]
    if HAVE_PYARROW:
        data = _pf(key).read_row_group(rg, columns=use).to_pydict()
    else:
        data = {c: _mp.read_column(path, c, rg) for c in use}
    n = len(next(iter(data.values()))) if data else 0
    for c in columns:
        if c not in data:
            data[c] = [None] * n
    return data


def iter_rows(data: dict[str, list]) -> Iterator[dict]:
    keys = list(data)
    for vals in zip(*(data[k] for k in keys)):
        yield dict(zip(keys, vals))


# ----------------------------------------------------------------------------- jsonl shards
class ShardWriter:
    """Writes gzip JSONL shards: <dir>/<prefix>-00000.jsonl.gz ..."""

    def __init__(self, out_dir: Path, prefix: str, shard_size: int = 50_000):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob(f"{prefix}-*.jsonl.gz"):
            old.unlink()
        self.prefix, self.shard_size = prefix, shard_size
        self.n_in_shard = self.shard_idx = self.total = 0
        self._fh: Optional[io.TextIOWrapper] = None

    def _open(self):
        p = self.dir / f"{self.prefix}-{self.shard_idx:05d}.jsonl.gz"
        self._fh = io.TextIOWrapper(gzip.open(p, "wb", compresslevel=3), encoding="utf-8")

    def write(self, rec: dict) -> None:
        if self._fh is None:
            self._open()
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n_in_shard += 1
        self.total += 1
        if self.n_in_shard >= self.shard_size:
            self._fh.close()
            self._fh = None
            self.shard_idx += 1
            self.n_in_shard = 0

    def close(self) -> int:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return self.total

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_shards(dir_: Path, prefix: str) -> Iterator[dict]:
    for p in sorted(Path(dir_).glob(f"{prefix}-*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ----------------------------------------------------------------------------- zip csv
def read_csv_from_zip(zip_path: Path, member_suffix: str = ".csv", **kw):
    import pandas as pd
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(member_suffix) and "__MACOSX" not in n)
        with z.open(name) as f:
            return pd.read_csv(f, **kw)


# ----------------------------------------------------------------------------- resume store
class ResumeStore:
    """SQLite key-value store: candidate id -> cleaned resume text + profile block.
    Used to join resume text into relevance pairs that only carry a resume_id."""

    def __init__(self, path: Path, readonly: bool = False):
        self.path = Path(path)
        uri = f"file:{self.path.as_posix()}?mode=ro" if readonly else None
        self.db = sqlite3.connect(uri, uri=True) if readonly else sqlite3.connect(str(self.path))
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=OFF")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS resumes (id INTEGER PRIMARY KEY, text TEXT, profile TEXT,"
                " has_parse INTEGER, meta TEXT)"
            )

    def put_many(self, rows: Iterable[tuple]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO resumes VALUES (?,?,?,?,?)", rows)
        self.db.commit()

    def get(self, rid: int) -> Optional[dict]:
        r = self.db.execute("SELECT text, profile, has_parse, meta FROM resumes WHERE id=?", (int(rid),)).fetchone()
        if not r:
            return None
        return {"text": r[0], "profile": r[1], "has_parse": bool(r[2]), "meta": json.loads(r[3] or "{}")}

    def get_many(self, rids: Iterable[int], chunk: int = 500) -> dict[int, dict]:
        """Batched get (Turn 53, review D4): one query per `chunk` ids instead of one per id."""
        ids = sorted({int(r) for r in rids})
        out: dict[int, dict] = {}
        for s in range(0, len(ids), chunk):
            part = ids[s:s + chunk]
            q = f"SELECT id, text, profile, has_parse, meta FROM resumes WHERE id IN ({','.join('?' * len(part))})"
            for r in self.db.execute(q, part):
                out[int(r[0])] = {"text": r[1], "profile": r[2], "has_parse": bool(r[3]),
                                  "meta": json.loads(r[4] or "{}")}
        return out

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]

    def close(self):
        self.db.close()
