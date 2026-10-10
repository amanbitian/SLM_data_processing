"""Near-duplicate detection with MinHash + LSH (numpy, no extra dependencies) — Turn 47.

    idx = MinHashIndex(threshold=0.8)
    idx.add_many(held_out_texts)                 # the side that must stay unseen
    hits = idx.query_many(training_texts)        # [(train_i, held_j, est_jaccard), ...]

Shingles are word 5-grams of a normalised text (lower case, alphanumerics), hashed to 64 bit; 128 permutations
in 32 bands x 4 rows (a pair with Jaccard 0.8 is caught with probability ~1 - (1 - 0.8^4)^32 ≈ 1.0, a pair at
0.4 only ~56% of the time and then rejected by the exact estimate).
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

_P = np.uint64((1 << 61) - 1)
_WORD = re.compile(r"[a-z0-9]+")


def norm_text(t: str) -> str:
    return " ".join(_WORD.findall((t or "").lower()))


def text_key(t: str) -> int:
    """64-bit key of the normalised text (exact-duplicate check)."""
    return int.from_bytes(hashlib.blake2b(norm_text(t).encode(), digest_size=8).digest(), "big", signed=True)


def shingles(t: str, k: int = 5) -> np.ndarray:
    w = _WORD.findall((t or "").lower())
    if len(w) < k:
        grams = [" ".join(w)] if w else []
    else:
        grams = [" ".join(w[i:i + k]) for i in range(len(w) - k + 1)]
    if not grams:
        return np.zeros(0, dtype=np.uint64)
    h = [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in set(grams)]
    return np.asarray(h, dtype=np.uint64) % _P


class MinHashIndex:
    def __init__(self, threshold: float = 0.8, n_perm: int = 128, bands: int = 32, seed: int = 7, k: int = 5):
        assert n_perm % bands == 0
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, (1 << 61) - 1, n_perm, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 61) - 1, n_perm, dtype=np.uint64)
        self.threshold, self.bands, self.rows, self.k = threshold, bands, n_perm // bands, k
        self.sigs: list[np.ndarray] = []
        self.buckets: list[dict] = [defaultdict(list) for _ in range(bands)]

    def signature(self, text: str) -> Optional[np.ndarray]:
        sh = shingles(text, self.k)
        if sh.size == 0:
            return None
        # a*x + b with 64-bit wrap-around: a fast multiply-add hash family, adequate for MinHash estimates
        v = (sh[:, None] * self.a[None, :] + self.b[None, :])
        return v.min(axis=0)

    def _bands(self, sig: np.ndarray):
        for i in range(self.bands):
            yield i, sig[i * self.rows:(i + 1) * self.rows].tobytes()

    def add(self, text: str) -> int:
        sig = self.signature(text)
        j = len(self.sigs)
        self.sigs.append(sig)
        if sig is not None:
            for i, key in self._bands(sig):
                self.buckets[i][key].append(j)
        return j

    def add_many(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add(t)

    def query(self, text: str) -> list[tuple[int, float]]:
        sig = self.signature(text)
        if sig is None:
            return []
        cand = set()
        for i, key in self._bands(sig):
            cand.update(self.buckets[i].get(key, ()))
        out = []
        for j in cand:
            est = float(np.mean(self.sigs[j] == sig))
            if est >= self.threshold:
                out.append((j, est))
        return out

    def query_many(self, texts: Iterable[str], limit: Optional[int] = None) -> list[tuple[int, int, float]]:
        hits = []
        for i, t in enumerate(texts):
            if limit is not None and i >= limit:
                break
            for j, est in self.query(t):
                hits.append((i, j, est))
        return hits
