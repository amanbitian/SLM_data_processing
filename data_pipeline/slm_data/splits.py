"""Deterministic, hash-based splits (stable across re-runs and machines)."""
from __future__ import annotations

import hashlib


def unit_hash(key: str) -> float:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") / 2**64


def pair_key(r: dict) -> str:
    """Stable id of ONE (query, candidate) pair. (query, resume_id) is not unique: external benchmark pairs
    have no resume_id, so every candidate of a JD would collide. The candidate text makes it unique."""
    s = "\x1f".join([str(r.get("query", "")), str(r.get("resume_id") or ""), str(r.get("candidate", ""))])
    return hashlib.blake2b(s.encode(), digest_size=12).hexdigest()


def split_of(key: str, val: float = 0.01, test: float = 0.01) -> str:
    u = unit_hash(key)
    if u < test:
        return "test"
    if u < test + val:
        return "val"
    return "train"
