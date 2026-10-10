from __future__ import annotations

import copy
import glob
import os
from pathlib import Path
from typing import Any

import yaml



REPO_ROOT = Path(__file__).resolve().parents[2]   # the data-preparation repository

class Cfg(dict):
    """dict with attribute access + path helpers."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    # ---- paths -------------------------------------------------------------
    @property
    def data_root_path(self) -> Path:
        return Path(os.environ.get("SLM_DATA_ROOT", self["data_root"]))

    @property
    def out_root_path(self) -> Path:
        # a relative out_root is relative to the repository root (not the shell's working directory)
        p = Path(os.environ.get("SLM_OUT_ROOT", self["out_root"]))
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def src(self, rel: str) -> Path:
        return self.data_root_path / rel

    def src_glob(self, pattern: str) -> list[Path]:
        return sorted(Path(p) for p in glob.glob(str(self.data_root_path / pattern)))

    def out(self, *parts: str) -> Path:
        p = self.out_root_path.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _set_dotted(d: dict, key: str, value: Any) -> None:
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        _set_dotted(raw, k.strip(), v)
    return Cfg(raw)
