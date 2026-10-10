"""The prompt / tokenizer format contract between the two repositories.

This repository's model/slm_model/{prompting,tokenizer}.py build the packed training examples; the training
repository's copies build prompts at inference. They must be the same code (comments / docstrings may differ):
pack writes format_digest() into packed/meta.json and the trainer refuses packed data with another digest.
This test compares the two copies directly when the training repository sits next to this one
(or SLM_TRAINING_REPO points to it).
Run:  python data_pipeline/tests/test_format_contract.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from slm_model.format_version import format_digest  # noqa: E402

TRAIN_REPO = Path(os.environ.get("SLM_TRAINING_REPO") or (ROOT.parent / "SLM_training"))


def main():
    mine = format_digest()
    other_dir = TRAIN_REPO / "model" / "slm_model"
    if not (other_dir / "prompting.py").exists():
        print(f"format digest {mine}; training repository not found at {TRAIN_REPO} -> comparison skipped")
        print("\nFORMAT CONTRACT TEST PASSED (partial)")
        return
    theirs = format_digest(other_dir)
    assert mine == theirs, (f"prompting.py / tokenizer.py differ between the repositories ({mine} vs {theirs}): "
                            f"copy the changed files across, then re-pack")
    print(f"format digest {mine} == training repository")
    print("\nFORMAT CONTRACT TEST PASSED")


if __name__ == "__main__":
    main()
