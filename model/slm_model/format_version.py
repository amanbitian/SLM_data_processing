"""
Format contract between the two repositories.

`SLM data cleaning_prep` (data preparation) and `SLM_training` (model, training, serving) each carry a copy of
`slm_model/prompting.py` and `slm_model/tokenizer.py`: the data side uses them to build the packed training
examples, the training / serving side to build prompts at inference. If the two copies drift apart, the model is
trained on one format and served with another — silently.

format_digest() is a hash of the CODE of those two files (comments and docstrings ignored), so:
  - `pack` writes it into packed/meta.json ("format_digest");
  - the trainer compares it with its own copy before training (train.on_format_mismatch: refuse | warn);
  - `python -m slm_model.format_version` prints it (compare the two repos by hand).
Torch-free on purpose (the data pipeline imports it).
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

FILES = ("prompting.py", "tokenizer.py")


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def format_digest(pkg_dir: Path | None = None) -> str:
    d = Path(pkg_dir) if pkg_dir else Path(__file__).resolve().parent
    h = hashlib.sha1()
    for name in FILES:
        tree = _strip_docstrings(ast.parse((d / name).read_text(encoding="utf-8")))
        h.update(name.encode())
        h.update(ast.dump(tree, annotate_fields=False, include_attributes=False).encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(format_digest())
