"""
Prompt format shared by the data pipeline (producer of training text) and the
inference engine (consumer). Torch-free on purpose so `slm_data` can import it.

Every task is a single flat token stream:

    <s><|task_X|> ...inputs... <|answer|> ...target... </s>

Loss is applied only after <|answer|> during SFT. Relevance answers ALWAYS start
with one of two dedicated verdict tokens, so P(relevant) is available from a single
forward pass (no decoding) — this is what makes relevance scoring fast:

    <|answer|><|relevant|>{"score":8,"failed":[],"reason":"..."}</s>
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional

# ---- special tokens (order is frozen: ids are assigned in this order) -------------
TASK_EXTRACT = "<|task_extract|>"
TASK_RELEVANCE = "<|task_relevance|>"
TASK_CLASSIFY = "<|task_classify|>"
QUERY = "<|query|>"
CANDIDATE = "<|candidate|>"
RESUME = "<|resume|>"
ANSWER = "<|answer|>"
RELEVANT = "<|relevant|>"
NOT_RELEVANT = "<|not_relevant|>"
SEP = "<|sep|>"
THINK = "<|think|>"          # optional reasoning span before the verdict ("thinking tokens")
THINK_END = "<|/think|>"

SPECIAL_TOKENS: list[str] = [
    TASK_EXTRACT, TASK_RELEVANCE, TASK_CLASSIFY,
    QUERY, CANDIDATE, RESUME, ANSWER,
    RELEVANT, NOT_RELEVANT, SEP, THINK, THINK_END,
]

VERDICT_TOKEN = {"relevant": RELEVANT, "not_relevant": NOT_RELEVANT}
RELEVANCE_THRESHOLD = 5  # score >= 5  <=> relevant (same rule as the production judge)


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---- extraction ---------------------------------------------------------------------
def extraction_prompt(resume_text: str) -> str:
    return f"{TASK_EXTRACT}{RESUME}{resume_text}{ANSWER}"


def extraction_target(target: dict) -> str:
    return _compact(target)


# ---- relevance ----------------------------------------------------------------------
def relevance_prompt(query: str, candidate_block: str, current_date: Optional[str] = None) -> str:
    header = query.strip()
    if current_date:
        header += f"\nCURRENT_DATE: {current_date}"
    return f"{TASK_RELEVANCE}{QUERY}{header}{CANDIDATE}{candidate_block}{ANSWER}"


def relevance_target(verdict: str, score: Optional[float] = None,
                     failed: Optional[Iterable[str]] = None, reason: Optional[str] = None,
                     think: Optional[str] = None) -> str:
    """Fast mode (think=None):  <|relevant|>{"score":8,"failed":[],"reason":"..."}
    Think mode:               <|think|>...reasoning...<|/think|><|relevant|>{"score":8,"failed":[]}"""
    if verdict not in VERDICT_TOKEN:
        raise ValueError(f"bad verdict {verdict!r}")
    prefix = f"{THINK}{think.strip()}{THINK_END}" if think else ""
    if think:
        reason = None  # the reasoning already explains the verdict
    body: dict[str, Any] = {}
    if score is not None:
        body["score"] = int(round(float(score)))
    if failed is not None:
        body["failed"] = sorted(set(failed))
    if reason:
        body["reason"] = reason
    return prefix + VERDICT_TOKEN[verdict] + (_compact(body) if body else "")


def parse_relevance_answer(text: str) -> dict:
    """Parse generated text that follows <|answer|> (verdict token may be included)."""
    out: dict[str, Any] = {"verdict": None}
    t = text.strip()
    if t.startswith(THINK) or THINK_END in t:
        th, _, t = t.partition(THINK_END)
        out["thinking"] = th.replace(THINK, "").strip()
        t = t.strip()
    for v, tok in VERDICT_TOKEN.items():
        if t.startswith(tok):
            out["verdict"] = v
            t = t[len(tok):]
            break
    try:
        out.update(json.loads(t[: t.rfind("}") + 1]) if "}" in t else {})
    except json.JSONDecodeError:
        out["parse_error"] = True
    return out


# ---- classification (external datasets with a single label) -------------------------
def classify_prompt(text: str, label_space: str) -> str:
    return f"{TASK_CLASSIFY}{label_space}{SEP}{RESUME}{text}{ANSWER}"
