"""
SentencePiece tokenizer wrapper + task encoders (torch-free).

The encoders are the ONLY place that turns task fields into token ids, and they are used
both by the data packer (training) and by the inference engine, so truncation behaves
identically at train and serve time.

SentencePiece (BPE + byte fallback) is used instead of a GPT-2 style BPE because llama.cpp
converts SentencePiece vocabularies for the `llama` architecture natively — the trained
model exports to GGUF and runs under the Dual-GPU orchestrator's llama-server.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from . import prompting as P

PII_TOKENS = ["[NAME]", "[EMAIL]", "[PHONE]", "[URL]"]
_BODY = "\u0000BODY\u0000"


class SLMTokenizer:
    def __init__(self, model_file: str | Path):
        import sentencepiece as spm

        self.path = Path(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.path))
        self.vocab_size = self.sp.get_piece_size()
        self.bos_id, self.eos_id = self.sp.bos_id(), self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.eos_id
        self.special = {t: self.sp.piece_to_id(t) for t in P.SPECIAL_TOKENS}
        for t, i in self.special.items():
            if i == self.sp.unk_id():
                raise ValueError(f"special token {t} missing from tokenizer vocab")
        self.relevant_id = self.special[P.RELEVANT]
        self.not_relevant_id = self.special[P.NOT_RELEVANT]
        self.answer_id = self.special[P.ANSWER]
        self.think_id = self.special[P.THINK]
        self.think_end_id = self.special[P.THINK_END]

    # ------------------------------------------------------------------ basics
    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: Sequence[str], num_threads: int = -1) -> list[list[int]]:
        return self.sp.encode(list(texts), out_type=int, num_threads=num_threads)

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode([int(i) for i in ids if int(i) not in (self.bos_id, self.eos_id, self.pad_id)])

    def id_to_piece(self, i: int) -> str:
        return self.sp.id_to_piece(int(i))

    # ------------------------------------------------------------------ task encoders
    def _wrap(self, template: str, body: str, budget: int) -> list[int]:
        pre, _, post = template.partition(_BODY)
        pre_ids, post_ids = self.encode(pre), self.encode(post)
        body_ids = self.encode(body)
        room = budget - 1 - len(pre_ids) - len(post_ids)  # 1 = BOS
        if room < 16:
            raise ValueError("prompt prefix too long for budget")
        if len(body_ids) > room:
            body_ids = body_ids[:room]
        return [self.bos_id] + pre_ids + body_ids + post_ids

    def encode_extraction(self, resume_text: str, budget: int) -> list[int]:
        return self._wrap(P.extraction_prompt(_BODY), resume_text, budget)

    def encode_relevance(self, query: str, candidate_block: str, budget: int,
                         current_date: Optional[str] = None, max_query_tokens: int = 2048) -> list[int]:
        q_ids = self.encode(query)
        if len(q_ids) > max_query_tokens:
            query = self.decode(q_ids[:max_query_tokens])
        return self._wrap(P.relevance_prompt(query, _BODY, current_date), candidate_block, budget)

    def encode_classify(self, text: str, label_space: str, budget: int) -> list[int]:
        return self._wrap(P.classify_prompt(_BODY, label_space), text, budget)

    def encode_answer(self, answer: str) -> list[int]:
        return self.encode(answer) + [self.eos_id]

    # ------------------------------------------------------------------ io
    def meta(self) -> dict:
        return {"vocab_size": self.vocab_size, "bos_id": self.bos_id, "eos_id": self.eos_id, "pad_id": self.pad_id,
                "special": self.special, "model_file": self.path.name}

    def save_meta(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.meta(), indent=2))
