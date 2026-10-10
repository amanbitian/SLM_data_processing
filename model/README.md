# model/slm_model (supplementary copy)

Supplementary code: a copy of the prompt format owned by the training repository (`../SLM_training/model/slm_model/`). See `../ARCHITECTURE.md` §0.

| File | What it is |
|---|---|
| `prompting.py` | Special tokens and task formats (torch-free) |
| `tokenizer.py` | SentencePiece wrapper and the task encoders (truncate the body only, never task tokens, query or `<|answer|>`) |
| `format_version.py` | `format_digest()`: hash of the code of the two files above; written into `packed/meta.json` by `pack` and checked by the trainer |

These files must stay identical in code to the training repository's copies. After changing one, copy it across, run `python data_pipeline/tests/test_format_contract.py`, and re-pack. The full model package (network, engine, export) is only in `SLM_training`.
