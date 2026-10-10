# SLM data cleaning & preparation

The data repository of the candidate-relevance / resume-extraction small language model. It turns the raw data into clean, PII-pseudonymised, labelled, split, leak-audited and tokenized files. The model, trainer, baselines and serving live in the sibling repository **`SLM_training`**, which reads this repository's `artifacts/data/packed/`.

```
<parent>/
├── SLM data cleaning_prep/        THIS repository
│   ├── README.md                  this file: layout, install, every command
│   ├── ARCHITECTURE.md            how the pipeline is built: step graph, splits, PII, labels, leak control, packed format
│   ├── data_for_training.md       the data plan: sources, problems found, split rules, cleaning, PII, token budget
│   ├── model_discussion.md        the full design discussion (append-only log; also kept in SLM_training)
│   ├── requirements.txt
│   ├── data_pipeline/
│   │   ├── data.py                ONE command for the whole data build
│   │   ├── configs/data_config.yaml   paths, cleaning, PII mode, judge weights, splits, tokenizer, packing
│   │   ├── slm_data/              inventory, build_relevance, registry, build_resumes, build_knowledge,
│   │   │                          build_external, external_more, audit, tokenizer_train, pack, step_keys, ...
│   │   ├── scripts/               download_external_data.py, t7_snapshot/ (pure-Python parquet reader + stats)
│   │   ├── tests/                 tests on invented data (no real data, no network)
│   │   ├── README.md              steps, outputs, settings (details)
│   │   ├── DATA_MODEL.md          ER diagram of every source table and output
│   │   └── DATA_SOURCES.md        public data sources, licences, download status
│   ├── model/slm_model/           SUPPLEMENTARY COPY of the prompt format: prompting.py, tokenizer.py, format_version.py
│   └── artifacts/data/            outputs (created by the build; git-ignored)
└── SLM_training/                  model, training, baselines, serving (see its README.md)
```

## Repositories and supplementary code

The project lives in **two repositories side by side**. Each owns its own code, and each also carries or uses a small amount of **supplementary code** from the other:

| Repository | Owns | Supplementary code |
|---|---|---|
| **`SLM data cleaning_prep`** (data preparation) | `data_pipeline/` (`data.py`, `slm_data/`, configs, scripts, tests), `data_for_training.md`, outputs in `artifacts/data/` | **A copy of the prompt format** from the training repository: `model/slm_model/{__init__,prompting,tokenizer,format_version}.py` (torch-free). `pack` uses it to build training examples exactly as the model will see them at inference. Three tests optionally use the training repository when it sits next to this one (`test_format_contract.py`, the contact-metric part of `test_resume_build_pii.py`, the eval-sharding part of `test_turn56_pipeline.py`); otherwise they skip that part. |
| **`SLM_training`** (model, training, baselines, serving) | `model/`, `training/`, `baselines/`, `first_run.py`, outputs in `artifacts/` (runs, leaderboard, reports) | **Torch-free helpers imported from the data repository** at run time: `slm_data.io_utils.read_shards`, `slm_data.splits.pair_key` / `unit_hash`, `slm_data.pack.soft_relevance_p` / `load_teacher`, `slm_data.resume_schema.grounding_score`, `slm_data.config.load_config` (used by the baselines' text export, teacher labelling, diagnostics, `long_context_probe`, `think_budget_sweep`, `hw_bench`, `first_run.py`). It finds them with `model/slm_model/repo_paths.py`. It also **reads the data repository's packed data** (`artifacts/data/packed`). |

**Rules that keep the two in step**

- **Prompt format.** `prompting.py` and `tokenizer.py` must be the same code in both repositories (comments and docstrings may differ). `format_version.format_digest()` hashes them. `pack` writes the digest into `packed/meta.json`, and the trainer refuses packed data with another digest (`train.on_format_mismatch: refuse`). Change them in `SLM_training`, copy them to the data repository, run `python data_pipeline/tests/test_format_contract.py`, then re-pack.
- **Helper signatures.** The data-repository functions listed above are an interface: change them only together with their callers in the training repository.
- **Locations.** By default the training repository looks for `../SLM data cleaning_prep`. Override with `SLM_DATA_REPO` (the repository), `SLM_DATA_OUT` (its `artifacts/data`) or `SLM_PACKED_DIR` (the packed data). The data tests look for `../SLM_training` (`SLM_TRAINING_REPO`).
- **Shared log.** `model_discussion.md` is kept in both repositories with identical, append-only entries.
- `Small_Language_model/` is the pre-split snapshot. It is no longer updated.

## Install

```bash
pip install -r requirements.txt          # pyarrow, pandas, numpy, pyyaml, sentencepiece, openpyxl (no torch)
```

Run the full build **with pyarrow, from a local disk** with ≥150 GB free (the outputs are ~80–100 GB). Point the outputs elsewhere with `SLM_OUT_ROOT=/local/disk/slm_data` if this repository sits on a small drive. Without pyarrow a slow pure-Python parquet reader is used (trials only).

New git repository (the folder was created without `.git`):

```bash
git init && git add -A && git commit -m "Initial import"
```

## Commands (every command of this repository)

Run from this folder (`SLM data cleaning_prep/`).

**Build**
```bash
export SLM_PSEUDO_SALT='<long random secret>'      # fake-identity key for PII; keep it, never commit it, same value every time
python data_pipeline/data.py                        # ALL steps, skipping the ones already done (per-step content keys)
python data_pipeline/data.py --set sources.limit_row_groups=1 --set workers=12   # quick trial on a slice
python data_pipeline/data.py registry audit         # only some steps (each named step is re-run)
python data_pipeline/data.py pack audit-packed      # re-pack and audit the final files
python data_pipeline/data.py --force                # redo everything
# steps: inventory relevance-scan registry resumes relevance-build knowledge external audit tokenizer pack audit-packed
```

**Same steps through the package**
```bash
export PYTHONPATH="$PWD/data_pipeline:$PWD/model"
python -m slm_data <step> [--force] [--set key=value]
python -m slm_data pack --only sft|pretrain          # re-pack one part (e.g. after teacher labelling)
python -m slm_data preview                           # print formatted examples
python -m slm_model.format_version                   # print the prompt-format digest (compare with SLM_training)
```

**Public data download** (sandbox-free machine with internet)
```bash
python data_pipeline/scripts/download_external_data.py --list | --dry-run
python data_pipeline/scripts/download_external_data.py --contact you@example.com [--only a,b] [--skip a,b] \
       [--fineweb-gb N] [--cosmopedia-gb N] [--wikipedia-gb N] [--checksums]
# MANIFEST.json is cumulative; capped sets sample shards across the whole source; a Wikidata page sequence cut by
# an error is kept as *.incomplete and not used. --contact is sent only to sec.gov / Wikimedia as their policies require.
```

**Tests** (invented data)
```bash
python data_pipeline/tests/test_pseudonymize.py
SLM_PSEUDO_SALT=test python data_pipeline/tests/test_resume_build_pii.py   # contact metrics part needs ../SLM_training
python data_pipeline/tests/test_registry_split.py
python data_pipeline/tests/test_turn53_pipeline.py    # step keys, batching, compact writer, tokenizer sampler, query families
python data_pipeline/tests/test_turn56_pipeline.py    # step-key DAG, pack filters / val / digests / atomic SFT, groups, downloader
python data_pipeline/tests/test_format_contract.py    # prompting / tokenizer identical to the training repository's copy
```

**Snapshot statistics** (of the T7 export; set the paths inside first, see `training_data/t7_snapshot/README.md`)
```bash
python data_pipeline/scripts/t7_snapshot/snapshot_stats.py <job>
python data_pipeline/scripts/t7_snapshot/ids_overlap.py
```

## After a build, check

- `artifacts/data/reports/registry_report.json` — test / val / dev counts per query;
- `artifacts/data/reports/leak_audit.json` and `leak_audit_packed.json` — both must say `"ok": true`;
- `artifacts/data/reports/query_families.csv` — review the automatic query families once (fix with `splits.query_family_overrides` / `query_split_overrides`);
- `artifacts/data/reports/resumes_stats.json` — `docs_with_leak` must be 0;
- `artifacts/data/reports/pack.json` — per-group tokens, `filters` (duplicates and held-out near-copies dropped), `format_digest`.

Steps, outputs and every setting: `data_pipeline/README.md`. How the pipeline is built: `ARCHITECTURE.md`. The data plan and which data is used when: `data_for_training.md`.

## Troubleshooting

| Message | Fix |
|---|---|
| `set SLM_PSEUDO_SALT` | Export the secret before a build; use the same value every time |
| `no split registry — run registry` | `python data_pipeline/data.py registry` (after `relevance-scan`) |
| `leak audit failed` | Read `artifacts/data/reports/leak_audit*.json` (problems + examples); usually a stale output — re-run the named step |
| `pretraining record with unknown group / source` | A new or misspelled source: add it to `slm_data/groups.py` `SOURCE_GROUP` |
| `train the tokenizer first` | Run the `tokenizer` step (needs `sentencepiece`) |
| Very slow build | Install pyarrow; run from a local disk, not the USB drive |
| `FORMAT CONTRACT` test fails | Copy the changed `prompting.py` / `tokenizer.py` to the other repository, then re-pack |
