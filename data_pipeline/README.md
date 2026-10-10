# data_pipeline (`slm_data`)

Turns the raw data into clean, labelled, split and tokenized training files. The inputs are:
- `/Volumes/New Volume/turing_candidate_dataset` (newest export);
- `/Volumes/New Volume/training_data` (older export, relevance judgments, eval assets, external data).

Run everything from the repository root (`SLM data cleaning_prep/`); `data.py` sets its own import paths, and for `python -m slm_data …` set `PYTHONPATH="$PWD/data_pipeline:$PWD/model"` (see the top-level README). Outputs go to `artifacts/data/` inside this repository; the training repository (`../SLM_training`) reads `artifacts/data/packed`.

Related docs:
- `../ARCHITECTURE.md`: how the pipeline is built (step graph, splits, PII, labels, leak control, packed format), and the **supplementary code** shared with the training repository.
- `DATA_MODEL.md`: ER diagram of every table and output.
- `DATA_SOURCES.md`: public sources.
- `../data_for_training.md`: the full data plan, split rules and token budget.

## Commands

```bash
export SLM_PSEUDO_SALT='<long random secret>'                 # fake-identity key for PII (keep it, never commit it)

python data_pipeline/data.py                                  # ALL steps, skipping the ones already done
python data_pipeline/data.py registry audit                   # only some steps (each named step is re-run)
python data_pipeline/data.py pack audit-packed                # re-pack and audit the final files
python data_pipeline/data.py --force                          # redo everything
python data_pipeline/data.py --set workers=12 --set sources.limit_row_groups=3     # quick trial on a slice

python -m slm_data <step> [--force]                           # same steps through the package CLI
python -m slm_data pack --only sft                            # re-pack SFT only (e.g. after teacher labelling)
python -m slm_data preview                                    # print formatted examples

python data_pipeline/scripts/download_external_data.py --list            # public data: list / dry-run / download
python data_pipeline/scripts/download_external_data.py --dry-run
python data_pipeline/scripts/download_external_data.py --contact you@example.com
python data_pipeline/scripts/download_external_data.py --only djinni_jobs,resume_job_fit

python data_pipeline/tests/test_pseudonymize.py               # tests on invented data (no real data needed)
SLM_PSEUDO_SALT=test python data_pipeline/tests/test_resume_build_pii.py
python data_pipeline/tests/test_registry_split.py
python data_pipeline/tests/test_turn53_pipeline.py             # step keys, batching, compact writer, tokenizer sampler, query families
```

**Step skipping (Turn 53, per-step since Turn 56).** A step is skipped only when `reports/<step>.done` holds the same key. Each step's key covers only what that step reads (`slm_data/step_keys.py`, `STEP_DEPS`):
- its own code (its module + everything it imports in `slm_data` / `slm_model`);
- its config sections;
- for steps that read raw data: every file under `sources` and `splits` (evaluation assets, override CSVs) by path, size, mtime and a content sample (first / middle / last 64 KiB);
- a one-way fingerprint of `SLM_PSEUDO_SALT` for the steps that pseudonymise;
- the keys of the steps it reads from.

So editing `pack.py` re-runs only `pack` and `audit-packed`, and changing the salt re-runs `resumes` and everything after it. AppleDouble `._*` files are ignored everywhere.

Without `pyarrow`, a slow pure-Python parquet reader is used. That is fine only for `sources.limit_row_groups` trials. Install pyarrow and run the full build from a local disk (not the USB drive).

## Steps

| Step | What it does | Output (under `artifacts/data/`) |
|---|---|---|
| `inventory` | checks every source | `reports/inventory.json` |
| `relevance-scan` | collects every judgment from `eval_consolidated` and the `evaluation/` CSVs, then builds a weighted judge consensus per (query, candidate). It adds score_mean / min / max / std, n_scores, verdict_flip, boundary and **risky** | `relevance/labels-*`, `candidates.sqlite`, `needed_resume_ids.txt` |
| `registry` | **two axes.** (1) Near-duplicate queries → families → train / dev / test (~80 / 10 / 10% of pairs; `reports/query_families.csv` for review). (2) **One split per talent_id**: eval assets → eval; ~20K test candidates across all queries; ~10K validation candidates (half `val`, half `dev`) plus 0.5% of everyone else → val; QA accounts → excluded; the rest → train. Accounts sharing an e-mail share the split | `splits/registry.sqlite`, `reports/registry_report.json` |
| `resumes` | newest export first, then the older one for missing ids. Cleans, quality-filters and **pseudonymises** PII (fake but consistent identities with check codes), drops documents where a real value survives, builds extraction targets (with a `contact` field, skills de-duplicated with the corrected taxonomy) and T7 domain labels. Held-out people never reach pretraining | `resumes/pretrain-*`, `extract-*`, `gold-*`, `classify-*`, `classify_t7-*`, `heldout_hashes.txt`, `resume_store.sqlite` |
| `relevance-build` | joins resume text into each pair. Its split is `pair_split(query, candidate)`: dev / test = unseen query families; train / val / dev_cand / test_cand = training-family queries by the candidate's split. Training pairs whose candidate text is a MinHash near-copy of a held-out candidate are dropped (`relevance.near_dup_decontam`). Only train is balanced (≤3:1 per query, ≤4,000 per query); names become placeholders | `relevance/pairs-*` |
| `knowledge` | KB text (group `knowledge`) + entity tables from O*NET, ESCO (TalentCLEF **training** files), ROR + OpenAlex, SEC, Wikidata (if present) and the skill taxonomy; `kb_names.txt` for routing Wikipedia articles | `knowledge/pretrain-*`, `kb_occupations-*`, `kb_institutions-*`, `kb_companies-*`, `kb_names.txt` |
| `external` | public text for training, every document tagged with a group (resume / jobs / knowledge / general). Includes Wikipedia (KB-entity articles → knowledge), FineWeb-Edu and Cosmopedia (→ general). HF resume-JD pairs (their test file = external benchmark), Kaggle resumes, JD titles, candidate roles, Djinni jobs and profiles, resume-job-fit text, WDC JobPosting (English, de-duplicated) | `external/{pretrain,pairs,classify}-*` |
| `audit` | **fails the build** on any of these:<br>- a talent_id on both sides of the candidate axis;<br>- any query family shared by training and dev / test, or a near-duplicate query across them;<br>- a held-out person in pretraining;<br>- an eval / QA account in training;<br>- an external JD in both HF train and dev / test;<br>- an external classification text in train and held-out;<br>- external pretraining text equal to a held-out query or candidate;<br>- MinHash near-duplicate candidate texts between training and held-out | `reports/leak_audit.json` |
| `tokenizer` | SentencePiece BPE 32K, trained on training people + public text only | `tokenizer/slm_sp.model` |
| `pack` | (Turn 56: SFT files are written to `sft.tmp` and swapped in complete; `meta.json` holds a content digest of every file) tokenizes everything into **one pretraining stream per group** (`pretrain_<group>_{train,val}.bin`, mixed by the trainer with `pack.pretrain_mix` or a task's `mix` / `stream`); drops pretraining docs whose text hash belongs to a held-out person; writes a key sidecar per file (normalised query / candidate hashes) | `packed/pretrain_{train,val}.bin`, `packed/sft/<task>_<split>.{bin,idx.npy,labels.npy,keys.npy}`, `meta.json` |
| `audit-packed` | the same query / candidate disjointness on the FINAL packed files (what the trainer reads) | `reports/leak_audit_packed.json` |

Relevance `labels.npy` columns are `is_relevant, judge_score, p_judge, query_group, p_teacher, risky, query_family_group` (column 6 since Turn 56: bootstrap CIs resample query families; column 3 stays the exact query for ranking batches), with `p_teacher = -1` when no teacher label exists. The teacher blend is applied at training time (`tasks[].teacher_weight`), so changing it needs no re-pack. Evaluation reports `stable_*` metrics on the non-risky pairs next to the overall ones.

## Splits

Relevance pairs combine the two axes:

| Pair split | Query family | Candidate | Used for |
|---|---|---|---|
| train | train | train | training |
| val | train | val | calibration, early stopping |
| **dev** | **dev (unseen)** | train / val / dev | decisions |
| dev_cand | train | dev | unseen-candidate check |
| **test** | **test (unseen)** | any (`strict_test: true` → only test candidates; others → `test_qonly`) | locked milestones |
| test_cand | train | test (~20K) | locked: unseen candidates on seen queries |

Candidate splits (all tasks):

| Split | Who | Used for |
|---|---|---|
| train | everyone not listed below | all training |
| val | half of the ~10K validation candidates + 0.5% of everyone else | early stopping, temperature / threshold calibration, pretraining perplexity |
| dev | the other half of the validation candidates | decisions: ablations, baselines, budget sweeps |
| test | ~20K candidates across all queries (each query keeps a share and both verdicts) | locked: milestone reports only |
| eval | evaluation assets: extraction gold, curated sets, fraud ids, JOB-110870, human gold | evaluation only |
| excluded | QA / test accounts | nothing |

## Key settings (`configs/data_config.yaml`)

| Setting | Default | Meaning |
|---|---|---|
| `sources.resume_parquet_globs` | turing_candidate_dataset, then resume_data | The first glob is preferred (`prefer_first_source`) |
| `sources.relevance_csv_globs` | `evaluation/**/*.csv` | Extra judged pairs |
| `sources.skill_taxonomy_csv` | `turing_candidate_dataset/Skill_Taxonomy_Corrected.csv` | Skill linking and synonyms |
| `splits.test_candidates` / `val_candidates` / `dev_share_of_val` | 20000 / 10000 / 0.5 | Held-out candidate counts |
| `splits.test_query_frac` / `dev_query_frac` / `min_query_families` | 0.10 / 0.10 / 2 | Query-family axis (share of pairs) |
| `splits.strict_test` | false | true = the test also needs an unseen candidate |
| `splits.query_family_overrides` / `query_split_overrides` | null | CSV `query,value` after reviewing `reports/query_families.csv` |
| `audit.near_dup` / `near_dup_threshold` / `near_dup_max_docs` | true / 0.8 / 200000 | MinHash near-duplicate check |
| `splits.eval_assets` | 8 asset files | Never trained on |
| `splits.exclude_email_patterns` | `@turing.com`, `+test` / `+qa` | QA accounts |
| `relevance.near_dup_decontam` / `near_dup_threshold` | true / 0.8 | Drop training pairs that near-duplicate a held-out candidate |
| `relevance.risky_policy` | soft | Keep uncertain labels as soft targets (`drop` = old behaviour) |
| `relevance.min_agreement` / `min_weight` | 0.75 / 0.9 | Below 0.75 agreement a pair is flagged `risky` (soft policy) |
| `relevance.max_candidate_chars` | 4000 | Matches what the judges saw |
| `cleaning.pii_mode` | pseudonymize | Or `redact` (placeholders). Needs `SLM_PSEUDO_SALT` |
| `cleaning.drop_on_leak` | true | Drop a document if any known real value survives |
| `extraction.include_contact` / `link_skills` / `domain_train_frac` | true / true / 0.3 | Contact field, taxonomy de-dup, share of T7 domain labels used |
| `extraction.min_grounding` | 0.5 | Share of parsed companies and skills that must appear in the text |
| `external.use_djinni` / `use_resume_job_fit` / `use_wdc` | true | External pretraining text |
| `pack.max_seq_len` / `think_frac` | 65536 / 0.35 | 64K context; share of relevance pairs with a think variant |
| `pack.pretrain_mix` | resume .50 · jobs .12 · knowledge .015 · general .365 | Default pretraining mix (stage 1); stages 2–3 set their own replay `mix` |
| `pack.global_dedupe_groups` | resume, jobs, knowledge, general | Exact normalised-text dedupe across all sources at pack time (on-disk SQLite set) |
| `pack.heldout_near_dup` / `_groups` / `_threshold` | true / resume / 0.8 | Drop training resumes (ours and public) that are MinHash near-copies of a held-out person's resume or a held-out relevance candidate |
| `pack.batch_max_chars` / `sft_batch_max_chars` | 2M / 400K | Encode batches are cut by characters as well as by count |
| `pack.chars_per_token_estimate` | 4.0 | Val budget (estimated tokens). Val documents are the lowest stable hashes of each group within the budget, so every source of the group is represented |
| `tokenizer.sample_max_files` | 64 | Tokenizer sample: random shard files per source, reservoir-sampled |
| `external.use_wikipedia` / `use_fineweb` / `use_cosmopedia` | true | General text (group `general`) |
| `sources.knowledge.*` | O*NET / TalentCLEF / ROR / OpenAlex / SEC / Wikidata paths | Inputs of the `knowledge` step |

## Where the code lives

- `data.py`: one-command entry point.
- `slm_data/registry.py`: splits.
- `slm_data/audit.py`: leak audit (`run` on shards, `run_packed` on packed files).
- `slm_data/near_dup.py`: MinHash + LSH and normalised text keys.
- `slm_data/pseudonymize.py`: PII fakes and leak audit per document.
- `slm_data/skill_taxonomy.py`: corrected taxonomy.
- `slm_data/build_relevance.py`: judgments, score stability, pairs.
- `slm_data/build_resumes.py`: resumes, extraction targets, domain labels.
- `slm_data/build_external.py` + `external_more.py`: public data (incl. general text).
- `slm_data/build_knowledge.py`: KB text + entity tables.
- `slm_data/groups.py`: pretraining groups.
- `slm_data/resume_schema.py`: canonical extraction schema.
- **Shared with inference:** `model/slm_model/prompting.py` + `tokenizer.py` (prompt format, special tokens) — a copy of the training repository's files; `format_version.py` hashes them, pack stores the digest in `meta.json`, and `tests/test_format_contract.py` compares the two copies.
- **Teacher labels:** `relevance/teacher-*.jsonl.gz` (from `slm_baselines.teacher_label`), keyed per pair and merged in `pack`.
- **Snapshot scripts:** `scripts/t7_snapshot/` (pure-Python parquet reader, id overlaps, aggregate stats).
- **Public data download:** `scripts/download_external_data.py`.
