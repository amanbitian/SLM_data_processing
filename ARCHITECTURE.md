# Data pipeline architecture

How the data repository (`SLM data cleaning_prep`) turns the raw exports into the packed files the model trains on: the step graph, identity and splits, cleaning and PII, labels, leak control, the tokenizer and the packed format. It describes the code as of Turn 58 of `model_discussion.md`.

- **How to run it:** `README.md` (every command).
- **Step details and every setting:** `data_pipeline/README.md`.
- **Which data feeds which stage, the token budget and the measured problems behind the design:** `data_for_training.md`.
- **Tables and keys:** `data_pipeline/DATA_MODEL.md` (ER diagram).
- **Public sources:** `data_pipeline/DATA_SOURCES.md`.
- **The model and trainer:** `../SLM_training/ARCHITECTURE.md`.

---

## Contents

0. [Repositories and supplementary code](#0-repositories-and-supplementary-code)
1. [At a glance](#1-at-a-glance)
2. [Step graph and step skipping](#2-step-graph-and-step-skipping)
3. [Identity and splits](#3-identity-and-splits)
4. [Cleaning and PII](#4-cleaning-and-pii)
5. [Labels](#5-labels)
6. [Public text and knowledge](#6-public-text-and-knowledge)
7. [Leak control](#7-leak-control)
8. [Tokenizer](#8-tokenizer)
9. [Packed format (what the trainer reads)](#9-packed-format)
10. [Performance design](#10-performance-design)
11. [Code map](#11-code-map)
12. [Known limitations and open items](#12-known-limitations-and-open-items)

---

## 0. Repositories and supplementary code

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

---

## 1. At a glance

| | |
|---|---|
| **Inputs** (read-only) | `turing_candidate_dataset/` (newest export: 5.55M talent_ids, resumes, work / education tables, corrected skill taxonomy); `training_data/` (older export, judged relevance pairs in `eval_consolidated` + `evaluation/` CSVs, evaluation assets, `external_data_1010/` public data) |
| **Outputs** | `artifacts/data/`: split registry, cleaned shards, reports, tokenizer, `packed/` |
| **Tasks prepared** | Pretraining text in four groups (resume / jobs / knowledge / general); relevance pairs (query × candidate, verdict + soft label + score + reason); resume extraction (resume → JSON); classification |
| **Scale** | ≈15B unique pretraining tokens (resume 5.6B, jobs 1.1–1.7B, knowledge 0.05B, general 8.2B; `data_for_training.md` §6b) |
| **Guarantees** | One split per person for every task; unseen query families in dev / test; no held-out text in training (exact and near-duplicate); no real PII in any training text; every step's output reproducible and keyed by its inputs |
| **One command** | `python data_pipeline/data.py` (11 steps; finished steps are skipped when their key is unchanged) |

---

## 2. Step graph and step skipping

```
inventory
relevance-scan ──► registry ──► resumes ──► relevance-build ─┐
                       │                                      ├──► audit ──► tokenizer ──► pack ──► audit-packed
knowledge ──► external ┘──────────────────────────────────────┘          (pack also reads the registry)
```

| Step | Reads | Writes (under `artifacts/data/`) |
|---|---|---|
| `inventory` | every configured source | `reports/inventory.json` |
| `relevance-scan` | `eval_consolidated` parquet + `evaluation/**/*.csv` | `relevance/labels-*` (weighted consensus + score stability), `candidates.sqlite`, `needed_resume_ids.txt` |
| `registry` | relevance labels, evaluation assets, e-mails | `splits/registry.sqlite` (two axes), `reports/registry_report.json`, `reports/query_families.csv` |
| `resumes` | both exports, registry, taxonomy, extraction gold | `resumes/pretrain-*`, `extract-*`, `gold-*`, `classify-*`, `classify_t7-*`, `heldout-*` (held-out people, for decontamination only), `heldout_hashes.txt`, `resume_store.sqlite` |
| `relevance-build` | labels, registry, resume store | `relevance/pairs-*` |
| `knowledge` | O*NET, TalentCLEF (ESCO), ROR, OpenAlex, SEC, Wikidata, taxonomy | `knowledge/pretrain-*`, `kb_occupations-*`, `kb_institutions-*`, `kb_companies-*`, `kb_names.txt` |
| `external` | Djinni, resume-job-fit, WDC, Kaggle, HF resume-JD, Wikipedia, FineWeb-Edu, Cosmopedia | `external/{pretrain,pairs,classify}-*` |
| `audit` | all shards + registry | `reports/leak_audit.json` (fails the build on a leak) |
| `tokenizer` | samples of training text | `tokenizer/slm_sp.model` |
| `pack` | all shards, registry, tokenizer | `packed/` (§9), `reports/pack.json` |
| `audit-packed` | packed key sidecars | `reports/leak_audit_packed.json` |

**Step keys** (`slm_data/step_keys.py`). A step is skipped only when `reports/<step>.done` holds the same key. The key hashes:

- the step's own code: its module plus everything it imports in `slm_data` / `slm_model`, found by parsing the imports;
- its config sections;
- for steps that read raw data: every file under `sources` and `splits` (evaluation assets, override CSVs) by path, size, mtime and a content sample (first, middle and last 64 KiB);
- a one-way fingerprint of `SLM_PSEUDO_SALT` for the steps that pseudonymise;
- the keys of the steps it reads from (the graph above).

So editing `pack.py` re-runs only `pack` and `audit-packed`, while changing the salt or a source file re-runs everything downstream of it. macOS `._*` files are ignored.

---

## 3. Identity and splits

**One key joins everything:** `talent_id` (also called `developer_id`, `resume_id` / `rid` and `id` in different tables; `DATA_MODEL.md`).

**Two axes** (`slm_data/registry.py`):

1. **Query families.** Near-duplicate queries are merged with union-find: token Jaccard ≥ 0.6, or ≥ 0.3 together with character similarity ≥ 0.8. Candidate pairs are found by exact prefix filtering on rarest-first tokens, which gives the same result as comparing every pair. Families are assigned train / dev / test by share of pairs (~80 / 10 / 10%, at least 2 families each), with manual overrides from `reports/query_families.csv`.
2. **Candidates.** One split per talent_id for **every** task:

| Split | Who |
|---|---|
| eval | evaluation assets (extraction gold, curated sets, fraud ids, JOB-110870, human gold) |
| test | ~20,000 relevance candidates across all queries (each query keeps a share and both verdicts) |
| val / dev | ~10,000 more, half each, plus 0.5% of everyone else in val |
| excluded | QA / test accounts (`@turing.com`, `+test` / `+qa`) |
| train | everyone else |

Accounts that share an e-mail share the split.

**Relevance pair split** = `pair_split(query, candidate)`:

| Pair split | Query family | Candidate | Used for |
|---|---|---|---|
| train | train | train | training |
| val | train | val | calibration, early stopping |
| dev | dev (unseen) | train / val / dev | decisions |
| dev_cand | train | dev | unseen-candidate check |
| test | test (unseen) | any (`strict_test: true` → only test candidates; the rest go to `test_qonly`) | locked milestones |
| test_cand | train | test | locked: unseen candidates on seen queries |

---

## 4. Cleaning and PII

- **Text cleaning** (`text_clean.py`): Unicode normalisation, control characters and PDF artefacts removed, length limits (300 to 240,000 characters), alphabetic and ASCII ratio filters, and de-duplication by talent_id and normalised-text hash. The newest export is preferred; the older one only fills missing ids, processed in input order so the result is deterministic.
- **PII pseudonymisation** (`pseudonymize.py`, `pii_mode: pseudonymize`):
  - Each person gets a realistic but fake name, e-mail, phone and LinkedIn URL, consistent across documents.
  - They are generated with HMAC from the secret `SLM_PSEUDO_SALT`. The mapping is one-way: the real values cannot be recovered.
  - Each fake value carries a check code (5-character base36; a Luhn digit for phones), so extraction tests can verify them.
  - Any document in which a known real value survives is dropped and counted (`docs_with_leak` must be 0).
- **Relevance inputs** use placeholders instead of names, so a name cannot act as a proxy in scoring. The candidate block uses the same 4,000-character resume view the judges saw.
- **Protected attributes** (gender, birth date, citizenship, salary) are never used.

---

## 5. Labels

- **Relevance** (`build_relevance.py`):
  - **Consensus.** The judge columns are combined into a weighted verdict and agreement (12 judges, weights 0.4–1.0).
  - **Score stability.** `score_mean / min / max / std`, `verdict_flip` and `boundary` are kept per pair. A pair is `risky` when a judge flipped, its scores straddle 5, or agreement is below 0.75. Risky pairs are kept as soft targets (`risky_policy: soft`).
  - **Balancing.** Only train is balanced: at most 3:1 per query and at most 4,000 pairs per query.
  - **Think variants.** 35% of pairs also get a `<|think|>` variant.
  - **Teacher labels.** These are merged at pack time, from `slm_baselines.teacher_label` in the training repository.
- **Extraction** (`build_resumes.py`, `resume_schema.py`):
  - **Targets** are grounded parse results: at least 50% of companies and skills must appear in the text.
  - **Skills** are de-duplicated with the corrected taxonomy.
  - **Contact field.** Each target includes a `contact` field with the injected pseudonyms.
  - **Length cap.** The answer is at most 4,096 tokens.
- **Classification:** domain / category / role labels (T7 domains at a 30% train share), plus external classification texts.

---

## 6. Public text and knowledge

Every pretraining document has a **group** (`groups.py`): resume, jobs, knowledge or general. An unknown source is an error, not silently `general`.

| Group | Sources | Default mix (stage 1) |
|---|---|---|
| resume | our resumes, Kaggle resumes, Djinni profiles, resume-job-fit resumes | 0.50 |
| jobs | Djinni JDs, WDC JobPosting, Kaggle JD titles, resume-job-fit JDs | 0.12 |
| knowledge | O*NET, ESCO / TalentCLEF, ROR + OpenAlex, SEC, Wikidata (companies → `kb_companies`, universities → `kb_institutions`), skill cards, Wikipedia articles about known entities | 0.015 |
| general | the rest of Wikipedia, FineWeb-Edu, Cosmopedia v2 | 0.365 |

The trainer mixes the groups with these weights (`pack.pretrain_mix`). Stages 2 and 3 set their own replay mixes in the training repository.

---

## 7. Leak control

| Layer | Where | What it guarantees |
|---|---|---|
| Registry | `registry` | One split per person (and per shared e-mail); unseen query families for dev / test |
| Held-out text hashes | `resumes` → `pack` | A held-out person's exact text is never trained on, even under another talent_id |
| Near-duplicate decontamination (relevance) | `relevance-build` | Training pairs whose candidate block is a MinHash near-copy (Jaccard ≥ 0.8) of a held-out candidate are dropped |
| Near-duplicate decontamination (pretraining) | `pack` | Training resume text, ours and public, that is a near-copy of a held-out person's resume or a held-out candidate is dropped (`pack.heldout_near_dup*`) |
| Global de-duplication | `pack` | Exact normalised-text duplicates across all sources of all four groups are dropped (on-disk set) |
| Shard audit | `audit` | Fails on a talent_id in two splits, a query family in train and dev / test, near-duplicate queries, external JD / classification / pretraining leaks, near-duplicate candidates |
| Packed audit | `audit-packed` | The same query and candidate disjointness on exactly the files the trainer reads (key sidecars) |

---

## 8. Tokenizer

SentencePiece BPE with 32K tokens, trained on **training people and public text only**:

- The sample takes 400K documents.
- For each source it uses up to 64 shard files in random order, with an unbiased reservoir sample from each.
- Longer prefixes are kept apart, so, for example, external JDs do not pull in general text.

Special tokens (task, verdict, think, answer and PII placeholder tokens) come from `model/slm_model/prompting.py` and `tokenizer.py`, which are the shared format files (§0).

---

## 9. Packed format

`artifacts/data/packed/` is self-contained: copy it to the GPU machine, or let the training repository read it in place.

| File | Contents |
|---|---|
| `pretrain_<group>_{train,val}.bin` | uint16 token streams, documents as `<s> … </s>`. Val is the lowest stable hashes per group within 2M estimated tokens, so every source of the group is represented |
| `sft/<task>_<split>.bin` | Concatenated examples (prompt + answer) |
| `sft/<task>_<split>.idx.npy` | int64 `[N, 3]` = offset, length, prompt length (loss on the answer only) |
| `sft/relevance_<split>.labels.npy` | float32 `[N, 7]` = is_relevant, judge score, P_judges(relevant), exact-query group, P_teacher (−1 if none), risky, query-family group |
| `sft/<task>_<split>.keys.npy` | int64 `[N, 2]` = query key, candidate / text key (for `audit-packed`) |
| `slm_sp.model` | The tokenizer |
| `meta.json` | dtype, special token ids, pretraining mix and per-group token counts, filters applied, **content digest of every file**, **`format_digest`** (§0) |

**Atomic writes.** SFT files are written into `sft.tmp` and swapped in only when complete, so a crash or a removed task cannot leave stale files next to fresh ones. The trainer's data fingerprint hashes `meta.json`, so a re-pack with different content never resumes an old data cursor.

---

## 10. Performance design

- **Parquet:** each file is opened once per process, and the handles are dropped after every step, so no USB drive is held busy. pyarrow is used when installed, with a pure-Python reader as the fallback for trials.
- **Workers:** results are consumed in input order (ordered `imap`), which makes builds reproducible.
- **Relevance build:** one SQLite range read per query, batched resume-store reads, and held-out blocks from the indexing pass reused when writing.
- **Pack:**
  - Encode batches are cut by characters as well as by count.
  - The SFT writer uses growable numpy arrays.
  - The val budget is enforced at queue time.
  - The held-out near-duplicate check runs inside the encode workers.
- **Query families:** prefix filtering instead of comparing every pair of queries.
- **Disk:** the full build writes roughly 80–100 GB. Use a local disk (`SLM_OUT_ROOT`) with ≥ 150 GB free.

---

## 11. Code map

| File | Contents |
|---|---|
| `data_pipeline/data.py` | One-command entry point (all steps, or named ones) |
| `slm_data/__main__.py` | Step runner with per-step content keys |
| `slm_data/step_keys.py` | Per-step dependency keys (code, config, inputs, salt, upstream) |
| `slm_data/config.py` | Config loading; relative `out_root` resolves inside this repository |
| `slm_data/io_utils.py` | Parquet access (cached handles), JSONL shards, resume store (**used by the training repository**) |
| `slm_data/inventory.py` | Source check |
| `slm_data/build_relevance.py` | Judgment consolidation, score stability, pair building, near-duplicate decontamination |
| `slm_data/registry.py` | Query families and the candidate split registry |
| `slm_data/build_resumes.py` | Resume corpus, extraction targets, domain labels, held-out shard |
| `slm_data/pseudonymize.py`, `pii.py` | Fake identities with check codes; PII patterns and redaction |
| `slm_data/text_clean.py` | Text normalisation and filters |
| `slm_data/resume_schema.py` | Canonical extraction schema, grounding score (**used by the training repository**) |
| `slm_data/skill_taxonomy.py` | Corrected taxonomy: synonyms, de-duplication, knowledge cards |
| `slm_data/build_knowledge.py` | KB text and entity tables |
| `slm_data/build_external.py`, `external_more.py` | Public text and pairs, general text, WDC |
| `slm_data/groups.py` | Pretraining groups (strict source mapping) |
| `slm_data/near_dup.py` | MinHash / LSH, normalised text keys |
| `slm_data/splits.py` | Stable hashes, pair keys (**used by the training repository**) |
| `slm_data/audit.py` | Shard and packed leak audits |
| `slm_data/tokenizer_train.py` | Tokenizer sample and training |
| `slm_data/pack.py` | Packing, global de-duplication, held-out near-duplicate filter, digests, atomic SFT directory (**`soft_relevance_p` / `load_teacher` used by the training repository**) |
| `model/slm_model/prompting.py`, `tokenizer.py`, `format_version.py` | **Supplementary copy** of the prompt format (§0) |
| `data_pipeline/scripts/download_external_data.py` | Public-data downloader (cumulative manifest, spread shard sampling, Wikidata completeness) |
| `data_pipeline/scripts/t7_snapshot/` | Pure-Python parquet reader and snapshot statistics |
| `data_pipeline/tests/` | Tests on invented data, plus `test_format_contract.py` |

---

## 12. Known limitations and open items

- **Near-duplicate persons** are handled at pack time for held-out people (MinHash). A registry-level grouping of near-duplicate people *before* splitting is still open.
- **General web text** (Wikipedia, FineWeb-Edu, Cosmopedia) is de-duplicated exactly but not near-duplicate-checked against held-out text. It is unlikely to contain our candidates' resumes.
- **Wikidata** files are still empty (rerun the downloader with `--contact`). **BLS** was blocked (HTTP 403).
- **Labels are LLM-judge consensus**, made on the first 4,000 characters of each resume. True 64K relevance needs labels made on full dossiers.
- **Arrow-native intermediates** (Parquet instead of JSONL, persisted MinHash signatures, a canonical cleaned-document table) would speed up rebuilds but are not done.
- **No real build yet.** The first real build needs pyarrow, sentencepiece and a local disk with ≥ 150 GB free.
