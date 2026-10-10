# Data for training: sources, cleaning, splits and leakage control

> **Turn 57 — two repositories.** This plan lives in the data repository (`SLM data cleaning_prep`). Training configs (`training/configs/*.yaml`), the model and the baselines mentioned below are in `../SLM_training`.

Written 2026-10-10 after a full profile of both data sources. Every number here was measured (exact counts unless marked "sample"). The aggregate snapshot of the T7 SSD lives in `training_data/t7_snapshot/` so this plan stays discussable after the SSD is unplugged. Related docs: `data_pipeline/DATA_SOURCES.md` (public data), `ARCHITECTURE.md` §19 (reasoning / KB design), `model_discussion.md` (Turn 39).

## 0. Which data is used when — read this first (Turn 51)

**Three kinds of data, three uses:**

- **Raw text** teaches the language: what resumes, job ads and the world of work look like. It feeds *pretraining*, sorted into four groups that are mixed with fixed weights.
- **Labelled examples** teach the two jobs: (a) read a resume into JSON, and (b) say whether a candidate fits a query. They feed *supervised training* (stages 2 and 3).
- **Held-out data** is never trained on. It measures progress: val / dev / test splits, evaluation assets and external benchmarks.

### 0.1 Every dataset and where it goes

| Dataset (location) | Size | Pretraining group | Supervised task | Evaluation |
|---|---|---|---|---|
| Our resumes: `turing_candidate_dataset` (+ `resume_data` for missing ids) | 5.54M resumes, ≈5.5B tokens | **resume** (training people only) | Extraction targets (3.5M parses → filtered); domain labels (670K) | Held-out people: val / dev / test |
| Work / education tables (`turing_candidate_dataset`) | 18.5M + 11.5M rows | – (they duplicate resume text) | Extraction labels / cross-checks | – |
| Relevance judgments: `eval_consolidated` + `evaluation/` CSVs | ~375K+ pairs, 164–280 queries | – | **Relevance** (train pairs; risky = soft targets) | val, dev (unseen queries), dev_cand, test / test_cand (locked) |
| Evaluation assets: extraction gold, curated sets, fraud ids, JOB-110870 | ~4K people | **never** | **never** | Extraction gold, case studies |
| Corrected skill taxonomy (46K) | – | **knowledge** (skill cards) | Skill de-dup in extraction targets | – |
| Djinni jobs (142K) | 0.07B tok | **jobs** | – | – |
| Djinni candidate profiles (210K) | 0.10B tok | **resume** | – | – |
| WDC JobPosting (14 parts, ~2M unique English postings est.) | ≈1.0–1.65B tok | **jobs** | (structured fields kept for a future query parser) | – |
| resume-job-fit (med2425) train files | small after de-dup | **resume** / **jobs** (text only, labels ignored) | – | – |
| HF resume-JD match (`jd & Resume External`) | 6,241 + 1,759 pairs | – | **Relevance** (train file, JD = query; JDs that occur in its test file are dropped) | Its test file: external dev / test |
| Kaggle resumes / JD titles / candidate roles | 2.5K / 2.3K / 1K | **resume** / **jobs** | Category / role classification | 5% val / 5% test (split by text) |
| O*NET 31.0 (1,016 occupations) | KB cards | **knowledge** | – | – |
| ESCO via TalentCLEF *training* files (3,039 occupations) | KB cards | **knowledge** | – | TalentCLEF test / validation kept for a future external eval |
| ROR + OpenAlex (30K education orgs, research stats) | KB cards | **knowledge** | – | – |
| SEC tickers (10.4K US-listed companies) | KB cards | **knowledge** | – | – |
| Wikidata companies / universities | empty: the download failed, fixed for a rerun | **knowledge** (when present) | – | – |
| Wikipedia (31 of 41 shards, 4.84M articles) | ≈3.3B tok | **general**; articles about known companies / universities / occupations → **knowledge** | – | – |
| FineWeb-Edu sample (2.9M docs) | ≈3.5B tok | **general** | – | – |
| Cosmopedia v2 (1.5M docs) | ≈1.4B tok | **general** | – | – |
| BLS Occupational Outlook | download blocked (HTTP 403): save manually in a browser | not used yet | – | – |

**Totals.** Unique pretraining text is now **≈15–15.5B tokens**:
- resume ≈5.6B;
- jobs ≈1.1–1.7B;
- knowledge ≈0.05B;
- general ≈8.2B.

Supervised data adds ≈0.4B relevance tokens, 0.4–1.4B extraction and ≈0.2B classification, but these reuse resume text.

### 0.2 When each one is used (training stages)

| Stage | What the model learns | Data (training split only) | Mix |
|---|---|---|---|
| **1 Pretrain** (4K, then 64K in 1b) | language of resumes, jobs, knowledge, general English | pretraining groups | resume 50% · jobs 12% · knowledge 1.5% · general 36.5% (`pack.pretrain_mix`; knowledge is small, so its share is kept low to avoid >4–5 repeats) |
| **2 Extraction** | resume → JSON (contact, skills with years, work, education) | extraction targets 75% + domain / category classification 5% + pretraining replay 20% | replay mix: resume 60 · jobs 20 · knowledge 15 · general 5 |
| **3 Relevance** | query + candidate → calibrated P(relevant), think, explain | relevance pairs (ours + HF) 45%, think variants 15%, extraction 25%, classification 5%, replay 10% | replay mix: resume 50 · jobs 25 · knowledge 20 · general 5 |
| **Evaluation** | progress and the comparison with Qwen3.5-4B | val (calibration), **dev** (unseen queries: decisions), dev_cand, locked test / test_cand, extraction gold, HF external test, human gold | never trained on |

How it is enforced: every pretraining document carries a `group`, and `pack` writes one stream per group (`packed/pretrain_<group>_{train,val}.bin`). The trainer mixes the streams by the weights above; a stage can override them with `mix:` or train on one group with `stream:`. The audits fail the build if any held-out person, query family or text appears in training.

## 1. What we have

### Source 1: `/Volumes/T7 Shield/turing_candidate_dataset` (33 GB, BigQuery export of Oct 2026)

| Table | Rows | What it holds | Notes |
|---|---|---|---|
| `resume_and other details/part_*.parquet` (7 parts) | 5,555,173 candidates, unique `talent_id` | resume text (`resume_plain`, 5,539,424 non-empty, 23.7 GB chars ≈ 5–6B tokens), OpenAI parse (`parsed_resume_data`, 3,507,029), work / education JSON, summary, role (257 values), domain / subdomain (12% filled), country, salary, contact fields | Primary candidate source. Contains heavy PII |
| `resume_and other details/fired.parquet` | 10,923 | same columns, candidates marked fired | Not in the main export; 103 are in the relevance pool. Sensitive |
| `work_experience/*.parquet` | 18,517,943 rows / 5,219,842 developers | position, company, dates, details, employment type, seniority (891 raw values), industry, department, company_id, duration | Structured ground truth for extraction |
| `education/*.parquet` | 11,510,118 rows / 5,903,692 developers | level, degree, college, dates, CGPA + scale, subject, university_id, tiers | Structured ground truth for extraction |
| `jd_resume_matching/*.parquet` | 1,165 | one job (JOB-110870, bioinformatics): candidates + in-house AI screening (`final_recommendation` 62 good / 83 review / 895 reject), fraud verdicts, job-fit reasons, 44 interview transcripts | Job title only, no JD text. 653 candidates overlap the relevance pool |
| `Skill_Taxonomy_Corrected.csv` | 46,338 skills | corrected taxonomy with confidence and change notes | Newer than the 43,340-row taxonomy in `training_data`; copied into the snapshot |

### Source 2: `/Volumes/New Volume/training_data`

| Folder | Size / count | Use |
|---|---|---|
| `resume_data/` | 7,460,148 rows, 5,635,011 with text, 3,451,678 parses (Sep 2026 export) | Older export of the same population (5,468,392 talent_ids shared with T7; 1,991,756 only here, mostly rows without text) |
| `eval_consolidated/` | 2,743,560 judge rows → 375,566 unique (query, resume) pairs, 164 normalised queries, 256,185 resumes; 12 judge columns | Relevance labels (consensus) |
| `evaluation/` | 26 GB of eval / A-B runs; 279 unique queries in its CSVs | up to ~115 more queries to consolidate (overlap with the 164 still to measure) |
| `resume_extraction/` | 1,960 candidates with GPT-5.x taxonomy extractions | Extraction gold (locked test) |
| `curated_dataset/`, `curated_dataset_35/` | 296 relevance / 250 extraction / 481 fraud; 35-row set | Eval assets |
| `Fraud/` | 969 unique talent_ids | Future fraud task |
| `id_universe/` | exact id lists of the above | Registry input |
| `jd & Resume External/`, `external_data_1010/` | HF resume-JD match, Kaggle resumes, JD titles, Djinni, WDC (downloading) | Public text, external evals |

How the sources relate: T7 and `resume_data` are two exports of the same candidate base. 252,360 of the 256,185 relevance resumes are in T7, all 1,960 extraction-gold ids are in `resume_data` (1,788 in T7), 912 of 969 fraud ids are in T7.

The T7 dataset is being copied to `/Volumes/New Volume/turing_candidate_dataset/`. The pipeline config now reads it from there, as the first (preferred) resume source.

### External data: `training_data/jd & Resume External/` (already wired into `build_external.py`)

| File | Rows | Used as | Notes |
|---|---|---|---|
| `Hugging face esume-jd-match/train-*.parquet` | 6,241 JD + resume pairs, labelled Good / Potential / No Fit | Relevance pairs (JD = query); Potential Fit → soft target 0.6; 2% val by JD | Public, small; a long-JD-query slice |
| `Hugging face esume-jd-match/test-*.parquet` | 1,759 | External benchmark only (half dev, half test, by JD) | Never trained on |
| `Resume.csv.zip` (Kaggle) | 2,484 resumes, 24 categories | Pretraining text + resume → category classification (5% val, 5% test, split by text) | PII replaced by placeholders (`redact`) |
| `job_title_des.csv.zip` | 2,277 JDs over 15 software titles | Pretraining JD text; seeds for synthetic queries | Narrow (software roles only) |
| `candidate_job_role_dataset.csv` | 1,000 synthetic rows (skills, qualification, level → role) | Tiny role-classification task | Templated, low value; keep at a very small weight |
| `data.zip` | 2,484 PDFs in 24 folders | Not used | The same Kaggle resumes as PDFs (duplicate of `Resume.csv`); only useful later to test PDF-to-text robustness |

`external_data_1010/` (Djinni 142K JDs + 210K profiles, WDC JobPosting, O*NET, ROR, OpenAlex, SEC, TalentCLEF, Wikipedia, FineWeb-Edu, Cosmopedia, resume-job-fit text) is downloaded and wired in (Turn 51, see §0). Wikidata is still empty (rerun the downloader) and BLS was blocked (403).

## 2. Problems found in the pipeline before Turn 45 (measured, not hypothetical)

*Historical: these were measured on the old pipeline and are all fixed — person / query-family registry (§4–5), leak audits (§8), pseudonymisation (§6). Kept as the reason for the design.*

1. **Near-duplicate queries straddle splits.** The current split hashes each normalised query on its own. Among the 164 queries there are 32 near-duplicate pairs (same text with a missing space, a trailing word, "3+" vs "6+ years", "Chinese + English" vs "…, remote"); 17 of those pairs land on different sides of train / dev / test. A model trained on one variant is then "tested" on its twin.
2. **Candidates are shared between train and held-out queries.** With today's split, 49.6% of the dev resumes and 54.5% of the test resumes also appear in training pairs under other queries, and 1,705 resumes sit in both dev and test. The relevance pool is also fully inside the pretraining corpus and the extraction training set (those use independent hashes `ext:`, `cls:`, `p:`), so the model has read every test resume before it is tested.
3. **Five independent split hashes** (`q:`/`qd:` queries, `p:` pairs, `ext:` extraction, `cls:` classification, `hf:`/`kag:`/`role:` external) with no shared registry, so one person can be train in one task and test in another.
4. **Eval assets come from the training pool.** `curated_dataset` (296 pairs) is drawn from `eval_consolidated`; the extraction gold, fraud ids and JOB-110870 candidates are inside the resume corpus. Nothing removes them from training.
5. **One person, several accounts.** 4,105 e-mail addresses are shared by 8,362 talent_ids (up to 50 on one address), and exact-duplicate resume texts exist (648 within a 93,757-row sample). Splitting by `talent_id` alone lets the same person be on both sides.
6. **Test / QA accounts.** 12,020 `+`-tagged or `@turing.com` addresses.
7. **Data-quality issues for labels** (samples): 33 of 58,942 parses are a JSON *schema*, not data; in work rows `end_year = 0` for 20% (current or unknown), `start_year = 0` for 3%, 1,483 end-before-start rows in a 1.85M sample, month fields in 6+ languages plus `"0"`, `""`, `"Invalid date"`; in education 5.8% end before start and CGPA on scales 4 / 5 / 10 / 100.
8. **PII and protected attributes everywhere.** 95% of resumes contain an e-mail and a phone, 43% a URL; parses carry `gender` and `birth_date` (77% of parses), plus citizenship, salary and the fired flag in the tables.

## 3. The strategy in one paragraph

Resolve every record to a **person** and every query to a **query family**, assign each person and each family to a split **once**, in a single versioned registry that every stage reads, and make the locked test **doubly unseen**: its queries and its candidates appear in no training stage at all (not pretraining, not extraction, not the tokenizer, not teacher or synthetic data). Then clean in a fixed order (identity → text → quality → PII → structured fields → labels → decontamination), keep a reason for every dropped record, and refuse to pack data unless an automatic leak audit passes.

## 4. Identity and grouping (the core of leakage control)

**Person group** = union-find over:
- `talent_id`, identical across T7 and `resume_data`;
- the normalised e-mail hash (lower-case; Gmail dots and `+tags` removed);
- `finger_print_id` (resume_data);
- normalised LinkedIn / GitHub handle;
- near-duplicate resume text: MinHash LSH on 5-word shingles, Jaccard ≥ 0.8, so re-uploads and edited copies join one group.

The group id is the smallest `talent_id` in the group. Groups larger than a cap (e.g. the 50-account address) go to manual review; such groups are usually test or agency accounts, not people.

**Query family** = union-find over normalised queries:
- lower-cased, with spacing, punctuation and quotes unified;
- linked when token Jaccard ≥ 0.6 or character similarity ≥ 0.8, or when the queries share a template differing only in numbers or a single constraint (3+ vs 6+ years, US vs Non-US).

A first automatic pass merges 164 queries into 138 families, but it also merges a few unrelated pairs ("…field jobs" vs "…legal"). So every automatic merge is reviewed once by a person (a CSV of ~30 pairs), and the reviewed families are frozen. The same applies to the ~115 extra queries from `evaluation/`.

**JD family** for external pairs: the hash of the first 200 normalised JD characters, plus MinHash on the full JD.

## 5. Split design

> **Decision of Turn 45 (implemented; replaces the query-family holdout below where they differ).**
> - **One command:** `python data_pipeline/data.py` prepares ALL data. That includes the T7 export on the pen drive, `resume_data`, every judged pair from `eval_consolidated` and the `evaluation/` CSVs, and all external data. Only the evaluation assets and the held-out candidates are kept out of training.
> - **Split by candidate:** the registry (`slm_data/registry.py`) gives every `talent_id` exactly one split, for every task.
>   - **test:** about 20,000 candidates drawn across all queries, so every query keeps a share and both verdicts.
>   - **val / dev:** about 10,000 more candidates, split in two halves. `val` (plus 0.5% of everyone else) is used for early stopping and calibration; `dev` is used for decisions (ablations, baselines, budget sweeps).
>   - **eval:** the evaluation assets.
>   - **excluded:** QA accounts.
>   - **train:** everyone else.
>   - Accounts that share an e-mail share the split.
> - **What the test measures:** queries are therefore seen in training (with other candidates), so the test measures new *candidates*, not new *queries*. An unseen-query slice can be added later if needed.
> - **Uncertain labels:** pairs whose score changed enough to flip the verdict keep `score_mean / score_min / score_max / score_std / n_scores` and are flagged `risky` (flip, scores on both sides of 5, or low judge agreement). They train as soft targets, and the eval also reports `stable_*` metrics without them.
> - **Leak check:** `audit` fails the build if a `talent_id` appears in two splits.
> - **Data model:** see the ER diagram in `data_pipeline/DATA_MODEL.md`.
>
> **Turn 47 (review fix): a query axis on top of the candidate axis.**
> - **Why:** the candidate split alone could not support an unseen-query claim.
> - **Query families:** near-duplicate queries are merged into families (token Jaccard ≥ 0.6 or character similarity ≥ 0.8, plus manual overrides). Each family is train / dev / test by share of pairs (~80 / 10 / 10%).
> - **Pair split:** `pair_split(query, candidate)`:
>   - **dev / test** = unseen query families;
>   - **val / dev_cand / test_cand** = held-out candidates on training-family queries (the ~20K test candidates live in `test_cand`);
>   - `strict_test: true` also requires an unseen candidate in `test`.
> - **Audit:** the shard audit and the packed-file audit (`audit-packed`) assert zero query-family overlap and no near-duplicate query across the two sides.

One registry, `artifacts/data/splits/registry_v1.parquet`, holds:
- `person_group → split`;
- `query_family → split`;
- the version, the seed and the content hash of the inputs.

Every builder (resumes, extraction, relevance, external, tokenizer sample, teacher labelling, synthetic generation, KB rendering) looks the split up there; nobody hashes on their own any more. Assignments are deterministic (`blake2b(salt:id)`), test membership is append-only, and any change bumps the version and forces re-baselining.

| Split | Queries | Candidates | Used for |
|---|---|---|---|
| **train** | train families | train persons | all training stages |
| **val** | train families | train persons, held-out *pairs* (2%) | early stopping, temperature / threshold fitting |
| **dev** | dev families (~10%) | any; every pair tagged `cand_seen` / `cand_unseen` | day-to-day decisions; both slices reported |
| **test (locked)** | test families (~10%) | test persons only, never seen anywhere | milestone reports only |
| **extraction test** | – | gold 1,960 + test persons | extraction metrics |
| **pretrain val** | – | a 0.5% person slice | perplexity |

Measured cost of the strict test, on today's pairs with the 138 families: holding the test-family candidates (38,193 persons) out of training removes 38,994 of 296,372 train pairs (13.2%). Holding out dev candidates too would remove 18.1%, so dev keeps seen candidates but reports the two slices separately. The removed pairs are not wasted: they become a "seen query, unseen candidate" diagnostic slice.

### Split at a glance (simulated on today's data, Turn 40)

Simulated on the 375,566 relevance pairs with the automatic query families and a plain hash at 80 / 10 / 10. Counts are before consensus filtering and balancing, so the final numbers will be lower. The real build also stratifies families by domain and size.

**Relevance pairs**

| Split | Query families (queries) | Pairs | Candidates | Rule |
|---|---|---|---|---|
| Train | 109 (129) | 250,648 | 196,272 | train family × train person |
| Val | same 109 | 5,116 | – | 2% of train pairs, by pair hash |
| Diagnostic (not trained on) | same 109 | 40,608 | test-pool persons | train family × held-out person ("seen query, unseen candidate") |
| Dev | 13 (14) | 33,830 | 33,109 | dev family × any person; 9,825 pairs have a candidate also in train, tagged and reported separately |
| Test (locked) | 16 (21) | 45,364 | 38,193 | test family × test person; neither seen in any training stage |

Check: 250,648 + 5,116 + 40,608 + 33,830 + 45,364 = 375,566.

**Persons and other tasks**

| Task / corpus | Train | Val / dev | Test |
|---|---|---|---|
| Pretraining text (~5.5M persons) | all persons except the held-out pool and pretrain-val | pretrain-val: 0.5% of persons (~28K) | not used for scoring; the held-out pool (~42K persons) is absent |
| Held-out person pool | – | – | 38,193 test-family candidates + 4,087 eval-asset persons (extraction gold, fraud, JOB-110870; 1,244 of them already in the pairs) ≈ 41,941 |
| Extraction (3.5M parses, filtered to 0.3–1M) | parsed train persons | 1% of train persons (by person hash) | 1,960 gold + parsed persons from the held-out pool |
| Tokenizer sample | train persons + public text | – | – |
| Teacher labels | cross-fitted folds over train families | – | never labels dev / test for training |
| External pairs (HF JD-match) | its train file, by JD family | 2% by JD family | its own test file (external benchmark) |

**Always held out, whatever their hash** (they define the held-out person pool):
- extraction gold (1,960);
- `curated_dataset` / `curated_dataset_35` pairs and candidates;
- the human gold set;
- fraud ids (969);
- the JOB-110870 candidates (1,165).

This costs well under 0.1% of the 5.5M candidates.

**Why "doubly unseen" matters here.** In production the candidate pool is mostly fixed, so dev with seen candidates matches day-to-day use. But a model that memorised resumes during pretraining can look good on seen candidates without understanding the query. The strict test is the only honest estimate of how the model handles new people, so it is the number we compare with Qwen3.5-4B and the baselines.

**Other leakage rules.**
- **Teacher labels:** cross-fitted folds built from train families only; the teacher never sees dev / test pairs (already enforced by `pair_key` folds; the folds must now come from the registry).
- **Reasons and thoughts:** judge reasons, checklist thoughts and KB cards come only from train pairs and public sources; statistics used to build the KB (e.g. "common skills for a role") are computed on train persons only.
- **Tokenizer:** trained on a sample of train persons and public text only.
- **External data:** MinHash / 13-gram decontamination against every held-out resume and every dev / test query before any external text enters training. `med2425` and the HF resume-JD test files stay out of our dev / test.
- **Time:** every pair keeps `judge_date`; we also report a "judged after <cutoff>" slice to see drift.

## 6. Cleaning pipeline (in this order)

Each stage writes a report with counts per rule; nothing is dropped silently.

**S0. Ingest and harmonise.**
- Build one candidate record per `talent_id`. T7 wins (newer export), and `resume_data` fills ids missing from T7.
- Map schema differences: T7 `name` vs `account_full_name`; T7 has no `resume_parse_status`; T7 adds the work / education JSON and tables.
- Parse every JSON field; drop parse payloads that are a JSON schema instead of data.
- Pipeline change needed: `build_resumes.COLS` expects the `resume_data` columns.

**S1. Identity resolution and de-duplication (section 4).**
- One canonical document per person for pretraining (the latest valid upload).
- Every id keeps its group so splits stay consistent.
- Test / QA accounts (`@turing.com`, `+test` / `+qa` tags, groups above the size cap) are dropped from training after review.

**S2. Text normalisation.**
- Keep the existing `clean_text` (NFKC, ligatures, control characters, bullets).
- Also strip BOM characters, which occur several times at the start of some resumes.
- Fix PDF artefacts: de-hyphenation, repeated headers / footers and "page x of y", letter-spaced words ("S o f t w a r e"), and two-column text that has been interleaved (flagged).

**S3. Quality filters.**
- Keep: the current minimum of 300 characters (removes 1.5% of the sample) and the cap.
- Keep, with changes:
  - alpha-ratio filter (removes 0.9%), with the threshold re-tuned on a reviewed sample;
  - replace the ASCII-ratio language filter with a language-ID model, keeping English plus the languages we decide to support.
- Add:
  - non-resume detection: cover letters, certificates, blank templates, JDs uploaded as resumes;
  - repetition / boilerplate rules (duplicate-line ratio, top n-gram share);
  - keyword-stuffing detection.

**S4. PII and sensitive attributes.**
- **Redact:**
  - e-mails, phones, URLs and handles;
  - names: account name, parsed first / middle / last, and e-mail local-part tokens;
  - street addresses and postcodes, dates of birth;
  - national IDs (Aadhaar, PAN, SSN, CPF, NIN patterns).
- **Keep city and country:** location is a legitimate query constraint.
- **Never used as input or target:** gender, birth date, citizenship, marital status / religion lines, salary fields (see the open decision on rate), the fired flag, fraud scores, interview transcripts.
- **Audit:** 1,000 redacted documents per release, regex recall plus a manual look; target residual PII below 0.1% of documents.
- **Pseudonymise rather than blank (Turn 43).** To keep the model familiar with real resume layouts (a name header, contact line, LinkedIn link), replace each real value with a realistic fake one rather than a bare placeholder. Each person gets one consistent fake name, e-mail, phone and handle across all of their documents. The model learns where contact details sit and how to copy them out of the input during extraction, but it never sees or memorises a real person's details.
  - Placeholders (`[NAME]`, `[EMAIL]`) stay for the candidate block in relevance, so names cannot act as a proxy for gender or ethnicity when scoring.
  - **AI interview transcripts:** only 44 exist on the SSD (JOB-110870). Pseudonymised, they are usable as a small conversational slice; a larger transcript source would need its own consent check.
- **Implementation (Turn 44): `slm_data/pseudonymize.py`.** It is logic only: the source files are never modified, and the fake values exist only in the build outputs under `artifacts/data`.
  - **Deterministic fake identity per person:** generated with HMAC from a secret salt (`SLM_PSEUDO_SALT`) and the person key (`talent_id` now, the person group once the registry exists). Names come from region-matched pools chosen by country. The mapping is one-way: with the salt you can recompute a person's fake identity, but never get back to the real one.
  - **Check codes ("coding logic"):**
    - e-mails and profile handles end with a 5-character code (3 random + 2 check characters);
    - phones keep their country code and layout, and the last digit is a Luhn check digit;
    - `verify(kind, value)` tells whether a string is one of our pseudonyms.
  - **What is replaced or removed:**
    - known values from the database columns (name, e-mail, phone, LinkedIn / GitHub, parse e-mails / numbers / portfolio links), plus pattern-detected e-mails, phones and profile URLs;
    - initials before a surname ("R. Verma" becomes "N. Mehta");
    - other people's e-mails (referees) become generic `contact.<code>@…` addresses that are not counted as the candidate's;
    - identity numbers (Aadhaar, PAN, SSN, CPF) and dates of birth are removed.
  - **Two tests come with every build:**
    1. **Pseudonymiser recall.** `audit_leaks()` searches each output for every known real value. Documents with a hit are counted and dropped (`drop_on_leak`), and the counts go to `reports/resumes_stats.json` as `leak_*` and `docs_with_leak`.
    2. **Extraction accuracy.** The injected values become an exact `contact` field in the extraction target (`include_contact`). `evaluate.contact_scores` reports `contact_name_exact`, `contact_emails_f1`, `contact_phones_f1`, `contact_links_f1` and `contact_hallucination` (the share of predicted contact values not present in the input).
  - **Known gap:** third-party names (referees) are not detected without a name list or NER model. Their e-mails and phones are replaced, but the names remain.
  - **Tests:** `data_pipeline/tests/test_pseudonymize.py` and `test_resume_build_pii.py` use invented data only, and both pass.

**S5. Structured fields (work / education tables).**
- Map months from all languages; turn `"0"`, `""` and `"Invalid date"` into null.
- `end_year = 0` or null becomes current or unknown, decided with `is_current` and `end_month`.
- Out-of-range years become null; swapped dates are fixed when clearly swapped, else dropped.
- Drop items with an empty position or company.
- De-duplicate repeated items.
- Merge overlapping jobs before computing total years of experience.
- Map the 891 raw seniority values to ~10 canonical levels.
- Normalise CGPA to its scale.
- Link company and college names to KB ids through `company_id` / `university_id` where present, then alias tables.

**S6. Extraction targets.**
- **Sources:**
  - the OpenAI parse (3.5M);
  - the structured tables;
  - the GPT-5.x gold.
- **Keep an item** only when it is grounded in the resume text (company, position, degree and college strings found by fuzzy match; the existing `min_grounding` 0.5 rule extends to per-item grounding) and parse and table agree, or the item is grounded.
- **Skills** link to the corrected taxonomy (46,338 skills, low-confidence rows flagged); years per skill may not exceed career length.
  - Implemented in `slm_data/skill_taxonomy.py`: corrected columns win; `alias` rows point at their merge target; a short curated abbreviation list (k8s, ML, Postgres, JS, golang…) is added, while ambiguous ones (CV, TF, RN, ES) are left out on purpose.
  - Synonyms collapse in extraction targets ("ReactJS", "React.js", "React" → one skill).
  - `card()` writes knowledge text for High / Medium-confidence skills (category, domain, parent, typical roles) for the KB and pretraining.
  - The same index serves query understanding later.
- **Protected attributes** are removed from the target schema.
- Start with a few hundred thousand high-agreement targets, not all 3.5M.

**S7. Relevance labels.**
- Consolidate `eval_consolidated` and the `evaluation/` CSVs.
- Key by (query family, person group).
- Keep the existing weighted judge consensus (minimum agreement 0.75) and drop `is_error` rows.
- Keep judge provenance.
- Build the candidate block with the same 4,000-character view the judges saw.
- Balance per query (cap 4,000, at most 3:1 positives to negatives).
- Send low-agreement pairs to teacher re-labelling, not straight to training.

**S8. Decontamination.**
- 13-gram and MinHash checks of every training corpus (pretraining, external text, extraction, relevance, synthetic, KB text) against held-out resumes and dev / test queries.
- Hits are removed from the training side.

**S9. Pack and audit.** `leak_audit` must pass before packing (section 8).
- **Global de-duplication at pack time (Turn 53; all four groups since Turn 56):** builders de-duplicate per source and per WDC part. Pack then drops exact normalised-text duplicates *across* sources (`pack.global_dedupe_groups`), with an on-disk set.
- **Held-out near-duplicates (Turn 56):** the resumes of test / eval people (`resumes/heldout-*`, never trained on) and the held-out relevance candidates form a MinHash index; any training resume, ours or public (Kaggle, Djinni, resume-job-fit), at Jaccard ≥ 0.8 to one of them is dropped.
- **Val selection (Turn 56):** per group, the validation documents are the lowest stable hashes within the budget (estimated at 4 characters per token), so the val stream samples every source of the group, whatever order the sources are read in.
- **Tokenizer sample (Turn 53):** random shard files of every source (up to 64), reservoir-sampled, instead of the first shards only.
- **Deterministic builds:** row groups are merged in input order (ordered `imap`), so which copy of a duplicate survives no longer depends on worker timing.

## 6b. Token budget and model size (measured Turn 46)

> **Update (Turn 51): general text is now downloaded.**
> - **New total:** unique pretraining text is about **15–15.5B tokens**: Wikipedia 3.3B + FineWeb-Edu 3.5B + Cosmopedia 1.4B on top of the ≈7B domain text, plus ~0.05B of knowledge cards.
> - **Effect on model size:** **650M is now comfortable** (≈26B tokens with ≤2 passes over the domain text). **1B is feasible** (≈28–30B tokens), with a compute cost of ~3–6 weeks.
> - **350M recommendation:** a 12–15B-token stage 1 is now mostly fresh text.
> - **Still small:** the knowledge corpus (~50M tokens). Wikidata (to re-download) and the full ESCO would grow it.
>
> The table below is the Turn 46 estimate before general text.

**How this was measured:** characters come from the parquet metadata (uncompressed column sizes) and a 12 MB slice of a WDC part. Tokens use **≈4.0 characters per token** (typical for a 32K BPE on English resumes; ±10%). The `pack` step reports the exact counts once the tokenizer is trained.

**Unique pretraining text available now:**

| Source | Characters | Tokens (approx.) |
|---|---|---|
| Resumes, turing_candidate_dataset (5.54M texts, 23.7 GB; after cleaning, filters and de-duplication ≈ −10%) | 21.3 G | **5.3 B** |
| Resumes only in the older export (estimate; most of its 1.99M extra ids have no text) | ~0.9 G | ~0.2 B |
| Djinni jobs + profiles | 0.67 G | 0.17 B |
| resume-job-fit text (unique share ~25%), HF resume-JD, Kaggle, JD titles | ~0.2 G | ~0.05 B |
| WDC JobPosting, 6 of 14 parts on disk (12 MB slice: 5,208 unique English postings, 1.36 JD chars per compressed byte) | 2.9 G | 0.7 B |
| **Total now** | | **≈ 6.5 B** |
| WDC when all 14 parts are in (before cross-part de-duplication) | 6.6 G | 1.65 B → total ≈ **7.4 B** |
| General text (FineWeb-Edu / Cosmopedia / Wikipedia) | not downloaded | 0 (planned 10–30% of the mix) |

**Supervised (task) tokens.** These reuse resume text, so they are not new unique tokens.

| Task | Size | Tokens |
|---|---|---|
| Relevance | ~375K pairs × ~1.1K tokens, before splits; ~10% fewer with the `evaluation/` CSVs merged and held-out candidates removed | ≈ 0.4 B |
| Extraction | 3.5M parses available; plan 0.3–1M examples × ~1.4K tokens | 0.4–1.4 B |
| Domain classification | 670K T7 labels, 30% used for training | ≈ 0.2 B |

**Not new data:** the work / education tables (7 GB of `details`) and the parse JSON (21.6 GB) are labels and duplicates of the resume text.

**What size model this supports.** The rule of thumb is ~20 tokens per parameter (Chinchilla) for compute-optimal training. Small models deployed for cheap inference are usually trained well beyond that (SmolLM2, Qwen3). Repeating the same data for up to ~4 epochs is nearly as good as fresh data (Muennighoff et al., 2023). So ~7B unique domain tokens give about **26–28B useful tokens**, plus general text.

| Model | Tokens / param at ~20–28B | Fits the data? | Training time on R9700 + 9070 XT (6·N·D +15%; 58–100 effective TFLOPS, i.e. 15–26% MFU) |
|---|---|---|---|
| **350M** (`slm-350m`) | 57–80 | Comfortably | 20B tokens: **~5.5–10 days** |
| **650M** (`slm-650m`) | 40 | Yes, with 4 passes over the resumes | 26B tokens: **~13–24 days** |
| **1B** | 28 | Only with general text added (FineWeb-Edu) | 28B tokens: **~22–39 days** |
| >1.5B | <20 | No: data-limited (>4 passes) and too slow on two consumer GPUs | – |

**Recommendation:**
- Train **350M first**, raising stage 1 from today's 3B tokens (`stage1_pretrain.yaml: total_tokens: 3.0e9`) to **~12–15B** (≈2–3 passes over the resumes + all JD text), about 4–7 days.
- Grow to **650M** only if dev results justify it (the function-preserving growth path of Turn 28).
- **1B** needs the general-text download and about a month of compute.
- Relevance quality is limited more by the **labels** (~250K training pairs, ~165–280 queries) than by pretraining tokens. Teacher labelling and synthetic queries add more there than a bigger model does.

## 7. Which data feeds which stage

| Stage | Data (train split only) | Approximate size |
|---|---|---|
| Tokenizer | sample of train persons' resumes + public text | 400K docs |
| 1. Pretraining | canonical resume per train person (~5–6B tokens); JD text (Djinni 142K, WDC, med2425 JDs, Kaggle JDs); knowledge text (O*NET, BLS, Wikipedia subset, KB-rendered cards); general text (FineWeb-Edu / Cosmopedia) at the DATA_SOURCES mix | 15–25B tokens incl. repeats |
| 2. Extraction SFT | high-agreement grounded targets from S6 | 0.3–1M examples |
| 3. Relevance SFT | consensus pairs (train families × train persons), think variants, teacher soft labels, query-grouped ranking batches; a small share of external JD pairs | ~250K pairs + external |
| Eval | val, dev (unseen query families), dev_cand (unseen candidates), locked test / test_cand, extraction gold, curated sets, JOB-110870 case study, human gold set, external benchmarks | – |

**How the trainer reads pretraining text (Turn 53).** Each epoch lays a grid of windows over every group's stream with a fresh random offset and shuffles it with a seed shared by both GPUs; each GPU reads every second window. Within an epoch no window is read twice and the two GPUs never read the same window (before: random offsets with replacement, independently per GPU). About 80% of windows are moved back to the nearest document start (at most half a window).

The work / education tables are not added as separate pretraining text (their `details` duplicate the resume). They serve as extraction labels and KB input.

## 8. Verification (automatic, fails the build)

`leak_audit` runs after every data build and writes `reports/leak_audit.json`. It checks:

- **Zero overlap:**
  - of person groups between train and test, across all tasks and the tokenizer sample;
  - of query families between train and dev / test;
  - between eval assets and any training file.
- **Near-duplicates:** the share of held-out resumes and queries with a MinHash or 13-gram hit in any training corpus, which must be 0 after S8.
- **Label health:** label balance, judge agreement and score distribution per split.
- **Coverage:** domain / country / role coverage of each split, so dev / test are not missing large slices (country is skewed: India 50%, US 11%).
- **PII:** residual PII rate on the redaction sample.
- **Memorisation probe, after pretraining:** the loss on held-out resumes vs train resumes. A large gap on the dev "seen candidate" slice but not on the strict test is expected; a gap of the wrong sign means leakage.

## 9. Decisions needed from you

1. **`fired.parquet`:** my recommendation is to not use it at all, neither as a label or feature nor to remove people; removing them would itself bias the data.
2. **Rate / salary:** several queries have rate limits ("Rate up to $100", "$40/hr"). Include the candidate's expected rate in the candidate block only if the production candidate block shows it.
3. **Languages to keep:** English only, or English plus Portuguese / Spanish (Brazil and LATAM are large).
4. **`+`-tagged addresses:** drop only `@turing.com` and explicit `test` / `qa` tags, or all `+` addresses.
5. **JOB-110870:** can you get the JD text? With it, the 1,165 candidates become a real end-to-end screening test (its labels come from the in-house AI screen, so treat them as a second opinion, not gold).
6. **Query families:** confirm the ~30 automatic merges once (I'll prepare the CSV).

## 10. Work order (status at Turn 46)

| # | Item | Status |
|---|---|---|
| 1 | Split registry: one split per talent_id, e-mail linking, eval-asset pool | **Done (code + synthetic tests).** Query families added in Turn 47 (dev / test = unseen families). Near-duplicate persons: held-out resumes are MinHash-decontaminated from all training resume text at pack time (Turn 56); a registry-level near-duplicate person grouping is still open |
| 2 | Ingest T7 (S0), identity / dedup (S1) | **Done:** T7 first, older export fills gaps, exact-text de-dup, held-out text hashes blocked at pack time |
| 3 | Cleaning S2–S4 and PII | **Done:** pseudonymisation with per-document leak audit. Open: language-ID model, non-resume detector, PDF artefact fixes beyond `clean_text` |
| 4 | Structured fields and extraction targets (S5–S6) | **Partly:** grounded parse targets + contact field + taxonomy de-dup. Open: work / education table cross-check and date repair |
| 5 | Relevance consolidation incl. `evaluation/` (S7) | **Done:** CSV judgments + score stability + risky flag |
| 6 | External data, decontamination (S8), audit and packing (S9) | **Done:** Djinni, resume-job-fit text, WDC, general text, audit, pack; cross-source dedupe of all groups and MinHash decontamination of all training resume text (ours + public) against held-out resumes at pack time (Turn 56). Open: near-duplicate checks of general web text against held-out text |
| 7 | First real data build (`python data_pipeline/data.py`) with pyarrow on a local disk | **Next** |
| 8 | Re-run baselines and our model on the new val / dev / test | After 7 (old numbers were measured on leaky splits) |

## 11. Snapshot of the T7 SSD (for when it is unplugged)

`training_data/t7_snapshot/` keeps:
- schemas with null counts;
- the export manifests;
- full-population category counts;
- sampled text and quality stats;
- work / education date checks;
- the JOB-110870 summary;
- sorted `talent_id` lists per table and all overlaps;
- the corrected skill taxonomy.

It holds no resume text and no personal data. Its README lists the key numbers. The scripts that produced it are in `data_pipeline/scripts/t7_snapshot/`.
