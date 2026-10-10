# Data sources for the SLM (curated, Turn 30)

What we have, what's public, what each source is for, and how far it gets us towards the ~20B tokens a 1B model wants (the training repository's ARCHITECTURE.md §19, Turn 29).

**Licences:** each one must be confirmed by you or legal before use. The notes below are what the sources state as of October 2026.

## The short version

- **There is plenty of text.** Job postings, resumes, occupation and skill taxonomies, and company and college knowledge are abundant.
- **Labelled relevance judgements are scarce everywhere.** By that I mean query ↔ candidate pairs with a verdict.
  - The largest public sets are a Chinese 2019 competition set (AliYun) and a new retrieval benchmark (PJB: ~300 queries, ~2K positive judgements).
  - **Your own `training_data/evaluation/` folder is the biggest untapped label source:** **279 unique queries** in its CSVs versus the 165 in `eval_consolidated`.
  - So labels will come from your own data, teacher labelling and synthetic queries; public data supplies text and knowledge.

## Tier 0: your own data not used yet (highest value, already yours)

| Source (`training_data/…`) | What it is | Use |
|---|---|---|
| `evaluation/` (26 GB: 677 CSV, 1,239 JSON, 47 XLSX) | Eval and A/B runs (Gemini auto-relevancy, shortlist reports, sequence-of-search, prod eval). CSVs alone contain **279 unique queries** | **More relevance queries and judge verdicts and reasons.** Consolidate into `relevance/` (dedupe by (query, resume_id), keep judge provenance). Roughly +70% queries, which matters most for dev/test CIs |
| `Skill_taxonomy - skill taxonomy.csv` | **43,340 skills** with canonical name, master skill, parent, category, taxonomy level, industry, domain, function, workflow, task and job roles | **Skills KB:** synonyms, hierarchy, role mapping. Feeds entity linking, query parsing ("React" ⇒ frontend) and checklist evidence |
| `curated_dataset/` + `curated_dataset_35/` | Curated eval sets: relevance 296 (74 queries × 4 score bands, stable GPT verdicts), extraction 250, fraud 481 | A **clean extra eval**, and the seed of the human gold set |
| `id_universe/` | Exact counts: 375,566 unique pairs, 122,747 with a GPT-5.4 verdict; overlap files | Sampling frames for teacher labelling and the gold set |
| `resume_extraction/` (1,960 domain-labelled) and `Fraud/` (970) | Gold extraction / domain labels; fraud labels | Gold eval; a possible later fraud task |
| `resume_data/` (in use) | ~5.2M usable resumes ≈ **5B tokens**; ~2.7M OpenAI parses (measured, Turn 29) | Pretraining + extraction SFT |

## Tier 1: public, permissive, high value

| Source | Size / content | Licence (as stated) | Use |
|---|---|---|---|
| **Djinni Recruitment Dataset** (`lang-uk` on Hugging Face) | ~230K anonymised candidate profiles + ~150K job postings, English and Ukrainian (UNLP 2024) | MIT | Pretraining text; JD ↔ profile pairs for teacher-labelled relevance; query synthesis |
| **Web Data Commons, schema.org JobPosting** (Common Crawl extractions) | JobPosting markup on ~4.1M URLs across ~50K sites (2022); yearly releases since 2013 | Common Crawl terms of use; page content stays with the publishers | **Millions of JDs with structured fields** (title, skills, location, salary, employment type). These fields are free labels for the query/constraint parser and add JD pretraining text |
| **O*NET** database | ~1K occupations: tasks, skills, knowledge, tools/tech, tens of thousands of alternate titles | CC BY 4.0 (attribution; "O*NET" trademark) | Title normalisation, seniority/function, skills, "how jobs work" text |
| **ESCO** (EU) | ~3K occupations, ~14K skills, 28 languages, ISCO-mapped; CSV/RDF/API | Free download; licence page not retrieved, so confirm | Skill/occupation taxonomy, multilingual aliases |
| **BLS Occupational Outlook Handbook** | US occupation descriptions: duties, education, pay, outlook | Public domain (cite BLS) | Knowledge text about jobs and careers |
| **ROR** (Research Organization Registry) | ~125K research organisations incl. universities: names, aliases, acronyms, country | CC0 (GeoNames-derived location fields: CC BY 4.0) | **College KB:** aliases ("IITB"), country, type |
| **OpenAlex institutions** | ROR-linked institutions with research output | CC0 | College KB enrichment (research strength as a tier signal) |
| **Wikidata** | Companies (industry, HQ, parent, employees, stock exchange), universities, places (country → region membership, e.g. LATAM) | CC0 | **Company + college + geography KB** (aliases, parents, types) |
| **SEC EDGAR** `company_tickers.json` | All SEC-registered tickers ↔ CIK | US government public data | US listed companies |
| **Wikipedia** | Articles on companies, universities, occupations, places | CC BY-SA 4.0 / GFDL | Knowledge pretraining subset (company / college / occupation / geography pages) |
| **FineWeb-Edu** | 1.3T tokens (score ≥3) / 5.4T (score ≥2) of educational web text | ODC-By 1.0 (+ Common Crawl terms of use) | General-knowledge mix (10–30% of pretraining) |
| **Cosmopedia v2** | Synthetic textbooks / how-to (SmolLM mix) | ODC-By 1.0 (v1: Apache-2.0) | Reasoning-style general text |
| **TalentCLEF 2025** (Zenodo) | Job-title matching (EN/ES/DE/ZH) and title → ESCO skill prediction; real, anonymised, human-annotated | Public (check the Zenodo licence) | Eval/training for title normalisation and skills |
| **PJB benchmark** (2026) | ~300 queries, ~200K resumes, ~2K positive judgements, 6 domains, reasoning-type labels | Paper CC BY 4.0; check the data licence | **External relevance eval** close to our task (reasoning-aware person–job retrieval) |
| **resume-job-fit-merged-v1** (`med2425`, HF) | 93,733 resume + JD pairs (~825 MB), labels from one LLM (Qwen2.5-32B) | No licence on the card; upstream sets unverified | **Text only** (resumes + JDs for context; JDs as query-synthesis seeds). Do not trust the labels or its synthetic test split. De-duplicate by hash first |
| HF resume-JD-match (in use) | Labelled fit pairs | (in use) | Relevance SFT |

## Tier 2: usable only after a terms-of-service / licence check

| Source | Note |
|---|---|
| Kaggle LinkedIn job postings 2023–24 (~124K, arshkon) and similar | Licence unclear on the mirror; LinkedIn's terms of use may apply |
| Naukri / Indian job-board scrapes (Kaggle snapshots, Apify actors) | Job-board terms of use; good Indian-market coverage if allowed |
| QS / THE rankings | Copyrighted lists; use derived tier labels internally only after a check |
| NIRF (India) | Public rank lists on nirfindia.org per category (scrape/parse); institutional data PDFs per college |
| AliYun person–job fit (2019) | Chinese; the only large public labelled PJF set; low fit for our language mix |

## Tier 3: synthetic (we generate it; see the training repository's ARCHITECTURE.md §19)

- **Queries:** 20–50K, teacher-generated and stratified by category and phrasing, with parsed constraint labels (self-consistency + validator).
- **Relevance labels:** teacher-labelled (cross-fitted) for Djinni / WDC JD × our resumes. Hard negatives come from near-miss constraints.
- **Checklists:** reasoning built from spec × profile × KB cards.
- **Rephrased text:** rephrased resumes and JDs (reported 1.5–5× effective data), plus knowledge text rendered from the KB.

## Getting to ~20B tokens (estimates)

| Block | Tokens |
|---|---|
| Our resumes (5B unique, up to 4 epochs) | 5B unique / up to ~20B passes |
| WDC JobPostings + Djinni + other JDs | ~1–3B (estimate; depends on the dedupe) |
| Knowledge: Wikipedia subset + O*NET / ESCO / BLS + KB-rendered text | ~0.5–1.5B |
| FineWeb-Edu / Cosmopedia mix | as much as needed (e.g. 5–10B) |
| Rephrased resumes / JDs | ×1.5–5 effective on the domain part |

So ~15–25B tokens of useful pretraining is realistic. That is enough for 650M, and for 1B with the general-text mix.

**Measured on 2026-10-10 (Turn 46; details in `../data_for_training.md` §6b):** about 6.5B unique tokens are on disk now.
- resumes ≈ 5.5B;
- WDC (6 of 14 parts) ≈ 0.7B;
- Djinni ≈ 0.17B;
- other external ≈ 0.05B.

That rises to ≈ 7.4B with all of WDC. General text is not downloaded yet.

**Download status (updated Turn 51, after the full download run):**
- **Downloaded and wired into the pipeline:**
  - Djinni jobs and profiles;
  - WDC JobPosting (all 14 parts);
  - O*NET 31.0;
  - ROR v2.14;
  - OpenAlex institutions (143,455);
  - SEC tickers;
  - TalentCLEF Task A / B (training files → knowledge; test / validation kept for evaluation);
  - Wikipedia (31 of 41 shards, 7.3 GB);
  - FineWeb-Edu (4 files, 8 GB);
  - Cosmopedia v2 (4 files, 4.4 GB);
  - resume-job-fit.
- **Failed:**
  - **Wikidata** returned no rows (endpoint timeouts). The downloader is fixed: items with an English Wikipedia article only, smaller pages, a proper User-Agent, and empty results no longer marked complete. Rerun `--only wikidata`.
  - **BLS** returned HTTP 403 (it blocks scripts). Save the XML from a browser; not used yet.
- **Unique tokens now:** ≈15–15.5B (`../data_for_training.md` §0 and §6b).

Earlier status (Turn 46):
- **Present:** Djinni jobs and profiles, resume-job-fit (med2425), WDC JobPosting parts 0–5 (part 6 in progress).
- **Not yet downloaded:** O*NET, BLS, ROR, OpenAlex, Wikidata, SEC, TalentCLEF, Wikipedia, FineWeb-Edu, Cosmopedia.

## Downloading Tier 1 (Turn 31)

```bash
python3 data_pipeline/scripts/download_external_data.py --list
python3 data_pipeline/scripts/download_external_data.py --contact you@example.com        # everything, into training_data/external_data_1010
python3 data_pipeline/scripts/download_external_data.py --only djinni_jobs,djinni_cvs,onet --dry-run
```

It is resumable and writes `MANIFEST.md` in the destination. The large general-text sources are capped by default (FineWeb-Edu 10 GB, Cosmopedia 5 GB, Wikipedia 8 GB; raise with `--fineweb-gb` etc.). `--contact` goes only to sec.gov, which requires it.

Manual: ESCO (the download link is e-mailed after a form) and PJB (find the data link in the paper). Tier 2 stays undownloaded until its terms are checked.

## Order of work

1. Consolidate `evaluation/` queries and verdicts, and load `Skill_taxonomy`. This is the biggest gain, needs no licence work, and is ours. **Done in code (Turn 44–45):** the `relevance-scan` CSV reader and `skill_taxonomy.py` with the corrected taxonomy.
2. KB v0: Wikidata companies + ROR/OpenAlex colleges + Skill_taxonomy + O*NET titles.
3. JD corpus: Djinni + WDC JobPosting (dedupe, language filter, PII scrub); structured fields become parser labels. **Done in code (Turn 45):** `external_more.py`.
4. General mix: FineWeb-Edu sample + Wikipedia knowledge subset.
5. External evals: PJB, TalentCLEF, curated_dataset.
6. Synthetic queries / labels / rephrasing with the teacher.
