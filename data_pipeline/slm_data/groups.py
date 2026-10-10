"""Pretraining text groups (Turn 51). Every pretraining document belongs to one group; pack writes one token
stream per group (packed/pretrain_<group>_{train,val}.bin) and the trainer mixes the groups with explicit
weights (pack.pretrain_mix, or a task's `mix` / `stream`), so e.g. 8B tokens of general text cannot drown
5.5B tokens of resumes, and the small knowledge corpus (~50M tokens) is not repeated dozens of times.

  resume     our resumes, Kaggle resumes, Djinni candidate profiles, resume-job-fit resumes
  jobs       job descriptions: Djinni, WDC JobPosting, Kaggle JD titles, resume-job-fit JDs
  knowledge  KB text rendered from O*NET, ESCO / TalentCLEF, ROR, OpenAlex, SEC, the skill taxonomy, and the
             Wikipedia articles whose title is a known company / university / occupation
  general    Wikipedia (rest), FineWeb-Edu, Cosmopedia v2
"""
GROUPS = ("resume", "jobs", "knowledge", "general")

SOURCE_GROUP = {
    "resume": "resume", "kaggle_resume": "resume", "djinni_profiles": "resume", "resume_job_fit_resume": "resume",
    "job_desc": "jobs", "djinni_jobs": "jobs", "wdc_jobposting": "jobs", "resume_job_fit_jd": "jobs",
    "onet": "knowledge", "esco_talentclef": "knowledge", "ror": "knowledge", "openalex": "knowledge",
    "sec": "knowledge", "wikidata": "knowledge", "skill_taxonomy": "knowledge", "wikipedia_kb": "knowledge",
    "wikipedia": "general", "fineweb_edu": "general", "cosmopedia_v2": "general",
}


def group_of(rec: dict) -> str:
    """Group of a pretraining record. Turn 56: an unknown source or group is an ERROR, not silently `general`
    (a misspelled or new source would otherwise change the training mix unnoticed). Add new sources to
    SOURCE_GROUP or write an explicit valid `group`."""
    g = rec.get("group")
    if g in GROUPS:
        return g
    src = rec.get("source") or ""
    if g is None and src in SOURCE_GROUP:
        return SOURCE_GROUP[src]
    raise ValueError(f"pretraining record with unknown group {g!r} / source {src!r}: add it to "
                     f"slm_data/groups.py SOURCE_GROUP (groups: {GROUPS})")
