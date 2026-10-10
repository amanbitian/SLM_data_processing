"""End-to-end check of the resume worker with pseudonymisation + skill linking, and of the contact metrics.
Synthetic rows only (no real data, no parquet needed). Run:  SLM_PSEUDO_SALT=test python data_pipeline/tests/test_resume_build_pii.py"""
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "data_pipeline"), str(ROOT / "model")]
# contact metrics live in the training repository (slm_train.evaluate); checked only when it sits next to this one
TRAIN_REPO = Path(os.environ.get("SLM_TRAINING_REPO") or (ROOT.parent / "SLM_training"))
os.environ.setdefault("SLM_PSEUDO_SALT", "TEST-ONLY")

import yaml  # noqa: E402
from slm_data import build_resumes as B  # noqa: E402

RESUME = ("RAVI VERMA\nravi.verma1990@gmail.com | +91 98765 43210 | linkedin.com/in/ravi-verma-0a1b2c\n"
          "Backend engineer with 6 years of Python, ReactJS and React.js experience at Infosys and Wipro.\n"
          "Infosys - Senior Engineer - 2019 to 2024. Wipro - Engineer - 2016 to 2019. B.Tech, IIT Delhi.\n" * 3)
PARSED = {"first_name": "Ravi", "last_name": "Verma", "full_name": "Ravi Verma", "country": "India",
          "personal_emails": ["ravi.verma1990@gmail.com"], "personal_numbers": ["+91 98765 43210"],
          "skills": ["Python", "ReactJS", "React.js", "React"],
          "experiences": [{"title": "Senior Engineer", "company": "Infosys", "starts_at": "2019-01"},
                          {"title": "Engineer", "company": "Wipro", "starts_at": "2016-01", "ends_at": "2019-01"}],
          "education": [{"degree_name": "B.Tech", "school": "IIT Delhi"}]}


def fake_rows(path, rg, cols):
    row = {c: None for c in cols}
    row.update({"talent_id": 4242, "resume_plain": RESUME, "parsed_resume_data": json.dumps(PARSED),
                "name": "Ravi Verma", "email": "ravi.verma1990@gmail.com", "phone_number": "+919876543210",
                "linkedin_url": "https://www.linkedin.com/in/ravi-verma-0a1b2c", "country": "India"})
    return {c: [row[c]] for c in cols}


def main():
    cfg = yaml.safe_load(open(ROOT / "data_pipeline" / "configs" / "data_config.yaml"))
    tx = cfg["sources"]["skill_taxonomy_csv"]
    if not Path(tx).exists():   # alternate mount path
        alt = "/sessions/awesome-happy-bardeen/mnt/New Volume/turing_candidate_dataset/Skill_Taxonomy_Corrected.csv"
        cfg["sources"]["skill_taxonomy_csv"] = alt if Path(alt).exists() else None
    B.read_row_group = fake_rows
    B._init(cfg, None)
    out = B._process_row_group(("x.parquet", 0))
    st = out["stats"]
    print("stats:", dict(st))
    assert st["docs_with_leak"] == 0 and len(out["items"]) == 1, st
    it = out["items"][0]
    txt = it["text"]
    for real in ("ravi.verma1990@gmail.com", "98765", "ravi-verma-0a1b2c", "Ravi", "Verma", "RAVI"):
        assert real not in txt, real
    tgt = it["target"]
    print("target:", json.dumps(tgt)[:600])
    assert list(tgt)[0] == "contact" and tgt["contact"]["emails"] and tgt["contact"]["phones"]
    assert tgt["contact"]["emails"][0] in txt and tgt["contact"]["name"].split()[0].lower() in txt.lower()
    if B._TX is not None:     # ReactJS / React.js / React collapse to one skill
        names = [s["name"] for s in tgt["skills"]]
        assert names == ["Python", "ReactJS"], names
    assert it["meta"]["pseudo_names"][0] == tgt["contact"]["name"].split()[0]

    # contact metrics (evaluate.py imports torch at module level -> stub it for this CPU-only check)
    if not (TRAIN_REPO / "training" / "slm_train").is_dir():
        print(f"contact metrics: skipped (training repository not found at {TRAIN_REPO}; set SLM_TRAINING_REPO)")
        print("\nRESUME BUILD + PII TESTS PASSED")
        return
    sys.path.append(str(TRAIN_REPO / "training"))
    sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=lambda: (lambda f: f)))
    from slm_train.evaluate import contact_scores
    gold = tgt["contact"]
    perfect = contact_scores(gold, gold, txt)
    assert perfect["contact_emails_f1"] == 1.0 and perfect["contact_hallucination"] == 0.0, perfect
    wrong = dict(gold, emails=["someone.else@gmail.com"])
    sc = contact_scores(wrong, gold, txt)
    assert sc["contact_emails_f1"] == 0.0 and sc["contact_hallucination"] > 0, sc
    print("contact metrics:", perfect, sc)
    print("\nRESUME BUILD + PII TESTS PASSED")


if __name__ == "__main__":
    main()
