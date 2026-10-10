"""Tests for slm_data.pseudonymize on an invented resume (no real person's data).
Run:  python data_pipeline/tests/test_pseudonymize.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm_data.pseudonymize import KnownPII, Pseudonymizer, audit_leaks, known_from_row, luhn_ok  # noqa: E402

RESUME = """RAVI KUMAR VERMA
Email: ravi.verma1990@gmail.com | Phone: +91 98765 43210 | Alt: 080-2345-6789
LinkedIn: https://www.linkedin.com/in/ravi-verma-0a1b2c  GitHub: github.com/raviverma90
Portfolio: https://raviverma.dev/projects
Date of Birth: 14/03/1990   Aadhaar: 2345 6789 0123   PAN: ABCDE1234F
Summary: R. Verma is a backend engineer who will lead API work. Ravi built systems at Infosys (www.infosys.com).
Experience: Infosys, Senior Engineer, 2019 - 2021; Wipro 2016-2019. Contact ravi for details.
Reference: Mr. Anil Mehta, anil.mehta@wipro.com
"""
ROW = {"name": "Ravi Kumar Verma", "email": "ravi.verma1990@gmail.com", "phone_number": "+919876543210",
       "linkedin_url": "https://www.linkedin.com/in/ravi-verma-0a1b2c", "github_url": "https://github.com/raviverma90"}
PARSED = {"first_name": "Ravi", "middle_name": "Kumar", "last_name": "Verma",
          "personal_emails": ["ravi.verma1990@gmail.com"], "personal_numbers": ["+91 98765 43210"],
          "portfolio_links": ["https://raviverma.dev/projects"]}


def main():
    ps = Pseudonymizer(allow_test_salt=True)
    known = known_from_row(ROW, PARSED)
    out, contact, st = ps.pseudonymize(RESUME, known, person_key=424242, country="India")
    print(out)
    print("contact:", contact)
    print("stats:", st)

    # 1) no known real value survives (this is the pseudonymiser-recall test)
    leaks = audit_leaks(out, known)
    assert sum(leaks.values()) == 0, leaks
    # identity numbers / DOB removed, non-PII URL and years kept, ordinary word "will" kept
    assert "2345 6789 0123" not in out and "ABCDE1234F" not in out and "14/03/1990" not in out
    assert "www.infosys.com" in out and "2019 - 2021" in out and "2016-2019" in out
    assert " will lead" in out
    # third-party e-mail (referee) is pseudonymised too, as a generic address, not as the candidate's contact
    assert "anil.mehta@wipro.com" not in out
    assert len(contact["emails"]) == 1 and contact["other_emails"][0].startswith("contact.")
    # the initial in "R. Verma" follows the fake first name
    assert f"{ps.profile(424242, 'India')['first'][0]}. {ps.profile(424242, 'India')['last']}" in out

    # 2) every injected value carries a valid check code
    for e in contact["emails"]:
        assert ps.verify("email", e), e
    for p in contact["phones"]:
        assert ps.verify("phone", p), p
    for u in contact["links"]:
        assert ps.verify("url", u), u
    assert not ps.verify("email", "ravi.verma1990@gmail.com")
    assert not ps.verify("url", ROW["linkedin_url"])
    # format kept: country code and grouping of the phone
    assert any(p.startswith("+91 ") and re.fullmatch(r"\+91 \d{5} \d{5}", p) for p in contact["phones"]), contact

    # 3) consistency: the same person gets the same identity in another document
    out2, contact2, _ = ps.pseudonymize("Ravi Verma - ravi.verma1990@gmail.com", known, 424242, "India")
    assert contact2["emails"][0] == contact["emails"][0], (contact2, contact)
    assert contact2.get("name") == contact.get("name")
    # a different person gets a different identity
    _, contact3, _ = ps.pseudonymize("Ravi Verma - ravi.verma1990@gmail.com", known, 777, "India")
    assert contact3["emails"][0] != contact["emails"][0]
    # different salt -> different identity (salt is the secret)
    _, contact4, _ = Pseudonymizer(salt="other").pseudonymize("ravi.verma1990@gmail.com", known, 424242, "India")
    assert contact4["emails"][0] != contact["emails"][0]

    # 4) name header in capitals is replaced keeping case, and the fake name is reported as ground truth
    prof = ps.profile(424242, "India")
    assert out.splitlines()[0] == f"{prof['first']} {prof['middle']} {prof['last']}".upper(), out.splitlines()[0]
    assert contact["name"] == f"{prof['first']} {prof['last']}"

    # 5) audit catches a planted miss
    assert audit_leaks(out + " ravi.verma1990@gmail.com", known)["email"] == 1
    assert luhn_ok("79927398713") and not luhn_ok("79927398710")
    print("\nALL PSEUDONYMIZE TESTS PASSED")


if __name__ == "__main__":
    main()
