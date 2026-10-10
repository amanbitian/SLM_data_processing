"""Consistent, verifiable pseudonymisation of candidate PII (Turn 44).

Real names, e-mails, phones and profile URLs are replaced by realistic FAKE values so that the model
sees normal resume layouts (and learns to copy contact details during extraction) without ever
seeing or memorising a real person's details.

"Coding logic": every fake value is generated deterministically from HMAC(secret salt, person key)
and carries a check code, so we can tell later, from the value alone, whether a string is one of OUR
pseudonyms:
  * e-mail   local part ends with a 5-char code  (3 random base36 chars + 2 check chars)
  * LinkedIn / GitHub / portfolio handles end with the same kind of 5-char code
  * phone    keeps the original layout and country code; the last digit is a Luhn check digit over
             all digits of the number
  * names    are drawn from region-matched name pools (no code: names must look natural); they are
             verified through `profile(person_key)`, which recomputes the person's fake identity.

That gives three tests for free:
  1. pseudonymiser recall  - `audit_leaks()` searches the output for every KNOWN real value
                              (from the database columns) -> any hit is a detection miss;
  2. extraction accuracy   - the injected fake values are exact ground truth for contact fields;
  3. hallucination check   - a value the model outputs that is not in its input, or fails
                              `verify()`, was invented (or is leaked real data).

Same person -> same fake identity in every document (keyed by person group / talent_id).
The salt must stay secret (env SLM_PSEUDO_SALT): with the salt and a person key one can recompute the
fake identity, but never the real one (the mapping is one-way).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .pii import PHONE, _phone_sub

B36 = "0123456789abcdefghijklmnopqrstuvwxyz"

# --------------------------------------------------------------------------- name pools (common names)
_POOLS = {
    "south_asia": (
        "Aarav Aditi Akash Amit Ananya Anil Arjun Deepa Divya Farhan Gaurav Imran Kavya Kiran Manish Meera Neha "
        "Nikhil Pooja Rahul Ravi Ritu Rohan Sana Sanjay Shreya Sneha Suresh Tanvi Varun Vikram Zara Ayesha Bilal "
        "Hamza Sadia Usman Nadia Tariq Rakib Nusrat Sujan Anjali",
        "Agarwal Bhatt Chopra Das Desai Gupta Iyer Jain Joshi Kapoor Khan Kumar Malik Mehta Menon Mishra Nair "
        "Patel Pillai Rao Reddy Saxena Shah Sharma Singh Sinha Trivedi Verma Qureshi Siddiqui Chaudhry Hossain "
        "Rahman Karki Perera Fernando Banerjee Ghosh"),
    "lusophone": (
        "Ana Beatriz Bruno Camila Carlos Daniel Eduardo Fernanda Gabriel Gustavo Helena Isabela Joao Julia Larissa "
        "Leonardo Lucas Luiza Marcos Mariana Mateus Paula Pedro Rafael Renata Rodrigo Thiago Vinicius Leticia Caio",
        "Almeida Alves Araujo Barbosa Cardoso Carvalho Costa Dias Ferreira Gomes Lima Lopes Martins Melo Moreira "
        "Nascimento Oliveira Pereira Ribeiro Rocha Rodrigues Santos Silva Souza Teixeira Vieira Correia Monteiro"),
    "hispanic": (
        "Alejandro Andrea Camilo Carolina Diego Daniela Esteban Fernando Gabriela Javier Jorge Juan Laura Lucia "
        "Luis Manuel Maria Mateo Natalia Pablo Paola Ricardo Santiago Sofia Valentina Ximena Andres Martina",
        "Castillo Castro Diaz Flores Garcia Gomez Gonzalez Gutierrez Hernandez Jimenez Lopez Martinez Mendoza "
        "Morales Moreno Navarro Ortiz Perez Ramirez Ramos Reyes Rodriguez Romero Ruiz Sanchez Torres Vargas Herrera"),
    "africa": (
        "Abena Adaeze Ade Amara Chidi Chinedu Ebuka Efua Emeka Esi Fatima Femi Grace Ifeoma Joseph Kwame Kofi "
        "Mercy Musa Ngozi Nnamdi Olu Samuel Tendai Wanjiru Kamau Otieno Achieng Amina Yaw",
        "Adeyemi Afolabi Agbaje Appiah Asante Bello Chukwu Eze Kamau Mensah Mutua Mwangi Ndlovu Nwosu Obi Ochieng "
        "Odhiambo Ogunleye Okafor Okeke Okonkwo Onyango Osei Owusu Tembo Wanjiku Adebayo Boateng Kariuki Abubakar"),
    "arabic": (
        "Ahmed Aisha Ali Amira Hassan Hana Ibrahim Karim Khaled Layla Mahmoud Mariam Mohamed Mona Mostafa Nour "
        "Omar Rania Salma Samir Tarek Yasmin Youssef Zainab Hadi Rami Dina Sara",
        "Abdallah Abdelaziz Amin Aziz Farouk Fathy Gamal Hamdi Hassan Ibrahim Kamal Khalil Mansour Moussa Nasser "
        "Radwan Saad Salem Sayed Shaker Soliman Taha Youssef Zaki Haddad Khoury Mahmoud Saleh"),
    "turkish": (
        "Ahmet Ayse Burak Can Cem Deniz Elif Emre Esra Fatma Hakan Irem Kaan Kerem Mehmet Merve Murat Ozan Selin "
        "Serkan Tolga Zeynep Yusuf Ece Baris",
        "Acar Aksoy Arslan Aydin Celik Demir Dogan Erdem Gunes Kaya Kilic Koc Kurt Ozdemir Ozturk Polat Sahin "
        "Tekin Yildirim Yildiz Yilmaz Aslan Cetin Korkmaz"),
    "east_se_asia": (
        "Adi Agus Bayu Dewi Dian Eka Fitri Hendra Indah Putri Rizki Siti Wahyu Anh Minh Linh Huy Thao Tuan "
        "Jose Mark Joy Angelica Aiko Haruto Wei Jun Mei Hao Ji-woo Min-jun",
        "Santoso Wijaya Hidayat Saputra Kusuma Pratama Nguyen Tran Le Pham Hoang Vu Dela Cruz Reyes Bautista "
        "Tanaka Sato Suzuki Chen Wang Zhang Liu Kim Park Lee Tan Lim Ong"),
    "anglo_euro": (
        "Adam Alex Amelia Anna Ben Chloe Daniel David Emily Emma Ethan Hannah Jack James Jessica Julia Kate Laura "
        "Liam Lucas Maria Matthew Michael Mia Noah Olivia Peter Ryan Sarah Sophie Thomas Tom Anna Lukas Marta Piotr",
        "Anderson Baker Brown Clark Davies Evans Fischer Harris Hughes Jackson Johnson Jones Kowalski Lewis Martin "
        "Miller Moore Muller Nowak Roberts Robinson Schmidt Smith Taylor Thomas Thompson Walker White Williams Wilson"),
}
POOLS = {k: (f.split(), l.split()) for k, (f, l) in _POOLS.items()}

_REGION = {}
for _r, _cs in {
    "south_asia": "India Pakistan Bangladesh Nepal Sri_Lanka Bhutan Maldives Afghanistan",
    "lusophone": "Brazil Portugal Angola Mozambique Cape_Verde",
    "hispanic": "Argentina Mexico Colombia Peru Chile Spain Venezuela Ecuador Bolivia Uruguay Paraguay Guatemala "
                "Costa_Rica Dominican_Republic Honduras El_Salvador Nicaragua Panama Cuba Puerto_Rico",
    "africa": "Nigeria Kenya Ghana Uganda Ethiopia South_Africa Tanzania Rwanda Zambia Zimbabwe Cameroon Senegal "
              "Ivory_Coast Malawi Botswana Namibia",
    "arabic": "Egypt Saudi_Arabia United_Arab_Emirates Morocco Tunisia Algeria Jordan Lebanon Iraq Syria Qatar "
              "Kuwait Oman Bahrain Libya Sudan Palestine Yemen",
    "turkish": "Turkey Azerbaijan",
    "east_se_asia": "Indonesia Malaysia Philippines Vietnam Thailand Singapore Japan China South_Korea Taiwan "
                    "Hong_Kong Cambodia Myanmar",
}.items():
    for _c in _cs.split():
        _REGION[_c.replace("_", " ").lower()] = _r

FREEMAIL = {"gmail.com", "googlemail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "hotmail.com", "outlook.com",
            "live.com", "icloud.com", "me.com", "aol.com", "protonmail.com", "proton.me", "rediffmail.com",
            "yandex.com", "mail.com", "gmx.com", "zoho.com", "msn.com", "hotmail.co.uk", "yahoo.com.br",
            "bol.com.br", "uol.com.br", "outlook.in", "qq.com", "163.com"}
_DOMAIN_WORDS = "blue nova apex terra lumen orbit pixel cedar quartz vertex harbor maple summit atlas prism".split()

# common words that are also names: only replaced when capitalised exactly like a name
NAME_STOPWORDS = {
    "will", "grace", "joy", "hope", "rich", "king", "young", "long", "mark", "bill", "dean", "bishop", "may", "june",
    "april", "august", "faith", "rose", "page", "hunter", "carter", "mason", "cook", "smith", "baker", "summer",
    "sky", "star", "love", "gold", "white", "black", "brown", "green", "gray", "grey", "bell", "ward", "case",
    "chance", "art", "sun", "son", "lead", "rock", "stone", "wood", "hill", "lake", "field", "ray", "max", "don",
    "drew", "earl", "frank", "jack", "pat", "rob", "sam", "sue", "ted", "victor", "major", "doctor", "senior",
    "junior", "data", "dev", "net", "java", "ruby", "python", "swift",
}
SOCIAL_HOSTS = ("linkedin.com", "github.com", "gitlab.com", "behance.net", "dribbble.com", "medium.com",
                "kaggle.com", "leetcode.com", "stackoverflow.com", "twitter.com", "x.com", "facebook.com",
                "instagram.com", "hackerrank.com", "codechef.com", "codeforces.com", "bitbucket.org")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+\s?@\s?[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+|\b(?:" + "|".join(re.escape(h) for h in SOCIAL_HOSTS)
                    + r")/[^\s<>()\"']*", re.I)
# identity numbers and dates of birth: removed (never needed, never kept)
ID_RES = [
    re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b"),              # Aadhaar (written in 4-4-4 groups)
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                    # PAN
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                     # SSN
    re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),             # CPF
]
DOB_RE = re.compile(r"(?i)\b(date\s+of\s+birth|d\.?\s?o\.?\s?b\.?|birth\s*date|born(?:\s+on)?)\s*[:\-]?\s*"
                    r"([0-3]?\d[./\- ](?:[01]?\d|[A-Za-z]{3,9})[./\- ](?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2}|"
                    r"[A-Za-z]{3,9}\.? [0-3]?\d,? (?:19|20)\d{2})")


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _luhn_digit(digits: str) -> str:
    """check digit d such that digits + d passes the Luhn test"""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def luhn_ok(digits: str) -> bool:
    return len(digits) >= 2 and _luhn_digit(digits[:-1]) == digits[-1]


@dataclass
class KnownPII:
    """Real values known from the database for ONE person (used to replace and to audit)."""
    names: list[str] = field(default_factory=list)        # full names / first / middle / last
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)         # linkedin / github / portfolio

    def name_tokens(self) -> list[str]:
        out = []
        for n in self.names:
            for t in re.split(r"[\s,]+", str(n or "")):
                t = t.strip(".-'")
                if len(t) >= 3 and t.replace("-", "").isalpha() and t.lower() not in out:
                    out.append(t.lower())
        return out


class Pseudonymizer:
    def __init__(self, salt: Optional[str] = None, allow_test_salt: bool = False):
        salt = salt or os.environ.get("SLM_PSEUDO_SALT")
        if not salt:
            if not allow_test_salt:
                raise RuntimeError("set SLM_PSEUDO_SALT (a long random secret, kept out of the repo)")
            salt = "TEST-ONLY-SALT"
        self.key = salt.encode()

    # ------------------------------------------------------------------ deterministic primitives
    def _h(self, *parts) -> bytes:
        return hmac.new(self.key, "\x1f".join(map(str, parts)).encode(), hashlib.sha256).digest()

    def _pick(self, seq, *parts):
        return seq[int.from_bytes(self._h(*parts)[:8], "big") % len(seq)]

    def _check(self, body: str) -> str:
        h = self._h("chk", body)
        return B36[h[0] % 36] + B36[h[1] % 36]

    def code(self, *parts) -> str:
        h = self._h("code", *parts)
        body = "".join(B36[b % 36] for b in h[:3])
        return body + self._check(body)

    def code_ok(self, code: str) -> bool:
        code = code.lower()
        return len(code) == 5 and all(c in B36 for c in code) and self._check(code[:3]) == code[3:]

    # ------------------------------------------------------------------ the fake identity of a person
    def profile(self, person_key, country: Optional[str] = None) -> dict:
        region = _REGION.get((country or "").strip().lower(), "anglo_euro")
        firsts, lasts = POOLS[region]
        first = self._pick(firsts, person_key, "first")
        last = self._pick(lasts, person_key, "last")
        middle = self._pick(firsts, person_key, "middle")
        if middle == first:
            middle = firsts[(firsts.index(first) + 1) % len(firsts)]
        return {"first": first, "middle": middle, "last": last, "region": region,
                "code": self.code(person_key, "id")}

    def fake_email(self, person_key, real: str, idx: int, prof: dict) -> str:
        dom = real.split("@")[-1].strip().lower()
        f, l = _ascii(prof["first"]), _ascii(prof["last"]).replace(" ", "")
        c = self.code(person_key, "email", idx)
        style = self._pick([0, 1, 2], person_key, "estyle", idx)
        local = (f"{f}.{l}.{c}", f"{f}{l}{c}", f"{f[0]}{l}.{c}")[style]
        if dom not in FREEMAIL:
            dom = f"{self._pick(_DOMAIN_WORDS, person_key, 'd1', dom)}{self._pick(_DOMAIN_WORDS, person_key, 'd2', dom)}.example"
        return f"{local}@{dom}"

    def fake_phone(self, person_key, real: str) -> str:
        digits = [c for c in real if c.isdigit()]
        if len(digits) < 2:
            return real
        keep = 0
        s = real.lstrip()
        if s.startswith("+"):        # keep the country code (1-3 digits before the first separator)
            m = re.match(r"\+\s*(\d{1,3})", s)
            keep = len(m.group(1)) if m else 0
        h = self._h("phone", person_key, "".join(digits))
        new = digits[:keep] + [str(h[i % len(h)] % 10) for i in range(keep, len(digits) - 1)]
        new.append(_luhn_digit("".join(new)))
        it = iter(new)
        return "".join(next(it) if c.isdigit() else c for c in real)

    def fake_url(self, person_key, real: str, prof: dict) -> str:
        low = real.lower()
        f, l = _ascii(prof["first"]), _ascii(prof["last"]).replace(" ", "")
        c = self.code(person_key, "url", low.split("/")[2] if "://" in low else low.split("/")[0])
        prefix = re.match(r"(?i)^(https?://)?(www\.)?", real).group(0)
        if "linkedin.com" in low:
            return f"{prefix}linkedin.com/in/{f}-{l}-{c}"
        for host in SOCIAL_HOSTS:
            if host in low:
                return f"{prefix}{host}/{f}{l[:1]}-{c}"
        return f"{prefix or 'https://'}{f}{l}-{c}.example"

    # ------------------------------------------------------------------ main entry
    def pseudonymize(self, text: str, known: KnownPII, person_key, country: Optional[str] = None) -> tuple[str, dict, dict]:
        """-> (new_text, contact ground truth (fake values present in new_text), stats)"""
        if not text:
            return text, {}, {}
        prof = self.profile(person_key, country)
        st = {"email": 0, "phone": 0, "url": 0, "name": 0, "id": 0, "dob": 0}
        contact = {"emails": [], "phones": [], "links": []}
        t = text

        # 1) identity numbers and dates of birth -> removed
        for rx in ID_RES:
            t, n = rx.subn("[ID]", t)
            st["id"] += n
        t, n = DOB_RE.subn(lambda m: m.group(1) + ": [DOB]", t)
        st["dob"] += n

        # 2) e-mails (known ones keep their index so all documents of a person agree)
        known_emails = [e.strip().lower() for e in known.emails if e and "@" in e]

        name_toks = known.name_tokens()

        def _email(m):
            raw = re.sub(r"\s", "", m.group(0))
            low = raw.lower()
            local = low.split("@")[0]
            st["email"] += 1
            if low in known_emails or any(tok in local for tok in name_toks if len(tok) >= 4):
                idx = known_emails.index(low) if low in known_emails else 100 + int.from_bytes(self._h(low)[:2], "big")
                fake = self.fake_email(person_key, raw, idx, prof)
                key = "emails"
            else:   # someone else's address (referee, employer): generic fake, not the candidate's contact
                dom = low.split("@")[-1]
                if dom not in FREEMAIL:
                    dom = f"{self._pick(_DOMAIN_WORDS, 'x1', dom)}{self._pick(_DOMAIN_WORDS, 'x2', dom)}.example"
                fake = f"contact.{self.code(person_key, 'xemail', low)}@{dom}"
                key = "other_emails"
            contact.setdefault(key, [])
            if fake not in contact[key]:
                contact[key].append(fake)
            return fake
        t = EMAIL_RE.sub(_email, t)

        # 3) URLs: social / profile links always; any other URL only if it contains the person's name

        def _url(m):
            raw = m.group(0).rstrip(".,;:")
            tail = m.group(0)[len(raw):]
            low = raw.lower()
            if any(h in low for h in SOCIAL_HOSTS) or any(tok in low for tok in name_toks if len(tok) >= 4) \
                    or any(u and u.lower().strip("/") in low for u in known.urls):
                fake = self.fake_url(person_key, raw, prof)
                st["url"] += 1
                if fake not in contact["links"]:
                    contact["links"].append(fake)
                return fake + tail
            return m.group(0)
        t = URL_RE.sub(_url, t)

        # 4) phones (known or pattern-detected; dates / years are left alone by _phone_sub)
        def _phone(m):
            if _phone_sub(m) == m.group(0):
                return m.group(0)
            raw = m.group(0)
            lead = len(raw) - len(raw.lstrip())
            fake = raw[:lead] + self.fake_phone(person_key, raw[lead:])
            st["phone"] += 1
            if fake.strip() not in contact["phones"]:
                contact["phones"].append(fake.strip())
            return fake
        t = PHONE.sub(_phone, t)

        # 5) names: map every real name token to a fake token, keep case; initials follow the token
        mapping = self._name_map(known, prof)
        for real_tok, fake_tok in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            stop = real_tok in NAME_STOPWORDS
            pat = re.compile(rf"(?<![\w@./-]){re.escape(real_tok)}(?![\w@-])", re.I)

            def _rep(m, fake_tok=fake_tok, stop=stop):
                s = m.group(0)
                if stop and not (s[:1].isupper()):
                    return s                    # "will", "grace" as ordinary words stay
                st["name"] += 1
                if s.isupper():
                    return fake_tok.upper()
                if s[:1].isupper():
                    return fake_tok[:1].upper() + fake_tok[1:]
                return fake_tok.lower()
            t = pat.sub(_rep, t)
        # initials of the real first name in front of the (now fake) last name: "R. Verma" -> "N. Mehta"
        for n in known.names:
            toks = [x for x in re.split(r"[\s,]+", str(n or "")) if x.strip(".")]
            if len(toks) >= 2 and toks[0][:1].isalpha():
                t, k = re.subn(rf"\b{re.escape(toks[0][0].upper())}\.(\s*)(?={re.escape(prof['last'])}\b)",
                               lambda m: prof["first"][0].upper() + "." + m.group(1), t)
                st["name"] += k
        full = f"{prof['first']} {prof['last']}"
        if re.search(rf"(?i)\b{re.escape(prof['first'])}\b", t) and re.search(rf"(?i)\b{re.escape(prof['last'])}\b", t):
            contact["name"] = full
        contact = {k: v for k, v in contact.items() if v}
        return t, contact, st

    def _name_map(self, known: KnownPII, prof: dict) -> dict[str, str]:
        """real token (lower) -> fake token. The first token of a full name maps to the fake first name,
        the last token to the fake last name, everything in between to the fake middle name."""
        m: dict[str, str] = {}
        for n in known.names:
            toks = [x.strip(".-'") for x in re.split(r"[\s,]+", str(n or "")) if x.strip(".-'")]
            toks = [x for x in toks if len(x) >= 3 and x.replace("-", "").isalpha()]
            if not toks:
                continue
            if len(toks) == 1:
                m.setdefault(toks[0].lower(), prof["first"])
                continue
            m.setdefault(toks[0].lower(), prof["first"])
            m.setdefault(toks[-1].lower(), prof["last"])
            for x in toks[1:-1]:
                m.setdefault(x.lower(), prof["middle"])
        return m

    # ------------------------------------------------------------------ checks
    def verify(self, kind: str, value: str) -> bool:
        """Is `value` one of our pseudonyms (by its check code)?"""
        v = (value or "").strip()
        if kind == "email":
            local = v.split("@")[0].lower()
            return self.code_ok(re.split(r"[.]", local)[-1][-5:])
        if kind == "phone":
            return luhn_ok("".join(c for c in v if c.isdigit()))
        if kind in ("url", "link", "linkedin", "github"):
            tail = re.split(r"[-/]", v.lower().rstrip("/").replace(".example", ""))[-1]
            return self.code_ok(tail[-5:])
        raise ValueError(kind)


def audit_leaks(text: str, known: KnownPII) -> dict:
    """Which KNOWN real values are still present in `text` (pseudonymiser misses)."""
    out = {"email": 0, "phone": 0, "url": 0, "name": 0}
    low = (text or "").lower()
    digits = re.sub(r"\D", "", text or "")
    for e in known.emails:
        if e and e.strip().lower() in low:
            out["email"] += 1
    for p in known.phones:
        d = re.sub(r"\D", "", p or "")
        if len(d) >= 8 and d[-8:] in digits:
            out["phone"] += 1
    for u in known.urls:
        slug = (u or "").lower().rstrip("/").split("/")[-1]
        if len(slug) >= 5 and slug in low:
            out["url"] += 1
    for tok in known.name_tokens():
        if tok in NAME_STOPWORDS:
            if re.search(rf"(?<![\w@./-]){re.escape(tok.capitalize())}(?![\w@-])", text or ""):
                out["name"] += 1
        elif re.search(rf"(?i)(?<![\w@./-]){re.escape(tok)}(?![\w@-])", text or ""):
            out["name"] += 1
    return out


def _dedupe(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def known_from_row(row: dict, parsed: Optional[dict] = None) -> KnownPII:
    """Collect the real values of one candidate from the database columns (T7 and resume_data schemas)."""
    k = KnownPII()
    for c in ("name", "account_full_name", "full_name"):
        if row.get(c):
            k.names.append(str(row[c]))
    for c in ("email", "account_email"):
        if row.get(c):
            k.emails.append(str(row[c]))
    for c in ("phone_number", "account_phone"):
        if row.get(c):
            k.phones.append(str(row[c]))
    for c in ("linkedin_url", "github_url"):
        if row.get(c):
            k.urls.append(str(row[c]))
    if parsed:
        for c in ("full_name", "first_name", "middle_name", "last_name"):
            if parsed.get(c):
                k.names.append(str(parsed[c]))
        for e in parsed.get("personal_emails") or []:
            k.emails.append(str(e))
        for p in parsed.get("personal_numbers") or []:
            k.phones.append(str(p))
        for c in ("linkedin_url", "github_url", "linkedin"):
            if parsed.get(c):
                k.urls.append(str(parsed[c]))
        for u in parsed.get("portfolio_links") or []:
            k.urls.append(str(u))
    k.names, k.emails, k.phones, k.urls = _dedupe(k.names), _dedupe(k.emails), _dedupe(k.phones), _dedupe(k.urls)
    return k
