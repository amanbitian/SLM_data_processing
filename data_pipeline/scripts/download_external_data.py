#!/usr/bin/env python3
"""Download the public datasets listed in data_pipeline/DATA_SOURCES.md (Tier 1).

Run it on a machine with normal internet access (your Mac / the GPU box):

    python3 data_pipeline/scripts/download_external_data.py --contact you@example.com
    python3 data_pipeline/scripts/download_external_data.py --list
    python3 data_pipeline/scripts/download_external_data.py --only djinni_jobs,onet --dry-run

Standard library only. Re-running is safe: finished files are skipped, partial
files resume (HTTP Range), and a source that fails does not stop the others.

Outputs go to  <dest>/<source>/...  (default: /Volumes/New Volume/training_data/external_data_1010)
plus  <dest>/MANIFEST.json  and  <dest>/MANIFEST.md  (status, size, URL, licence).

Licences are the ones the publishers state; confirm them before training on a source.
Not downloaded on purpose: Tier 2 sources (Kaggle LinkedIn / Naukri scrapes, QS/THE, NIRF,
AliYun) need a terms-of-service check or credentials first. ESCO needs an e-mailed link and
PJB's data location is not in its abstract: both are listed as MANUAL.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_DEST = "/Volumes/New Volume/training_data/external_data_1010"
UA = "slm-data-download/1.0 (research; resumable)"
HF = "https://huggingface.co"

# ----------------------------------------------------------------------------- http helpers


def _req(url, headers=None, method="GET"):
    h = {"User-Agent": UA}
    h.update(headers or {})
    return urllib.request.Request(url, headers=h, method=method)


def http_json(url, headers=None, tries=4, data=None):
    last = None
    for i in range(tries):
        try:
            r = _req(url, headers)
            if data is not None:
                r.data = data
            with urllib.request.urlopen(r, timeout=60) as f:
                return json.loads(f.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def remote_size(url, headers=None):
    try:
        with urllib.request.urlopen(_req(url, headers, "HEAD"), timeout=60) as f:
            n = f.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:  # noqa: BLE001
        return None


def fmt(n):
    n = float(n or 0)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


class Ctx:
    def __init__(self, a):
        self.dest = Path(a.dest)
        self.dry = a.dry_run
        self.contact = a.contact
        self.cap = {"fineweb_edu": a.fineweb_gb, "cosmopedia_v2": a.cosmopedia_gb, "wikipedia_en": a.wikipedia_gb}
        self.wiki_config = a.wiki_config
        self.fw_config = a.fineweb_config
        self.files = []  # (relative path, bytes) for the current source
        self.notes = []
        self.rows = {}   # file -> row count, where known (Wikidata)

    def out(self, src):
        p = self.dest / src
        if not self.dry:
            p.mkdir(parents=True, exist_ok=True)
        return p


def download(ctx, url, path, headers=None, expect=None):
    """Resumable download. Returns bytes on disk (or the expected size when --dry-run)."""
    path = Path(path)
    size = expect if expect is not None else remote_size(url, headers)
    if ctx.dry:
        print(f"    [dry] {url} -> {path}  ({fmt(size) if size else '?'})")
        ctx.files.append((str(path.relative_to(ctx.dest)), size or 0))
        return size or 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size) and path.stat().st_size > 0:
        print(f"    skip (complete) {path.name}")
        ctx.files.append((str(path.relative_to(ctx.dest)), path.stat().st_size))
        return path.stat().st_size
    part = path.with_name(path.name + ".part")
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        h = dict(headers or {})
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(_req(url, h), timeout=90) as f:
                if have and f.status != 206:  # server ignored Range: restart
                    have = 0
                mode = "ab" if have else "wb"
                total = (size or 0) or (int(f.headers.get("Content-Length", 0)) + have)
                t0, got = time.time(), have
                with open(part, mode) as o:
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        o.write(b)
                        got += len(b)
                        if time.time() - t0 > 5:
                            print(f"\r    {path.name}: {fmt(got)} / {fmt(total) if total else '?'}   ", end="", flush=True)
                            t0 = time.time()
            if size and part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != expected {size}")
            part.replace(path)
            print(f"\r    ok   {path.name}  {fmt(path.stat().st_size)}                    ")
            ctx.files.append((str(path.relative_to(ctx.dest)), path.stat().st_size))
            return path.stat().st_size
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # already complete
                part.replace(path)
                ctx.files.append((str(path.relative_to(ctx.dest)), path.stat().st_size))
                return path.stat().st_size
            if e.code in (401, 403, 404):
                raise RuntimeError(f"HTTP {e.code} for {url}") from e
            time.sleep(3 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            print(f"\n    retry {attempt + 1}: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def hf_list(repo, prefix=""):
    """List files (path, size) of a HF dataset repo under prefix, recursively."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(prefix)}?recursive=true&expand=false"
    items, out = http_json(url), []
    for it in items:
        if it.get("type") == "file":
            out.append((it["path"], int(it.get("size") or 0)))
    return out


def spread(files, seed: str = "slm-shards"):
    """Deterministic order that SAMPLES the whole source (Turn 56): with a byte cap, taking shards in name order
    gave e.g. the first 31/41 Wikipedia and 4/104 Cosmopedia shards (topic / time bias if upstream sharding is
    not random). Shards already on disk keep their place first, so a rerun with the same cap is stable."""
    import hashlib
    return sorted(files, key=lambda f: hashlib.sha1(f"{seed}:{f[0]}".encode()).hexdigest())


def hf_download(ctx, repo, files, dst_dir, cap_gb=None):
    if cap_gb is not None:
        have = {p.name for p in Path(dst_dir).glob("*")} if Path(dst_dir).exists() else set()
        files = [f for f in files if Path(f[0]).name in have] + [f for f in spread(files)
                                                                 if Path(f[0]).name not in have]
    total = 0
    for p, sz in files:
        if cap_gb is not None and total + sz > cap_gb * 1e9 and total > 0:
            ctx.notes.append(f"stopped at cap {cap_gb} GB ({fmt(total)} downloaded)")
            break
        url = f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(p)}"
        download(ctx, url, Path(dst_dir) / Path(p).name, expect=sz or None)
        total += sz


# ----------------------------------------------------------------------------- sources


def s_djinni_jobs(c):
    d = c.out("djinni_jobs")
    hf_download(c, "lang-uk/recruitment-dataset-job-descriptions-english", [("data/train-00000-of-00001.parquet", 145859589), ("README.md", 0)], d)


def s_djinni_cvs(c):
    d = c.out("djinni_candidate_profiles")
    hf_download(c, "lang-uk/recruitment-dataset-candidate-profiles-english", [("data/train-00000-of-00001.parquet", 237415736), ("README.md", 0)], d)


def s_resume_job_fit(c):
    # text only is the point (resumes + JDs); the Qwen2.5-32B labels are not trusted (see DATA_SOURCES.md)
    d = c.out("resume_job_fit_merged_v1")
    files = [("data/train-00000-of-00002.parquet", 0), ("data/train-00001-of-00002.parquet", 0), ("data/test-00000-of-00001.parquet", 0), ("README.md", 0)]
    hf_download(c, "med2425/resume-job-fit-merged-v1", files, d)
    c.notes.append("text use only; labels are single-LLM; many repeated resumes/JDs across pairs, so de-duplicate by hash before use")


def s_wdc_jobposting(c):
    base = "https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific/JobPosting/"
    d = c.out("wdc_jobposting_2024-12")
    for n in ["JobPosting_lookup.csv", "JobPosting_sample.txt", "JobPosting_domain_stats.csv"] + [f"part_{i}.gz" for i in range(14)]:
        download(c, base + n, d / n)


def s_onet(c):
    d = c.out("onet_31_0")
    last = None
    for name in ("db_31_0_text.zip", "db_31_0_excel.zip"):  # text = tab-delimited files; excel = same tables as .xlsx
        try:
            download(c, f"https://www.onetcenter.org/dl_files/database/{name}", d / name)
            return
        except RuntimeError as e:
            last = e
    raise last


def s_bls_ooh(c):
    d = c.out("bls_ooh")
    download(c, "https://www.bls.gov/ooh/xml-compilation.xml", d / "ooh-xml-compilation.xml", headers={"User-Agent": "Mozilla/5.0 (research download)"})


def _zenodo_download(c, query=None, community=None, dst="", max_gb=3.0, title_has=None):
    d = c.out(dst)
    if community:
        r = http_json(f"https://zenodo.org/api/communities/{community}/records?sort=newest&size=1")
    else:
        r = http_json("https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": query, "size": 10, "sort": "bestmatch"}))
    hits = r.get("hits", {}).get("hits", [])
    if title_has:
        hits = [h for h in hits if title_has.lower() in json.dumps(h.get("metadata", {})).lower()]
    if not hits:
        raise RuntimeError("no Zenodo record found")
    rec = hits[0]
    c.notes.append(f"Zenodo record {rec.get('id')}: {rec.get('metadata', {}).get('title')}  (licence: {rec.get('metadata', {}).get('license', {}).get('id')})")
    for f in rec.get("files", []):
        if f["size"] > max_gb * 1e9:
            c.notes.append(f"skipped {f['key']} ({fmt(f['size'])} > {max_gb} GB)")
            continue
        link = f.get("links", {}).get("self") or f"https://zenodo.org/records/{rec['id']}/files/{f['key']}?download=1"
        download(c, link, d / f["key"], expect=f["size"])


def s_ror(c):
    _zenodo_download(c, community="ror-data", dst="ror", max_gb=2.0)


def s_talentclef(c):
    _zenodo_download(c, query="TalentCLEF", dst="talentclef", max_gb=3.0, title_has="talentclef")


def s_openalex_institutions(c):
    d = c.out("openalex_institutions")
    out = d / "institutions.jsonl.gz"
    if c.dry:
        print("    [dry] OpenAlex API cursor paging -> institutions.jsonl.gz (~100K records)")
        return
    done = d / "institutions.DONE"
    if done.exists():
        print("    skip (complete) institutions.jsonl.gz")
        c.files.append((str(out.relative_to(c.dest)), out.stat().st_size))
        return
    cursor, n = "*", 0
    tmp = d / "institutions.jsonl.gz.part"
    with gzip.open(tmp, "wt", encoding="utf-8") as o:
        while cursor:
            r = http_json("https://api.openalex.org/institutions?" + urllib.parse.urlencode({"per-page": 200, "cursor": cursor}))
            for x in r["results"]:
                o.write(json.dumps(x, ensure_ascii=False) + "\n")
                n += 1
            cursor = r["meta"].get("next_cursor")
            if n % 5000 < 200:
                print(f"\r    {n} institutions", end="", flush=True)
            time.sleep(0.15)
    tmp.replace(out)
    done.write_text(str(n))
    print(f"\r    ok   {n} institutions")
    c.files.append((str(out.relative_to(c.dest)), out.stat().st_size))


WIKIDATA_QUERIES = {
    # direct instance-of only, and only items with an English Wikipedia article (the unrestricted classes have
    # millions of items and time out on the public endpoint); no ORDER BY (it forces a full sort per page)
    "companies": """SELECT ?item ?itemLabel ?alt ?industryLabel ?countryLabel ?hqLabel ?parentLabel ?inception ?employees ?website WHERE {{
  {{ SELECT ?item WHERE {{ ?item wdt:P31 wd:{cls} . ?article schema:about ?item ;
       schema:isPartOf <https://en.wikipedia.org/> . }} LIMIT {lim} OFFSET {off} }}
  OPTIONAL {{ ?item wdt:P452 ?industry. }} OPTIONAL {{ ?item wdt:P17 ?country. }} OPTIONAL {{ ?item wdt:P159 ?hq. }}
  OPTIONAL {{ ?item wdt:P749 ?parent. }} OPTIONAL {{ ?item wdt:P571 ?inception. }} OPTIONAL {{ ?item wdt:P1128 ?employees. }}
  OPTIONAL {{ ?item wdt:P856 ?website. }}
  OPTIONAL {{ ?item skos:altLabel ?alt. FILTER(LANG(?alt)='en') }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }}
}}""",
    "universities": """SELECT ?item ?itemLabel ?alt ?countryLabel ?cityLabel ?inception ?students ?website ?ror WHERE {{
  {{ SELECT ?item WHERE {{ ?item wdt:P31 wd:{cls} . ?article schema:about ?item ;
       schema:isPartOf <https://en.wikipedia.org/> . }} LIMIT {lim} OFFSET {off} }}
  OPTIONAL {{ ?item wdt:P17 ?country. }} OPTIONAL {{ ?item wdt:P131 ?city. }} OPTIONAL {{ ?item wdt:P571 ?inception. }}
  OPTIONAL {{ ?item wdt:P2196 ?students. }} OPTIONAL {{ ?item wdt:P856 ?website. }} OPTIONAL {{ ?item wdt:P6782 ?ror. }}
  OPTIONAL {{ ?item skos:altLabel ?alt. FILTER(LANG(?alt)='en') }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }}
}}""",
}
WIKIDATA_CLASSES = {"companies": ["Q4830453", "Q891723", "Q6881511", "Q783794"], "universities": ["Q3918", "Q875538", "Q62078547"]}  # business, public co., enterprise, company / university, public univ., ...


def s_wikidata(c):
    d = c.out("wikidata")
    if c.dry:
        print("    [dry] Wikidata SPARQL paging -> companies_*.json, universities_*.json")
        return
    for kind, classes in WIKIDATA_CLASSES.items():
        for cls in classes:
            out = d / f"{kind}_{cls}.jsonl.gz"
            if out.exists() and _gz_rows(out) == 0:            # an empty result from an earlier failed run
                out.unlink()
            if out.exists():
                print(f"    skip (complete) {out.name}")
                c.rows[out.name] = _gz_rows(out)
                c.files.append((str(out.relative_to(c.dest)), out.stat().st_size))
                continue
            off, lim, rows, failed = 0, 2000, 0, False
            ua = {"Accept": "application/sparql-results+json",
                  "User-Agent": f"slm-research-data/1.0 ({c.contact or 'no contact given'})"}  # Wikimedia UA policy
            tmp = out.with_name(out.name + ".part")
            with gzip.open(tmp, "wt", encoding="utf-8") as o:
                while True:
                    q = WIKIDATA_QUERIES[kind].format(cls=cls, lim=lim, off=off)
                    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q, "format": "json"})
                    try:
                        r = http_json(url, headers=ua, tries=5)
                    except RuntimeError as e:
                        c.notes.append(f"{kind} {cls} stopped at offset {off}: {e}")
                        failed = True
                        break
                    b = r["results"]["bindings"]
                    for x in b:
                        o.write(json.dumps({k: v["value"] for k, v in x.items()}, ensure_ascii=False) + "\n")
                    rows += len(b)
                    print(f"\r    {kind} {cls}: {rows} rows", end="", flush=True)
                    if len(b) == 0 or off > 400000:
                        break
                    off += lim
                    time.sleep(1.0)
            print()
            if rows == 0:                                     # keep nothing: a rerun must try again
                tmp.unlink(missing_ok=True)
                c.notes.append(f"{kind} {cls}: no rows (endpoint error or timeout); rerun later")
                continue
            if failed:   # Turn 56: a page sequence cut by an error is INCOMPLETE -> not promoted to the final name
                bad = out.with_name(out.name + ".incomplete")
                tmp.replace(bad)
                c.notes.append(f"{kind} {cls}: incomplete ({rows} rows before the error) kept as {bad.name}, "
                               f"not used; rerun to complete")
                continue
            c.rows[out.name] = rows
            tmp.replace(out)
            c.files.append((str(out.relative_to(c.dest)), out.stat().st_size))


def _gz_rows(path) -> int:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except (OSError, EOFError):
        return 0


def s_sec_tickers(c):
    if not c.contact:
        raise RuntimeError("SEC requires an identifying User-Agent: pass --contact you@example.com")
    d = c.out("sec_edgar")
    h = {"User-Agent": f"slm-research {c.contact}", "Accept-Encoding": "identity"}
    for n in ("company_tickers.json", "company_tickers_exchange.json"):
        download(c, f"https://www.sec.gov/files/{n}", d / n, headers=h)


def s_wikipedia(c):
    repo = "wikimedia/wikipedia"
    files = [f for f in hf_list(repo, c.wiki_config) if f[0].endswith(".parquet")]
    hf_download(c, repo, sorted(files), c.out("wikipedia_en"), cap_gb=c.cap["wikipedia_en"])
    c.notes.append(f"config {c.wiki_config}; shards sampled across the source (hash order) up to the cap")


def s_fineweb_edu(c):
    repo = "HuggingFaceFW/fineweb-edu"
    files = [f for f in hf_list(repo, c.fw_config) if f[0].endswith(".parquet")]
    hf_download(c, repo, sorted(files), c.out("fineweb_edu"), cap_gb=c.cap["fineweb_edu"])
    c.notes.append(f"config {c.fw_config}; capped; raise --fineweb-gb for more")


def s_cosmopedia(c):
    repo = "HuggingFaceTB/smollm-corpus"
    files = [f for f in hf_list(repo, "cosmopedia-v2") if f[0].endswith(".parquet")]
    hf_download(c, repo, sorted(files), c.out("cosmopedia_v2"), cap_gb=c.cap["cosmopedia_v2"])
    c.notes.append("cosmopedia-v2 config of smollm-corpus; capped")


# name: (function, licence, est size, description)
SOURCES = {
    "djinni_jobs": (s_djinni_jobs, "MIT", "0.15 GB", "Djinni job descriptions (HF lang-uk)"),
    "djinni_cvs": (s_djinni_cvs, "MIT", "0.24 GB", "Djinni candidate profiles (HF lang-uk)"),
    "resume_job_fit": (s_resume_job_fit, "none stated on the card; upstream datasets unverified", "0.3 GB", "med2425 resume-job-fit-merged-v1 (resume + JD text; labels not trusted)"),
    "wdc_jobposting": (s_wdc_jobposting, "Common Crawl terms of use; content belongs to publishers", "4.8 GB", "WDC schema.org JobPosting, Oct-2024 crawl, 14 parts"),
    "onet": (s_onet, "CC BY 4.0 (attribute O*NET)", "~0.05-0.1 GB", "O*NET 31.0 database"),
    "bls_ooh": (s_bls_ooh, "Public domain (cite BLS)", "0.013 GB", "BLS Occupational Outlook Handbook, full XML"),
    "ror": (s_ror, "CC0", "~0.1-0.3 GB", "ROR research-organisation registry (latest Zenodo release)"),
    "openalex_institutions": (s_openalex_institutions, "CC0", "~0.3 GB", "OpenAlex institutions via API"),
    "wikidata": (s_wikidata, "CC0", "~0.2-1 GB", "Wikidata companies + universities via SPARQL (best effort)"),
    "sec_tickers": (s_sec_tickers, "US government public data", "<0.01 GB", "SEC company tickers (needs --contact)"),
    "talentclef": (s_talentclef, "check Zenodo record", "<3 GB", "TalentCLEF 2025 (Zenodo search; best effort)"),
    "wikipedia_en": (s_wikipedia, "CC BY-SA 4.0 / GFDL", "capped (default 8 GB; full en ~20 GB)", "Wikipedia English (HF wikimedia/wikipedia)"),
    "fineweb_edu": (s_fineweb_edu, "ODC-By 1.0 + Common Crawl terms", "capped (default 10 GB)", "FineWeb-Edu sample (HF)"),
    "cosmopedia_v2": (s_cosmopedia, "ODC-By 1.0", "capped (default 5 GB)", "Cosmopedia v2 synthetic textbooks (HF smollm-corpus)"),
}
MANUAL = {
    "esco": ("https://esco.ec.europa.eu/en/use-esco/download", "ESCO sends the download link by e-mail: pick version v1.2.1, content 'classification', language en (and others), file type csv, then save the zip into external_data_1010/esco/. Licence: confirm on the ESCO site."),
    "bls_ooh_if_403": ("https://www.bls.gov/ooh/xml-compilation.xml", "BLS blocks scripted downloads (HTTP 403). Open the link in a browser and save it as external_data_1010/bls_ooh/ooh-xml-compilation.xml (13 MB). Not used by the pipeline yet."),
    "pjb_benchmark": ("https://arxiv.org/abs/2603.17386", "The abstract names no data link. Open the paper PDF and look for the data / code link (GitHub / HF), then save it into external_data_1010/pjb/."),
    "tier2_not_downloaded": ("DATA_SOURCES.md Tier 2", "Kaggle LinkedIn / Naukri job postings (Kaggle login + dataset terms), QS/THE rankings, NIRF lists, AliYun PJF: need a terms-of-service check first."),
}


def _sha1(path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(dest, results):
    (dest / "MANIFEST.json").write_text(json.dumps(results, indent=2))
    lines = ["# external_data_1010 manifest", "", f"Written {time.strftime('%Y-%m-%d %H:%M')} by download_external_data.py. Licences are as stated by the publishers; confirm before use.", "",
             "| Source | Status | Size | Files | Licence | Notes |", "|---|---|---|---|---|---|"]
    for k, r in results["sources"].items():
        lines.append(f"| {k} | {r['status']} | {fmt(r.get('bytes', 0))} | {len(r.get('files', []))} | {r.get('licence', '')} | {'; '.join(r.get('notes', []) + ([r['error']] if r.get('error') else []))} |")
    lines += ["", "## Manual / not downloaded", ""]
    for k, (u, t) in MANUAL.items():
        lines.append(f"- **{k}** ({u}): {t}")
    (dest / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--only", default="", help="comma-separated source names (see --list)")
    ap.add_argument("--skip", default="", help="comma-separated source names to skip")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="show what would be fetched; writes nothing")
    ap.add_argument("--contact", default=os.environ.get("SLM_CONTACT", ""), help="your e-mail, sent only to sec.gov as the required User-Agent contact")
    ap.add_argument("--fineweb-gb", type=float, default=10.0)
    ap.add_argument("--cosmopedia-gb", type=float, default=5.0)
    ap.add_argument("--wikipedia-gb", type=float, default=8.0)
    ap.add_argument("--fineweb-config", default="sample/10BT", help="path inside the HF repo, e.g. sample/10BT or sample/100BT")
    ap.add_argument("--wiki-config", default="20231101.en")
    ap.add_argument("--checksums", action="store_true", help="record a SHA-1 per downloaded file in MANIFEST.json")
    a = ap.parse_args()

    if a.list:
        for k, (_, lic, size, desc) in SOURCES.items():
            print(f"{k:24s} {size:34s} {lic:48s} {desc}")
        print("\nManual:")
        for k, (u, t) in MANUAL.items():
            print(f"  {k}: {t}")
        return 0

    names = [n for n in (a.only.split(",") if a.only else SOURCES) if n]
    names = [n for n in names if n not in a.skip.split(",")]
    bad = [n for n in names if n not in SOURCES]
    if bad:
        print("unknown source(s):", bad, "(see --list)")
        return 2
    dest = Path(a.dest)
    if not a.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest).free
        print(f"free space at {dest}: {fmt(free)}")
    ctx = Ctx(a)
    results = {"dest": str(dest), "sources": {}}
    # Turn 56: the manifest is cumulative — a `--only x` run updates x and keeps every other source's entry
    try:
        prev = json.loads((dest / "MANIFEST.json").read_text())
        results["sources"].update(prev.get("sources") or {})
    except (OSError, ValueError):
        pass
    for n in names:
        fn, lic, size, desc = SOURCES[n]
        print(f"\n== {n}: {desc}  [{size}]")
        ctx.files, ctx.notes, ctx.rows = [], [], {}
        t0 = time.time()
        try:
            fn(ctx)
            status = "dry-run" if a.dry_run else ("ok" if ctx.files else "empty")
            err = None
        except Exception as e:  # noqa: BLE001
            status, err = "FAILED", str(e)
            print(f"    FAILED: {e}")
        if status == "ok" and n == "wikidata" and not any(ctx.rows.values()):
            status = "empty"                                  # files present but no rows: not ok
        results["sources"][n] = {"status": status, "error": err, "licence": lic, "files": [f for f, _ in ctx.files],
                                 "bytes": sum(b for _, b in ctx.files), "notes": list(ctx.notes),
                                 "seconds": round(time.time() - t0), "rows": dict(ctx.rows),
                                 "sha1": {f: _sha1(dest / f) for f, _ in ctx.files if (dest / f).exists()}
                                 if a.checksums else {},
                                 "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if not a.dry_run:
            write_manifest(dest, results)
    print("\nSummary")
    for n, r in results["sources"].items():
        print(f"  {n:24s} {r['status']:8s} {fmt(r['bytes']):>10s}  {r.get('error') or ''}")
    print("\nManual steps: esco, pjb_benchmark (see MANIFEST.md)")
    return 1 if any(r["status"] == "FAILED" for r in results["sources"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
