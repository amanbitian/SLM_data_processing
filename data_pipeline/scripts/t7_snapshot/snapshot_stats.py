"""Aggregate statistics of the T7 dataset (no raw text or PII is written; emails are only hashed in memory).
usage: python3 snapshot_stats.py <job> ; results -> snap/<job>.json"""
import sys, os, json, re, glob, hashlib, collections, statistics
sys.path.insert(0, os.path.dirname(__file__))
import pq

T7 = '/sessions/awesome-happy-bardeen/mnt/turing_candidate_dataset'
OUT = os.path.join(os.path.dirname(__file__), 'snap'); os.makedirs(OUT, exist_ok=True)
EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE = re.compile(r'\+?\d[\d\s().-]{8,}\d')
URL = re.compile(r'(https?://|www\.|linkedin\.com|github\.com)', re.I)


def dec(x):
    return x.decode('utf-8', 'replace') if isinstance(x, bytes) else x


def pct(v, ps=(1, 5, 25, 50, 75, 95, 99)):
    v = sorted(v)
    if not v:
        return {}
    return {f'p{p}': v[min(len(v) - 1, int(len(v) * p / 100))] for p in ps} | {'n': len(v), 'mean': round(sum(v) / len(v), 1)}


def rgs(path):
    m = pq._FCACHE.get(path) or pq.footer(path)
    pq._FCACHE[path] = m
    return len(m[4])


def col(path, c, rg):
    return pq.read_column(path, c, rg, 10**9)


def cat_counts(paths, cols, every=1, top=60):
    out = {c: collections.Counter() for c in cols}
    for p in paths:
        for rg in range(0, rgs(p), every):
            for c in cols:
                out[c].update(dec(x) if x is not None else None for x in col(p, c, rg))
    return {c: {'null': v.pop(None, 0), 'distinct': len(v), 'top': v.most_common(top)} for c, v in out.items()}


def job_resume_cats():
    paths = sorted(glob.glob(f'{T7}/resume_and other details/part_*.parquet'))
    r = cat_counts(paths, ['country', 'role', 'domain', 'subdomain', 'industry', 'is_experience_in_frontier_geo', 'salary_local_currency'])
    ft = []
    for p in paths:
        for rg in range(rgs(p)):
            ft += [x for x in col(p, 'full_time_experience_in_months', rg) if x is not None]
    r['full_time_experience_in_months'] = pct(ft)
    return r


def job_resume_emails():
    """people with several accounts: same e-mail (hashed in memory) across talent_ids."""
    paths = sorted(glob.glob(f'{T7}/resume_and other details/part_*.parquet'))
    h = collections.Counter()
    test_acc = 0
    for p in paths:
        for rg in range(rgs(p)):
            for e in col(p, 'email', rg):
                if not e:
                    continue
                s = dec(e).strip().lower()
                if '+' in s.split('@')[0] or s.endswith('@turing.com'):
                    test_acc += 1
                local, _, dom = s.partition('@')
                if dom in ('gmail.com', 'googlemail.com'):
                    local = local.split('+')[0].replace('.', '')
                h[hashlib.blake2b((local + '@' + dom).encode(), digest_size=8).digest()] += 1
    multi = [v for v in h.values() if v > 1]
    return {'emails': len(h), 'emails_with_multiple_talent_ids': len(multi), 'talent_ids_in_shared_email_groups': sum(multi),
            'max_accounts_per_email': max(multi) if multi else 1, 'plus_or_turing_addresses (likely test/QA)': test_acc}


def job_resume_text(every=25):
    paths = sorted(glob.glob(f'{T7}/resume_and other details/part_*.parquet'))
    L, sl, nwork, nedu = [], [], [], []
    flags = collections.Counter()
    keys = collections.Counter()
    wkeys, ekeys = collections.Counter(), collections.Counter()
    hashes = collections.Counter()
    n = 0
    for p in paths:
        for rg in range(0, rgs(p), every):
            rp = col(p, 'resume_plain', rg)
            pr = col(p, 'parsed_resume_data', rg)
            we = col(p, 'work_experience', rg)
            ed = col(p, 'education', rg)
            sm = col(p, 'summary', rg)
            for i in range(len(rp)):
                n += 1
                t = dec(rp[i]) or ''
                L.append(len(t))
                if t:
                    hashes[hashlib.blake2b(re.sub(r'[^a-z0-9]', '', t.lower()).encode(), digest_size=8).digest()] += 1
                    a = sum(c.isascii() for c in t) / len(t)
                    flags['ascii<0.7'] += a < 0.7
                    flags['has_email'] += bool(EMAIL.search(t))
                    flags['has_phone'] += bool(PHONE.search(t))
                    flags['has_url'] += bool(URL.search(t))
                    flags['bom_or_ctrl'] += ('﻿' in t) or bool(re.search(r'[\x00-\x08]', t))
                    flags['lt300'] += len(t) < 300
                    flags['gt60k'] += len(t) > 60000
                    al = sum(c.isalpha() for c in t) / len(t)
                    flags['alpha<0.45'] += al < 0.45
                if sm[i]:
                    sl.append(len(sm[i]))
                if pr[i]:
                    try:
                        j = json.loads(pr[i])
                        if isinstance(j, dict):
                            keys.update(j.keys())
                            if 'type' in j and 'properties' in j:
                                flags['parse_is_json_schema_not_data'] += 1
                        flags['parse_ok'] += 1
                    except Exception:
                        flags['parse_bad_json'] += 1
                for src, cnt, kc in ((we[i], nwork, wkeys), (ed[i], nedu, ekeys)):
                    if src:
                        try:
                            j = json.loads(src)
                            cnt.append(len(j))
                            for it in j[:3]:
                                kc.update(it.keys())
                        except Exception:
                            flags['struct_bad_json'] += 1
    exact_dup = sum(v - 1 for v in hashes.values() if v > 1)
    return {'sampled_rows': n, 'sample_every_n_row_groups': every, 'resume_chars': pct(L), 'summary_chars': pct(sl),
            'flags': dict(flags), 'exact_text_duplicates_in_sample': exact_dup,
            'parsed_keys': keys.most_common(60), 'work_items_per_candidate': pct(nwork), 'edu_items_per_candidate': pct(nedu),
            'work_item_keys': wkeys.most_common(20), 'edu_item_keys': ekeys.most_common(20)}


def job_work(every=10):
    paths = sorted(glob.glob(f'{T7}/work_experience/*.parquet'))
    r = cat_counts(paths, ['start_month', 'end_month', 'employment_type', 'seniority', 'work_mode', 'industry', 'department'], every=every, top=40)
    sy, ey, dl = [], [], []
    bad = collections.Counter()
    for p in paths:
        for rg in range(0, rgs(p), every):
            s, e, d, pos, comp = (col(p, c, rg) for c in ('start_year', 'end_year', 'details', 'position', 'company'))
            for i in range(len(s)):
                bad['rows'] += 1
                a, b = s[i], e[i]
                sy.append(a)
                if b is not None:
                    ey.append(b)
                bad['start_year_0'] += a == 0
                bad['start_year_out_of_range'] += a is not None and a != 0 and not 1950 <= a <= 2026
                bad['end_year_null_(current?)'] += b is None
                bad['end_year_0'] += b == 0
                bad['end_year_out_of_range'] += b is not None and b != 0 and not 1950 <= b <= 2030
                bad['end_before_start'] += (a and b and 1950 <= a <= 2026 and 1950 <= b <= 2030 and b < a) or 0
                bad['empty_position'] += not pos[i]
                bad['empty_company'] += not comp[i]
                bad['empty_details'] += not d[i]
                if d[i]:
                    dl.append(len(d[i]))
    r['year_checks'] = dict(bad)
    r['details_chars'] = pct(dl)
    return r


def job_edu(every=10):
    paths = sorted(glob.glob(f'{T7}/education/*.parquet'))
    r = cat_counts(paths, ['level', 'subject', 'start_month', 'end_month'], every=every, top=40)
    bad = collections.Counter()
    sc = collections.Counter()
    for p in paths:
        for rg in range(0, rgs(p), every):
            s, e, deg, colg, cs = (col(p, c, rg) for c in ('start_year', 'end_year', 'degree', 'college', 'cgpa_scale'))
            for i in range(len(s)):
                bad['rows'] += 1
                bad['start_year_0'] += s[i] == 0
                bad['end_year_0'] += e[i] == 0
                bad['end_before_start'] += (s[i] and e[i] and e[i] < s[i]) or 0
                bad['year_out_of_range'] += any(x and not 1940 <= x <= 2035 for x in (s[i], e[i]))
                bad['empty_degree'] += not deg[i]
                bad['empty_college'] += not colg[i]
                sc[cs[i]] += 1
    r['checks'] = dict(bad)
    r['cgpa_scale'] = sc.most_common(10)
    return r


def job_jdmatch():
    p = f'{T7}/jd_resume_matching/jd_resume_matching_part_001.parquet'
    cols = ['job_code', 'pipeline_source', 'is_discarded', 'final_recommendation', 'fraud_verdict', 'fraud_bucket', 'fraud_status',
            'job_fit_status', 'interview_status', 'interview_recommended_action', 'country_compliance_status', 'role', 'country']
    r = cat_counts([p], cols)
    fit = []
    dec_ = collections.Counter()
    for rg in range(rgs(p)):
        for x in col(p, 'job_fit_reason', rg):
            if x and dec(x).startswith('{'):
                try:
                    dec_[json.loads(dec(x)).get('decision')] += 1
                except Exception:
                    dec_['bad'] += 1
    r['job_fit_reason_decisions(json-form only)'] = dec_.most_common()
    return r


def job_fired():
    p = f'{T7}/resume_and other details/fired.parquet'
    return cat_counts([p], ['country', 'role', 'domain'])


JOBS = {'resume_cats': job_resume_cats, 'resume_emails': job_resume_emails, 'resume_text': job_resume_text,
        'work': job_work, 'edu': job_edu, 'jdmatch': job_jdmatch, 'fired': job_fired}

if __name__ == '__main__':
    j = sys.argv[1]
    res = JOBS[j]()
    json.dump(res, open(os.path.join(OUT, j + '.json'), 'w'), indent=1, default=str, ensure_ascii=False)
    print(j, 'done')
