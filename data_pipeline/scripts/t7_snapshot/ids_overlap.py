"""Collect id sets from every source and measure overlaps. Writes /tmp-free results to outputs/ids/*.npy + report json."""
import sys, os, json, time, glob, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import pq

T7 = '/sessions/awesome-happy-bardeen/mnt/turing_candidate_dataset'
TD = '/sessions/awesome-happy-bardeen/mnt/New Volume/training_data'
OUT = os.path.join(os.path.dirname(__file__), 'ids'); os.makedirs(OUT, exist_ok=True)


def col_all(path, col, conv=None):
    m = pq._FCACHE.get(path) or pq.footer(path)
    pq._FCACHE[path] = m
    vals = []
    for i in range(len(m[4])):
        v = pq.read_column(path, col, i, max_values=10**9)
        vals.extend(v if conv is None else [conv(x) for x in v])
    return vals


def ids(paths, col, name, conv=None):
    f = os.path.join(OUT, name + '.npy')
    if os.path.exists(f):
        return np.load(f)
    t0, allv = time.time(), []
    for p in paths:
        v = col_all(p, col, conv)
        allv.extend(x for x in v if x is not None)
        print(f'  {name}: {os.path.basename(p)} {len(v)} ({time.time()-t0:.0f}s)', flush=True)
    a = np.asarray(allv, dtype=np.int64)
    np.save(f, a)
    return a


def to_int(x):
    if x is None: return None
    try: return int(float(x.decode() if isinstance(x, bytes) else x))
    except Exception: return None


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'small':
        ids([f'{T7}/jd_resume_matching/jd_resume_matching_part_001.parquet'], 'talent_id', 't7_jdmatch')
        ids([f'{T7}/resume_and other details/fired.parquet'], 'talent_id', 't7_fired')
    elif which == 't7res':
        ids(sorted(glob.glob(f'{T7}/resume_and other details/part_*.parquet')), 'talent_id', 't7_resume')
    elif which == 'td_res':
        ids(sorted(glob.glob(f'{TD}/resume_data/cluster_fraud_data_parquet/part_*.parquet')), 'talent_id', 'td_resume')
    elif which == 'work':
        ids(sorted(glob.glob(f'{T7}/work_experience/*.parquet')), 'developer_id', 't7_work_dev')
    elif which == 'edu':
        ids(sorted(glob.glob(f'{T7}/education/*.parquet')), 'developer_id', 't7_edu_dev')
