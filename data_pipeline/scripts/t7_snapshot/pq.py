"""Minimal pure-Python parquet reader (for machines without pyarrow).
footer(path) -> schema, row counts, codecs, column stats.
read_column_sample(path, col, rg, max_values) -> decoded values (SNAPPY/GZIP/UNCOMPRESSED; PLAIN + dictionary)."""
import struct, sys, json, gzip, zlib, io

# ---------------- thrift compact protocol (generic) ----------------
class R:
    def __init__(s, b, p=0): s.b, s.p = b, p
    def byte(s):
        v = s.b[s.p]; s.p += 1; return v
    def varint(s):
        sh = r = 0
        while True:
            x = s.byte(); r |= (x & 0x7F) << sh; sh += 7
            if not x & 0x80: return r
    def zz(s):
        v = s.varint(); return (v >> 1) ^ -(v & 1)
    def binary(s):
        n = s.varint(); v = s.b[s.p:s.p + n]; s.p += n; return bytes(v)
    def val(s, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3:
            v = s.byte(); return v - 256 if v > 127 else v
        if t in (4, 5, 6): return s.zz()
        if t == 7:
            v = struct.unpack('<d', s.b[s.p:s.p + 8])[0]; s.p += 8; return v
        if t == 8: return s.binary()
        if t in (9, 10):
            h = s.byte(); n = h >> 4; et = h & 15
            if n == 15: n = s.varint()
            if et in (1, 2):
                return [s.byte() == 1 for _ in range(n)]
            return [s.val(et) for _ in range(n)]
        if t == 11:
            n = s.varint()
            if n == 0: return {}
            kt = s.byte(); kv, vt = kt >> 4, kt & 15
            return {s.val(kv): s.val(vt) for _ in range(n)}
        if t == 12: return s.struct()
        raise ValueError(f"type {t}")
    def struct(s):
        out, last = {}, 0
        while True:
            h = s.byte()
            if h == 0: return out
            d, t = h >> 4, h & 15
            fid = last + d if d else s.zz()
            last = fid
            out[fid] = s.val(t)

TYPES = {0: 'BOOLEAN', 1: 'INT32', 2: 'INT64', 3: 'INT96', 4: 'FLOAT', 5: 'DOUBLE', 6: 'BYTE_ARRAY', 7: 'FIXED'}
CODECS = {0: 'UNCOMPRESSED', 1: 'SNAPPY', 2: 'GZIP', 3: 'LZO', 4: 'BROTLI', 5: 'LZ4', 6: 'ZSTD', 7: 'LZ4_RAW'}

def footer(path):
    with open(path, 'rb') as f:
        f.seek(-8, 2); n, magic = struct.unpack('<i4s', f.read(8))
        assert magic == b'PAR1', magic
        f.seek(-8 - n, 2); meta = R(f.read(n)).struct()
    return meta

def describe(path):
    m = footer(path)
    schema = m[2]
    cols = []
    for se in schema[1:]:
        cols.append({'name': se[4].decode(), 'type': TYPES.get(se.get(1)), 'rep': se.get(3), 'children': se.get(5, 0),
                     'converted': se.get(6), 'logical': list(se.get(10, {}).keys())})
    rgs = m.get(4, [])
    stats = {}
    for rg in rgs:
        for cc in rg[1]:
            md = cc[3]; name = '.'.join(x.decode() for x in md[3])
            st = stats.setdefault(name, {'codec': CODECS.get(md[4]), 'nulls': 0, 'values': 0, 'csize': 0, 'usize': 0, 'min': None, 'max': None})
            st['values'] += md[5]; st['csize'] += md[7]; st['usize'] += md[6]
            s = md.get(12, {})
            st['nulls'] += s.get(3, 0) or 0
            mn, mx = s.get(6, s.get(2)), s.get(5, s.get(1))
            if mn is not None and (st['min'] is None or mn < st['min']): st['min'] = mn
            if mx is not None and (st['max'] is None or mx > st['max']): st['max'] = mx
    return {'rows': m[3], 'row_groups': len(rgs), 'created_by': (m.get(6) or b'').decode(errors='replace'),
            'kv': {k.get(1, b'').decode(): (k.get(2) or b'')[:200].decode(errors='replace') for k in m.get(5, [])}, 'schema': cols, 'stats': stats, '_meta': m}

# ---------------- snappy (pure python) ----------------
def snappy_decompress(b):
    r = R(b); n = r.varint(); out = bytearray(); p = r.p
    while p < len(b):
        tag = b[p]; p += 1; t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(b[p:p + nb], 'little'); p += nb
            ln += 1; out += b[p:p + ln]; p += ln
        else:
            if t == 1:
                ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | b[p]; p += 1
            elif t == 2:
                ln = (tag >> 2) + 1; off = int.from_bytes(b[p:p + 2], 'little'); p += 2
            else:
                ln = (tag >> 2) + 1; off = int.from_bytes(b[p:p + 4], 'little'); p += 4
            start = len(out) - off
            if off >= ln:
                out += out[start:start + ln]
            else:
                for i in range(ln): out.append(out[start + i])
    assert len(out) == n, (len(out), n)
    return bytes(out)

def decompress(codec, data, usize):
    if codec == 0: return data
    if codec == 1: return snappy_decompress(data)
    if codec == 2: return gzip.decompress(data)
    raise NotImplementedError(CODECS.get(codec))

# ---------------- page decoding ----------------
def rle_bp(buf, bw, count):
    """RLE / bit-packed hybrid -> list of ints."""
    r = R(buf); out = []
    while len(out) < count and r.p < len(buf):
        h = r.varint()
        if h & 1:
            ng = h >> 1; nbytes = ng * bw; chunk = buf[r.p:r.p + nbytes]; r.p += nbytes
            val = int.from_bytes(chunk, 'little'); mask = (1 << bw) - 1
            for i in range(ng * 8):
                out.append((val >> (i * bw)) & mask)
        else:
            n = h >> 1; w = (bw + 7) // 8; v = int.from_bytes(buf[r.p:r.p + w], 'little'); r.p += w
            out.extend([v] * n)
    return out[:count]

def plain(ptype, buf, n):
    out, p = [], 0
    for _ in range(n):
        if p >= len(buf): break
        if ptype == 6:
            ln = struct.unpack_from('<i', buf, p)[0]; p += 4; out.append(buf[p:p + ln]); p += ln
        elif ptype == 1:
            out.append(struct.unpack_from('<i', buf, p)[0]); p += 4
        elif ptype == 2:
            out.append(struct.unpack_from('<q', buf, p)[0]); p += 8
        elif ptype == 5:
            out.append(struct.unpack_from('<d', buf, p)[0]); p += 8
        elif ptype == 4:
            out.append(struct.unpack_from('<f', buf, p)[0]); p += 4
        elif ptype == 0:
            out.append(bool((buf[p // 8] >> (p % 8)) & 1)); p += 1
        else:
            raise NotImplementedError(ptype)
    return out

_FCACHE = {}

def read_column(path, colname, rg_index=0, max_values=200, max_def=1):
    m = _FCACHE.get(path)
    if m is None:
        m = _FCACHE[path] = footer(path)
    rg = m[4][rg_index]
    cc = next(c for c in rg[1] if '.'.join(x.decode() for x in c[3][3]) == colname)
    md = cc[3]; ptype, codec = md[1], md[4]
    start = md.get(11) or md[9]
    total = md[7]
    with open(path, 'rb') as f:
        f.seek(start); blob = f.read(total)
    p, dictv, vals = 0, None, []
    while p < len(blob) and len(vals) < max_values:
        r = R(blob, p); ph = r.struct(); p = r.p
        csize, usize, ptyp = ph[3], ph[2], ph[1]
        raw = blob[p:p + csize]; p += csize
        if ptyp == 2:  # dictionary page
            dh = ph[7]; data = decompress(codec, raw, usize); dictv = plain(ptype, data, dh[1]); continue
        if ptyp == 0:  # data page v1
            dh = ph[5]; n = dh[1]; enc = dh[2]
            data = decompress(codec, raw, usize); q = 0
            defs = [max_def] * n
            if max_def > 0:
                ln = struct.unpack_from('<i', data, q)[0]; q += 4
                defs = rle_bp(data[q:q + ln], 1, n); q += ln
            nn = sum(1 for d in defs if d == max_def)
            if enc in (2, 8):  # dictionary indices
                bw = data[q]; q += 1; idx = rle_bp(data[q:], bw, nn); it = iter(dictv[i] for i in idx)
            else:
                it = iter(plain(ptype, data[q:], nn))
            for d in defs:
                vals.append(next(it, None) if d == max_def else None)
        elif ptyp == 3:  # data page v2
            dh = ph[8]; n = dh[1]; nnull = dh[2]; enc = dh[4]; dl, rl = dh[5], dh[6]
            levels = raw[:dl + rl]; body = raw[dl + rl:]
            if dh.get(7, True): body = decompress(codec, body, usize - dl - rl)
            defs = rle_bp(levels[rl:rl + dl], 1, n) if (max_def > 0 and dl) else [max_def] * n
            nn = sum(1 for d in defs if d == max_def)
            if enc in (2, 8):
                bw = body[0]; idx = rle_bp(body[1:], bw, nn); it = iter(dictv[i] for i in idx)
            else:
                it = iter(plain(ptype, body, nn))
            for d in defs:
                vals.append(next(it, None) if d == max_def else None)
    return vals[:max_values]

if __name__ == '__main__':
    for path in sys.argv[1:]:
        d = describe(path)
        print(f"\n### {path}\nrows={d['rows']} row_groups={d['row_groups']} created_by={d['created_by']}")
        for k, v in d['kv'].items():
            if k != 'ARROW:schema': print(f"  kv {k}: {v[:150]}")
        for c in d['schema']:
            st = d['stats'].get(c['name'], {})
            mn, mx = st.get('min'), st.get('max')
            fmtv = lambda v: (v[:40].decode(errors='replace') if isinstance(v, bytes) else v)
            print(f"  {c['name']:40s} {str(c['type']):10s} rep={c['rep']} ch={c['children']} conv={c['converted']} "
                  f"codec={st.get('codec')} nulls={st.get('nulls')} vals={st.get('values')} usize={st.get('usize')} "
                  f"min={fmtv(mn)!r} max={fmtv(mx)!r}")
