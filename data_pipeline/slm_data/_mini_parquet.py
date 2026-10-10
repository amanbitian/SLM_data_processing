"""
Pure-python, dependency-free Parquet reader used ONLY as a fallback when pyarrow is not
installed. Supports flat schemas, SNAPPY/GZIP/UNCOMPRESSED codecs, PLAIN and
(RLE_)DICTIONARY encodings, data page v1 + v2, BYTE_ARRAY / INT32 / INT64 / FLOAT /
DOUBLE / BOOLEAN. It is slow (pure python) — install pyarrow for real runs.
"""
from __future__ import annotations

import struct
import zlib
from functools import lru_cache


class _R:
    __slots__ = ("b", "i")

    def __init__(self, b, i=0):
        self.b = b
        self.i = i

    def byte(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def varint(self):
        r = sh = 0
        while True:
            x = self.byte()
            r |= (x & 0x7F) << sh
            sh += 7
            if not x & 0x80:
                return r

    def zz(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def binary(self):
        n = self.varint()
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def val(self, t):
        if t in (1, 2):
            return t == 1
        if t == 3:
            return struct.unpack("b", bytes([self.byte()]))[0]
        if t in (4, 5, 6):
            return self.zz()
        if t == 7:
            v = struct.unpack("<d", self.b[self.i:self.i + 8])[0]
            self.i += 8
            return v
        if t == 8:
            return self.binary()
        if t in (9, 10):
            return self.lst()
        if t == 11:
            return self.mp()
        if t == 12:
            return self.struct()
        raise ValueError(f"thrift type {t}")

    def lst(self):
        h = self.byte()
        n, et = h >> 4, h & 0xF
        if n == 15:
            n = self.varint()
        return [(self.byte() == 1) if et in (1, 2) else self.val(et) for _ in range(n)]

    def mp(self):
        n = self.varint()
        if n == 0:
            return {}
        kv = self.byte()
        return {self.val(kv >> 4): self.val(kv & 0xF) for _ in range(n)}

    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0:
                return out
            t, d = h & 0xF, h >> 4
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.val(t)


def _snappy(b: bytes) -> bytes:
    r = _R(b)
    r.varint()
    i, out = r.i, bytearray()
    while i < len(b):
        tag = b[i]
        i += 1
        t = tag & 3
        if t == 0:
            n = tag >> 2
            if n >= 60:
                nb = n - 59
                n = int.from_bytes(b[i:i + nb], "little")
                i += nb
            n += 1
            out += b[i:i + n]
            i += n
            continue
        if t == 1:
            n = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | b[i]
            i += 1
        elif t == 2:
            n = (tag >> 2) + 1
            off = int.from_bytes(b[i:i + 2], "little")
            i += 2
        else:
            n = (tag >> 2) + 1
            off = int.from_bytes(b[i:i + 4], "little")
            i += 4
        s = len(out) - off
        if off >= n:
            out += out[s:s + n]
        else:
            for k in range(n):
                out.append(out[s + k])
    return bytes(out)


def _decomp(codec, b):
    if codec == 0:
        return b
    if codec == 1:
        return _snappy(b)
    if codec == 2:
        return zlib.decompress(b, 16 + 15)
    raise NotImplementedError(f"codec {codec} needs pyarrow (pip install pyarrow)")


def _rle(b, i, bw, count):
    out, bytew = [], (bw + 7) // 8
    mask = (1 << bw) - 1
    while len(out) < count and i < len(b):
        r = _R(b, i)
        h = r.varint()
        i = r.i
        if h & 1:
            groups = h >> 1
            nbytes = groups * bw
            chunk = int.from_bytes(b[i:i + nbytes], "little")
            i += nbytes
            out.extend((chunk >> (k * bw)) & mask for k in range(groups * 8))
        else:
            run = h >> 1
            v = int.from_bytes(b[i:i + bytew], "little")
            i += bytew
            out.extend([v] * run)
    return out[:count]


_FIXED = {1: ("<i", 4), 2: ("<q", 8), 4: ("<f", 4), 5: ("<d", 8)}


def _plain(b, n, ptype, i=0):
    if ptype == 6:
        out = []
        for _ in range(n):
            L = struct.unpack_from("<i", b, i)[0]
            i += 4
            out.append(bytes(b[i:i + L]).decode("utf-8", "replace"))
            i += L
        return out
    if ptype == 0:
        return [bool((b[i + k // 8] >> (k % 8)) & 1) for k in range(n)]
    fmt, w = _FIXED[ptype]
    return list(struct.unpack_from("<" + fmt[1] * n, b, i))


@lru_cache(maxsize=64)
def _footer(path):
    with open(path, "rb") as f:
        f.seek(-8, 2)
        n = struct.unpack("<i", f.read(4))[0]
        f.seek(-8 - n, 2)
        m = _R(f.read(n)).struct()
    schema = {}
    for e in m[2][1:]:
        if 1 in e:
            schema[e[4].decode()] = (e[1], e.get(3, 1) == 1)  # (physical type, optional)
    return m, schema


def num_row_groups(path) -> int:
    return len(_footer(str(path))[0][4])


def num_rows(path) -> int:
    return _footer(str(path))[0][3]


def columns(path) -> list[str]:
    return list(_footer(str(path))[1])


def read_column(path, name, rg_idx):
    m, schema = _footer(str(path))
    ptype, optional = schema[name]
    rg = m[4][rg_idx]
    cc = next(c[3] for c in rg[1] if b".".join(c[3][3]).decode() == name)
    codec, start, total = cc[4], cc.get(11) or cc[9], cc[7]
    with open(path, "rb") as f:
        f.seek(start)
        buf = f.read(total)
    i, dic, vals = 0, None, []
    while i < len(buf):
        r = _R(buf, i)
        ph = r.struct()
        i = r.i
        body = buf[i:i + ph[3]]
        i += ph[3]
        if ph[1] == 2:  # dictionary page
            dic = _plain(_decomp(codec, body), ph[7][1], ptype)
            continue
        if ph[1] == 0:  # data page v1
            d = _decomp(codec, body)
            nv, enc = ph[5][1], ph[5][2]
            j = 0
            if optional:
                L = struct.unpack_from("<i", d, 0)[0]
                defs = _rle(d, 4, 1, nv)
                j = 4 + L
            else:
                defs = [1] * nv
        elif ph[1] == 3:  # data page v2
            h = ph[8]
            nv, enc, dl, rl = h[1], h[4], h[5], h.get(6, 0)
            defs = _rle(body[rl:rl + dl], 0, 1, nv) if optional else [1] * nv
            d = body[rl + dl:]
            if h.get(7, True):
                d = _decomp(codec, d)
            j = 0
        else:
            continue
        nn = sum(defs)
        if enc in (8, 2):
            bw = d[j]
            vv = [dic[k] for k in _rle(d, j + 1, bw, nn)]
        else:
            vv = _plain(d, nn, ptype, j)
        it = iter(vv)
        vals.extend(next(it) if x else None for x in defs)
    return vals
