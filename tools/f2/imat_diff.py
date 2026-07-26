#!/usr/bin/env python3
# Control check for D1: are the two imatrices identical everywhere except blk.64?
# If any shared tensor's data differs, arms 2 and 3 differ by more than the MTP data
# and the comparison is invalid.
import struct, sys, hashlib

def load(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    nt = struct.unpack("<Q", f.read(8))[0]
    nkv = struct.unpack("<Q", f.read(8))[0]
    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def rval(t):
        if t in FIXED: return f.read(FIXED[t])
        if t == 8: return rstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rval(et) for _ in range(n)]
    align = 32
    for _ in range(nkv):
        k = rstr(); t = struct.unpack("<I", f.read(4))[0]; v = rval(t)
        if k == "general.alignment": align = struct.unpack("<I", v)[0]
    infos = []
    for _ in range(nt):
        nm = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((nm, dims, ttype, off))
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    return f, data_start, infos

def tsize(dims, ttype):
    n = 1
    for d in dims: n *= d
    # imatrix files hold f32 sums and f32/i32 counts
    return n * 4

fa, sa, ia = load(sys.argv[1])
fb, sb, ib = load(sys.argv[2])
da = {nm: (dims, tt, off) for nm, dims, tt, off in ia}
db = {nm: (dims, tt, off) for nm, dims, tt, off in ib}

shared = sorted(set(da) & set(db))
only_a = sorted(set(da) - set(db))
only_b = sorted(set(db) - set(da))

diff = []
for nm in shared:
    dims, tt, off = da[nm]
    dimsb, ttb, offb = db[nm]
    if dims != dimsb or tt != ttb:
        diff.append((nm, "shape/type")); continue
    sz = tsize(dims, tt)
    fa.seek(sa + off); ha = hashlib.sha256(fa.read(sz)).hexdigest()
    fb.seek(sb + offb); hb = hashlib.sha256(fb.read(sz)).hexdigest()
    if ha != hb:
        diff.append((nm, "data"))

print(f"shared tensors: {len(shared)}")
print(f"only in A ({sys.argv[1].split('/')[-1]}): {len(only_a)}")
for n in only_a[:20]: print("   ", n)
print(f"only in B ({sys.argv[2].split('/')[-1]}): {len(only_b)}")
print(f"shared tensors that DIFFER: {len(diff)}")
for n, why in diff[:10]: print("   ", n, why)
print()
print("VERDICT:", "CLEAN - files differ ONLY by the blk.64 entries"
      if not diff and not only_b and all(n.startswith("blk.64.") for n in only_a)
      else "NOT CLEAN - investigate above")
