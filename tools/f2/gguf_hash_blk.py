#!/usr/bin/env python3
# Hash selected tensors' raw bytes from two GGUFs, to prove they actually differ.
# Used to check that --exclude-weights really changed blk.64's quantized values,
# and that blk.0-63 are untouched between arms.
import struct, sys, hashlib

GGML_SIZES = {0:(4,1),1:(2,1),2:(18,32),3:(20,32),6:(22,32),7:(24,32),8:(34,32),
              9:(36,32),10:(84,256),11:(110,256),12:(112,256),13:(144,256),14:(210,256),
              15:(292,256),16:(66,256),17:(74,256),18:(98,256),19:(50,256),20:(54,256),
              24:(1,1),25:(2,1),26:(4,1),27:(4,1),28:(8,1),29:(52,256),30:(2,1)}

def load(path):
    f = open(path, "rb"); assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    nt = struct.unpack("<Q", f.read(8))[0]; nkv = struct.unpack("<Q", f.read(8))[0]
    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8","replace")
    FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def rval(t):
        if t in FIXED: return f.read(FIXED[t])
        if t == 8: return rstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            return [rval(et) for _ in range(n)]
    align = 32
    for _ in range(nkv):
        k = rstr(); t = struct.unpack("<I", f.read(4))[0]; v = rval(t)
        if k == "general.alignment": align = struct.unpack("<I", v)[0]
    infos = {}
    for _ in range(nt):
        nm = rstr(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack("<I", f.read(4))[0]; off = struct.unpack("<Q", f.read(8))[0]
        infos[nm] = (dims, tt, off)
    pos = f.tell()
    return f, (pos + align - 1)//align*align, infos

def nbytes(dims, tt):
    n = 1
    for d in dims: n *= d
    bs, bn = GGML_SIZES.get(tt, (None, None))
    if bs is None: return None
    return n // bn * bs

fa, sa, ia = load(sys.argv[1])
fb, sb, ib = load(sys.argv[2])
pref = sys.argv[3] if len(sys.argv) > 3 else "blk.64."

same = diff = skipped = 0
print(f"{'tensor':<38} {'type':>5} {'A sha':>10} {'B sha':>10}  same?")
for nm in sorted(n for n in ia if n.startswith(pref)):
    if nm not in ib: continue
    da, ta, oa = ia[nm]; db, tb, ob = ib[nm]
    sz = nbytes(da, ta)
    if sz is None or ta != tb:
        skipped += 1; continue
    fa.seek(sa+oa); ha = hashlib.sha256(fa.read(sz)).hexdigest()[:10]
    fb.seek(sb+ob); hb = hashlib.sha256(fb.read(sz)).hexdigest()[:10]
    ok = ha == hb
    same += ok; diff += (not ok)
    print(f"{nm:<38} {ta:>5} {ha:>10} {hb:>10}  {'SAME' if ok else 'DIFFER'}")
print(f"\nprefix {pref}: identical={same} differing={diff} skipped={skipped}")
