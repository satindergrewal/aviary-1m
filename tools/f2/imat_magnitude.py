#!/usr/bin/env python3
# How BIG is the difference the MTP pass makes to non-MTP layers' imatrix values?
# ~1e-7 relative = floating-point reduction-order noise, harmless.
# Large = the patch is changing what the main model computes, which would be a real bug.
import struct, sys, array

def load(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
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

fa, sa, ia = load(sys.argv[1])
fb, sb, ib = load(sys.argv[2])

def read_f32(f, start, dims, off):
    n = 1
    for d in dims: n *= d
    f.seek(start + off)
    a = array.array("f"); a.frombytes(f.read(n*4))
    return a

targets = [n for n in ia if n.endswith(".in_sum2") and not n.startswith("blk.64.")]
targets = sorted(targets)[:8] + sorted(targets)[-4:]

print(f"{'tensor':<44} {'max|rel diff|':>14} {'mean|rel diff|':>15}")
worst = 0.0
for nm in targets:
    if nm not in ib: continue
    da, _, oa = ia[nm]; db, _, ob = ib[nm]
    if da != db: continue
    va = read_f32(fa, sa, da, oa); vb = read_f32(fb, sb, db, ob)
    mx = 0.0; tot = 0.0; cnt = 0
    for x, y in zip(va, vb):
        d = abs(x - y); s = max(abs(x), abs(y), 1e-30)
        r = d/s
        mx = max(mx, r); tot += r; cnt += 1
    worst = max(worst, mx)
    print(f"{nm:<44} {mx:>14.3e} {tot/max(1,cnt):>15.3e}")

print()
print(f"worst relative difference across sampled tensors: {worst:.3e}")
print("READING:", "float reduction-order noise, harmless" if worst < 1e-4
      else "LARGE - the MTP pass is changing main-model computation, investigate")
