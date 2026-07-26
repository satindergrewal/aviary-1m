#!/usr/bin/env python3
# Dependency-free GGUF header reader: tensor names + selected KV.
# Used to answer: does this model carry nextn/MTP layers (i.e. can --spec-type draft-mtp work)?
import struct, sys, re

path = sys.argv[1]
f = open(path, "rb")
assert f.read(4) == b"GGUF", "not a gguf"
ver = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
print(f"gguf v{ver}  tensors={n_tensors}  kv={n_kv}")

def rstr():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

FIXED = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
def rval(t):
    if t in FIXED:
        return f.read(FIXED[t])
    if t == 8:
        return rstr()
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [rval(et) for _ in range(n)]
    raise ValueError(f"unknown kv type {t}")

interesting = {}
for _ in range(n_kv):
    k = rstr()
    t = struct.unpack("<I", f.read(4))[0]
    v = rval(t)
    kl = k.lower()
    if "nextn" in kl or "mtp" in kl or "block_count" in k or "expert" in kl:
        if t in (4, 5):   # uint32/int32
            v = struct.unpack("<I", v)[0]
        interesting[k] = v

names = []
for _ in range(n_tensors):
    nm = rstr()
    nd = struct.unpack("<I", f.read(4))[0]
    for _ in range(nd):
        f.read(8)
    f.read(4)   # type
    f.read(8)   # offset
    names.append(nm)

print("sample tensors:", names[:3])
hits = [n for n in names if "nextn" in n.lower() or "mtp" in n.lower()]
print(f"nextn/MTP tensors: {len(hits)}", hits[:6])
idx = [int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.", n)] if m]
print("max blk index:", max(idx) if idx else None)
print("relevant KV:", {k: v for k, v in interesting.items() if not isinstance(v, bytes)})
print("VERDICT:", "MTP-capable (draft-mtp testable)" if hits else "NO MTP layers -> --spec-type draft-mtp NOT testable with this file")
