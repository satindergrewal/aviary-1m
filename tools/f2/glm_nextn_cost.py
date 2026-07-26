#!/usr/bin/env python3
# What does the nextn (MTP) q8_0 pin actually cost in this artifact, and what would
# dropping it to q2_K return? D1 measured that this change is output-lossless
# (byte-identical text) and costs only draft acceptance / throughput.
import struct, sys

# (block_size_bytes, elements_per_block)
T = {0:("f32",4,1), 1:("f16",2,1), 2:("q4_0",18,32), 3:("q4_1",20,32), 6:("q5_0",22,32),
     7:("q5_1",24,32), 8:("q8_0",34,32), 9:("q8_1",36,32), 10:("q2_K",84,256),
     11:("q3_K",110,256), 12:("q4_K",144,256), 13:("q5_K",176,256), 14:("q6_K",210,256),
     15:("q8_K",292,256), 16:("iq2_xxs",66,256), 17:("iq2_xs",74,256), 18:("iq3_xxs",98,256),
     19:("iq1_s",50,256), 20:("iq4_nl",18,32), 24:("i8",1,1), 25:("i16",2,1), 26:("i32",4,1),
     29:("iq1_m",52,256), 30:("bf16",2,1), 21:("iq3_s",110,256), 22:("iq2_s",82,256),
     23:("iq4_xs",136,256)}

path = sys.argv[1]
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
kv = {}
for _ in range(nkv):
    k = rstr(); t = struct.unpack("<I", f.read(4))[0]; v = rval(t)
    if t in (4,5): v = struct.unpack("<I", v)[0]
    kv[k] = v

rows = []
for _ in range(nt):
    nm = rstr(); nd = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
    tt = struct.unpack("<I", f.read(4))[0]; off = struct.unpack("<Q", f.read(8))[0]
    rows.append((nm, dims, tt))

def sz(dims, tt):
    n = 1
    for d in dims: n *= d
    if tt not in T: return None, None
    name, bs, bn = T[tt]
    return name, n // bn * bs

print("relevant KV:", {k:v for k,v in kv.items() if "nextn" in k.lower() or "block_count" in k or "expert_count" in k})
mtp = [r for r in rows if "nextn" in r[0].lower()]
blocks = sorted({r[0].split(".")[1] for r in mtp if r[0].startswith("blk.")})
print(f"\nnextn-named tensors: {len(mtp)}   MTP block index/es: {blocks}")

# the whole MTP block, not just the nextn-named tensors
target_blk = blocks[0] if blocks else None
blk_rows = [r for r in rows if target_blk and r[0].startswith(f"blk.{target_blk}.")]
print(f"tensors in blk.{target_blk}: {len(blk_rows)}\n")

cur = alt = 0
print(f"{'tensor':<42} {'type':>9} {'current MiB':>12} {'as q2_K MiB':>12}")
for nm, dims, tt in sorted(blk_rows):
    tname, b = sz(dims, tt)
    if b is None: continue
    n = 1
    for d in dims: n *= d
    quant = tt not in (0, 1, 30)          # norms stay f32/f16, never requantized
    b2 = (n // 256 * 84) if quant and n % 256 == 0 else b
    cur += b; alt += b2
    if quant:
        print(f"{nm:<42} {tname:>9} {b/1048576:>12.1f} {b2/1048576:>12.1f}")
print(f"\nMTP block total   current {cur/1073741824:.2f} GiB   as q2_K {alt/1073741824:.2f} GiB")
print(f"RECLAIMABLE       {(cur-alt)/1073741824:.2f} GiB  ({(cur-alt)/1048576:.0f} MiB)")
