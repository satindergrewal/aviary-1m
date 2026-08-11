import json, struct, os, re, collections

MASTER = "<BOX>/kimi-k3"
MAP = "<BOX>/reap80_kept_indices.json"

def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

DT = {"F64":8,"I64":8,"F32":4,"I32":4,"BF16":2,"F16":2,"I16":2,"U8":1,"I8":1,"BOOL":1,"U32":4}

kept = {int(k): set(v) for k, v in json.load(open(MAP)).items()}
wm = json.load(open(os.path.join(MASTER, "model.safetensors.index.json")))["weight_map"]
hdrs = {}
for s in sorted(set(wm.values())):
    hdrs[s] = read_header(os.path.join(MASTER, s))

expert_re = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w\d)\.(weight_packed|weight_scale)$")

# logical param counts: MXFP4 packed U8 holds 2 values/byte; weight_scale is metadata
groups = collections.Counter()
bytes_g = collections.Counter()
logical = collections.Counter()

for name, sh in wm.items():
    m = hdrs[sh][name]
    nb = DT[m["dtype"]]
    for d in m["shape"]:
        nb *= d
    e = expert_re.search(name)
    if e:
        il, ei, w, kind = int(e.group(1)), int(e.group(2)), e.group(3), e.group(4)
        tag = "expert_KEPT" if (il not in kept or ei in kept[il]) else "expert_DROPPED"
        bytes_g[tag] += nb
        if kind == "weight_packed":
            logical[tag] += 2 * (m["shape"][0] * m["shape"][1])   # 2 x 4-bit per byte
        continue
    if "vision_tower" in name or "mm_projector" in name:
        tag = "vision"
    elif "block_sparse_moe.shared_experts" in name:
        tag = "shared_experts"
    elif "block_sparse_moe.routed_expert_" in name:
        tag = "routed_bottleneck"
    elif "block_sparse_moe.gate" in name:
        tag = "router"
    elif "self_attn" in name:
        tag = "attention"
    elif "embed_tokens" in name or "lm_head" in name:
        tag = "embed/head"
    else:
        tag = "other_dense"
    bytes_g[tag] += nb
    n = 1
    for d in m["shape"]:
        n *= d
    logical[tag] += n

print(f"{'group':<18} {'params':>16} {'disk GB':>10}")
for k in sorted(bytes_g, key=lambda x: -bytes_g[x]):
    print(f"{k:<18} {logical[k]:>16,} {bytes_g[k]/1e9:>10.1f}")

kept_p = sum(v for k, v in logical.items() if k != "expert_DROPPED")
kept_b = sum(v for k, v in bytes_g.items() if k != "expert_DROPPED")
full_p = sum(logical.values())
print()
print(f"FULL  K3        : {full_p/1e9:8.1f} B params   {sum(bytes_g.values())/1e9:8.1f} GB")
print(f"REAP80 (ours)   : {kept_p/1e9:8.1f} B params   {kept_b/1e9:8.1f} GB")
print(f"  router-only exp: {logical['expert_KEPT']/1e9:8.1f} B  (already MXFP4 4.25bpw)")
print()
print("KT ladder projections on %.1f B params:" % (kept_p/1e9))
for name, bpw in (("MXFP4 passthrough (cell zero)", 4.25), ("IQ3_KT", 3.15),
                  ("IQ2_KT", 2.145), ("IQ1_KT", 1.75)):
    # experts at bpw, dense held at Q8_0 8.5bpw
    dense = kept_p - logical["expert_KEPT"]
    gb = (logical["expert_KEPT"] * bpw + dense * 8.5) / 8 / 1e9
    gb_all = kept_p * bpw / 8 / 1e9
    print(f"  {name:<32} experts@bpw+dense_q8 = {gb:7.1f} GB   |  uniform = {gb_all:7.1f} GB")
