#!/usr/bin/env python3
"""Verify already-written subsetter shards WITHOUT waiting for the run to finish.

Checks, per completed shard:
  1. header parses, declared sizes match file length
  2. renumbered expert tensors point at the RIGHT master expert - byte-for-byte
  3. router rows (gate.weight, e_score_correction_bias) equal master's kept rows

Byte equality is the whole claim: survivors must be bit-exact with the master.
"""
import json, os, re, struct, sys, hashlib, random

MASTER = "<BOX>/kimi-k3"
OUT = "<BOX>/bigmodels/kimi-k3-reap80-ours"
MAP = "<BOX>/reap80_kept_indices.json"

DT = {"F64":8,"I64":8,"F32":4,"I32":4,"BF16":2,"F16":2,"I16":2,"U8":1,"I8":1,"BOOL":1,"U32":4}


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def nbytes(m):
    n = DT[m["dtype"]]
    for d in m["shape"]:
        n *= d
    return n


def blob(path, off, size):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(size)


def main():
    kept = {int(k): v for k, v in json.load(open(MAP)).items()}
    mwm = json.load(open(os.path.join(MASTER, "model.safetensors.index.json")))["weight_map"]
    mhdr = {}

    shards = sorted(f for f in os.listdir(OUT) if f.endswith(".safetensors"))
    # last shard may still be growing - only check complete ones
    complete = []
    for s in shards:
        p = os.path.join(OUT, s)
        try:
            hdr, base = read_header(p)
        except Exception as e:
            print(f"{s}: header unreadable ({e}) - still being written"); continue
        need = base + max(v["data_offsets"][1] for v in hdr.values())
        have = os.path.getsize(p)
        if have < need:
            print(f"{s}: {have/1e9:.1f}/{need/1e9:.1f} GB - still being written"); continue
        complete.append((s, hdr, base))
        print(f"{s}: COMPLETE, {len(hdr)} tensors, {have/1e9:.1f} GB, size matches header")

    if not complete:
        print("no complete shards yet"); return

    rng = random.Random(1234)
    expert_re = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")
    checked_e = checked_r = 0
    fails = []

    for s, hdr, base in complete:
        path = os.path.join(OUT, s)
        names = list(hdr)
        experts = [n for n in names if expert_re.search(n)]
        routers = [n for n in names if n.endswith("gate.weight") or n.endswith("e_score_correction_bias")]

        for name in rng.sample(experts, min(12, len(experts))):
            m = expert_re.search(name)
            il, new_i = int(m.group(1)), int(m.group(2))
            old_i = kept[il][new_i]
            src = name.replace(f".experts.{new_i}.", f".experts.{old_i}.")
            sm = mwm[src]
            if sm not in mhdr:
                mhdr[sm] = read_header(os.path.join(MASTER, sm))
            smh, sbase = mhdr[sm]
            a, b = smh[src]["data_offsets"]
            want = blob(os.path.join(MASTER, sm), sbase + a, b - a)
            oa, ob = hdr[name]["data_offsets"]
            got = blob(path, base + oa, ob - oa)
            ok = (want == got) and (smh[src]["shape"] == hdr[name]["shape"])
            checked_e += 1
            if not ok:
                fails.append((name, f"expert bytes differ from master {src}"))

        for name in routers:
            il = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
            if il not in kept:
                continue
            if mwm.get(name) is None:
                fails.append((name, "not found in master")); continue
            sm = mwm[name]
            if sm not in mhdr:
                mhdr[sm] = read_header(os.path.join(MASTER, sm))
            smh, sbase = mhdr[sm]
            mm = smh[name]
            rows = mm["shape"][0]
            rb = nbytes(mm) // rows
            oa, ob = hdr[name]["data_offsets"]
            got = blob(path, base + oa, ob - oa)
            want = b"".join(blob(os.path.join(MASTER, sm), sbase + mm["data_offsets"][0] + r * rb, rb)
                            for r in kept[il])
            checked_r += 1
            if got != want or hdr[name]["shape"][0] != len(kept[il]):
                fails.append((name, f"router rows differ (shape {hdr[name]['shape']})"))

    print(f"\nchecked {checked_e} expert tensors byte-for-byte, {checked_r} router tensors row-for-row")
    if fails:
        print("FAIL:")
        for f in fails[:10]:
            print("  ", f)
        sys.exit(1)
    print("PASS - every checked tensor is bit-exact with the correct master source")


if __name__ == "__main__":
    main()
