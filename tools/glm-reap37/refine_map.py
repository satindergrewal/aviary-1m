#!/usr/bin/env python3
"""Refine the REAP37 kept-expert map against the gate.weight router matrix.

The e_score_subsequence recovery is exact where F32 scalars are distinct, but off-by-one
where two master experts share a near-tie e_score (pipenetwork's own subsequence shifted
one index). gate.weight (BF16 [256, hidden]) is a second, independent signal with corr==1.0
on the correct row - it has far more dimensions than a scalar, so it disambiguates.

For each MoE layer, run the recovered map through a local hill-climb: for any slot whose
gate corr vs its mapped master row < 0.99, search the whole master index range for a
higher-corr candidate, subject to monotonicity (prev_slot+1 <= candidate <= next_slot-1).
A corr==1.0 row is exact truth, not a tie.

No full download. Range-fetches gate.weight per layer. No GPU. Idempotent.
"""
import json, urllib.request, struct as st, sys
import numpy as np

REAP_BASE = "https://huggingface.co/pipenetwork/GLM-5.2-REAP37-MLX-4bit/resolve/main/"
MASTER_BASE = "https://huggingface.co/zai-org/GLM-5.2/resolve/main/"
GATE_W = "model.layers.{il}.mlp.gate.weight"


def fetch_range(url, a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()


class Store:
    def __init__(self, base):
        self.base = base
        idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())
        self.wm = idx["weight_map"]
        self.hh = {}
        self.cache = {}

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        shard = self.wm[name]
        if shard not in self.hh:
            hlen = st.unpack("<Q", fetch_range(self.base + shard, 0, 7))[0]
            self.hh[shard] = (json.loads(fetch_range(self.base + shard, 8, 8 + hlen - 1)), hlen)
        hdr, hlen = self.hh[shard]
        m = hdr[name]
        a, b = m["data_offsets"]
        raw = fetch_range(self.base + shard, 8 + hlen + a, 8 + hlen + b - 1)
        dt = m["dtype"]
        if dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
        elif dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32).reshape(m["shape"])
        else:
            raise ValueError(f"{name} dtype {dt}")
        self.cache[name] = arr
        return arr


def refine_layer(rg, mg, kept_map):
    """Hill-climb a recovered map slot-by-slot against gate corr, monotonic-preserving."""
    rn = rg / (np.linalg.norm(rg, axis=1, keepdims=True) + 1e-12)
    mn = mg / (np.linalg.norm(mg, axis=1, keepdims=True) + 1e-12)
    sim = rn @ mn.T                                            # [160, 256]
    n = len(kept_map)
    out = list(kept_map)
    nfix = 0
    for i in range(n):
        lo = (out[i-1] + 1) if i > 0 else 0
        hi = (out[i+1] - 1) if i + 1 < n else mg.shape[0] - 1
        current = out[i]
        if current < lo or current > hi or sim[i, current] < 0.99:
            # find best candidate in [lo, hi]
            sub = sim[i, lo:hi + 1]
            j_rel = int(np.argmax(sub))
            j = lo + j_rel
            if sim[i, j] > sim[i, min(max(current, lo), hi)]:
                out[i] = j
                nfix += 1
    return out, nfix


def main():
    map_path, out_path = sys.argv[1], sys.argv[2]
    layer_arg = sys.argv[3] if len(sys.argv) > 3 else "all"

    raw = json.load(open(map_path))
    kept = {int(k): v for k, v in raw.items()}
    reap, master = Store(REAP_BASE), Store(MASTER_BASE)

    layers = sorted(kept)
    if layer_arg != "all":
        lo, hi = layer_arg.split("-")
        layers = [il for il in layers if int(lo) <= il <= int(hi)]

    refined = {}
    total_fix = 0
    for il in layers:
        rg = reap.get(GATE_W.format(il=il))
        mg = master.get(GATE_W.format(il=il))
        if rg.shape[0] != len(kept[il]):
            raise ValueError(f"layer {il}: reap gate has {rg.shape[0]} rows, map has {len(kept[il])}")
        res, nfix = refine_layer(rg, mg, kept[il])
        refined[str(il)] = res
        total_fix += nfix
        if nfix:
            print(f"layer {il}: fixed {nfix} off-by-one slots")

    json.dump(refined, open(out_path, "w"), indent=1)
    print(f"\nWROTE {out_path}: {len(refined)} layers; {total_fix} slots corrected")


if __name__ == "__main__":
    main()
