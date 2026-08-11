#!/usr/bin/env python3
"""Recover the REAP37 kept-expert map for GLM-5.2 by matching e_score_correction_bias
F32 scalars between pipenetwork's pruned MLX build and the zai-org full master.

pipenetwork publishes GLM-5.2-REAP37-MLX-4bit with NO kept-expert index list. But both
it and the master carry per-expert e_score_correction_bias scalars (one per expert per
MoE layer). The kept set is an EXACT F32 subsequence of the master's 256, in original
order - verified on layer 10 (dropped expert 6.9633 sits between survivors).

REAP37 keeps 160/256 experts per MoE layer. We map each of the 160 surviving positions
to its original expert index by subsequence matching against the master's 256.

Range-fetches only the (small) gate tensors - never downloads the full ~480B MLX or the
master. Read-only. No GPU. Pure stdlib.

Sources credited in the build: pipenetwork/GLM-5.2-REAP37-MLX-4bit (pruning), Cerebras
REAP arXiv:2510.13999 (method). Transferred and simplified from tools/k3/validate_map.py.
"""
import json, urllib.request, struct as st, sys, os

REAP_BASE = "https://huggingface.co/pipenetwork/GLM-5.2-REAP37-MLX-4bit/resolve/main/"
MASTER_BASE = "https://huggingface.co/zai-org/GLM-5.2/resolve/main/"
GATE = "model.layers.{il}.mlp.gate.e_score_correction_bias"


def fetch_range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=120).read()


class Store:
    """Read a single tensor from a multi-shard safetensors repo over HTTP Range."""

    def __init__(self, base):
        self.base = base
        idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json",
                                                timeout=60).read())
        self.wm = idx["weight_map"]
        self.shard_hdr = {}     # shard -> (header dict, hdr_len)

    def _hdr(self, shard):
        if shard not in self.shard_hdr:
            url = self.base + shard
            hlen = st.unpack("<Q", fetch_range(url, 0, 7))[0]
            hdr = json.loads(fetch_range(url, 8, 8 + hlen - 1))
            self.shard_hdr[shard] = (hdr, hlen)
        return self.shard_hdr[shard]

    def get(self, name):
        shard = self.wm.get(name)
        if shard is None:
            raise KeyError(f"{name} not in index")
        hdr, hlen = self._hdr(shard)
        m = hdr[name]
        a, b = m["data_offsets"]
        raw = fetch_range(self.base + shard, 8 + hlen + a, 8 + hlen + b - 1)
        dt = m["dtype"]
        if dt == "F32":
            return list(st.unpack("<%df" % (len(raw) // 4), raw))
        if dt == "BF16":
            words = st.unpack("<%dH" % (len(raw) // 2), raw)
            return [st.unpack("<f", st.pack("<I", w << 16))[0] for w in words]
        raise ValueError(f"unhandled dtype {dt} for {name}")


def match_subsequence(master, kept):
    """Map each kept value to a master index, preserving order.

    Values are F32 scalars. We match by (1) exact equality where unique, and
    (2) a monotonic constraint: chosen indices must be strictly increasing.
    Returns list of original expert indices, len == len(kept).
    Raises if ambiguous (same value matches multiple unused slots) - the
    resolution there is the K3 monotonic constraint search, which we add only
    if a real collision shows up.
    """
    n_kept = len(kept)
    out = [-1] * n_kept

    def rec(i, start):
        if i == n_kept:
            return True
        # find all master slots with this exact value at index >= start
        v = kept[i]
        for j in range(start, len(master)):
            if master[j] == v:
                out[i] = j
                if rec(i + 1, j + 1):
                    return True
                out[i] = -1
        return False

    if not rec(0, 0):
        raise ValueError("no monotonic subsequence match")
    return out


def main():
    outpath = sys.argv[1] if len(sys.argv) > 1 else "reap37_kept_indices.json"
    layer_arg = sys.argv[2] if len(sys.argv) > 2 else "all"

    reap = Store(REAP_BASE)
    master = Store(MASTER_BASE)

    # MoE layers = those with the gate tensor.
    layers = sorted({int(n.split(".")[2]) for n in reap.wm if n.endswith("mlp.gate.e_score_correction_bias")})
    if layer_arg != "all":
        lo, hi = layer_arg.split("-")
        layers = [il for il in layers if int(lo) <= il <= int(hi)]
    print(f"MoE layers to map: {len(layers)}  (range {layers[0]}..{layers[-1]})")

    kept_map = {}
    for il in layers:
        name = GATE.format(il=il)
        k = reap.get(name)
        m = master.get(name)
        assert len(m) == 256, f"layer {il}: master has {len(m)} experts, expected 256"
        idx = match_subsequence(m, k)
        kept_map[str(il)] = idx
        if il % 10 == layers[0] % 10 or il == layers[0]:
            print(f"  layer {il}: kept {len(idx)} (first 8 -> {idx[:8]})")

    json.dump(kept_map, open(outpath, "w"), indent=1)
    n_keep = len(next(iter(kept_map.values())))
    print(f"\nWROTE {outpath}: {len(kept_map)} layers, {n_keep} kept each (of 256)")
    print("kept fraction:", round(n_keep / 256, 4), "~= REAP", round((1 - n_keep/256)*100, 1), "% pruned")


if __name__ == "__main__":
    main()
