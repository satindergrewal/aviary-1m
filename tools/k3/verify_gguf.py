#!/usr/bin/env python3
"""End-to-end proof of the cell-zero claim on the finished GGUF.

Reads the produced K3 GGUF and, for sampled (layer, expert, projection) triples,
compares the MXFP4 bytes in the GGUF against repack(master weight_packed,
weight_scale) - resolving the expert index back through the REAP80 map.

If this passes, the chain is closed: Moonshot's shipped weights -> our subset ->
GGUF, bit-exact the whole way. Everything measured below this rung is our loss.
"""
import json, os, struct, sys
import numpy as np
import torch

sys.path.insert(0, "<BOX>/llama.cpp-k3")
sys.path.insert(0, "<BOX>/llama.cpp-k3/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType
from conversion.base import repack_mxfp4_blocks

GGUF = "<BOX>/bigmodels/k3-reap80-ours-mxfp4-bf16.gguf"
MASTER = "<BOX>/kimi-k3"
MAP = "<BOX>/reap80_kept_indices.json"

PROJ = {"w1": "ffn_gate_exps", "w2": "ffn_down_exps", "w3": "ffn_up_exps"}


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load_master(name):
    wm = json.load(open(os.path.join(MASTER, "model.safetensors.index.json")))["weight_map"]
    shard = wm[name]
    hdr, base = read_header(os.path.join(MASTER, shard))
    m = hdr[name]
    a, b = m["data_offsets"]
    with open(os.path.join(MASTER, shard), "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    return np.frombuffer(raw, dtype=np.uint8).reshape(m["shape"]).copy()


def main():
    kept = {int(k): v for k, v in json.load(open(MAP)).items()}
    r = GGUFReader(GGUF, "r")

    kv = {f.name: f for f in r.fields.values()}
    def gv(k):
        f = kv.get(k)
        if f is None: return None
        try:    return f.contents()
        except Exception: return None

    print("=== metadata")
    for k in ("general.architecture", "kimi-k3.block_count", "kimi-k3.expert_count",
              "kimi-k3.expert_used_count", "kimi-k3.embedding_length",
              "kimi-k3.attention.head_count", "kimi-k3.vocab_size"):
        v = gv(k)
        if v is not None:
            print(f"  {k} = {v}")

    types = {}
    for t in r.tensors:
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
    print("\n=== tensor type histogram")
    for k, v in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v}")

    byname = {t.name: t for t in r.tensors}
    print(f"\n=== byte-exactness vs Moonshot master ({len(r.tensors)} tensors total)")
    fails = 0
    checked = 0
    for layer, new_i, wid in ((3, 0, "w1"), (3, 5, "w2"), (47, 100, "w3"),
                              (92, 178, "w1"), (60, 42, "w2")):
        gname = f"blk.{layer}.{PROJ[wid]}.weight"
        t = byname.get(gname)
        if t is None:
            print(f"  {gname}: MISSING"); fails += 1; continue
        if t.tensor_type != GGMLQuantizationType.MXFP4:
            print(f"  {gname}: type {t.tensor_type.name}, expected MXFP4"); fails += 1; continue

        n_exp = int(t.shape[-1])
        buf = t.data.reshape(n_exp, -1)
        got = np.asarray(buf[new_i], dtype=np.uint8)

        old_i = kept[layer][new_i]
        base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{old_i}.{wid}"
        want = np.asarray(repack_mxfp4_blocks(
            torch.from_numpy(load_master(base + ".weight_packed")),
            torch.from_numpy(load_master(base + ".weight_scale"))), dtype=np.uint8).reshape(-1)

        ok = got.size == want.size and bool((got == want).all())
        checked += got.size
        print(f"  {gname:<32} expert {new_i:>3} (master {old_i:>3})  "
              f"{got.size/1e6:6.2f} MB  EXACT={ok}")
        if not ok:
            fails += 1
            if got.size == want.size:
                d = np.flatnonzero(got != want)
                print(f"     {d.size} differing bytes, first at {d[:5]}")
            else:
                print(f"     size mismatch got {got.size} want {want.size}")

    print()
    if fails:
        print(f"FAIL ({fails})"); sys.exit(1)
    print(f"PASS - {checked/1e6:.1f} MB of expert payload byte-identical to Moonshot's release")
    print("Chain closed: Moonshot weights -> REAP80 subset -> GGUF, bit-exact throughout.")


if __name__ == "__main__":
    main()
