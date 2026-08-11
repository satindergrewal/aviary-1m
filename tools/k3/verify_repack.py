#!/usr/bin/env python3
"""Empirically verify that PR #26185's repack_mxfp4_blocks is LOSSLESS on OUR weights.

The cell-zero claim for our K3 ladder is "MXFP4 passthrough is byte-exact with
Moonshot's release". That rests entirely on this function being a pure bit repack.
The docstring says so; the risk is nibble-order, so measure it.

Path A - decode the compressed-tensors source directly:
    byte i: low nibble = element 2i, high nibble = element 2i+1
    code: sign = bit 3, magnitude = E2M1[code & 7]
    value = (-1)^sign * mag * 2^(e-127)

Path B - run the repack, decode as ggml block_mxfp4:
    17 bytes/block: [0]=e, [1..16]=qs
    element j = kvalues[qs[j] & 0xF],  element j+16 = kvalues[qs[j] >> 4]
    value = kvalue * 2^(e-128)          (kvalues doubled, scale halved)

Exact equality is required - every representable value is dyadic.
"""
import json, os, struct, sys
import numpy as np
import torch

sys.path.insert(0, "<BOX>/llama.cpp-k3")
from conversion.base import repack_mxfp4_blocks

MASTER = "<BOX>/kimi-k3"

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float64)


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load(name, d=MASTER):
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    shard = wm[name]
    hdr, base = read_header(os.path.join(d, shard))
    m = hdr[name]
    a, b = m["data_offsets"]
    with open(os.path.join(d, shard), "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    return np.frombuffer(raw, dtype=np.uint8).reshape(m["shape"]).copy()


def decode_source(packed, scale):
    rows, pc = packed.shape
    cols = pc * 2
    nb = cols // 32
    lo = packed & 0x0F
    hi = packed >> 4
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    mag = E2M1[(codes & 7).astype(np.int64)]
    sign = np.where((codes & 8) != 0, -1.0, 1.0)
    exp = np.repeat(scale.astype(np.int64), 32, axis=1) - 127
    return sign * mag * np.exp2(exp.astype(np.float64))


def decode_ggml(raw, rows, cols):
    nb = cols // 32
    blk = raw.reshape(rows, nb, 17)
    e = blk[:, :, 0].astype(np.int64)
    qs = blk[:, :, 1:]
    lo = KVALUES[(qs & 0x0F).astype(np.int64)]
    hi = KVALUES[(qs >> 4).astype(np.int64)]
    out = np.concatenate([lo, hi], axis=2)                 # [rows, nb, 32]
    scale = np.exp2((e - 128).astype(np.float64))[:, :, None]
    return (out * scale).reshape(rows, cols)


def main():
    tested = 0
    for layer, exp_i, proj in ((3, 0, "w1"), (3, 0, "w2"), (47, 123, "w3"), (92, 178, "w1")):
        base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{exp_i}.{proj}"
        packed = load(base + ".weight_packed")
        scale = load(base + ".weight_scale")
        rows, cols = packed.shape[0], packed.shape[1] * 2

        a = decode_source(packed, scale)
        raw = repack_mxfp4_blocks(torch.from_numpy(packed), torch.from_numpy(scale))
        b = decode_ggml(np.asarray(raw, dtype=np.uint8), rows, cols)

        exact = np.array_equal(a, b)
        nz = np.count_nonzero(a)
        print(f"L{layer:>2} e{exp_i:<3} {proj}  shape {rows}x{cols}  "
              f"nonzero {nz/a.size:5.1%}  distinct {len(np.unique(a)):>4}  "
              f"EXACT={exact}")
        if not exact:
            bad = np.argwhere(a != b)[:5]
            print("   MISMATCH at", bad.tolist())
            print("   a:", [a[tuple(i)] for i in bad], " b:", [b[tuple(i)] for i in bad])
            sys.exit(1)
        tested += a.size

    print(f"\nLOSSLESS CONFIRMED on {tested:,} real weights - repack is a pure bit move.")
    print("=> MXFP4 passthrough is a legitimate byte-exact cell zero for our ladder.")


if __name__ == "__main__":
    main()
