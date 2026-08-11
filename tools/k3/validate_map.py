#!/usr/bin/env python3
"""SECOND-SOURCE validation of the recovered REAP80 kept-expert map.

The map was recovered by matching e_score_correction_bias F32 scalars. That is one
signal. This validates it against a completely different tensor: the ROUTER MATRIX.

REAP80's gate.weight is MLX 8-bit affine (U32, 4 vals/word, group 64, BF16
scales+biases). Master's gate.weight is BF16 [896, 7168]. If the map is right,
dequantised REAP row r must match master row kept[r] to within 8-bit quant noise,
and must NOT match a random other row.

Reads a few MB. No GPU. Read-only on both trees.
"""
import json, struct, os, sys
import numpy as np

MASTER = "<BOX>/kimi-k3"
REAP   = "<BOX>/kimi-k3-reap80"
MAP    = "<BOX>/reap80_kept_indices.json"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Store:
    def __init__(self, d):
        self.d = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def _h(self, shard):
        if shard not in self.hdr:
            self.hdr[shard] = read_header(os.path.join(self.d, shard))
        return self.hdr[shard]

    def get(self, name):
        shard = self.wm[name]
        hdr, base = self._h(shard)
        m = hdr[name]
        a, b = m["data_offsets"]
        with open(os.path.join(self.d, shard), "rb") as f:
            f.seek(base + a)
            raw = f.read(b - a)
        dt = m["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dt == "U32":
            arr = np.frombuffer(raw, dtype=np.uint32)
        elif dt in ("U8", "I8"):
            arr = np.frombuffer(raw, dtype=np.uint8)
        else:
            raise ValueError(dt)
        return arr.reshape(m["shape"])


def mlx_dequant_8bit(qw, scales, biases):
    """qw: [R, W] uint32, 4 uint8 values per word, little-endian byte order."""
    q = qw.view(np.uint8).reshape(qw.shape[0], -1).astype(np.float32)   # [R, C]
    R, C = q.shape
    g = C // scales.shape[1]
    q = q.reshape(R, scales.shape[1], g)
    return (q * scales[:, :, None] + biases[:, :, None]).reshape(R, C)


def main():
    kept = {int(k): v for k, v in json.load(open(MAP)).items()}
    m, r = Store(MASTER), Store(REAP)
    rng = np.random.default_rng(0)

    layers = sorted(kept)
    probe = [layers[0], layers[1], layers[len(layers)//4], layers[len(layers)//2],
             layers[3*len(layers)//4], layers[-2], layers[-1]]
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        probe = layers

    print(f"probing {len(probe)} layers of {len(layers)}\n")
    print(f"{'layer':>5} {'n':>4} {'median_corr':>12} {'min_corr':>10} {'median_relerr':>14} {'ctrl_corr':>10}")
    bad = []
    for il in probe:
        pre = f"model.layers.{il}.block_sparse_moe.gate"
        qw  = r.get(f"{pre}.weight")
        sc  = r.get(f"{pre}.scales")
        bi  = r.get(f"{pre}.biases")
        deq = mlx_dequant_8bit(qw, sc, bi)                       # [179, 7168]
        ref = m.get(f"language_model.model.layers.{il}.block_sparse_moe.gate.weight")  # [896, 7168]

        idx = np.array(kept[il])
        tgt = ref[idx]                                           # [179, 7168]

        dn = deq / (np.linalg.norm(deq, axis=1, keepdims=True) + 1e-12)
        tn = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-12)
        corr = (dn * tn).sum(axis=1)

        relerr = np.linalg.norm(deq - tgt, axis=1) / (np.linalg.norm(tgt, axis=1) + 1e-12)

        # control: same rows against a random WRONG assignment
        wrong = rng.permutation(896)[:len(idx)]
        wn = ref[wrong]
        wn = wn / (np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12)
        ctrl = float(np.median((dn * wn).sum(axis=1)))

        print(f"{il:>5} {len(idx):>4} {np.median(corr):>12.6f} {corr.min():>10.6f} "
              f"{np.median(relerr):>14.6f} {ctrl:>10.6f}")
        if corr.min() < 0.99:
            bad.append((il, int(np.argmin(corr)), float(corr.min())))

    print()
    if bad:
        print("FAIL rows (corr < 0.99):", bad[:20])
        sys.exit(1)
    print("PASS - every probed row matches its mapped master row on an independent tensor")


if __name__ == "__main__":
    main()
