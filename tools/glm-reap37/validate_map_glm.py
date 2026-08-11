#!/usr/bin/env python3
"""SECOND-SIGNAL validation of the recovered REAP37 kept-expert map (GLM-5.2).

The map was recovered by matching e_score_correction_bias F32 scalars (one signal). This
validates it against a completely different tensor: the ROUTER WEIGHT MATRIX.

pipenetwork kept the router high-precision: mlp.gate.weight in REAP37 is BF16 [160, 6144],
NOT 4-bit (only the experts in mlp.switch_mlp got 4-bit). Master's mlp.gate.weight is
F32/BF16 [256, 6144]. If the map is right, REAP37 row r correlates with master row
kept[r] near 1.0, and a random wrong row correlates near 0. Same control-first design as
tools/k3/validate_map.py. Range-fetches one layer's gate.weight at a time - no full
download. No GPU.
"""
import json, urllib.request, struct as st, sys
import numpy as np

REAP_BASE = "https://huggingface.co/pipenetwork/GLM-5.2-REAP37-MLX-4bit/resolve/main/"
MASTER_BASE = "https://huggingface.co/zai-org/GLM-5.2/resolve/main/"
GATE_W = "model.layers.{il}.mlp.gate.weight"
MAP_PATH = "<BOX>/glm-reap37/reap37_kept_indices.json"


def fetch_range(url, a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()


class Store:
    def __init__(self, base):
        self.base = base
        idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())
        self.wm = idx["weight_map"]
        self.shard_hdr = {}

    def _hdr(self, shard):
        if shard not in self.shard_hdr:
            url = self.base + shard
            hlen = st.unpack("<Q", fetch_range(url, 0, 7))[0]
            self.shard_hdr[shard] = (json.loads(fetch_range(url, 8, 8 + hlen - 1)), hlen)
        return self.shard_hdr[shard]

    def get(self, name):
        shard = self.wm.get(name)
        if shard is None:
            raise KeyError(name)
        hdr, hlen = self._hdr(shard)
        m = hdr[name]
        a, b = m["data_offsets"]
        raw = fetch_range(self.base + shard, 8 + hlen + a, 8 + hlen + b - 1)
        dt = m["dtype"]
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(m["shape"])
        if dt == "U32":
            return np.frombuffer(raw, dtype=np.uint32).reshape(m["shape"])
        if dt == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(m["shape"])
        raise ValueError(f"{name} dtype {dt}")


def main():
    kept = {int(k): v for k, v in json.load(open(sys.argv[1] if len(sys.argv) > 1 else MAP_PATH)).items()}
    reap = Store(REAP_BASE)
    master = Store(MASTER_BASE)
    rng = np.random.default_rng(0)

    layers = sorted(kept)
    probe = [layers[0], layers[len(layers)//4], layers[len(layers)//2], layers[3*len(layers)//4], layers[-1]]
    if len(sys.argv) > 2 and sys.argv[2] == "--all":
        probe = layers

    print(f"probing {len(probe)} layers of {len(layers)}")
    print(f"{'layer':>5} {'n':>4} {'median_corr':>12} {'min_corr':>10} {'ctrl_corr':>10}")
    bad = []
    for il in probe:
        deq = reap.get(GATE_W.format(il=il))          # [160, 6144] BF16 (high-precision)
        ref = master.get(f"model.layers.{il}.mlp.gate.weight")  # [256, hidden]
        if deq.shape[1] != ref.shape[1]:
            raise ValueError(f"layer {il}: hidden mismatch {deq.shape[1]} vs {ref.shape[1]}")

        idx = np.array(kept[il])
        tgt = ref[idx]

        dn = deq / (np.linalg.norm(deq, axis=1, keepdims=True) + 1e-12)
        tn = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-12)
        corr = (dn * tn).sum(axis=1)

        wrong = rng.permutation(ref.shape[0])[:len(idx)]
        wn = ref[wrong]
        wn = wn / (np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12)
        ctrl = float(np.median((dn * wn).sum(axis=1)))

        print(f"{il:>5} {len(idx):>4} {np.median(corr):>12.6f} {corr.min():>10.6f} {ctrl:>10.6f}")
        if corr.min() < 0.99:
            bad.append((il, int(np.argmin(corr)), float(corr.min())))

    print()
    if bad:
        print("FAIL rows (corr < 0.99):", bad[:20]); sys.exit(1)
    print("PASS - probed rows match their mapped master rows on an independent tensor; control ~0")


if __name__ == "__main__":
    main()


class Store:
    def __init__(self, base):
        self.base = base
        idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())
        self.wm = idx["weight_map"]
        self.shard_hdr = {}

    def _hdr(self, shard):
        if shard not in self.shard_hdr:
            url = self.base + shard
            hlen = st.unpack("<Q", fetch_range(url, 0, 7))[0]
            self.shard_hdr[shard] = (json.loads(fetch_range(url, 8, 8 + hlen - 1)), hlen)
        return self.shard_hdr[shard]

    def get(self, name):
        shard = self.wm.get(name)
        if shard is None:
            raise KeyError(name)
        hdr, hlen = self._hdr(shard)
        m = hdr[name]
        a, b = m["data_offsets"]
        raw = fetch_range(self.base + shard, 8 + hlen + a, 8 + hlen + b - 1)
        dt = m["dtype"]
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(m["shape"])
        if dt == "U32":
            return np.frombuffer(raw, dtype=np.uint32).reshape(m["shape"])
        if dt in ("U8", "I8"):
            return np.frombuffer(raw, dtype=np.uint8).reshape(m["shape"])
        raise ValueError(f"{name} dtype {dt}")


def mlx_dequant_4bit(qw, scales, biases):
    """qw: [R, C/8] uint32 (8 4-bit vals/word little-endian), group = C/scales.shape[1]."""
    q32 = qw.astype(np.uint64)
    vals = np.stack([(q32 >> (4 * k)) & 0xF for k in range(8)], axis=-1)  # [R, C/8, 8]
    q = vals.reshape(qw.shape[0], -1).astype(np.float32)                   # [R, C]
    R, C = q.shape
    g = C // scales.shape[1]
    q = q.reshape(R, scales.shape[1], g)
    return (q * scales[:, :, None] + biases[:, :, None]).reshape(R, C)


def main():
    kept = {int(k): v for k, v in json.load(open(sys.argv[1] if len(sys.argv) > 1 else MAP_PATH)).items()}
    reap = Store(REAP_BASE)
    master = Store(MASTER_BASE)
    rng = np.random.default_rng(0)

    layers = sorted(kept)
    probe = [layers[0], layers[len(layers)//4], layers[len(layers)//2], layers[3*len(layers)//4], layers[-1]]
    if len(sys.argv) > 2 and sys.argv[2] == "--all":
        probe = layers

    print(f"probing {len(probe)} layers of {len(layers)}")
    print(f"{'layer':>5} {'n':>4} {'median_corr':>12} {'min_corr':>10} {'ctrl_corr':>10}")
    bad = []
    for il in probe:
        qw = reap.get(GATE_W.format(il=il))          # [R, C/8] uint32
        sc = reap.get(GATE_SC.format(il=il))         # [R, groups]
        bi = reap.get(GATE_BI.format(il=il)) if GATE_BI.format(il=il) in reap.wm else np.zeros_like(sc)
        deq = mlx_dequant_4bit(qw, sc, bi)            # [160, hidden]
        ref = master.get(f"model.layers.{il}.mlp.gate.weight")  # [256, hidden]

        idx = np.array(kept[il])
        tgt = ref[idx]

        dn = deq / (np.linalg.norm(deq, axis=1, keepdims=True) + 1e-12)
        tn = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-12)
        corr = (dn * tn).sum(axis=1)

        wrong = rng.permutation(ref.shape[0])[:len(idx)]
        wn = ref[wrong]
        wn = wn / (np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12)
        ctrl = float(np.median((dn * wn).sum(axis=1)))

        print(f"{il:>5} {len(idx):>4} {np.median(corr):>12.6f} {corr.min():>10.6f} {ctrl:>10.6f}")
        if corr.min() < 0.95:
            bad.append((il, int(np.argmin(corr)), float(corr.min())))

    print()
    if bad:
        print("FAIL rows (corr < 0.95):", bad[:20]); sys.exit(1)
    print("PASS - probed rows match their mapped master rows on an independent tensor; control ~0")


if __name__ == "__main__":
    main()
