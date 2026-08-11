#!/usr/bin/env python3
"""Build a REAP80-equivalent Kimi-K3 from OUR OWN full master checkpoint.

pipenetwork publishes pruned WEIGHTS but no index lists, and their harness is
MLX/Apple-only. We recovered the kept-expert map from e_score_correction_bias
scalars (monotonic constraint search that COUNTS solutions to prove uniqueness)
and confirmed it on a second independent tensor: dequantised MLX router rows
match master rows at corr 0.99995+ while wrong rows sit at <=0.15.

This emits plain HF in the MASTER's own naming, NOT MLX's fused switch_mlp
layout, so the existing K3 GGUF converter works unmodified.

Per MoE layer:
  - keep only listed experts' w1/w2/w3 (weight_packed + weight_scale), renumbered 0..N-1
  - slice gate.weight rows and gate.e_score_correction_bias to the kept set
  - copy every other tensor byte-for-byte

Expert payloads are copied as RAW BYTES: no dequant, no requant. Survivors stay
bit-exact with the master, which is what makes this reproducible and auditable.
"""
import json, os, re, struct, sys, shutil, time

SHARD_LIMIT = 40 * 1024**3


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


DT = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2,
      "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U32": 4, "U16": 2}


def nbytes(meta):
    n = DT[meta["dtype"]]
    for d in meta["shape"]:
        n *= d
    return n


def human(b):
    return f"{b/1e9:.1f} GB"


def build_ops(master, kept):
    """Return list of ops: (dst_name, dtype, dst_shape, shard, abs_off, keep_rows, row_bytes, size)."""
    wm = json.load(open(os.path.join(master, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(wm.values()))
    hdrs = {}
    for i, s in enumerate(shards):
        hdrs[s] = read_header(os.path.join(master, s))
    shard_ord = {s: i for i, s in enumerate(shards)}

    expert_re = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")
    router_re = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$")

    ops = []
    n_exp = n_router = n_pass = 0
    for name, shard in wm.items():
        hdr, base = hdrs[shard]
        meta = hdr[name]
        off = base + meta["data_offsets"][0]
        shape = list(meta["shape"])

        m = expert_re.search(name)
        if m:
            il, ei = int(m.group(1)), int(m.group(2))
            keep = kept.get(il)
            if keep is None:                       # layer not pruned: pass through
                ops.append((name, meta["dtype"], shape, shard, off, None, 0, nbytes(meta)))
                n_pass += 1
                continue
            pos = keep.get(ei)
            if pos is None:                        # dropped expert
                continue
            dst = name.replace(f".experts.{ei}.", f".experts.{pos}.")
            ops.append((dst, meta["dtype"], shape, shard, off, None, 0, nbytes(meta)))
            n_exp += 1
            continue

        r = router_re.search(name)
        if r:
            il = int(r.group(1))
            keep = kept.get(il)
            if keep is not None:
                rows = shape[0]
                rb = nbytes(meta) // rows
                order = [e for e, _ in sorted(keep.items(), key=lambda kv: kv[1])]
                shape[0] = len(order)
                ops.append((name, meta["dtype"], shape, shard, off, order, rb, len(order) * rb))
                n_router += 1
                continue

        ops.append((name, meta["dtype"], shape, shard, off, None, 0, nbytes(meta)))
        n_pass += 1

    # sequential-read order: by source shard then source offset
    ops.sort(key=lambda o: (shard_ord[o[3]], o[4]))
    return ops, dict(expert=n_exp, router=n_router, passthrough=n_pass, shards=len(shards))


def write_out(master, outdir, ops, dry=False):
    os.makedirs(outdir, exist_ok=True)
    groups, cur, cur_b = [], [], 0
    for op in ops:
        if cur and cur_b + op[7] > SHARD_LIMIT:
            groups.append((cur, cur_b)); cur, cur_b = [], 0
        cur.append(op); cur_b += op[7]
    if cur:
        groups.append((cur, cur_b))

    total = sum(g[1] for g in groups)
    print(f"plan: {len(ops)} tensors -> {len(groups)} shards, {human(total)}")
    if dry:
        return None

    weight_map, written = {}, 0
    t0 = time.time()
    handles = {}
    for gi, (grp, gb) in enumerate(groups, 1):
        fname = f"model-{gi:05d}-of-{len(groups):05d}.safetensors"
        hdr, off = {}, 0
        for (dst, dt, shape, _s, _o, _k, _rb, size) in grp:
            hdr[dst] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + size]}
            off += size
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        blob += b" " * ((8 - len(blob) % 8) % 8)

        path = os.path.join(outdir, fname)
        with open(path, "wb", buffering=1 << 22) as w:
            w.write(struct.pack("<Q", len(blob))); w.write(blob)
            for (dst, dt, shape, shard, so, keep, rb, size) in grp:
                f = handles.get(shard)
                if f is None:
                    f = handles[shard] = open(os.path.join(master, shard), "rb", buffering=1 << 22)
                if keep is None:
                    f.seek(so)
                    left = size
                    while left:
                        chunk = f.read(min(1 << 24, left))
                        if not chunk:
                            raise IOError(f"short read {dst}")
                        w.write(chunk); left -= len(chunk)
                else:
                    for row in keep:
                        f.seek(so + row * rb)
                        d = f.read(rb)
                        if len(d) != rb:
                            raise IOError(f"short row {dst}")
                        w.write(d)
                weight_map[dst] = fname
        written += gb
        el = time.time() - t0
        print(f"  [{gi}/{len(groups)}] {fname} {human(gb)}  "
              f"({human(written)} done, {written/el/1e6:.0f} MB/s, {el/60:.1f} min)", flush=True)

    for f in handles.values():
        f.close()
    with open(os.path.join(outdir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": written}, "weight_map": weight_map}, f, indent=1)
    return written


def write_aux(master, outdir, kept, n_layers=93):
    for fn in sorted(os.listdir(master)):
        src = os.path.join(master, fn)
        if not os.path.isfile(src):
            continue
        if fn.endswith(".safetensors") or fn.endswith("index.json"):
            continue
        shutil.copy2(src, os.path.join(outdir, fn))

    cfg_path = os.path.join(outdir, "config.json")
    cfg = json.load(open(cfg_path))
    tgt = cfg["text_config"] if "text_config" in cfg else cfg
    n_keep = len(next(iter(kept.values())))
    tgt["num_experts"] = n_keep
    tgt["expert_counts"] = [len(kept.get(i, {})) for i in range(n_layers)]
    cfg["_reap_provenance"] = {
        "source": "satindergrewal local full-precision K3 master",
        "recipe": "REAP80 kept-expert map recovered from pipenetwork/Kimi-K3-REAP80 "
                  "e_score_correction_bias, uniqueness proven by solution-counting "
                  "monotonic search; confirmed independently against dequantised "
                  "MLX router rows (corr >= 0.99995 vs <= 0.15 for wrong rows)",
        "experts_kept_per_moe_layer": n_keep,
        "experts_original": 896,
        "expert_payloads": "raw byte copy, bit-exact with master (no dequant/requant)",
    }
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"aux: config num_experts -> {n_keep}, expert_counts written")


def verify(outdir, kept, n_exp_expect):
    wm = json.load(open(os.path.join(outdir, "model.safetensors.index.json")))["weight_map"]
    hdrs = {}
    bad = []
    per_layer = {}
    for name, shard in wm.items():
        if shard not in hdrs:
            hdrs[shard] = read_header(os.path.join(outdir, shard))[0]
    for name in wm:
        m = re.search(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.", name)
        if m:
            per_layer.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        r = re.search(r"\.layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$", name)
        if r:
            il = int(r.group(1))
            if il in kept:
                sh = hdrs[wm[name]][name]["shape"]
                if sh[0] != len(kept[il]):
                    bad.append((name, sh))
    n_keep = len(next(iter(kept.values())))
    for il, s in per_layer.items():
        if il in kept and (len(s) != n_keep or max(s) != n_keep - 1 or min(s) != 0):
            bad.append((f"layer {il} experts", sorted(s)[:5] + ["..."] + sorted(s)[-2:]))
    print(f"verify: {len(wm)} tensors, {len(per_layer)} MoE layers, "
          f"contiguous 0..{n_keep-1} experts each")
    if bad:
        print("VERIFY FAIL:", bad[:10]); return False
    print("VERIFY PASS")
    return True


def main():
    if len(sys.argv) < 4:
        print("usage: reap_subset.py <master_dir> <kept_indices.json> <out_dir> [--plan]")
        sys.exit(1)
    master, idxfile, outdir = sys.argv[1:4]
    dry = "--plan" in sys.argv

    raw = json.load(open(idxfile))
    kept = {int(k): {int(e): p for p, e in enumerate(v)} for k, v in raw.items()}
    counts = {len(v) for v in kept.values()}
    print(f"map: {len(kept)} layers, keep-count(s) {sorted(counts)}")

    ops, stats = build_ops(master, kept)
    print(f"ops: expert={stats['expert']} router={stats['router']} "
          f"passthrough={stats['passthrough']} (from {stats['shards']} source shards)")

    written = write_out(master, outdir, ops, dry=dry)
    if dry:
        return
    write_aux(master, outdir, kept)
    ok = verify(outdir, kept, stats["expert"])
    print(f"TOTAL {human(written)} -> {outdir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
