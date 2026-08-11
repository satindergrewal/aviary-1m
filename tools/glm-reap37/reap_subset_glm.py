#!/usr/bin/env python3
"""Build a REAP37 GLM-5.2 from the zai-org full master using a recovered kept-expert map.

Adapted from tools/k3/reap_subset.py for GLM's MoE naming: experts live at
`model.layers.N.mlp.experts.<e>.{gate,up,down}_proj.weight`, router at
`model.layers.N.mlp.gate.weight` + `model.layers.N.mlp.gate.e_score_correction_bias`.
(K3 was `block_sparse_moe`; concept identical.)

Per MoE layer:
  - keep only listed experts' gate/up/down_proj, renumbered 0..N-1
  - slice mlp.gate.weight rows and mlp.gate.e_score_correction_bias to the kept set
  - copy every other tensor byte-for-byte (no dequant/requant; survivors stay
    bit-exact with the master)

Emits plain HF safetensors in the master's own naming so the standard GLM GGUF
converter works unmodified. Read-only on the master. No GPU.
"""
import json, os, re, struct, sys, shutil, time

SHARD_LIMIT = 40 * 1024**3
# GLM-5.2: transformer layers 0..77 (dense 0-2, MoE 3..77 = 75 MoE layers), plus the
# NextN/MTP block at layer 78 which IS a full MoE layer (HF index: 771 expert keys +
# gate.weight + e_score_correction_bias; verified 2026-08-09 against the zai-org index
# and the GGUF master). The REAP37 MLX drops the MTP head entirely.
# DONOR RULE: layer 78 is sliced with layer 77's kept set. Passthrough would leave it
# 256-wide against config n_routed_experts=160, and llama.cpp create_tensor() builds
# every MoE layer at the hparams shape -> hard load failure. Donor risk is bounded to
# MTP drafter acceptance, never target output quality.


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


expert_re = re.compile(r"\bmlp\.experts\.(\d+)\." )   # captures expert idx; layer handled separately


def build_ops(master, kept):
    """kept: {layer_int: {orig_expert_int: new_pos_int}}. Layer = model.layers.N."""
    wm = json.load(open(os.path.join(master, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(wm.values()))
    hdrs = {s: read_header(os.path.join(master, s)) for i, s in enumerate(shards)}
    shard_ord = {s: i for i, s in enumerate(shards)}

    # split into layer prefix + expert/router detection
    layer_re = re.compile(r"\blayers\.(\d+)\.")

    ops = []
    n_exp = n_router = n_pass = 0
    for name, shard in wm.items():
        hdr, base = hdrs[shard]
        meta = hdr[name]
        off = base + meta["data_offsets"][0]
        shape = list(meta["shape"])

        lm = layer_re.search(name)
        il = int(lm.group(1)) if lm else -1

        m = expert_re.search(name)
        if m and il >= 0:
            ei = int(m.group(1))
            keep = kept.get(il)
            if keep is None:
                ops.append((name, meta["dtype"], shape, shard, off, None, 0, nbytes(meta)))
                n_pass += 1
                continue
            pos = keep.get(ei)
            if pos is None:                       # dropped expert
                continue
            dst = name.replace(f".experts.{ei}.", f".experts.{pos}.")
            ops.append((dst, meta["dtype"], shape, shard, off, None, 0, nbytes(meta)))
            n_exp += 1
            continue

        if il >= 0 and (name.endswith("mlp.gate.weight") or name.endswith("mlp.gate.e_score_correction_bias")):
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
                    f.seek(so); left = size
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


def write_aux(master, outdir, kept):
    for fn in sorted(os.listdir(master)):
        src = os.path.join(master, fn)
        if not os.path.isfile(src):
            continue
        if fn.endswith(".safetensors") or fn.endswith("index.json"):
            continue
        shutil.copy2(src, os.path.join(outdir, fn))

    cfg_path = os.path.join(outdir, "config.json")
    cfg = json.load(open(cfg_path))
    tgt = cfg.get("text_config", cfg)
    n_keep = len(next(iter(kept.values())))
    for key in ("moe_num_experts", "num_experts", "n_routed_experts"):
        if key in tgt:
            tgt[key] = n_keep
    # kept map is keyed by ACTUAL MoE layer index (3..77 on GLM-5.2) + donor entry 78
    # (NextN block, sliced with layer 77's set). Dense layers 0-2 have no router;
    # expert_counts covers ONLY the MoE layers, keyed by their real index.
    # num_hidden_layers stays 78.
    tgt["expert_counts"] = {str(il): len(kept[il]) for il in sorted(kept)}
    cfg["_reap_provenance"] = {
        "source": "satindergrewal local zai-org GLM-5.2 full-precision master",
        "recipe": "REAP37 kept-expert map recovered from pipenetwork/GLM-5.2-REAP37-MLX-4bit "
                  "via e_score_correction_bias F32 subsequence match (exact, monotonic); "
                  "Cerebras REAP arXiv:2510.13999 method",
        "experts_kept_per_moe_layer": n_keep,
        "experts_original": 256,
        "expert_payloads": "raw byte copy, bit-exact with master (no dequant/requant)",
        "mtp_head_layer": "donor-sliced with layer 77 kept set (layer 78 NextN IS a full "
                          "MoE layer; uniform 160 required by llama.cpp create_tensor)",
    }
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"aux: config expert-count fields -> {n_keep}, expert_counts written for MoE layers 3..77")


def verify(outdir, kept):
    wm = json.load(open(os.path.join(outdir, "model.safetensors.index.json")))["weight_map"]
    hdrs = {}
    bad = []
    for name, shard in wm.items():
        if shard not in hdrs:
            hdrs[shard] = read_header(os.path.join(outdir, shard))[0]
    per_layer = {}
    for name in wm:
        m = expert_re.search(name)
        l = re.search(r"\blayers\.(\d+)\.", name)
        if m and l:
            per_layer.setdefault(int(l.group(1)), set()).add(int(m.group(1)))
        if l and name.endswith("mlp.gate.e_score_correction_bias"):
            il = int(l.group(1))
            if il in kept:
                sh = hdrs[wm[name]][name]["shape"]
                if sh[0] != len(kept[il]):
                    bad.append((name, sh))
    n_keep = len(next(iter(kept.values())))
    for il, s in per_layer.items():
        if il in kept and (len(s) != n_keep or max(s) != n_keep - 1 or min(s) != 0):
            bad.append((f"layer {il} experts", sorted(s)[:3] + ["..."] + sorted(s)[-2:]))
    print(f"verify: {len(wm)} tensors, {len(per_layer)} MoE layers, contiguous 0..{n_keep-1} experts each")
    if bad:
        print("VERIFY FAIL:", bad[:10]); return False
    print("VERIFY PASS")
    return True


def main():
    if len(sys.argv) < 4:
        print("usage: reap_subset_glm.py <master_dir> <kept_indices.json> <out_dir> [--plan]")
        sys.exit(1)
    master, idxfile, outdir = sys.argv[1:4]
    dry = "--plan" in sys.argv

    raw = json.load(open(idxfile))
    kept = {int(k): {int(e): p for p, e in enumerate(v)} for k, v in raw.items()}
    # DONOR (2026-08-09): layer 78 (NextN/MTP) is a full MoE layer; slice it with
    # layer 77's kept set so the whole model is uniformly 160-wide. See header note.
    kept[78] = kept[77]
    counts = {len(v) for v in kept.values()}
    print(f"map: {len(kept)} layers, keep-count(s) {sorted(counts)}")

    ops, stats = build_ops(master, kept)
    print(f"ops: expert={stats['expert']} router={stats['router']} "
          f"passthrough={stats['passthrough']} (from {stats['shards']} source shards)")

    written = write_out(master, outdir, ops, dry=dry)
    if dry:
        return
    write_aux(master, outdir, kept)
    ok = verify(outdir, kept)
    print(f"TOTAL {human(written)} -> {outdir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
