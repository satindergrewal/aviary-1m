# Kimi-K3 REAP80 pipeline

Builds a REAP80-equivalent Kimi-K3 (601 B params) from the **full K3 master**, and a
GGUF whose expert weights are **byte-identical to Moonshot's release**.

Built and verified 2026-07-29. The 402 GB HF intermediate and the 350 GB pipenetwork
MLX download were deleted afterwards; everything needed to rebuild them is here.

## Why this exists

pipenetwork publishes REAP-pruned K3 weights but no kept-expert index lists, and their
harness is MLX/Apple-only. We recovered the map from the checkpoint itself, then rebuilt
the pruned model from the owner's own master so the result is ours end to end.

## The model's real shape

| | params | disk |
|---|---:|---:|
| full K3 | 2779.9 B | 1560.9 GB |
| REAP80 (this) | 601.1 B | 403.4 GB |

Experts 543.9 B (289 GB), attention **36.2 B / 72.4 GB BF16 — the second biggest block**,
shared_experts 12.2 B, routed bottleneck 4.7 B, embed/head 2.3 B, vision 0.45 B.

Three architecture facts that drive every decision here:

1. **Experts ship already MXFP4** (`weight_packed` U8 + E8M0 `weight_scale`, group 32).
   No BF16 expert weights exist. A bf16 round trip would need ~5.5 TB — never do it.
2. **Low-rank routed-expert bottleneck**: hidden 7168 → `routed_expert_down_proj`
   [3584, 7168] → experts run at 3584 with intermediate 3072 → `routed_expert_up_proj`
   [7168, 3584]. Those three are **shared, not per-expert**. Do not slice them.
3. 93 layers: layer 0 dense, 92 MoE layers × 179 kept experts (from 896).

## Rebuild

```bash
MASTER=<BOX>/kimi-k3
MAP=<BOX>/bigmodels/reap80_kept_indices.json     # also on <BOX>
OUT=<BOX>/bigmodels/kimi-k3-reap80-ours
PY=~/miniconda3/envs/llama/bin/python                  # torch 2.6.0+cpu, tiktoken, blobfile

# 1. subset: ~36 min, 402.4 GB, 101,436 tensors, pure CPU/disk, master read-only
python3 reap_subset.py $MASTER $MAP $OUT

# 2. verify the subset against master (byte-for-byte experts, row-for-row routers)
python3 verify_subset.py

# 3. GGUF, MXFP4 passthrough + BF16 dense: ~9 min, 401.8 GB, 2573 tensors
cd <BOX>/llama.cpp-k3          # branch k3-arch = upstream master + PR #26185
$PY convert_hf_to_gguf.py $OUT \
    --outfile <BOX>/bigmodels/k3-reap80-ours-mxfp4-bf16.gguf --outtype bf16

# 4. prove the GGUF experts are byte-identical to Moonshot's weights
$PY verify_gguf.py

# 5. quantize (note: nthreads is POSITIONAL, and --allow-requantize is required)
cd <BOX>/llama.cpp-k3kt        # branch k3-kt = k3-arch + our kt-quants
./build-cuda/bin/llama-quantize --allow-requantize \
    <BOX>/bigmodels/k3-reap80-ours-mxfp4-bf16.gguf \
    <BOX>/bigmodels/k3-reap80-ours-IQ2_KT.gguf IQ2_KT 16
```

## Scripts

| file | what it does |
|---|---|
| `reap_subset.py` | the subsetter. Keeps listed experts per layer, renumbers to 0..178, slices `gate.weight` + `e_score_correction_bias` rows, copies everything else. Expert payloads are a **raw byte copy** — no dequant, no requant. Ops sorted by (source shard, offset) so reads stay sequential |
| `validate_map.py` | validates the expert map on a tensor independent of how it was recovered — dequantises the MLX 8-bit router and compares to master rows. **Needs the MLX download, which is deleted**; kept for the record |
| `verify_subset.py` | byte-for-byte expert + row-for-row router check of the subset vs master. Works on partial runs, so you can verify completed shards mid-flight |
| `verify_gguf.py` | reads the finished GGUF, resolves each renumbered expert back through the map, compares MXFP4 bytes to master |
| `verify_repack.py` | proves PR #26185's `repack_mxfp4_blocks` is lossless by decoding real weights two ways |
| `param_count.py` | exact param/byte census by group, and KT ladder size projections |

## The cell zero

`repack_mxfp4_blocks` is a **pure bit repack**, measured not assumed: 44,040,192 real
weights decoded two ways (compressed-tensors source semantics vs repack-then-ggml-
block-mxfp4) matched exactly. Mechanism: ggml's `kvalues_mxfp4` are *doubled* E2M1 and
`e8m0_to_fp32_half` is 2^(e-128), so the doubling and halving cancel.

So the MXFP4-passthrough GGUF is byte-identical to Moonshot's own weights, and **every
loss measured below it is provably ours**. For contrast, the public AtomicChat K3 GGUFs
are a MXFP4 → Q8_0 conversion with lower quants requantized from that Q8_0 — lossy
before the ladder starts.

## Ladder projections (601.1 B params)

experts@bpw + dense@q8: MXFP4-passthrough 349.7 / IQ3_KT 274.9 / IQ2_KT 206.6 /
IQ1_KT 179.8 GB. Uniform-bpw: 319 / 237 / 161 / 132 GB.

At ~180 GiB of VRAM, IQ2_KT only fits if the dense side drops below q8 — and the
72 GB of BF16 attention is where that has to come from.

## Gotchas

- `llama-quantize` has **no `--threads` flag**. nthreads is the 4th positional arg.
  Passing `--threads` dumps the usage text and looks like a clean run in a log.
- `--allow-requantize` is required: the source's experts are already MXFP4.
- **KT types need no imatrix.** `tensor_requires_imatrix()` lists only IQ3_XXS / IQ2_XXS /
  IQ2_XS / IQ2_S / IQ1_M / IQ1_S (+ Q2_K inside Q2_K_S). IQ1/2/3/4_KT fall through to
  false, so the whole ladder builds on CPU with no calibration GPU window.
- `tiktoken` + `blobfile` must be installed — K3 ships a tiktoken tokenizer.
- The converter is text-only; it skips `vision_tower.` / `mm_projector.`.
- Emit **master-native HF naming**. PR #26185 matches
  `language_model.…block_sparse_moe.experts.N.wX.weight_packed` verbatim; MLX's fused
  `switch_mlp` layout is rejected.
- Byte-matching MLX expert weights to identify experts is **invalid** — MLX repacks
  MXFP4 into its own layout. Match `e_score_correction_bias` scalars instead.
- 125 GB RAM vs a 402 GB model: CPU smoke tests work via mmap but thrash. Load hits ~20,
  kswapd climbs, ssh can time out. It settles.

## Evidence

`../../evidence/k3-reap80/` — the 92-layer map validation, subsetter run, conversion log,
and GGUF byte-verification output.
