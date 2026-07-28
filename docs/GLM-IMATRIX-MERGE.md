# GLM 5.2: a merged imatrix with indexer data nobody had

Produced 2026-07-28. `/mnt/data/glm-calib/glm52-merged.imatrix`, 1.1 GB.

## What was found

The imatrix that built the served model is Unsloth's, read from the production
GGUF's own provenance keys (`unsloth_calibration_GLM-5.2.txt`, 88 chunks x 9216 =
811,008 tokens). It is good calibration and the earlier claim that it was thin
English-only wikitext was wrong, see `GLM-CALIBRATION-CORPUS-DESIGN.md`.

But it has a hole:

| | Unsloth | ours (fleet build) |
|---|---|---|
| entries | 2004 | 2130 |
| calibration tokens | **811,008** | 102,400 |
| **`.indexer.` entries** | **0** | **126** |

**Unsloth's imatrix contains no importance data for the DSA lightning-indexer
tensors at all.** `blk.N.indexer.{attn_k,attn_q_b,proj}` across 21 layers
(0, 1, 2, then every fourth to 74) were quantized with every column treated as
equally important.

Ours has them because the fleet tree carries the DSA work; the older `mtpim` tree
did not collect them either. This was found by accident while diagnosing an
unrelated problem, by diffing tensor-name sets between the two files.

## The merge

Neither file dominates. Unsloth has 8x our calibration tokens on the shared 2004
entries; we have 126 entries they lack entirely. So take the better of each:

```bash
python3 tools/f2/imatrix_merge.py \
  --base    glm52-imatrix-unsloth.gguf \
  --overlay glm52-mtp.imatrix \
  --out     glm52-merged.imatrix \
  --take    'indexer'
```

```
taken from overlay : 126 entries
kept from base     : 2004 entries
merged             : 2130 entries, blocks 0-77
partner check      : OK, every in_sum2 travels with its own counts
orphaned entries   : 0
```

The partner check matters: an imatrix stores `<tensor>.in_sum2` and
`<tensor>.counts` as separate entries, and if those came from different runs the
consumer would divide summed activations by the wrong count and silently produce
garbage. The tool aborts rather than emit such a file.

## What is NOT claimed

**That this improves the quant.** It has not been measured. The merged file
contains strictly more information than either input, which is a fact about the
file, not about output quality. Establishing improvement needs a requant plus an
eval with a lossless floor as cell zero, and after three retractions in one night
the standard here is measure-then-claim.

Two honest wrinkles:

- The merged file's KV metadata reports `chunk_size 9216, chunk_count 88`, which
  describes the **base** only. The 126 indexer entries came from a 512-token,
  200-chunk run. `imatrix.merged_from` records the provenance, but the chunk
  fields do not describe both halves.
- The original goal, MTP/`blk.78` data, is **still not solved**. See
  `f2-mtp-imatrix` in memory: the loader fix was necessary but not sufficient, and
  blk.78 is absent even from the final save. This merge delivers the indexer half
  and nothing about MTP.

## Why the indexer might matter

GLM-DSA's indexer selects which keys the sparse attention attends to. Its weights
are small relative to the experts, so upgrading their fidelity is cheap in bytes.
Whether importance data changes their quantized values enough to matter is exactly
the untested part above.
