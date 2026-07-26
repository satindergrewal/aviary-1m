# F2: giving MTP (nextn) layers real imatrix data

Why `llama-imatrix` silently produces nothing for a model's MTP layers, and the patch
that fixes it.

**Date:** 2026-07-26. **Tree:** mainline `ggml-org/llama.cpp`.

---

## The gap

Models with multi-token-prediction carry one or more extra "nextn" blocks. Nobody has
importance data for them, so every published quant pins them to Q8_0 (or drops them).
On a large MoE that is real weight budget spent on a layer nobody measured.

The usual assumption is that `llama-imatrix` lacks a flag. It does not. The cause is
structural:

```cpp
uint32_t llama_hparams::n_layer() const {
    return n_layer_all - n_layer_nextn;
}
```

Every graph-build loop runs `for (il = 0; il < n_layer; ++il)`, so the nextn blocks are
**excluded from the forward graph entirely**. Their tensors are never multiplied during
a normal decode, so there is nothing for the collector's op-callback to observe. No
warning is printed; the tensors simply never appear in the output file.

This matters more than "one small projection" suggests. An MTP block is a full
transformer layer. On Qwen3.6-27B, `blk.64` holds 15 tensors against a normal layer's
11:

```
attn_q, attn_k, attn_v, attn_output          <- 4 matmuls
ffn_up, ffn_gate, ffn_down                   <- 3 matmuls
nextn.eh_proj                                <- 1 matmul
attn_norm, attn_q_norm, attn_k_norm,
post_attention_norm, nextn.enorm,
nextn.hnorm, nextn.shared_head_norm          <- norms (imatrix does not collect these)
```

So **8 quantizable matmuls** per MTP block get no calibration data.

## What already existed

No new decode path had to be invented. Mainline already has both halves:

- `LLAMA_CONTEXT_TYPE_MTP` builds a context over *only* the nextn layers (the loader
  filter flips to `il >= hparams.n_layer()`). It runs against the same already-loaded
  model, with `ctx_other` pointing at the target context, so there is no second model
  load.
- `llama_set_embeddings_nextn` / `llama_get_embeddings_nextn` (in `src/llama-ext.h`)
  expose the target's hidden state from before the final output norm, which is exactly
  the input an MTP head consumes.

`common/speculative.cpp`'s `draft-mtp` implementation is the working reference for
driving the pair, and the patch mirrors it.

## The patch

Three files:

- `common/common.h`, `common/arg.cpp`: a `--no-mtp` opt-out.
- `tools/imatrix/imatrix.cpp`: when `llama_model_n_layer_nextn(model) > 0`, create the
  MTP context, enable nextn embeddings on both contexts, and after each main
  `llama_decode` run a second decode on the MTP context over the same tokens.

The second batch carries **both** token ids and embeddings: the token ids supply the
`t+1` token embedding, and `batch.embd` supplies the target hidden state that produced
token `t`, shifted right by one position within each sequence.

Two traps worth naming, both inherited from how the speculative path works:

- `llama_batch_init` allocates only *one* of `token`/`embd`. The MTP graph needs both,
  so `batch.token` is malloc'd separately (speculative.cpp does the same).
- The right-shift is **per sequence**, not across the whole batch. imatrix packs
  `n_seq` sequences into one batch as `row = seq*batch_size + k`, so shifting the flat
  buffer would bleed one sequence's hidden state into the next one's first row. A
  per-sequence `pending_h` carries the last row across batches, and is zeroed when the
  KV cache is cleared at each chunk boundary.

The collector attaches to the MTP context for free: `common_context_params_to_llama`
copies `cb_eval` from the shared params.

## Verification

Measured on Qwen3.6-27B-Q4_K_M (arch `qwen35`, 65 blocks, `nextn_predict_layers: 1`, MTP
block at `blk.64`), Metal, 2 chunks at `-c 512`. The `--no-mtp` flag reproduces the old
behaviour exactly, so control and treatment are the **same binary, model and corpus**,
differing only in whether the second pass runs:

```
CONTROL   (--no-mtp)   blk.64 matmuls with imatrix data: 0
TREATMENT (patch on)   blk.64 matmuls with imatrix data: 8
                         blk.64.attn_q     blk.64.ffn_up
                         blk.64.attn_k     blk.64.ffn_gate
                         blk.64.attn_v     blk.64.ffn_down
                         blk.64.attn_output
                         blk.64.nextn.eh_proj
```

Zero before, all eight after, which is every quantizable matmul in the block. The tool
also now says which case you are in, instead of failing silently:

```
main: collecting data for 1 MTP (nextn) layer(s) via a second pass
main: model has 1 MTP (nextn) layer(s) but --no-mtp was given; their tensors will have no imatrix data
```

A regression run on a model with no MTP layers (Qwen3-4B) collects its usual 504 entries
and is unaffected, since the whole path is behind `n_layer_nextn > 0`.

Note this verification run used an already-quantized model, which is fine for proving the
mechanism collects. Production calibration should still be run against the full-precision
weights.

## D1: what a low-bit MTP block actually costs

Collection working is not the same as the data being *useful*. Three arms, quantized from
the same Qwen3.6-27B bf16 to the same Q4_K_M target, differing **only** in how `blk.64`
was treated:

| arm | `blk.64` | represents |
|---|---|---|
| 1 | `q8_0` | what publishers do today |
| 2 | `q2_K`, imatrix data **excluded** | default behaviour without this patch |
| 3 | `q2_K`, imatrix data **used** | what this patch enables |

`q2_K` rather than `q4_K` on purpose: at `q4_K` the two low-bit arms would be
indistinguishable and the comparison would prove nothing by construction.

**Controls, all verified rather than assumed:**

- Arms 2 and 3 use the *same* imatrix file, with `--exclude-weights blk.64` removing only
  the MTP entries. The quantizer confirms it: `loaded 504 ... have 496`, i.e. exactly the
  8 blk.64 matmuls dropped.
- `blk.63` (an ordinary layer) is **byte-identical across all three arms** (11/11 tensors),
  so the arms really do differ only in the MTP block.
- Within `blk.64`, arms 2 and 3 differ in exactly the 8 quantized matmuls and match on all
  7 norms, proving `--exclude-weights` changed the weights it was supposed to.
- Arms 2 and 3 came out at exactly the same file size (16,686,850,624 bytes), as they must,
  since only the values differ.

**Results** (6 fixed prompts, 256 tokens, temp 0, single RTX PRO 6000):

| arm | `blk.64` | size (bytes) | pooled acceptance | tok/s | output |
|---|---|---|---|---|---|
| baseline, no speculation | q8_0 | 16,998,720,064 | n/a | 69.6 | reference |
| 1 | q8_0 | 16,998,720,064 | **0.6625** | 110.0 | identical to arms 2,3 |
| 3 | q2_K + data | 16,686,850,624 | **0.5688** | 103.9 | identical to arms 1,2 |
| 2 | q2_K, no data | 16,686,850,624 | **0.5698** | 103.6 | identical to arms 1,3 |

### Three findings

**1. Quantizing the MTP block does not change output at all.** All three arms produced
byte-identical text on all six prompts, despite `blk.64` going from q8_0 to q2_K. This is
the concrete answer to the "risk of unintended quality degradation" worry: because
llama.cpp verifies drafts against the target, a degraded MTP block costs **throughput,
not quality**. Speculation is a speed mechanism, and damaging it slows you down rather
than making you wrong.

A related detail worth stating precisely: spec-on output is *not* bit-identical to
spec-off. Two of six prompts matched the no-spec baseline and four did not. That is a
property of the speculative path itself (verification happens under different batch
shapes, so near-tied logits can flip), and it is **identical across all three arms**,
which is exactly why it does not confound the comparison.

**2. The cost of the q8_0 → q2_K drop is real but modest:** 297 MiB saved on one MTP layer
of a 27B, for a 14% relative fall in acceptance (0.6625 → 0.569) and 5.7% in throughput
(110.0 → 103.7 tok/s). Speculation is still a large win over none: 103.7 vs 69.6 tok/s.

**3. Negative result, stated plainly: the imatrix data made no measurable difference
here.** Arm 3 (with data) scored 0.5688 and arm 2 (without) 0.5698. The difference is
noise, and if anything it favours the arm with no data. So on this model, at this quant
level, F2's data did not improve draft quality.

That is worth knowing before anyone builds a pipeline assuming it will. What the patch
definitely does is close a silent hole: those tensors previously got quantized with no
importance data and no warning. What it does *not* do, on this evidence, is make a
low-bit MTP block measurably better.

### Limits of this measurement

One model (27B dense), one quant level, six prompts, and an imatrix collected from an
already-quantized copy rather than bf16 weights. A stronger calibration, a MoE at scale
(where the MTP block is far larger), or a more aggressive quant could all move the third
finding. The first two findings are about mechanism and are less likely to be
model-specific, but they have still only been shown on one model.

## Honest scope

This fixes *collection*, and D1 above shows collection alone did not buy accuracy on this
model. The size argument should be kept in proportion too: 297 MiB measured on a 27B, and
extrapolation to a ~744B MoE is arithmetic, not measurement. Real, and it stacks with
other savings, but not a context-ceiling unlock.
