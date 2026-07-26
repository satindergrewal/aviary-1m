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

## Honest scope

This fixes *collection*. It does not by itself prove that a low-bit MTP layer is safe:
that is the separate measurement (quantize the MTP block down with the new data and
compare quality against the Q8 pin). The size argument should also be kept in
proportion. Freeing one MTP layer from Q8 on a ~744B model is on the order of single-digit
gigabytes. Real, and it stacks with other savings, but it is not a context-ceiling unlock.
