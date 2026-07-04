# Speculative decoding on hybrid linear-attention models: the rollback problem

Notes from porting DeepSeek's [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (DSpark draft training and evaluation) to the Qwen3.5-family hybrid architecture used by [Ornith-1.0](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B). Written because we hit this wall at 2 AM and the error message will probably find someone else the same way.

## The problem

Speculative decoding verifies a block of draft tokens with the target model, accepts a prefix of them, and **rolls back** the rejected suffix. With standard attention this is trivial: the KV cache has one entry per token, so rollback is `cache.crop(accepted_len)`.

Hybrid models (Qwen3.5, Qwen3-Next, Jamba, Nemotron-H style) interleave full-attention layers with linear-attention layers (Gated DeltaNet in this family). Linear attention keeps a **fixed-size recurrent state** that absorbs every token it sees. There is no per-token entry to crop. Once the state has eaten a rejected draft token, you cannot un-eat it.

Concretely, DeepSpec's evaluator does:

```python
past_key_values_target = DynamicCache()   # attention-only cache
...
past_key_values_target.crop(start)        # rollback after each verify round
```

On a Qwen3.5 target this crashes at load time with:

```
ValueError: `has_previous_state` can only be called on LinearAttention layers,
and the current Cache seem to only contain Attention layers.
```

and even if you swapped in the model's hybrid cache class, `crop()` would silently corrupt the linear states: they would contain contributions from tokens the verifier rejected.

This is not a GPU-size problem. No amount of VRAM fixes it. It is bookkeeping.

## Fix 1: cache-less verification (what this repo uses for measurement)

For measuring draft quality (acceptance length), correctness matters and speed does not. So we wrap the target model in a shim that ignores the cache entirely and re-forwards the full accepted sequence each verification round, reconstructing token placement from the absolute `position_ids` the evaluator already passes:

```python
offset = int(position_ids[0, 0])                      # where this chunk starts
self._ids = torch.cat([self._ids[:, :offset], input_ids], dim=1)
out = self._model(input_ids=self._ids, use_cache=False, ...)
logits = out.logits[:, -input_ids.shape[1]:, :]        # slice back to the chunk
```

Truncating the buffer to `offset` before appending makes rollback implicit: a rejected suffix is simply never referenced again. Acceptance statistics are exact. Wall-clock speed is meaningless in this mode (quadratic recompute), so speedups must be projected from acceptance length, not timed.

Implementation: `deepspec/eval/nocache_wrapper.py` in our DeepSpec working copy, ~50 lines, applied only when `model_type` starts with `qwen3_5`.

## Fix 2: state checkpointing (the production answer)

The real solution, which vLLM already ships for Qwen3-Next/Qwen3.6 MTP speculative decoding: **checkpoint the linear-attention states**. Recurrent states are small (megabytes per layer, not gigabytes), so you can afford to store them per draft round, or even per token. vLLM's implementation keeps per-token intermediate states in a sliding ring buffer, so rejecting tokens costs only the rejected-token work. Reported production numbers on Qwen3.6-27B MTP: acceptance rate ~0.72 and ~2.5x decode speedup, which demonstrates hybrid-architecture speculative decoding is not just possible but already fast in the wild.

The same approach would drop into DeepSpec's evaluator: snapshot linear states before each verify forward, restore on rollback, replay accepted tokens (linear-time, cheap). We estimate this at a day of work and will do it if cache-less measurement becomes the bottleneck.

## What this means if you are doing this yourself

1. Draft **training** is unaffected: it reads precomputed hidden-state caches, no runtime cache involved.
2. Draft **quality measurement** works today with the cache-less wrapper above.
3. Draft **deployment** on hybrid targets should ride engines that already solved rollback (vLLM's speculative decoding machinery for this architecture family), rather than reinventing it.
4. Do not buy a bigger GPU for this problem. It will not help.

## Prior art and references

- [vLLM x Qwen3-Next: hybrid attention and MTP](https://medium.com/data-science-in-your-pocket/vllm-x-qwen3-next-hybrid-attention-multi-token-prediction-and-thinking-controls-for-a0f6b3dcc120)
- [MTP speculative decoding for Qwen3.6, incl. the DeltaNet rollback ring-buffer design](https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/)
- [Gated DeltaNet explainer](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md)
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (the codebase being ported)

Full porting playbook for DeepSpec on Qwen3.5 (backbone nesting, config nesting, capture-layer constraints, 32GB VRAM memory ladder, CPU-offloaded BF16 optimizer) lives in this repo's history and will be published alongside the draft models once evaluation completes.
