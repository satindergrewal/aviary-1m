# --kv-paged: port status

Branch `kv-paged` in our fork, based on current upstream mainline `0e4a03622`.
Two commits: matiaslin's original `7e0805654`, then `627c53494` carrying the
rebase fixes.

## Why this exists

Satinder hit the concurrency wall running the GLM 5.2 agentic build: at `-np 1`
every sub-agent, plus his own queries, serialised behind the running task while
nvtop showed **40-58% GPU utilisation**. Idle bandwidth with blocked callers is a
serving-concurrency limit, not a compute limit. Decode is memory-bandwidth bound,
so batched requests could share the weight read instead of queueing for it.

llama.cpp allocates **static slots**. vLLM and SGLang use PagedAttention plus
continuous batching: KV lives in a shared pool of fixed-size pages that a request
grabs as it grows and frees the instant it finishes.

matiaslin's draft PR #22569 implements exactly that for llama.cpp and is stuck on
maintainer disagreement rather than on code. Benchmarked A10G, Llama-3-8B f16:
unified OOMs at 26 sequences; paged runs **247 at 1256 tok/s against 496**, about
2.5x throughput for roughly 3% low-concurrency overhead.

## What the port required

The branch was written 2026-04-08 against upstream `660b1b4bd`. Mainline has moved
**294 commits** since. 17 conflict blocks across 11 files. Most were additive, but
four would have compiled into silently wrong behaviour:

1. **A switch fallthrough in the CUDA dispatch.** Both sides added a `case` and
   shared the trailing `break;`, so keeping both let `GGML_OP_LIGHTNING_INDEXER`
   fall through into `GGML_OP_PAGED_ATTN`. Needed an explicit break.
2. **A stale constructor call.** Their `else` branch called the April
   `llama_kv_cache` ctor; mainline's has since gained `hparams` and `filter`.
   Took their if/else structure with HEAD's call.
3. **`hparams.n_layer` became a method** in mainline (it returns `n_layer_all`
   minus `n_layer_nextn`, the same function at the root of the F2 MTP gap). The
   branch used it as a field.
4. **Three conflicts split a function or class body** with the closing brace
   outside the conflict region, so both sides relied on it. Concatenating them
   nested one definition inside the other. `common.cpp` additionally carried the
   April `common_init_result` signature alongside mainline's `model_only` one.

Verified afterwards that ops.cpp, llama-graph.h and llama-graph.cpp differ from
upstream by **added lines only**, so the brace repairs did not disturb neighbours.

## Where it stands

| check | result |
|---|---|
| builds on macOS, Metal enabled | yes |
| `test-paged-kv` (unit: block manager, allocator, scheduler, deadlock detection, oversize-prompt rejection) | **ALL PASSED** |
| `test-paged-kv-e2e` primary check: top-5 overlap >= 4 | **PASS, 5/5, argmax identical (2217)** |
| `test-paged-kv-e2e` secondary check: first 4 tokens identical | **FAIL at token 1** (ref 7826, paged 12785). Control proves PRE-EXISTING, not from the rebase |
| CUDA path | not yet built or run |
| throughput | not measured, no number claimed |

## The token-1 divergence: pre-existing, and the port is faithful

Token 0 matches exactly and the logit distributions agree 5/5 on top-5, but the
second token diverges. That is the first decode step reading back KV written
during prefill.

**Control run, settled.** matiaslin's unmodified branch was built in a separate
worktree and run against the same fixture on the same machine with the same flags:

```
matiaslin original:  argmax 2217 = 2217, overlap 5/5, FAIL token 1: ref=7826 paged=12785
our kv-paged rebase: argmax 2217 = 2217, overlap 5/5, FAIL token 1: ref=7826 paged=12785
```

Identical, down to the token IDs. Two conclusions:

1. **The divergence is pre-existing in the branch.** The rebase did not cause it.
2. **The port is behaviourally faithful.** Reproducing the original bit-for-bit
   across 294 commits of upstream drift is the strongest available evidence that
   the 17 conflict resolutions were correct.

Two things to weigh before deciding how much the failure matters:

- This ran the **CPU reference** paged-attention path, which the code itself
  labels "for correctness validation only and is not optimized". The author's
  benchmarks were CUDA, so the CPU path may simply be less exercised. Running the
  same test on CUDA is the next step and would separate "CPU reference is sloppy"
  from "the paged read path is wrong".
- The test's own comment concedes that "minor floating-point accumulation
  differences between the paged and non-paged paths can flip individual tokens
  after a handful of decode steps". Failing at token 1 is earlier than that
  excuse comfortably covers, so it should not be waved away as FP noise without
  the CUDA result.

## Constraints inherited from the prototype

- **Single device.** `validate_paged_kv_placement` refuses a model split across
  devices and tells the user to pin with `-sm none -mg <id>`. GLM 5.2 needs
  `--tensor-split` across both cards, so this cannot serve the production model
  as it stands.
- `n_batch` must equal `n_ubatch`; no SWA; full offload only.
- CUDA is the only optimised backend. The CPU path exists but is a reference.

## Note on what mainline already gives us

Separately from paged KV, mainline's `--kv-unified` (now the default) sets the
per-sequence cap to the **full** `n_ctx` rather than `n_ctx / n_parallel`:

```c
// src/llama-context.cpp
if (cparams.kv_unified) { cparams.n_ctx_seq = cparams.n_ctx; }
else                    { cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max; }
```

Measured on the GLM server: 4 slots each reporting the full 65,536. So the
`-np 1` serialisation that motivated this work is already substantially improved
on the fleet build, without paged KV. Paged KV is for vLLM-class concurrency
(247 sequences against 26), not for the basic case.
