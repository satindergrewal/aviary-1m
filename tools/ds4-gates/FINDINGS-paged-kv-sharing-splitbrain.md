# KV-sharing architectures run SPLIT-BRAIN under `--kv-paged`

**Status: ROOT-CAUSED IN SOURCE, 2026-08-07. Reproducible. Found by `arch_serve_gate.sh` on
`gemma-4-E2B-it`, isolated with no draft model involved.**

## What was measured

Prompt `"The capital of France is"`, greedy, `-c 4096`, `DS4P_PAGED_SWA=1`:

| vehicle | static | paged | |
|---|---|---|---|
| `gemma-4-E2B-it` Q4_K_M | `" Paris."` | **`" a."`** | FAIL |
| `gemma4-26B-A4B` Q4_K_M | `" Paris."` | identical | PASS |

Same arch string (`gemma4`), same binary, same code path. **Checkpoint-dependent** — the second time
in one day (the first was fused-QKV on `starcoder`).

`static-path warns = 0`, `cap-fails = 0`, `headdim-fails = 0`, `DS4P-CONSUME = 45` over **layers 0–14
only**.

## Root cause

`gemma-4-E2B-it` sets `n_layer_kv_from_start = 15` of 35 layers. `llama-hparams.cpp:272`:

```cpp
bool llama_hparams::has_kv(uint32_t il) const {
    if (n_layer_kv_from_start >= 0) {
        return il < (uint32_t) n_layer_kv_from_start;
    }
    ...
```

`gemma4.cpp` branches on it, and the `else` states the situation in its own words:

```cpp
} else {
    // reuse KV cache of earlier layers -- DELIBERATELY NOT PAGED. This layer supplies no
    // new K/V (Qcur only), and the paged op FUSES the KV write, so it has nothing to
    // write and no slots of its own. Wiring it would be pattern-matching the branch above.
    cur = build_attn(inp_attn, ..., Qcur, nullptr, nullptr, ...);
}
```

`inp_attn` is `build_attn_inp_kv_iswa()` — **the static cache**.

So under `--kv-paged`:

| layers | path | effect |
|---|---|---|
| 0–14 (`has_kv`) | paged consumer | **write** the paged pool |
| 15–34 (shared) | `build_attn(inp_attn, …)` | **read the static ISWA cache, which nothing ever fills** |

Twenty layers attend over empty KV; fifteen are correct. That is why the output is
**coherent-but-wrong** rather than gibberish — `" a."`, not `<unused25>` spam.

**And it explains the checkpoint-dependence exactly:** `26B-A4B` has `per_layer_input = 0` and an
array `head_count_kv`, shares nothing, so nothing ever reads the static cache — and it passes on the
same binary.

## ⚠ The comment was correct when written, and is now stale

*"the paged op FUSES the KV write, so it has nothing to write and no slots of its own"* — true at the
time. The paged op could not express a read-only call.

**Read-only paged attention was implemented the same day** (`cce5d6959`): geometry now comes from the
pool rather than `k_new`, and the write phase is skipped when K/V are null. The capability that makes
this branch wireable did not exist when the comment was written.

## Four mechanisms eliminated before this one, recorded so they are not re-tried

| hypothesis | killed by |
|---|---|
| SWA layers never get a paged context | layer-id overlap: every layer both pages and falls back → temporal, not structural |
| `gemma4` is broken in the fork | a *different* gemma4 checkpoint passes on the same binary |
| `inp_attn->mctx` vs `mctx` (a real anti-pattern, aligned anyway) | changed **nothing** — identical output, identical counts |
| an instrumentation gap hid 20 layers | `has_kv(il)` — those layers have no KV; the counters were right |

⚠ **A withdrawn number.** "45 graph nodes but only 18 metal dispatches, unexplained" was reported
twice before being checked. The known-good control settles it: `ernie4_5` **passes** with 180 consume
nodes and **2** dispatch lines. That log line is per-encode or per-unique-geometry, not per-op. Nodes
≫ dispatches is normal. It was never evidence.

## Blast radius

Any architecture with `n_layer_kv_from_start >= 0`, i.e. any KV-sharing / layer-reuse design. On the
19-model list this is at minimum the `gemma4` E2B/E4B class. **`gemma4-assistant`'s only viable target
is `gemma-4-E2B-it`** (its `embedding_length_out = 1536` hard-couples it), so that row now depends on
this defect *as well as* on paged speculation.

## What would close it

**Cheap and loud, today:** refuse `--kv-paged` at init when the model reports
`n_layer_kv_from_start >= 0`, with stable message text (the ds4-gates refusal branch matches on
message text, so it becomes exit-3 rather than a false FAIL). Wrong-and-loud beats wrong-and-quiet —
right now the answer is silently wrong.

**The real fix:** wire the shared branch to the paged pool as a **read-only** call, indexed at the
*source* layer's pool entry rather than `il`. That is precisely the machinery built today: no new K/V,
geometry from the pool, write phase skipped.

★ **And the index mapping is not an open question — it already exists**, `llama-model.cpp:2431`, as
the `layer_reuse_cb` the STATIC cache uses:

```cpp
reuse = [&](uint32_t il) {
    GGML_ASSERT(hparams.n_layer_kv_from_start >= 2);
    if (il >= (uint32_t) hparams.n_layer_kv_from_start) {
        return hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
    }
    return -1;
};
```

So for E2B (15): SWA shared layers reuse layer 13, non-SWA shared layers reuse layer 14. The paged
pool needs the same mapping and the shared branch needs a read-only call at `reuse(il)`.

## ⚠ The guard's first version broke a working configuration

It tested only `n_layer_kv_from_start >= 0` and aborted `gemma4-26B-A4B` too — a model that owns KV on
every layer and had been **passing**. The flag is set-but-equal-to-`n_layer` there, meaning "all
layers have KV", i.e. no sharing. **A guard that fires on a flag's PRESENCE rather than on the
CONDITION IT IMPLIES breaks working configurations.** The correct test is `< n_layer()`.

That near-miss also nearly cost the finding: seeing *both* models abort, the first reading was "the
hypothesis is dead". It was the guard that was wrong, not the mechanism — and only running the
known-good control immediately made the difference.

Controls, all three:

| vehicle | result |
|---|---|
| `gemma-4-E2B-it` | REFUSED, gate **exit 3** (designed refusal, not a failure) |
| `gemma4-26B-A4B` | **PASS**, 240 consume — the config the first guard broke |
| `ernie4_5` | **PASS**, 180 consume — non-gemma arch untouched |

⚠ The gates match designed refusals on the **assert string**, and the first pattern used `KV-SHARING`
while the assert says `KV-sharing` — a case mismatch that scored a working guard as a plain failure
(exit 1 instead of 3). Both gates now match case-insensitively.

Not estimated. An estimate with no measurement behind it becomes a schedule.
