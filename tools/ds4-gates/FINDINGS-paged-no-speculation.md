# Speculative decoding is structurally absent from the paged decode loop

**Status: CONFIRMED IN SOURCE, 2026-08-07. Not a bug to bisect — a missing feature with a named
location. Found by `draft_pair_gate.sh` on its first complete run.**

## What was measured

Qwen3.5-4B (target) + Qwen3.5-4B-DFlash (draft), `--spec-type draft-dflash`, `-c 4096`,
`DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1`, greedy, `cache_prompt: false`:

| arm | `draft_n` | `accepted` | output |
|---|---|---|---|
| a — target alone, static | 0 | 0 | reference |
| b — target + draft, static | **27** | **13** | identical to (a) |
| c — target + draft, **paged** | **0** | **0** | identical to (a) |

Paging was live in arm (c): `DS4P-CONSUME = 192`.

## Root cause, counted rather than eyeballed

```
tools/server/server-context.cpp:3328
    void update_slots() {
        // 4d: the scheduler owns batching entirely in paged mode
        if (paged_sched) {
            update_slots_paged();
            ...
            return;          // <-- returns before any speculation code
        }
        ...
        // line 4349: if (slot.can_speculate()) { common_speculative_begin(...); }
```

| function | speculation references | reached by the paged loop? |
|---|---|---|
| `update_slots_paged()` (lines 3068–3326, 259 lines) | **0** | it *is* the paged loop |
| `pre_decode()` | **8** | **no** |
| `post_decode()` | **13** | **no** |
| `handle_last_sampled_token()` | **5** | **no** |

The static `update_slots()` calls `pre_decode()` and `post_decode()`. `update_slots_paged()` calls
neither, and contains no speculation reference of its own. All the drafting lives in helpers the paged
loop never enters.

⚠ **A correction to the first version of this document.** It reported "0 in the paged loop vs 22 in
the static loop". The 22 was an artifact: the count used `awk '/^    void update_slots\(\)/,0'`,
whose range runs to **end of file**, so it swept every function that follows. Correctly scoped by
brace matching, `update_slots()` itself contains **0** speculation references too — the code is in
`pre_decode`/`post_decode`, which is why the corrected table names them. The load-bearing number was
always the **0** in the paged loop; the 22 was decoration, and it was wrong.

The paged scheduler owns batching entirely, and speculation was never wired into it. Nothing is
disabled and no flag is off — the code is not there.

This explains every symptom precisely: the implementation registers
(`adding speculative implementation 'draft-dflash'`), the slot advertises `"speculative": true`, and
the statistics line reads `#calls(b,g,a) = 0 0 0`. Initialised, advertised, never invoked.

## Why it matters, in the owner's terms

His bar: *"when I run the latest models, they don't error out unexpectedly **or are slow** despite we
put all efforts make them fast."*

Under `--kv-paged`, any model relying on speculative decoding loses **100% of its speedup, silently,
with correct output**. MTP, dflash, eagle3, gemma4-assistant, dspark. On the pair above the draft was
accepting 13 of 27 proposals under static and proposes zero under paging.

**Three of the nineteen are draft heads whose entire purpose is speculative decoding**:
`gemma4-assistant`, `eagle3`, `dflash`. They cannot be "verified working under paging" in any
meaningful sense while the paged loop cannot speculate. What is verifiable today is that they load,
that the pair is dimensionally compatible, and that the *target* pages correctly — which is not what
those rows are for.

⇒ The blocker on those three is **not** the draft-pair harness. It is wiring speculation into
`update_slots_paged()`.

## ⚠ Why this was invisible until now

The output is **identical in all three arms**. Correct text, correct paging, 192 consume events, no
warning, no error. Every check that existed before `draft_pair_gate.sh` would have passed it.

The `draft_n > 0` requirement is the entire difference between finding this and shipping it. Text
agreement cannot detect inert drafting, because greedy speculative decoding is **lossless** — the text
matches whether the draft worked perfectly or was never consulted.

## Two prerequisites discovered on the way, both silent

1. **`-md` alone is inert in this fork.** It loads the draft, logs `loading draft model`, prints its
   arch — and drafts nothing. `common/speculative.cpp:2488` logs *"no implementations specified for
   speculative decoding"* and returns `nullptr`. `--spec-type <name>` is required. Measured: `-md`
   alone produces **no `draft_n` key at all**; `-md --spec-type draft-dflash` produces `39 / 25`.
   Valid names (`common/speculative.cpp:2186`): `draft-simple` `draft-eagle3` `draft-mtp`
   `draft-dflash` `draft-dspark` `ngram-simple` `ngram-map-k` `ngram-map-k4v` `ngram-mod`
   `ngram-cache`.

2. **Qwen3.5-4B is both hybrid and SWA**, so arm (c) needs
   `DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1`. Both refusals were caught by matching the guard's *message
   text* rather than an arch list — `llama-model.cpp:2262` and `:2478`.

## A SECOND, DISTINCT blocker on the same rows: the DRAFT builds its own paged pool, and it aborts

Found running the `gemma4-assistant` pair (target `gemma-4-E2B-it` Q4_K_M + assistant F16). Arm (c)
never reached the speculation question — it failed to serve:

```
llama-kv-cache-paged.cpp:314: GGML_ASSERT(buf_gpu && "Failed to allocate GPU KV cache buffer") failed
```

Sequence from the log, by line number:

| line | event |
|---|---|
| 2076 | target `gemma4` loaded |
| 3506 | **target's** paged KV init — `n_gpu_blocks=384`, succeeds |
| 3691 | `common_speculative_init_result: loading draft model` |
| 3771 | draft `gemma4-assistant` loaded |
| 3980 | **a SECOND paged KV init, also `n_gpu_blocks=384`** — aborts |

The draft model inherits `--kv-paged` and constructs its **own** paged pool, sized with the target's
block count. It is not a capacity problem: the fitter reported `free_vram=107152 MiB` and
`VRAM would have allowed 165736 blocks (88.5 GiB)` against a 384-block request (~220 MB).

⚠ This is **independent of** the missing-speculation gap above. Fixing one does not fix the other:
even once `update_slots_paged()` can speculate, a draft that aborts during KV allocation never gets
that far. Both need addressing before any draft-head row can be answered.

Consistent with the `dflash` pair, where arm (c) *did* serve — that draft is a standalone-shaped
model, so its context construction differs from `gemma4-assistant`'s, which additionally throws
`requires ctx_other to be set` during memory fitting immediately before the abort.

## Confirmed on the way: the `draft-mtp` type for `gemma4-assistant`

`spec_type_for()` in the harness guessed `draft-mtp` for this arch, flagged in-comment as a guess.
Arm (b) measured **`draft_n = 33`, `accepted = 11`**, output identical to (a). The guess was right,
and it is now a measurement — on a second speculative type, which also shows the losslessness
property holding independently of type.

## What would close it

Wire speculation into `update_slots_paged()`: draft generation at prompt-eval completion, draft tokens
into the paged batch, and acceptance/rollback against the paged block table. The rollback is the
interesting part — rejected draft tokens have already been written into paged blocks, so the write
slots must be reclaimed, which the static path does not have to think about.

Not estimated here. An estimate with no measurement behind it becomes a schedule.
