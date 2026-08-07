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

| loop | references to `common_speculative` / `spec_draft` / `can_speculate` |
|---|---|
| `update_slots_paged()` | **0** |
| static `update_slots()` | **22** |

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

## What would close it

Wire speculation into `update_slots_paged()`: draft generation at prompt-eval completion, draft tokens
into the paged batch, and acceptance/rollback against the paged block table. The rollback is the
interesting part — rejected draft tokens have already been written into paged blocks, so the write
slots must be reclaimed, which the static path does not have to think about.

Not estimated here. An estimate with no measurement behind it becomes a schedule.
