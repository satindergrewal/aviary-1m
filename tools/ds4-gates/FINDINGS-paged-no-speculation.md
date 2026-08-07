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
into the paged batch, and acceptance against the paged block table.

⚠⚠ **A CORRECTION TO THE FIRST VERSION OF THIS PARAGRAPH.** It said *"rejected draft tokens have
already been written into paged blocks, so the write slots must be reclaimed."* **That is wrong, and
it contradicted an invariant already proven in this lane.** `DS4P_SLOT_COVER` asserts
`slots[t] == btab[pos/bs]*bs + pos%bs` — the write slot is a pure function of POSITION — and it held
over ~50,000 tokens with 0 mismatches. So a rejected token's slot is simply **overwritten** by the
next token at that position. Nothing to reclaim. Left unchallenged, that sentence would have anchored
a design around a problem that does not exist.

⚠ **A SECOND CORRECTION, from reading the BODIES rather than the signatures.** The version above said
*"the scheduler's request/step API is one-token-per-sequence and forward-only… the API is the work."*
That was signature-level reading and it overstated the problem. The batch is **already multi-token per
sequence** — chunked prefill uses exactly that path, and `update()` advances `n_past` by
`curr_info.batch_lens[i]`, not by 1.

What is actually one-token is narrower, and it is four specific lines:

| # | what assumes a single decode token | where |
|---|---|---|
| A | decode allocates exactly one slot: `allocate(1, *group)`, guarded by `required_capacity = n_past + 1` | `llama-paged-scheduler-impl.cpp:460,467` |
| B | a decoding group contributes `logical_seq.back()` once | `llama-paged-scheduler-impl.cpp:798` |
| C | logits are requested only on the sequence's LAST batch token: `batch.logits[...] = !mid_prefill && (token_idx == new_tokens - 1)` — verification needs logits on **every** drafted position | `llama-paged-scheduler-impl.cpp:805` |
| D | `update()` appends exactly one sampled token (`logical_seq.push_back(new_tokens[i])`) and advances `n_past` by tokens **SUBMITTED**, not tokens **ACCEPTED** | `llama-paged-scheduler-impl.cpp:876–878` |

**D is the whole rollback story**, and it confirms the slot-cover correction above: with 4 drafted and
2 accepted, `n_past` must advance by 2 rather than 4, and the KV written at the two rejected positions
is simply overwritten by the next step at those same positions. No blocks are reclaimed, nothing is
freed — the rollback is `n_past` plus `set_seq_max_pos`, both already present in `update()`.

The public signature change follows from D: `llama_paged_scheduler_update()` needs a per-sequence
**accepted count** alongside the tokens. `n_batch == n_ubatch` (`llama-model.cpp:2624`) still bounds
how wide a verify batch may be.

So: **an API extension plus four localised changes, in that order** — not a rewrite of the scheduler's
contract. Reading the bodies made the job smaller than reading the signatures did, which is the
opposite of the usual direction and worth noticing.

Still not estimated, but now scoped by what was read rather than by what was assumed.

## The complete change list, read out of the bodies (2026-08-07)

Both halves now read. **The loop half is narrower than feared and the scheduler half is four lines.**

### Loop half — `update_slots_paged()`

| what | state |
|---|---|
| the token mirror the draft reads (`slot.prompt.tokens`) | **already maintained per token**, `server-context.cpp:3287`, with a comment explaining exactly why. The obvious failure mode — desynced mirror feeds the draft wrong context, acceptance craters, `draft_n > 0` still passes — **does not apply.** |
| drafting calls | **absent.** The loop invokes none of `pre_decode` (8 spec refs), `post_decode` (13), `handle_last_sampled_token` (5). |
| sampling | samples **one token per sequence**, from the last batch row: `tok_idx = batch_offsets[i] + batch_lens[i] - 1`. Verification needs a token per **drafted position**, then the accept-prefix. |

### Scheduler half — `llama-paged-scheduler-impl.cpp`

| # | line | assumes one decode token | state |
|---|---|---|---|
| A | `:460,467` | `allocate(1, *group)`, guarded by `required_capacity = n_past + 1` | **DONE** |
| B | `:798` | a decoding group contributes `logical_seq.back()` once | **DONE** |
| C | `:805` | logits only on the last batch row: `token_idx == new_tokens - 1` | **DONE** |
| D | `:876–878` | appends one sampled token; advances `n_past` by tokens **SUBMITTED**, not **ACCEPTED** | **DONE** |
| **E** | `:874–875` | `set_seq_max_pos` from the last **SUBMITTED** position, not the last **ACCEPTED** | **DONE** |
| **input channel** | — | `step()` builds decode rows from `logical_seq.back()`; drafted tokens have **no way in** | **DONE** — `llama_paged_scheduler_set_draft()` |
| **server loop** | `update_slots_paged()` | invokes no drafting at all | **NOT STARTED** — the only thing left |
| **wrapper layout** | `llama-paged-scheduler.cpp` | one token layout per call, not per row | **DONE** |

⚠ **This list started as "four localised lines" and is now seven items.** E was five lines from D and
invisible while fixing D. The input channel is a new public entry point and the largest single piece.
Every scoping claim on this rock has been true of the layer read and silent about the next one down —
three times, in both directions. The response is not another qualified claim; it is to stop publishing
scope until the layer below has been read.

### ⚠ The scheduler half is exercised on BOTH branches — and getting there took two vacuities

`test-paged-kv-e2e` stages real drafts and **prints which branch it took**:

```
draft probe took the REJECT branch (n_acc=1 of 2)
multi-accept probe n_past 24 -> 26, logical 25 -> 27 (want +2, +2)
```

**The first probe covered only the reject shape.** It staged `draft_tok = next` — the token just
appended — so acceptance needed the model to repeat itself immediately. The proof was already in the
mutation result and went unread: `n_past += n_sub` went red, which is only possible when
`n_acc != n_sub`, i.e. the reject branch.

**And the multi-accept probe passed a mutation before it was finished.** Capping the append loop at
one token still went green, because the probe only asserted `n_past` — which advances by `n_acc`
*whatever the append does*. Advance-count and append-count are different quantities; a probe checking
one is blind to the other. `llama_paged_seq_state::n_logical` exists for exactly this, and the header
says so.

Two placement failures on the way, both caught by the test rather than by inspection:

| attempt | failure |
|---|---|
| gate the probe on "token stream complete" | **vacuous** — the reject probe runs early, when the stream is short, so the gate skipped the accept path on every run and printed PASSED |
| fire it after the stopping `update()` | the group was already FINISHED and removed from `id_to_group`, so `set_draft` returned false |

⇒ **Mutate each invariant separately.** A mutation of the counter says nothing about the append.
⇒ **Make the run testify which branch it took.** One `printf` converts an inference into a
measurement, and it found all four vacuities in this session.

### Loop half: WIRED, and blocked on the drafter's own KV (2026-08-07)

Fork `40aa45e45`. Regression clean; **speculation still does not run.**

**Landed and necessary:** `llama_set_embeddings` before the paged decode · `common_speculative_process`
after it · `common_speculative_accept(…, accepted.size() - 1)` (the last entry is the target's bonus
token, not a draft acceptance) · the verify on `common_sampler_sample_and_accept_n` ·
`n_draft_total`/`n_draft_accepted` reporting.

**⚠ THREE PIECES OF SLOT STATE THE PAGED LOOP DID NOT MAINTAIN**, each found by the drafter's own
counters rather than by reading:

| state | symptom |
|---|---|
| `SLOT_STATE_GENERATING` | my condition tested it; **the loop never sets it** — the whole draft block was DEAD CODE, and the only occurrence of that constant in the function was my own test |
| `common_speculative_begin` | never called → `#calls(b,…) = 0` |
| `slot.sampled` — the drafter's `id_last` | read by my params, **never written here** → `draft()` ran 22 times and generated **zero** |

Same class every time: **the paged loop enters neither `pre_decode` nor `post_decode`, so every piece
of slot state those maintain is silently absent** — and absence produces a *working server that drafts
nothing*. Correct output, no error, no crash. Only `draft_n` and the drafter's statistics line show
anything at all.

⇒ **Before adding any further speculation call to this loop, enumerate the slot state it reads.** Three
of four failures were a field the static path writes elsewhere.

**⚠ THE BLOCKER, and it is not a missing line:**

```
W draft: llama_decode returned -1
```

**Counted, not sampled:**

| | |
|---|---|
| `get_n_draft_max` calls | **23** — values 22, 21, 20 … 3, 2, 1, 0 |
| draft decode failures | **22** — every single arming attempt |

⇒ **The arming is correct.** The block runs every step and computes a sane `n_max`. The failure is
entirely inside `ctx_dft`.

⚠ **A correction to the first reading of this.** Seeing `max possible draft: 2, 1, 0` I concluded the
drafter was being starved. That was `tail -3` of a healthy 23-call series starting at 22 — the
decrease is just `n_remaining` shrinking toward the 24-token limit. The full distribution took one
command and says the opposite. **`tail -3` is a partial view, and a partial view produced a coherent
wrong story for the sixth time this session.**

What dflash does (`common/speculative.cpp:1145–1164`):

```cpp
n = dp.n_past;                                  // I pass s.prompt.n_tokens()
for (i = 0; i < n_block_tokens; ++i)
    common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
llama_decode(ctx_dft, batch);                   // -> -1, 22/22
```

It writes at positions `n..n+k` into the **draft** context; `-1` is "no KV slot for the batch".

⚠ **Leading candidate, labelled as a candidate.** dflash conditions on the **target's post-norm
embeddings**, delivered via `common_speculative_process()`. That call is made and never returns false
— but *"did not fail"* is not *"produced embeddings"*. Whether the **paged attention path yields valid
post-norm embeddings at all** is unverified, and whether `llama_set_embeddings` does anything on that
path is likewise unconfirmed.

Absence-vs-presence again: proof the call did not error, no proof it produced anything.

**Eliminated, by reading and by experiment:**

| candidate | how it died |
|---|---|
| layer-input capture not configured under paging | `speculative.cpp:1000` sets `llama_set_embeddings_layer_inp(ctx_tgt, …)` in the dflash impl's **constructor** — identical in both arms |
| target not producing layer inputs | `get_embeddings_layer_inp` `GGML_ASSERT`s on `has_data()`. **No abort fires** and `process()` returns true, so the target *is* producing them under paging |
| the draft-static fix (`604c2858a`) masking it | **experiment**: disabled the fix, rebuilt, re-ran → decode failures **22 with it, 22 without**, pool aborts **0 either way**. Neither causing nor masking. |

★ **And that experiment sharpened three things that had been one blur.** The pool-allocation abort I
fixed **never occurred on the dflash pair at all** — 0 aborts even with the fix disabled. It was
specific to `gemma4-assistant`. So these are **three independent problems**, not one family:

| | scope | state |
|---|---|---|
| draft builds its own paged pool and aborts | `gemma4-assistant` only | **FIXED** |
| `llama_decode(ctx_dft)` returns −1 | `dflash` | **OPEN**, unrelated to the above |
| speculation absent from the paged loop | all | wired, blocked by the row above |

⚠ **Why the experiment was worth running:** "draft inherits kv_paged — FIXED" had been banked for
hours. When a later symptom looked like it could be that fix in disguise, testing it cost one rebuild
and one gate run. **A banked-wrong FIXED is worse than an open blocker, because nobody re-examines
it.** Restored afterwards and verified by an **empty git diff**, not by trusting the copy.

### ★ ROOT CAUSE OF THE −1, from instrumenting rather than guessing

```
W draft: llama_decode returned -1 | n_tokens=4 n_ctx_dft=4096 n_ubatch=512
W draft:   row 0: tok=271    pos=13 seq=0 logits=1
W draft:   row 1: tok=248077 pos=14 seq=0 logits=1     (248077 = dflash's mask token)
W draft:   row 2: tok=248077 pos=15 seq=0 logits=1
W draft:   row 3: tok=248077 pos=16 seq=0 logits=1
```

Four tokens at positions 13–16, seq 0, against **4096 free**. **Not capacity** — the draft cache will
not accept those positions for that sequence, because nothing occupies 0..12 in its decoder KV.

**Self-perpetuating**, which is why 22 attempts failed *identically* rather than degrading: the first
lands at pos 13 against an empty draft KV, fails, writes nothing; the next lands at pos 14 against the
same gap.

Encode side is fine in both arms — **0** `llama_encode(ctx_dft)` failures paged, 0 static. Decode
failures: **0 static, 22 paged**.

⇒ **The static path maintains the DRAFT CONTEXT's KV state between attempts, and
`update_slots_paged()` does neither half of it:**

```cpp
use_ckpt_dft = (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

⚠ **That is the 17-line `ctx_dft` bucket the audit classified as "applies ~unchanged".** The
classification was right and I then ported **none** of it. *Applies-unchanged still has to be
applied* — "this carries over" is a statement about difficulty, not a statement that it is done.

### ★★ THE −1 WAS AN OFF-BY-ONE (fork `cb3062141`)

Not KV priming, not checkpoints, not the draft-static fix, not embeddings — **one integer.**

The static path arms the draft **before** pushing the sampled token into the prompt mirror.
`get_n_draft_max`'s own comment says so: *"slot.prompt is not yet expanded with the `id` token sampled
above."* `update_slots_paged()` pushes **first** and arms after, so `n_tokens()` was already +1.

| arm | first draft decode |
|---|---|
| static | `n_tokens=4 pos0=12 seq0=0 n_ctx_dft=4096` |
| paged | `n_tokens=4 pos0=**13**` |

The draft cache refused the write, and every later attempt landed one further along the same gap —
which is why 22/22 failed **identically** and looked structural. A uniform wall of failures from a
single integer.

**Second fix, exposed once drafts actually generated:** `speculative.cpp:2625` asserts
`GGML_ASSERT(!dp.drafting || dp.result->empty())`. The static path satisfies it *structurally*, by
only arming in the `else` of `if (!spec_draft.empty())` — a leftover partial draft is REUSED, never
re-armed over. The paged path consumes the whole draft each step, so clearing is correct.

| | before | after |
|---|---|---|
| `#gen drafts` | 0 | **6** |
| decode failures | 22 | **16** |
| abort | — | gone |

⚠ **Speculation still does not run end to end.** `draft_n` reports 0; six drafts generate; sixteen
attempts still fail; none reach the verify as multi-row batches.

### ⚠⚠ METHOD: an error-only probe cannot show you what success looks like

Three passes of theorising — KV priming, the draft-static fix, embeddings extraction — and none was
the answer. **The answer came from logging the SUCCEEDING arm at the same site and diffing one
integer.**

And the first attempt at that probe landed on a *different* `llama_decode` site, so neither arm
printed. That read as "the probe is broken" when it meant **"the probe is in the wrong place"** — the
same class again, inside the tool being used to escape the class.

⇒ **When two arms differ, instrument BOTH.** The failing one tells you what broke; only the working
one tells you what it should have been.

### ⚠ CLOSE-OUT LIST — three of my own guards become FALSE the moment the loop half works

Written now so the landing commit cannot forget them. Wrong-and-loud beats wrong-and-quiet, including
when the loud thing is a stale warning of mine:

| item | where | action in the SAME commit that wires drafting |
|---|---|---|
| load-time warning *"speculative decoding is NOT implemented in the paged decode loop… the draft will NEVER be consulted"* | `server-context.cpp`, fork `7b614a636` | **remove or make conditional** |
| slot API reports `"speculative": true` under `--kv-paged` | `server-context.cpp:802` | resolves itself — the statement becomes true |
| *"no speculation under paged"* assertions | this doc, `kv-paged-facts` memory, `ds4-ports-lane` memory pointer | rewrite all three |
| `draft_pair_gate.sh` arm (c) expected to VOID at `draft_n=0` | `tools/ds4-gates/` + several commit messages | arm (c) flipping from VOID to a real acceptance number **is the event this whole rock is pointed at** |

### The 68-line audit — sorted by WHICH CONTEXT each line touches

The sort key that makes this tractable comes from a fix landed earlier the same day: `604c2858a`
forces the **draft context static**. So draft-side machinery carries over nearly unchanged, and only
target-side KV operations can be wrongly ported.

| bucket | lines | verdict |
|---|---|---|
| `ctx_dft` — draft-context checkpoints, `use_ckpt_dft`, draft params | 17 | **applies ~unchanged** — that KV is static by construction |
| `ctx_tgt` — target KV ops | 13 | **mostly DROPS OUT**, see below |
| neither — draft generation + bookkeeping | 55 | the straightforward part (`common_speculative_draft`, draft params, partial-draft reuse) |

**The `ctx_tgt` bucket largely evaporates, for two independent reasons:**

1. **Context shift is not reachable on the paged path.** The cluster at `:3550`
   (`seq_rm(slot.id, n_keep, n_keep + n_discard)` + `seq_add`) is the context-shift block. Measured on
   `update_slots_paged()`:

   | token | in the paged loop? |
   |---|---|
   | `n_keep` | **no** |
   | `n_discard` | **no** |
   | `shift` / "context shift" | **no** |

   The scheduler owns positions and never shifts, so this whole cluster is out of scope. One read, one
   decision, a third of the risky bucket gone.

2. **Target-side rollback checkpoints are unnecessary.** `use_ckpt_tgt` exists because some contexts
   cannot do a partial `seq_rm` and must snapshot instead — it is rollback machinery. The paged path
   has **no rollback**: rejected tokens leave KV past the new `n_past` and the next step overwrites
   them (`DS4P_SLOT_COVER`). The accepted count *is* the rollback.

⇒ What remains is draft generation plus draft-context bookkeeping, on a context that is now static.
**The audit collapsed the bucket that could have produced a wrong port.**

### ⚠ THE PREFIX-TRUNCATION CONTRACT — the loop half must size against `batch_lens`, never its own draft

B's chunker **clamps**:

```cpp
const int32_t want = 1 + (int32_t) group->pending_draft.size();
chunk_tokens[i] = std::min(want, std::max(1, budget - reserve_after));
```

So the rows actually emitted can be a **PREFIX** of the staged draft when the batch budget is tight —
`set_draft()` caps against `n_batch`, but that check cannot see how much of the budget the other
candidates in this step have taken.

**Consequences the loop half must respect:**

| rule | why |
|---|---|
| verify against `info->batch_lens[i] - 1` drafted rows, **never** against the caller's copy of the draft length | the emitted count is authoritative; the staged count is a request |
| build `idxs` for `common_sampler_sample_and_accept_n` with exactly `batch_lens[i]` entries, and pass a draft slice of `batch_lens[i] - 1` | the helper asserts `idxs.size() == draft.size() + 1` and will abort on a mismatch |
| a truncated tail is **silently dropped** — `update()` clears `pending_draft` unconditionally | correct: drafts are per-step advisory, not a queue. Stated so nobody "fixes" it into a carry-over. |

Getting this wrong is an off-by-tail that only appears under batch pressure — invisible at `np=1`
with a short prompt, which is every checkpoint run so far.

### ★ The verify helper, read at the body (`common/sampling.cpp:658`)

```cpp
GGML_ASSERT(idxs.size() == draft.size() + 1);
for (i = 0; i < draft.size(); i++) {
    id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);
    if (draft[i] != id) break;          // the accept-prefix
}
if (i == draft.size()) { /* bonus token at idxs[i] */ }
```

Two properties that are **invisible from the signature** and both load-bearing:

1. **It takes an explicit `idxs` vector** — it does *not* reach into `slot.spec_i_batch` or any
   static-slot bookkeeping. The paged loop passes `batch_offsets[i] + 0..n_draft` and it works
   verbatim.
2. **It accepts into the sampler chain as it goes**, including the rejected token it stops on. The
   caller must **not** re-accept the returned tokens — pattern-matching the non-speculative path would
   double-accept.

Its return length is 1..n_draft+1 and **is** the accepted count step D consumes. The two halves were
designed independently and meet exactly.

### The server-loop half — read, not yet written

The three consumption call sites, read before designing:

| site | today | needed |
|---|---|---|
| `sampled` (`:3220`) | sized `n_seq`, one per sequence | **batch-offset sized**, matching step D's layout contract |
| `stops` (`:3362`) | one per sequence | **unchanged** — a sequence stops or it does not; this is not a per-row property |
| `populate_token_probs(..., idx)` | already takes an index | unchanged — each accepted token reports its own probabilities |

⚠ I had been picturing `stops` becoming ragged alongside `sampled` "for symmetry". It must not.
That change would look tidy and would quietly redefine what a stop means.

**The drafting machinery the paged loop skips, quantified:**

| function | size | spec-related lines |
|---|---|---|
| `pre_decode` | 721 | **68** |
| `post_decode` | 275 | **18** |

★ **Two findings that shrink the job:**

1. `common_sampler_sample_and_accept_n(smpl, ctx_tgt, spec_i_batch, …)` **already exists** — the
   verify loop (sample each row, compare to the draft, stop at the first mismatch) does not need
   writing.
2. The static path's rollback — `n_rollback = spec_draft.size() + 1 - accepted.size()` followed by
   `seq_rm` — **is not needed on the paged path at all.** Rejected tokens leave KV past the new
   `n_past` and the next step overwrites them (`DS4P_SLOT_COVER`: the write slot is a pure function of
   position). The rollback *is* the accepted count, which step D already carries. The paged loop skips
   the most delicate piece of the static one.

⚠ **And what keeps it from being "add three calls":** those 68 lines are not all draft generation.
They include context-shift interaction (`seq_rm(slot.id, n_keep, n_keep + n_discard)`), checkpoint
handling on both target and draft contexts, and partial-draft reuse across steps. **Whether each
applies to the paged loop is a per-item question**, and being wrong in either direction is a defect:
skip one that matters and drafts go stale; port one that does not and `seq_rm` gets called on a paged
cache.

That per-item audit is the remaining work. No estimate — the scheduler half took six items, two
retractions and four mutation tests, and it never touched the drafter's own KV lifecycle.

### ⚠ A FIFTH piece the "four lines" scoping did not contain: the draft-token INPUT CHANNEL

`step()` builds decode rows from `logical_seq.back()` — one token, from state the scheduler **already
owns**. For B, the N drafted tokens have to get *into* the scheduler before `prepare_batch` runs.
There is no entry point for that today.

So B needs a per-group pending-draft buffer plus a way to set it: a **new public API surface**, larger
than any single line in A–D. Naming it here so "four localised lines" does not get banked the way
"the API is the work" almost did — the same error in the opposite direction.

### ⚠ SEMANTICS DECISION THAT GATES B/C — advance-count ≠ append-count

Learned by shipping the bug. The original `update()` advances `n_past` by `batch_lens[i]` and appends
**exactly one** token, always. Those are two different numbers and fusing them corrupts
`logical_seq` on every multi-token prefill chunk.

When `n_accepted` IS provided, a **final-prefill-chunk row still needs advance=lens / append=1**.
Decide before writing B/C:

- either `n_accepted` applies to **decode rows only**, prefill rows keeping legacy semantics, or
- a per-row discriminator distinguishes prefill-final from decode.

**DECIDED: a sentinel, not a struct change.** `prefill_pending[i]` discriminates *mid*-chunk rows, but
the **final** prefill chunk has `prefill_pending = 0` and still needs advance=lens / append=1 — so it
does not separate the case on its own.

```
n_accepted[i] < 0   ->  legacy semantics for this row (advance by batch_lens[i], append ONE)
n_accepted[i] >= 1  ->  speculative row: advance by n_accepted[i], append that many
```

Self-describing, needs no addition to `llama_paged_batch_info`, and the caller always knows which rows
it drafted for. A whole-array `nullptr` stays legacy for every row, so step D's checkpoint is
unaffected.

⚠ Under the `np=1` scope discipline a batch holds one sequence and is therefore either prefill or
decode, never mixed — so this distinction is not exercised at first. It is decided now anyway, because
the mixed case arrives with `-np>1` and a rule invented then would be invented while looking at a
failure.

### The draft-token INPUT CHANNEL — designed

`llama_paged_batch_info` is **output only** (scheduler → caller): `write_slots`, `block_table`,
`context_lens`, `batch_offsets`, `batch_lens`, `prefill_pending`, `seq_ids`. There is no inbound path,
which is why `step()` can only build decode rows from `logical_seq.back()`.

Proposed surface — one function, mirroring `add_request`'s shape:

```c
// Stage drafted tokens for a request. The NEXT prepare_batch() emits 1 + n_draft rows for it
// (the last accepted token plus the draft) instead of a single row. Cleared by update().
LLAMA_API bool llama_paged_scheduler_set_draft(struct llama_paged_scheduler * sched,
                                               int32_t             request_id,
                                               const llama_token * draft,
                                               int32_t             n_draft);
```

Stored as a per-group `pending_draft` vector beside `logical_seq`. `step()` then emits
`1 + pending_draft.size()` rows for that group (change **B**) with logits on all of them (change
**C**), `allocate(1 + n_draft)` (change **A**), and `update()` consumes the accepted count (change
**D**, already landed).

⚠ **This is the largest single piece of the rock** — a new public entry point, bigger than any of
A–D individually — and the "four localised lines" scoping did not contain it.

### ⚠ THE CHECKPOINT MUST USE A MULTI-CHUNK PROMPT

Four short-prompt arch runs passed the broken version green. Nothing in a single short request reads
the **interior** of `logical_seq` — decode reads `.back()`. The interior is read by the
preemption/recompute requeue (`:417–424`, which replays it into KV), fork inheritance
(`llama-kv-cache-paged.cpp:414`), the prefix-share matcher (`:199`) and `on_finish` (`:348`).

So the standing checkpoint for every remaining step is:

```
test-paged-kv-e2e -m <model>                      scheduler-level, the only one in the tree
arch_serve_gate.sh on 3+ verified archs           short prompt
arch_serve_gate.sh with AG_PROMPT = 30+ tokens    MULTI-CHUNK prefill  <- the one that catches this class
```

### Public API

`llama_paged_scheduler_update()` needs a per-sequence **accepted count** beside the tokens. Make it
nullable so the existing one-token semantics survive, and grep the callers first so the blast radius
is a fact rather than an assumption. Semantics to state explicitly: `n_past += accepted`,
`logical_seq` extends by **all** accepted tokens, `seq_max_pos` = last **accepted** position, and
EOS-mid-run truncates the accepted count at the stop.

### Scope discipline

**`np=1` greedy first.** The `-np>1` absolute-offset corruption is a separate open defect; coupling
them makes both undebuggable. `n_batch == n_ubatch` (`llama-model.cpp:2624`) bounds verify width.

### Pre-registered acceptance criterion, with baselines already measured

`draft_n > 0` is necessary and **nowhere near sufficient** — a draft fed wrong context still proposes.
The discriminating check is **arm (c) paged acceptance ≈ arm (b) static acceptance on the same pair**:

| pair | arm (b) static acceptance |
|---|---|
| `dflash` + Qwen3.5-4B | **13 / 27** |
| `gemma4-assistant` + gemma-4-E2B-it | **11 / 33** |

**⚠ "Comparable" is not a criterion. The tolerance is fixed HERE, before the feature exists:**

> **PASS** = arm (c) paged acceptance ratio within **±0.05 absolute** of arm (b) static acceptance
> ratio on the same pair. Outside that = FAIL, investigate the mirror or the context.

⚠ **The tolerance was ±0.15 by guess. It is now ±0.05 by MEASUREMENT, and the debt is paid.** The
pre-registration required measuring arm (b)'s own rep-to-rep spread before the feature lands. Done,
`dflash` + Qwen3.5-4B, 5 reps, `temperature 0`, fixed seed, `cache_prompt: false`:

```
rep 1  draft_n=27  accepted=13  ratio=0.4815
rep 2  draft_n=27  accepted=13  ratio=0.4815
rep 3  draft_n=27  accepted=13  ratio=0.4815
rep 4  draft_n=27  accepted=13  ratio=0.4815
rep 5  draft_n=27  accepted=13  ratio=0.4815
```

**Spread is ZERO.** Greedy acceptance is fully deterministic within a path, so *any* arm-to-arm
difference is a real difference and not noise. The residual ±0.05 does not cover run-to-run variance
— there is none — it covers **fp divergence between the static and paged KV paths flipping a marginal
accept/reject decision**. At 27 drafts that is at most ~1 flipped decision; anything larger is a bug,
not arithmetic.

This is the *"limit calibrated on unchecked runs"* scar run in reverse: a bound set by guess, then
measured, and the measurement made it **four times tighter**. Had I skipped the measurement I would
have shipped a criterion that could not distinguish a working feature from a badly broken one.

Leaving "comparable" undefined until after the run means defining the bar while looking at the answer.
That is the impossible-bar scar from the other direction, and it is the easiest one to walk into
because it never feels like cheating.

**Pre-registered mutation test** — the one that could not be built for read-only, and can be built
here. Temporarily force `accepted = submitted` (accept everything, verify nothing). Output MUST
diverge from static within a few tokens, because the `dflash` baseline says roughly half its proposals
are wrong. Three states, same shape as the contiguity assert:

| state | expected |
|---|---|
| feature + check | PASS, output identical to static |
| accept-everything mutation | check goes RED |
| no check at all | silently wrong, which is today |

On record before the feature exists so it can embarrass me.

⚠ The `gemma4-assistant` pair additionally needs the KV-sharing fix — its only viable target is
`gemma-4-E2B-it`, which is now REFUSED under `--kv-paged`. `dflash` + Qwen3.5-4B is the usable pair.

 An estimate with
no measurement behind it becomes a schedule.
