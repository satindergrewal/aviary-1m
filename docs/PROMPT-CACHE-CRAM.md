# Prompt cache sizing (`--cache-ram`) — measured, three regimes

**Status:** mechanism and ratio MEASURED (Fable session, 2026-07-30). GLM byte counts are
arithmetic on our own measured KV/token, not a GLM serve log. Disk tier does **not** exist.

## Why this matters

A cache **hit** replaces a full re-prefill. Measured on Qwen3-4B Q8_0 / Metal:
**prefill 28,785 ms → 26.6 ms (1,024×), wall 28.95 s → 0.45 s (64×).**

For the mission — max context on our own hardware — this is the single cheapest TTFT lever
we have, because it is a **flag**, not a code change. But the shipped default puts a 4-slot
long-context config in the *worst* of three regimes, not the best.

## The three regimes

Harness: `tools/prompt-cache/cram_ab.sh`, `cram_thrash.sh` (raw per-request JSON alongside). Protocol is 3 requests on one slot —
long prompt A, then long prompt B (which displaces A), then A again. A and B differ from
token 1 so slot-level prefix matching cannot confound the result.

**Witness is `timings.prompt_n` on request 3.** Not `tokens_evaluated` — that reads 19,277
in *every* arm including the misses, which is precisely the false-pass trap recorded in
`spec-decode-measurement-gotchas`.

| arm | pool vs state | req 3 | `prompt_n` | prefill | wall |
|---|---|---|---|---|---|
| `-cram 1` | fits nothing | MISS | 19,277 | 28,785 ms | 28.95 s |
| **`-cram 3000`** | **fits 1, not 2** | **MISS** | **19,277** | **28,315 ms** | **28.68 s** |
| `-cram 8192` | fits both | **HIT** | **1** | **26.6 ms** | **0.45 s** |

Each regime has its own log signature at `-lv 4` (invisible below it):

```
-cram 1     alloc: prompt state 2712.034 MiB exceeds cache size limit 1.000 MiB, skipping
            prompt cache update took 0.66 ms          <- refused BEFORE the copy
-cram 3000  update: prompt 19284 tokens, 2712.034 MiB           (166.24 ms — A saved)
            alloc: making room, removing oldest entry (size = 2712.034 MiB)   <- A EVICTED
            update: prompt 20185 tokens, 2838.747 MiB           (199.33 ms — B saved)
            => req 3 re-prefills. The 166 ms spent saving A was pure waste.
-cram 8192  both resident -> req 3 prompt_n = 1
```

### The counter-intuitive part

Over-cap saves do **not** waste bandwidth. `server_prompt_cache::alloc()` runs *before*
`llama_state_seq_get_data_ext()`, returns `nullptr`, and `prompt_save()` bails without
copying — measured at **0.66 ms**. The expensive failure mode is **fits-one-but-not-two**,
where the copy happens and is then evicted by the next save.

## Source facts (both trees: box `llama.cpp-idxfilter`, Mac `llama.cpp-dspark-metal`)

| fact | location |
|---|---|
| default cap **8192 MiB** | `common/common.h:626` / `:636` — `int32_t cache_ram_mib = 8192;` |
| flag `-cram, --cache-ram` | `common/arg.cpp:1628` |
| over-cap entry **dropped** (with `SRV_WRN`) | `tools/server/server-task.cpp:1678-1683` |
| eviction loop to make room | `server-task.cpp:1698-1706` |
| **RAM-only**, no disk tier | `server-task.h:633-641` — `std::list<server_prompt_cache_state> states;` |
| real bulk state copy | `server-context.cpp:238` — `llama_state_seq_get_data_ext` |
| `alloc()` before copy (why over-cap is cheap) | `server-context.cpp:233-236` then `:238` |
| `--cache-idle-slots` **default true** | `common/common.h:633` |
| every idle slot saved on each task launch | `server-context.cpp:2393` |

## Mapping onto our GLM-5.2 config

Using our own measured **93.9 KiB/token** (post indexer-KV fix):

- one full **64K** slot state = **6,010 MiB (5.87 GiB)**
- default cap 8192 MiB → holds **exactly one**; two need 12,019 MiB → **eviction**
- **so the approved 64K × 4-slot config sits in the middle row of the table above**
- measured copy bandwidth 2712 MiB / 166 ms = **16.0 GB/s** → a GLM 64K state costs
  **~368 ms** to save
- `--cache-idle-slots` saves *every* idle slot per launch → with 3 idle slots, **~1.1 s of
  copying per task launch**, most of it evicted before it can be used

To hold N slot states:

| N | need | flag |
|---|---|---|
| 2 | 12,019 MiB | `-cram 12288` |
| 3 | 18,029 MiB | `-cram 18432` |
| 4 | 24,038 MiB | `-cram 24576` |

## Recommendation (not applied — approved config untouched)

**`-cram 12288`** today. Holds 2 states, doubles current capacity, costs 12 GiB.

Sized rather than maxed because the box was checked first: **125 GiB total, 6 free, 29 GiB
already in swap**, with a 16 GiB `llama-quantize` resident. PSI reads
`some avg10=0.00 avg60=0.00` so nothing is stalling *now* and that swap is historical from
the big model loads — but `-cram 24576` should wait until the chain and quantize release.

Alternative if RAM is tight: `--no-cache-idle-slots` stops paying the ~368 ms/slot save cost
that the current cap mostly discards. That trades the *possibility* of a 1,024× hit for
lower launch latency — worth measuring, not assuming.

## Not established

- **Disk tier**: does not exist in either tree. Grep finds only `prompt_save`, `prompt_load`,
  `server_prompt_cache::load`. The "3–19 s NVMe restore vs 25–40 min re-prefill, 100–500×"
  figure is an inherited proposal, **not** measured here.
- **GLM runtime numbers**: the ratio and mechanism are measured on 4B/Metal. The GLM byte
  counts above are arithmetic. A GLM serve log needs a GPU window.

## Harness note

Raw per-request records are committed next to the scripts (`req_cram1.jsonl`,
`req_cram3000.jsonl`, `req_cram8192.jsonl`) so the table above can be re-derived without
re-running anything.

The first version of `cram_ab.sh` **exited 0 while all six requests returned HTTP 400** on a
context overflow (19,277-token prompts into `-c 16384`). Fixed: the request helper now exits
non-zero and prints the server's error body, and the arm prints
`ARM ... HAD FAILED REQUESTS -- results are NOT valid`. Same false-pass family as the
`grep -c` misread and `| tail -3 && echo PUSHED`.

---

# Addendum: the lookup is prefix-based, and its selection rule matters once `-cram` is raised

Source: `tools/server/server-task.cpp:1742-1790` (`server_prompt_cache::load`).

```c
const int   lcp_cur    = it->prompt.tokens.get_common_prefix(tokens_new); // LONGEST COMMON PREFIX
const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();  // how much of the CACHED prompt survives
const float sim_cur    = float(lcp_cur) / tokens_new.size();         // how much of the NEW prompt is covered

if (f_keep_cur < 0.25f) { continue; }                        // "don't trash large prompts"
if (f_keep_best < f_keep_cur && sim_best < sim_cur) { ... }  // must beat the incumbent on BOTH
```

**1. It is not exact-match — it is longest-common-prefix.** This is the important correction to
the 3-request harness above, which only tested *exact* repeats. The real agentic pattern is a
prompt that **grows** each turn, and that pattern *is* served: turn N+1 reuses turn N's cached
prefix and prefills only the delta. So raising `-cram` helps the actual workload, not just an
artificial repeat.

**2. There is a 25% floor.** A cached entry is skipped unless at least a quarter of it is a
prefix of the new prompt. Restoring a 100K state to serve a 10K prompt would waste the
remainder, so the heuristic refuses. Consequence: a long session that gets *truncated* or
restarted short will not reuse the long entry.

**3. The selection is greedy and order-dependent — and this only becomes relevant at
larger `-cram`.** The winner must beat the incumbent on **both** `f_keep` and `sim`
(`&&`, not a combined score), and both bests are updated during the scan. A candidate that is
strictly better on `sim` but marginally worse on `f_keep` is skipped, so which entry wins can
depend on `states` iteration order. With the shipped 8 GiB cap and 64K states the pool holds
one entry and this never matters. **The recommendation to raise `-cram` is precisely what makes
it matter** — worth watching at `-lv 4`, where every candidate is printed with its `lcp`,
`f_keep` and `sim`.

## Restore cost is symmetric with save

`llama_state_seq_set_data_ext` at `:1780` copies the state back in. Measured from the hit arm:
**~170 ms for 2,712 MiB**, matching the 16.0 GB/s save figure. So a GLM 64K state costs ~368 ms
each way. Still trivially better than a full re-prefill.

## Where the hit-request wall time actually goes

The 0.45 s hit is fully accounted for, and **prefill is the smallest part**:

```
prompt eval   26.59 ms /  1 token     <- server forces >=1 token: "n_past was set to 19276"
eval         147.14 ms /  8 tokens    <- generating the 8 output tokens (18.4 ms/tok)
restore      ~170 ms                  <- llama_state_seq_set_data_ext, 2712 MiB
tokenize        5.6 ms                <- measured separately, see below
```

So the honest headline for a user is the **64× wall-clock**, not the 1,024× prefill number:
generation and the restore copy are irreducible. Both are reported above for that reason.

## Tokenization is a non-issue — measured, and it closes a board item

`tokenize_mixed` (`server-common.cpp:627`) calls `common_tokenize()` on the whole string every
request. There is **no cache, no memo, no prefix reuse** — the quarantined claim is true as a
statement about the code. It is also irrelevant to latency:

| chars | tokens | ms | tok/s |
|---|---|---|---|
| 1,954 | 908 | 0.5 | 1.7 M |
| 9,770 | 4,734 | 1.7 | 2.8 M |
| 39,080 | **19,277** | **5.6** | **3.4 M** |

Linear, and 5.6 ms for a 19K-token prompt. Extrapolated to 200K that is ~58 ms. **This closes
the "tokenizer multi-core / redundant-pass deletion, ~3× from deletion alone" board item as a
latency lever** — 3× of 5.6 ms is a 3.7 ms saving. Measured on the `/tokenize` endpoint,
CPU-only (`-ngl 0`), best of 3 per size, 5 reps at full size (5.6/5.6/5.7/5.7/5.8 ms).

**Escape hatch worth knowing anyway:** `tokenize_mixed` accepts an *array* mixing strings and
pre-tokenized integer token IDs (`:633-655`). A client can tokenize a stable prefix once and
send IDs — client-side prefix reuse with no llama.cpp change. Given the 5.6 ms measurement,
there is no reason to bother.

---

# Addendum 2: growing-prompt reuse MEASURED (the agentic case), and it compounds with context

Addendum 1 asserted growing-prompt reuse from a **source read**. That was a gap, so it is now
measured. Harness `tools/prompt-cache/cram_growing.sh`, raw records `req_growing.jsonl`.

**Design point that makes or breaks this test:** every growing turn is preceded by a *different*
prompt that displaces the slot. Without that displacement the slot still holds the prefix and
ordinary slot-level `n_past` continuation would serve the request — the measurement would credit
the prompt cache for something the slot did.

| request | prefilled | total | reused | log |
|---|---|---|---|---|
| T1 base (cold) | 5,116 | 5,116 | — | — |
| D1 displace | 19,610 | 19,610 | — | pushes T1 into the cache |
| **T2 grown** | **6,237** | **11,353** | **5,116 = 45.1%** | `f_keep = 0.999, sim = 0.451` |
| **D2 repeat** | **1** | **19,610** | **~100%** | `f_keep = 1.000, sim = 1.000` |
| **T3 grown more** | **6,236** | **17,589** | **11,353 = 64.5%** | `f_keep = 1.000, sim = 0.645` |

The logged metrics match the arithmetic exactly — `5116/11353 = 0.451`, `11353/17589 = 0.645` —
so the mechanism is pinned, not inferred. Each growing turn prefills **only its delta**
(~6,237 tokens both times). D2, an exact repeat, went **28.256 s → 0.305 s = 93× wall**.

## ★ The property that matters for a max-context mission

**Reuse rose 45.1% → 64.5% as the session grew**, because the delta is constant while the prefix
grows. The win therefore *increases* with context length:

| session | 2K turn delta | served from cache |
|---|---|---|
| 32K | 2K | 93.8% |
| 64K | 2K | **96.9%** |
| 128K | 2K | 98.4% |
| 246K (our new ceiling) | 2K | **99.2%** |

This is the opposite of a diminishing return: the longer the context we push, the more of every
turn the prompt cache can serve. It is the cheapest lever we have on long-context TTFT, and it
is a flag.

## ★★ And this is where the two findings compound against us

Growing-prompt reuse **requires the cached entry to survive displacement**. In this run the three
states were 720 + 1,597 + 2,758 MiB = 5,075 MiB, all resident under `-cram 8192`, so nothing was
evicted and reuse worked.

At the approved GLM config a single 64K state is **6,010 MiB** and the cap is **8,192 MiB**. One
state fits; the second save evicts the first. So the default does not merely lose the
exact-repeat hit documented above — **it also destroys the growing-session reuse that would
otherwise serve ~97% of every turn at 64K**, and it pays ~368 ms per save for the privilege.

That is the argument for `-cram 12288`, now resting on measurement rather than on a source read.
