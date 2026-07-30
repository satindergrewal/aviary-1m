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
