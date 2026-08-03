# DESIGN — ds4-ports Phase A thought experiments: P0 tier + 3b

**Status: FOR THE OWNER'S READ. No code until he has read this (standing order).**
Author: Fable-DS4 | Date: 2026-08-03 | Grounding: every claim below is against read code in
`<BOX>/wt-ds4-ports` @ `70ef59be1` (= current fleet) or `/tmp/ds4` @ 82d2a6f, with
file:line. Provenance: control-flow reads, no runtime measurements yet — every design here
leads with a MEASURE gate before its first line of implementation code.

The single biggest Phase A finding: **three of the five items shrank on contact with the
code.** Upstream's server refactor already carries much of what ds4 taught — the ports become
"verify, close the specific gap, gate it" rather than "transplant a subsystem." That is good
news for effort and risk, and it changes the sequencing recommendation at the end.

---

## 1. P0-3 dead-client abort — ✅ CLOSED ZERO-CODE 2026-08-03 ~02:10 (measured, as designed)

**The gate ran on the Mac CPU (no GPU needed — the chain is engine-agnostic and slow CPU
prefill is the ideal kill window) and the current tree PASSES:** detection 0.55-0.58 s for
ALL THREE death modes (FIN, RST via SO_LINGER, SIGKILL) = within the 1-s
`HTTP_POLLING_SECONDS` tick; slot freed 7.8-9.4 s = next batch boundary (one in-flight ~7 s
CPU batch drains, nothing new scheduled). Scaled to the box: a dead client at 1M context
wastes ~2-3 s, not 16 minutes. The ECONNRESET-night class is closed — that night's binary
predates the modularized cancel wiring (candidate 1 in §1.2 below was the answer).
Artifact: `tools/ds4-gates/abort_paths_gate.sh` + `abort_client.py` + results/abort-*.txt.
It took FIVE gate iterations to get a clean measurement (oversized prompts, prompt-cache
hits, SIGINT-ignored-in-background, natural-completion masquerade, macOS wc padding) — every
trap is now encoded in the script header. Step 2/3 of the design below: NOT NEEDED.

### (original design, kept for the record)

### 1.1 What the tree already has (all read, file:line)
- httplib's `req.is_connection_closed` — a non-blocking socket peek, functionally ds4's
  `MSG_PEEK|MSG_DONTWAIT` probe — is captured into every request as `should_stop`
  (server-http.cpp:590/637/654).
- The HTTP handler thread polls it: `rd.next(should_stop)` →
  `recv_with_timeout(ids, polling_interval_seconds)` → on each timeout tick calls
  `should_stop()` (server-queue.cpp:391-398). Interval = `HTTP_POLLING_SECONDS` constant
  (server-context.cpp:3976; value TBC when drafting the gate script).
- Dead client → handler returns → `~server_response_reader` → `stop()` → `SERVER_TASK_TYPE_CANCEL`
  posted to the queue FRONT (server-queue.cpp:443-456).
- Queue loop: tasks (incl. CANCEL → `slot.release()`, server-context.cpp:2465-2474) are pumped
  BEFORE `callback_update_slots()` each iteration (server-queue.cpp:163). A prefill spans many
  `update_slots` calls, so batch-boundary abort is structurally present.
- Decode-level abort is explicitly unwired: `TODO: handle ret == 2 (abort) when we start
  aborting` (server-context.cpp:3662).

### 1.2 Why his ECONNRESET night still happened — candidate causes, each testable
1. That night's binary predates this modularized cancel wiring (most likely).
2. `HTTP_POLLING_SECONDS` too coarse for the workload.
3. Half-open connections: a peek detects FIN/RST arrivals, NOT silent drops (NAT timeout,
   pulled cable). No socket API detects those without a write or keepalive.
4. `update_slots` internal chunking: verify one call cannot process multiple `n_batch`
   chunks without returning to the task pump (the retry pattern at ctx:3640-3690).

### 1.3 Design
**Step 0 (the actual first deliverable): a measurement gate, not code.**
`abort_paths_gate.sh` (name borrowed from ds4's suite): serve V4 paged config, launch a 256K
prefill, kill the client three ways — clean FIN (curl SIGINT), RST (close without shutdown),
kill -9 — and measure slot-free latency from the server log. PASS = slot frees ≤ 1 ubatch +
poll interval (≈ 2-4 s at ub 2048 / ~1000 t/s prefill). Run for streaming AND non-streaming.
**If this passes on the current tree, P0-3 is a no-op and closes with a gate script as its
artifact.**

**Step 1 (only what the gate shows):** likely candidates are tuning `HTTP_POLLING_SECONDS`,
wiring cancel-check earlier in the slot's prompt-processing state, or nothing.

**Step 2 (probably never):** mid-ubatch abort via `llama_set_abort_callback` landing at the
ret==2 TODO. At our prefill speeds a ubatch is ~2 s; the extra machinery buys seconds and adds
a hot-path callback. Deferred with trigger: only if a future model's ubatch wall-time exceeds
~10 s.

**Explicit non-goal:** silent-drop detection. That is TCP keepalive / SSE-ping configuration
(`sse_ping_interval` exists for streams, ctx:4298), not engine code.

### 1.4 Pressure tests
- 16 agents, one dies mid-prefill: CANCEL is queue-front, handler scan is O(slots) — fine.
- 1M context: prefill ≈ 16 min; without the working chain a dead client burns ALL of it —
  this is why the gate matters at exactly our scale.
- Live clients: zero slot-loop cost by construction (polling lives on handler threads).

### 1.5 Kill gates
(1) slot-free ≤ 1 ubatch + poll tick for FIN and RST deaths; (2) surviving concurrent request
byte-identical output vs undisturbed run; (3) zero decode-curve regression (trivially true if
no code changes).

---

## 2. P0-2 prefix re-validation — the gap is ONE missing binding, not a missing system

### 2.1 What is already validated (read)
- **Claim vs request:** full token-array LCP compare — slot selection (ctx:1611), n_past reuse
  (ctx:3200), prompt-cache alloc/load (task:1662/1687/1742/1753), mtmd chunks by id+size
  (server-common.cpp:471+). This is ds4's byte-memcmp, in token space, already everywhere.
- **State parse:** restore rejects, loudly and fail-safe: layer-count, cell-capacity, v_trans,
  per-layer K/V type + row size, DSA k_idx presence/layout (llama-kv-cache.cpp state_read_data);
  file paths additionally check LLAMA_STATE_SEQ_MAGIC/VERSION (llama-context.cpp:3197).
  On any failure the server already degrades to recompute (`prompt_load` false →
  `prompt_clear()`, ctx:1676-1678).

### 2.2 The one thing NOBODY validates
The binding **token-claim ↔ state-bytes**. `server_prompt_cache::load` restores the payload,
then adopts the entry's token record on trust (`prompt = std::move(it_best->prompt)`,
task:1800+). In-RAM this binding was created in-process at save; nothing detects
post-save corruption or entry cross-contamination — and the moment states hit DISK (P1-5),
stale/corrupt/foreign files inherit this trust. Truth: KV-correctness-for-tokens cannot be
verified without recompute; what CAN be cheaply guaranteed:

1. **Payload integrity:** hash (xxh3/64) over state bytes at save, verify at load.
2. **Identity:** tuple (arch, model desc, KV K/V types, n_layer, head_dim) stored per entry,
   compared at load — catches cross-model/config reuse before the parse layer's per-field
   errors, and gives one counter instead of scattered logs.
3. **Pairing sanity:** state's own cell_count cross-checked against the token record length
   at SAVE time (the state header already carries it) — a wrong-save becomes loud at the
   moment it happens, which is the class the solar-night suspicion lived in.

### 2.3 Cost arithmetic (why full hash is affordable)
64K V4 state ≈ 6 GiB → xxh3 at ~30 GB/s ≈ 0.2 s per save/load, vs 125 s cold prefill it
replaces. 1M state ≈ 7.4 GiB ≈ 0.25 s vs ~16 min prefill. Hit path (token compare only) is
untouched — zero overhead where ds4's gate demands it. A sampled-hash mode (head+tail+strides)
stays a flag if measurement ever shows the full pass mattering at 16-agent save bursts.

### 2.4 Kill gates
(1) zero measurable overhead on cache hits; (2) bit-flip a cached payload under
`DS4P_REVALIDATE=1` → recompute, correct output, `reval_hash_fail=1`; (3) exactness counters
in `/metrics`: `ds4p_reval_{saves,loads,hash_fail,identity_fail,cell_mismatch}` — a run is
valid only if the counters say the path executed (announce-vs-verify scar).

### 2.5 Open questions for the owner
- Full-hash default with sampled-hash flag, or the reverse?
- Identity tuple: include the GGUF file identity (path+size+mtime — cheap) or model desc
  string only?

---

## 3. P0-1 yield-quench governor — a controller over counters that ALREADY exist

### 3.1 The substrate (read; this item shrank the most)
- Per-slot stats already tracked and reset per task: `n_draft_total`, `n_draft_accepted`,
  `n_draft_verif_steps`, `n_accepted_per_pos` (ctx:328-331, 357-360).
- Per-SEQUENCE control seam already exists: `common_speculative_get_draft_params(spec, seq_id)`
  exposes a mutable `n_max` override ("can be used to constrain the max draft") and a
  `drafting` flag; accept feedback flows through `common_speculative_accept(spec, seq_id, n)`
  (common/speculative.h:32-71). The drafter-implementation CHAIN (external draft model, MTP/
  nextn — `need_embd_nextn` ctx:406) sits BELOW this seam.
- ⇒ The governor is server-side state + one gating decision at draft time. **No engine
  surgery, and it is drafter-agnostic by construction** — it governs GLM MTP, DSpark heads,
  and the incoming giants' MTP identically (the doctrine holds: never frame this via GLM
  alone).

### 3.2 Design — semantics VERIFIED against ds4 source (ds4.c:34815-34827, read tonight)
The plan's digest drifted from the source in three ways; this design follows the SOURCE:

```
yewma = guard                                  // init AT break-even: neutral start,
                                               // a request must PROVE itself bad
yewma = (1-α)·yewma + α·yield                  // α = 1/8
debt  accumulates shortfall vs guard
quench iff:  steps ≥ minev  AND  yewma < guard  AND  debt > budget    // CONJUNCTION
```

The conjunction is load-bearing: debt-only triggering (what the digest implied) would quench a
request that had one bad patch and RECOVERED (ewma back above guard). Both conditions + the
`minev` floor = currently-bad AND cumulatively-bad AND enough evidence. Quench is TERMINAL for
the request (`draft_params.n_max = 0` for that seq; ds4 measured re-arming to lose —
`DS4P_QUENCH_REARM=1` kept as an experiment flag). ds4 ships this default-ON behind
`DS4_DSPARK_QUENCH` with a forced-quench env for gates (ds4.c:34988-35001) — we mirror that
shape under `DS4P_*`.

**The numbers that must be OURS: guard (break-even), minev, budget.** Even ds4's own artifacts
show guard as a calibrated PARAMETER, not a constant (2.16 in the plan's digest, 2.3 in their
D4.4 gate print at ds4.c:28645; their MTP arms and DSpark arms carry separate calibrations,
ds4.c:22571). Our own data says it is target-dependent (same head: 0.60× on the 66.8 t/s box
target, 1.65× on the 7.4 t/s Mac target). Options: (a) static per-serve flag
`--ds4p-quench-guard`, measured once per config by a gate script; (b) auto-calibrate at first
verify steps of each serve (verify-step cost vs decode-step cost). (a) is honest and simple;
(b) is self-tuning but adds a warm-up regime. **Recommendation: (a) first, (b) as a later
flag.**

### 3.3 Pressure tests / failure modes surfaced
- 16 agents, mixed workloads: per-seq isolation is the point — a repetitive-workload agent
  keeps its speedup while a creative-writing agent quenches. Governor state is per slot, no
  cross-talk.
- The F1 pathology (q8 draft KV → 0-2% acceptance): an ungoverned server ships garbage-speed
  forever; the governor auto-quenches it in ~4 steps. This makes P0-1 a SAFETY NET for a
  measured bug class, not just an optimizer.
- Short requests (< 4 verify steps): governor never fires — correct, no evidence.
- EWMA poisoning across tasks: counters reset per task (ctx:357) — begin each request armed.
- MTP-on-giants: nothing in the design references drafter type — the seam is above the chain.

### 3.4 Kill gates (from the plan, sharpened)
(1) forced-quench (`DS4P_QUENCH_FORCE=1`) byte-identical tokens vs `--spec-off`, overhead
≤ 1.004×; (2) acceptance-0.60 workload on the fast target auto-quenches to ≤ 1.01× of
no-spec baseline; (3) acceptance-0.85+ workload must NOT quench (0 quench events in the
counter); (4) `/metrics`: `ds4p_quench_{seqs,steps,debt_max}` + per-slot end-of-task log line.

### 3.5 Open question for the owner
Break-even calibration: static flag per serve config (recommended) vs startup
auto-calibration?

---

## 4. 3b kv-paged for hybrid archs — root cause found at the dispatch

### 4.1 The bug, precisely (read)
`create_memory` (llama-model.cpp:2054+): INKLING is `llm_arch_is_hybrid` → takes the
`llama_memory_hybrid`(_iswa) branch (filter special-case for FALCON_H1 || INKLING at :2171).
The `cparams.kv_paged` branch (:2334-2347) lives in the plain-attention leg of `default:` —
**a hybrid arch can never reach the paged constructor. `--kv-paged` is silently ignored** and
the hybrid's attention sub-cache allocates full `-c` statically = the measured 48 GiB wall
(B3). Second half: the fitter (`common_fit_paged_kv_blocks`, common/common.cpp:1217) budgets
`n_layers = llama_model_n_layer(model)` with uniform attention head_dim — for a hybrid,
attention-KV layers ≠ n_layer, so its pool estimate is the wrong family even once the
constructor is reachable (19.5 GiB estimate vs 48 GiB actual, measured 2026-08-02).

### 4.2 Design options
- **(A) RECOMMENDED — surgical:** under `cparams.kv_paged`, `llama_memory_hybrid` constructs
  its ATTENTION sub-cache as `llama_kv_cache_paged` (conv/recurrent state unchanged; it is
  hardcoded GGML_TYPE_F32 and per-seq fixed-size). Fitter fix in the same change: derive
  bytes-per-block from the CACHE's own composition — expose it from `llama_kv_cache_paged` /
  the memory object rather than re-deriving from hparams (the head_dim bug's lesson,
  cc4059bfd: duplicated arithmetic drifts; make the cache the single source of truth).
- (B) Generalize the paged pool to per-layer-type sub-pools — bigger surgery, only worth it
  with P2-8 continuous batching. Deferred, revisit when P2-8 starts.

### 4.3 Pressure tests / open technical questions (to answer in implementation gates)
- iswa hybrid variant: same construction point, verify SWA window interaction with block
  eviction.
- Paged scheduler asserts/assumptions about ALL layers being paged (llama_paged_scheduler_init
  rejected GLM_DSA loudly — verify its layer-filter tolerance for hybrid).
- Conv state per-seq size for inkling: measure and publish (it bounds dynamic admission later;
  vLLM's ~11 GiB/seq figure was their implementation, NOT ours).
- Fused-ops interaction: hybrid+paged must be swept under `-fa auto` (explicit `-fa on`
  disables the probe — standing rule), and watched for the resolve_fused_ops global-disable
  class on any CPU-resident layer.
- B4 (compute buffer ∝ ub × total ctx) is UNTOUCHED by 3b — it remains the 1M gate and its
  own item. 3b makes slot count dynamic; B4 makes 1M fit at all. Do not conflate the gates.

### 4.4 Kill gates (from the plan + walls doc)
(1) inkling serve with `--kv-paged` logs `creating llama_kv_cache_paged`, pool sized
on-demand; (2) decode curve 8K→512K within 5% of non-paged; (3) 4×1M with q8 KV boots without
OOM once B4 lands (until then: 4×64K parity vs today's serve); (4) non-hybrid archs
byte-identical vs pre-change build (regression fence).

---

## 5. P0-4 methodology — the process the other four run inside

- **Kill switches:** every ported feature behind `DS4P_<FEATURE>` env + `--ds4p-*` flag,
  default OFF. One feature, one switch, no compound flags.
- **Same-boot ABBA** for every perf number (A/B/B/A within one server boot where possible;
  across boots, pin clocks and note them). No cross-binary comparisons presented as A/B
  (the 63.6-vs-66.8 confound stays a worked example of why).
- **Exactness counters over log lines:** a gate passes only on its counter/artifact —
  byte-identical token streams (greedy, fixed seed, same batch composition) hashed and
  compared; `/metrics` counters proving the path executed. Announce-vs-verify scar is the
  design input here.
- **Gate scripts live in `ornith-1m/tools/ds4-gates/`** (his repo — survives llama.cpp branch
  churn), named after their ds4 ancestors (`abort_paths_gate`, `reval_gate`, `quench_gate`,
  `hybrid_paged_gate`) so the suite reads as the checklist ds4's names already are.
- **Provenance marks** on every reported claim (VERIFIED BY ME / control-flow read /
  arithmetic / inherited) — already policy, restated as part of the definition-of-done.

---

## 6. Revised sequencing + effort (consequence of Phase A findings)

| order | item | revised effort | why it moved |
|---|---|---|---|
| 1 | P0-4 scaffolding (env-gate helper + gate-script skeleton) | 0.5 d | unchanged |
| 2 | P0-3 **gate first** | 0.5 d, possibly closes with NO code | chain already exists |
| 3 | P0-2 binding | 0.5-1 d | narrower than planned: one hash + identity tuple + save-time cell check |
| 4 | P0-1 governor | 1-1.5 d | seam exists; break-even measurement is the real work |
| 5 | 3b hybrid paging | 1-3 d (unchanged) | cache-internals work, bounded, precedent exists |

Total P0+3b: **~3.5-6 focused days** (was 4-7.5), GPU only for gates in windows around the
regen. Everything lands on `ds4-ports`, merges to `fleet` only working+stable+tested and only
on his explicit approval (standing).

## 7. Open questions for the owner (the complete list)

1. P0-2: full-hash default vs sampled-hash default? Identity tuple with or without GGUF file
   identity?
2. P0-1: break-even as a static per-config flag (recommended) vs startup auto-calibration?
3. P0-3: if the current tree PASSES the abort gate, is a gate-script-only close acceptable
   (no code), with the ret==2 mid-ubatch abort left as a triggered deferral?
4. 3b: agree to option (A) surgical hybrid paging, deferring (B) per-layer-type pools to
   P2-8?
5. Gate scripts location `ornith-1m/tools/ds4-gates/` — good, or prefer them on the
   llama.cpp branch?
