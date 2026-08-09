# PENDING — what is left, and which of it needs the GPU

**2026-08-09 · the answer to "what is pending, and can it be done on CPU?"**

The split below is the useful one: **GPU-bound** means it needs the Metal device and a loaded model,
so it queues behind whatever measurement is running. **CPU-only** means it can proceed on the Mac
right now, in parallel, with one caveat recorded at the bottom that cost a measurement today.

---

## The bar

Owner's bar, verbatim: *"equal or better than the static, and the ranges are 256k to 1M context
ranges, not 2k, 32k, 64k etc."*

## ★★★★ DECODE vs CONTEXT — the complete curve (2026-08-10 04:00)

Five rungs, one model, one kernel, `ignore_eos`, 4-arm ABBA each with its own drift bound.

| ctx | ratio paged/static | verdict | |
|---|---|---|---|
| 8,192 | **0.8451** | CLEARS | paged 15.5% slower |
| 32,768 | **0.9061** | CLEARS | paged 9.4% slower |
| 65,536 | **0.9131** | CLEARS | paged 8.7% slower |
| 131,072 | **1.0242** | **UNREADABLE** | effect 2.4% vs static drift 10.3% |
| 262,144 | **1.3471** | CLEARS | paged 34.7% faster |

**Monotone across five rungs, crossing 1.0 at ~128k.** The UNREADABLE at 128k is the gate being
**right**: at the crossover the effect is near zero by construction, so it cannot clear any drift.
**An instrument that reports "unreadable" exactly where the effect vanishes is working, not failing.**

### The mechanism — read from the ABSOLUTES, not the ratio

```
ctx        static tg  paged tg        decay per DOUBLING:  static  paged
  8,192      68.82      58.16         8k  -> 32k            1.050  1.014
 32,768      62.41      56.55         32k -> 64k            1.163  1.154
 65,536      53.65      48.99         64k -> 128k           1.421  1.267
131,072      37.76      38.68         128k-> 256k           1.911  1.453
262,144      19.76      26.62
```

⇒ **BOTH paths accelerate; static accelerates harder.** 1.05 → 1.16 → 1.42 → 1.91 against
1.01 → 1.15 → 1.27 → 1.45. **Smooth and progressive — no jump, no threshold.** A
bandwidth/working-set story, graduated.

⇒ **Pure span-bound decode decays toward 2.0×/doubling. Static's final step is 1.911 — 96% of that
limit. Paged is 1.453.** ⇒ **At 256k static decode is almost entirely span-bound; paged is roughly
half so.** That is the result in one line, and it is a measurement rather than a mechanism.

⚠ **THREE FRAMINGS DIED GETTING HERE, each stated too confidently before the next rung:**
*"paging removes a cost that grows with L"* → *"static falls off a cliff"* → **"both bend, static
bends harder."** Also dead: my pre-registered model that `1/ratio` is linear in `1/L` with intercept
1.0 — refuted outright, since `1/ratio = 0.7423 < 1` at 256k is impossible under a fixed-overhead
story. **The shape prediction held and the mechanism was wrong: right verdict, wrong mechanism.**

⚠ **NOT SHOWN:** five rungs, **n=2 per arm**, one model, one quant, one kernel, one box. **Prefill
was UNREADABLE at every rung** — that column is not read and is not data.

### ★★★ 2026-08-10 — PREFILL WAS NOT UNREADABLE. I WAS READING IT AS AN ENDPOINT AVERAGE.

**The whole prefill curve has been in every log this lane has ever produced.** `slot print_timing`
emits a cumulative pair every `-b` tokens; differencing consecutive lines converts the printed
running average (which lags, and therefore understates decay) into a true **interval rate**:

```
rate_i = (n_i - n_{i-1}) / (t_i - t_{i-1})
```

⇒ **One 400k prefill contains ~400 samples of a 5-7× decay curve, and I was collapsing it to a
single number and then comparing two such numbers against a 14.8% drift band.** That is why the
column read as noise. `prefill_curve.py` extracts it. **Nothing was measured to get this — it is a
re-read of artifacts already on disk.**

**512k ABBA, pos3 (paged, complete) vs pos4 (static), matched depth inside ONE prompt:**

```
depth        paged tok/s   static tok/s   static/paged
  50-100k       296.5         305.6          1.031
 100-150k       197.7         191.3          0.967
 150-200k       147.8         139.5          0.943
 200-250k       118.0         111.4          0.944   <- the 4th point, and it FLATTENED
 within-arm decay, full buckets:  paged 4.04x   static 4.67x
```

⚠⚠ **I PUBLISHED "THE RATIO TRENDS MONOTONICALLY" ON THREE POINTS AND THE FOURTH PLATEAUED IT.**
Committed at `588c911`, corrected ~20 minutes later when arm 4 crossed 250k. **0.943 → 0.944 is not
a continuation of a trend, it is a floor.** Three points can only ever show a trend; a plateau needs
a fourth to be visible at all. **This is the fourth framing to die in this section and the first one
I caught before the run ended rather than at the next rung.**

⇒ **THE CORRECTED READING, and it is a better mechanism than the one it replaces:**

```
0-100k     static FASTER   (ratio > 1)     paging's indirection is visible while FFN still dominates
~100-125k  CROSSOVER
150k+      PLATEAU 0.944   paged 5.6% faster, and NOT widening
```

★ **A PLATEAU IS WHAT THE PHYSICS PREDICTS AND "WIDENING FOREVER" IS NOT.** In prefill each batch is
512 tokens, so attention is a matmul over (512 × context) and the per-token FFN cost is amortised.
Shallow, FFN dominates and the paged block-table indirection shows as overhead. Deep, attention
dominates and **both arms scale with the same asymptote** — so the ratio converges to the constant
ratio of the two attention kernels' deep-context throughput, which is what 0.944 is.

⇒ **CONSISTENCY CHECK, and it lands exactly.** The endpoint prefill numbers for this run are
static 124.9 / paged 132.3 tok/s → **1.059**, i.e. static/paged = **0.944** — agreeing with the
plateau **to three decimals**. The endpoint average is the depth-weighted mean of the curve, and
most of a 400k prompt is deep, so it sits on the plateau.

⚠ **I FIRST WROTE "by two INDEPENDENT routes" AND THAT IS WRONG.** The endpoint average is the
depth-weighted mean of **the same curve**, off **the same server timer** — it is the aggregate of
the very samples being differenced, so it cannot independently confirm them. What it *does* check
is that my parser agrees with the server's own arithmetic, which is worth having and is not what I
claimed. **Consistent, not independent.** The real second instrument on the real data is Grok's
reproduction with a different regex and different differencing, which matched the table; the
known-answer control is `--selftest`. **Three checks, three different things, and I had merged two
of them into a word that overstated both.**

⚠⚠ **AND THIS CONSTRAINS A CLAIM ELSEWHERE IN THIS FILE: "+35% at 256k with the gap widening" is a
DECODE result and must not be read as a prefill one.** Prefill's gap does **not** widen past 150k.
The two are not in conflict — decode reads the *entire* KV cache for *one* token and is far more
bandwidth-bound, which is exactly where the paged layout's locality pays; prefill amortises that
over 512 tokens per batch. **Same run, same kernels, opposite shapes, and the mechanism says why.**

⇒ **Falsifiable at 1M:** if the plateau is the attention-kernel asymptote, the 1M prefill ratio is
also **~0.94**, not something larger. A widening prefill ratio at 1M refutes this mechanism.

★ **AND THE TREND SURVIVES THE COLD-ARM CONFOUND WHERE A LEVEL WOULD NOT.** A cold penalty scales
an arm **uniformly**, so it cannot manufacture a depth-**dependent** ratio. **The LEVEL of this
comparison is confounded (pos3 vs pos4); the SLOPE is not.** That makes the slope the strongest
claim in this document — the only one that does not depend on the prelude, the arm order, or the
cold-arm correction.

⇒ **Independent agreement with the rung table:** the rung crossover sits at **~128k** across six
server launches; the within-run ratio crosses **1.0 between 100k and 150k** inside a single prompt.
**Two measurements with almost nothing in common landing on the same crossover.**

⚠ **TWO BUCKETS ARE EXCLUDED AND ONE NEARLY BECAME A HEADLINE.** The first run reported **1.263**
at 0-50k. Artifact, not measurement: static's first progress line lands at **4096**, paged's at
**19968**, and the rate falls 650→300 tok/s across exactly that region — **static's bucket contained
the fastest 16k of the prompt and paged's did not.** The comparison was crediting static with a head
start it was never measured over. Every bucket now carries its covered fraction and anything under
80% is refused **by name**; the same guard drops static's still-filling 200-250k.
**Arms must differ in ONE thing — applied to the x-axis.**

⚠ **THE TWO ARMS PRINT PROGRESS IN DIFFERENT FORMATS** (`n_tokens = X,` vs `n_tokens = X / total`),
so the first regex matched 409 static lines and **zero** paged lines. Caught only because the
parser VOIDs below 3 samples instead of returning a one-armed "comparison". *Instrument both arms* —
in the one file whose entire purpose is to compare two arms.

⚠ **HARNESS DEFECT FOUND BY TRYING TO USE IT:** `static.log` / `paged.log` are reused **per arm
TYPE** while `.res` files are per **POSITION**. **pos4 overwrites pos1.** So the comparison that
would separate cold-penalty from drift — pos1's curve against pos4's — is destroyed by the harness,
and **every within-arm diagnostic in this lane has only ever covered the LAST arm of each type.**
Per-position log files: staged patch.

⇒ **PRE-REGISTERED before arm 4 reaches that depth:** static's **350-400k** interval rate lands in
**62-69 tok/s** (ratio 0.85-0.94) against paged's measured **73.3**. **If static comes back above
73.3 there, "static decays harder" is refuted at the depth that actually matters.**

⚠ **THE BAND WAS SET FROM THE TREND READING AND THE PLATEAU NOW PREDICTS ITS TOP EDGE.** A plateau
at 0.944 puts static at **69.2 tok/s** — one tenth of a point *outside* the band I registered.
**Recorded rather than widened.** A band moved after seeing which way the data went is not a
prediction, and this lane already has a scar for a limit calibrated on unchecked runs. So:
**0.85-0.94 was the trend model's band and it is about to miss high; the plateau model says
~0.94 exactly.** Whichever lands, one of the two was wrong in a way that was written down first.

⇒ **THE BAR, complete:** paged decode costs **9-15% below 64k**, is **parity at ~128k**, and is
**+35% at 256k with the gap widening**. **The owner's range starts at 256k. Inside his range, paging
wins, and wins harder the deeper it goes.**

---

## ★★★★ 512k VERDICT (2026-08-10 07:35) — MET ON ALL THREE METRICS, and the confound was ~1%

```
pos1 static  wall=3255.6  pp=124.94  tg= 8.480   n=399181  pred_n=512
pos2 paged   wall=3047.7  pp=132.30  tg=16.882   n=399181  pred_n=512
pos3 paged   wall=3046.1  pp=132.39  tg=16.642   n=399181  pred_n=512
pos4 static  wall=3273.4  pp=124.22  tg= 8.570   n=399181  pred_n=512

DECODE   1.9662   effect 96.6%  vs drift 1.4%   CLEARS x67
PREFILL  1.0623   effect  6.2%  vs drift 0.6%   CLEARS x11
WALL     1.0714   effect  7.1%  vs drift 0.6%   CLEARS x13
```

**★ COLD-ARM LINE FIRST, per the locked procedure: static pos1 vs pos4 agree to +1.06% on decode
and +0.6% on prefill. NO positional signature in either direction.** `prompt_n` is 399,181 on all
four arms and `pred_n` is the achieved 512 on all four, not the requested value.

### ⚠⚠ PROVENANCE CORRECTION, FOUND AFTER THE RUN: THE HEADER NAMES THE WRONG COMMIT

```
header stamped   tip: 074672e33            <- what the SOURCE TREE said
binary reported  build 10667 (35be827f3)   <- what actually RAN, one commit behind
```

The gate stamps `git rev-parse HEAD`, which describes the **source**. The binary reports its own
build commit on its first log line, and they disagreed. **The verdict stands** — the single missing
commit (`074672e33`, "`--kv-paged` on a model that cannot page must REFUSE, not abort") changes
behaviour only on models that *cannot* page, and this one pages: 128 consume events, pool built,
needle PASS on all four arms. **But nothing in the artifact could have told anyone that**, and a
reader would have attributed 1.9662 to code the measurement never executed.

⚠ **`gate_tip_stamp` DOES NOT CLOSE THIS, and I had it queued believing it did.** It appends
`+dirty(N)` for uncommitted edits — a different hole. **A stale BINARY built from a CLEAN tree
produces a clean stamp and a wrong claim.** Provenance has to come from the artifact that did the
work, not from the tree beside it.

⇒ **FIXED in `arm()`: read the binary's own build commit, compare to the source tip, VOID on
mismatch** (override `PP_ALLOW_STALE_BIN=1`). It runs seconds after startup, so refusing costs
nothing while discovering it after a 3.5 h ABBA costs the ABBA. **Controlled both directions on
real logs:** the 512k log extracts `35be827f3` and would VOID; a fresh log extracts `9eb5b1241`,
equal to the tip, and proceeds; differing abbreviation lengths compare on the shorter.

⇒ **Q1 — does the ABBA-mean decode ratio clear 1.8? YES, 1.9662, and not marginally.** The
pre-registered threshold was "clears 1.8 unless pos4 exceeds **+19.6%**". It came in at **+1.06%**.

⇒ **Q2 — how far does the prelude correction push it down? IT DOESN'T. The correction is ~1%.**
Q2 was declared **UNBOUNDED** before the run, because the warm-up prelude's validated envelope is
32-64k and 512k is far outside it. **That was the honest label given what was known, and the data
answered it instead of an argument.** The prelude works at 512k.

### ★★ PREFILL AND WALL RESOLVE FOR THE FIRST TIME AT ANY RUNG — AND THE EFFECT NEVER CHANGED

At 256k prefill was **"1.7% inside a 14.8% band"**. Here the same column reads **6.2% against a
0.58%/0.07% band**. The warm-up prelude, `ignore_eos`, and the champion geometry together collapsed
the drift by roughly **25x**, turning an invisible effect into an 11x clearance.
**Nothing about the machine changed. The instrument did.** That is the single most useful sentence
in this file for anyone about to call a result "unreadable".

### ⚠ THREE THINGS TO MARK AGAINST MYSELF, NOT ONE

**1 · Both branch tables priced a scenario that did not occur.** Mine and the foreman's spanned
+6.5% to +25% cold penalty; reality was +1.06%. **Pre-registering was still right — it is what made
"clears 1.8" a decision rather than a reaction — but neither of us registered the branch that
actually happened, which is that the confound would be ABSENT.** A branch table that spans only the
"it's bad" range is a one-sided prior wearing a table's clothes.

**2 · "Plateau at 0.944" was two buckets. With five it is 0.940 ± 1.12%.**

```
150k 0.943 · 200k 0.944 · 250k 0.950 · 300k 0.929 · 350k 0.936
mean 0.9404, spread ±1.12%   vs endpoint prefill 0.9412  ->  agree to 0.08%
```

**The plateau claim survives; the precision I implied did not.** 0.944 was one sample of a band,
quoted as a constant — the same shape as the 0.6% drift figure, on the same day, by the same hand.

**3 · The pre-registered 350-400k bucket came in at 68.6 tok/s.** The trend band was 62-69 and
**contains it**; the plateau point estimate was 69.2 and **missed by 0.9%**. ⚠ **A 7-wide band
containing the answer is WEAKER evidence than a point estimate within 1%**, so the honest scoring is
that **the plateau model won and the trend band passed by being vague.** Recorded that way rather
than claiming both were right.

### THE BAR, ACROSS HIS RANGE

```
256k  decode 1.347      512k  decode 1.966
```

**Inside the owner's stated range paging is not a tie. It is roughly double at 512k, and every
metric moves the same direction.**

⚠ **Scope, unchanged and still binding: `-np 1`.** Every number above is single-slot. The reason is
**not** a live defect — see the corrected block above — it is that **multi-slot at long context is an
empty composition cell** (`warm_multislot` closed it at 8k/block 16/short prompts; this runs
`-np 1`/block 64/400k fill). Costed at ~1.5 h and boarded.

---

## ★★★ 256k VERDICT (2026-08-10 03:20) — the bar is MET, and this is the first defensible number

The first ABBA with every instrument correct at once: champion kernel, real 512-token decode window,
warm-up prelude, per-metric drift bounds, cold-arm check, full provenance in the header.

```
pos1 static  wall 856.1  pp 235.8  tg 19.16      pred_n=512 (26.7s)
pos2 paged   wall 915.0  pp 218.2  tg 27.00      pred_n=512 (19.0s)
pos3 paged   wall 783.3  pp 256.0  tg 26.24      pred_n=512 (19.5s)
pos4 static  wall 817.8  pp 246.7  tg 20.36      pred_n=512

WALL     1.0146   effect  1.5% vs drift 14.4%  ->  UNRESOLVED
PREFILL  0.9830   effect  1.7% vs drift 14.8%  ->  UNREADABLE
DECODE   1.3471   effect 34.7% vs drift  5.9%  ->  paged FASTER, CLEARS
```

Needle PASS on every arm, and the read was **pre-registered before the data** and matched branch for
branch.

⚠ **`PREFILL 0.9830 → UNREADABLE` IS NOW KNOWN TO BE A LIMIT OF THE READING, NOT OF THE DATA.** The
512k run shows the prefill column has a within-run structure that endpoint averages destroy: the
matched-depth ratio plateaus at **0.944**, and **that plateau reproduces the endpoint average of the
same run to three decimals**. So 0.983 here is very likely a real depth-weighted mean too, sitting
between the shallow region (static faster) and the deep plateau — 256k has proportionally more
shallow prefill than 400k does, which is the right direction for 0.983 vs 0.944.

⛔ **It cannot be checked, because this run's logs no longer exist** — `$D` was a fixed path and
four later invocations overwrote them (edit D in the staged applier). **The instrument that would
read it now exists and the data it needed is gone.** Recorded as a hole rather than back-filled by
inference: the next 256k ABBA gets this for free and this one does not.

### ⚠⚠ "ALL FOUR ARMS PROVED CONSUMPTION" WAS AN INFERENCE FROM SILENCE. IT IS NOW A MEASUREMENT.

The first version of this section said consumption was proven. It was **inferred from the engine's
no-consumer alarm staying quiet**, and the gate that wrote it **could not see the branch that was
actually firing**. Grok caught it in a log this gate had already scored clean:

```
fails the paged capability contract     0    <- the only branch the gate counted
took the STATIC path -- no paged context  110  <- never counted, and firing,
                                               on exactly layers 3,7,11,...,39
```

⇒ **Resolved by a `-lv 5` decider, split on the request marker rather than on startup:**

```
                        whole    AFTER the request starts
DS4P-CONSUME banded       640          640
DS4P-CONSUME auto           0            0
took the STATIC path      210            0
layers consuming after request start:    [3,7,11,15,19,23,27,31,35,39]
layers on static path after request start: []
```

**All 210 static-path warnings are RESERVE-TIME graph builds** — llama-server builds ~21 reserve
graphs before the paged context exists. **Every full-attention layer consumes at request time.**
The resolution is **WHEN, not WHETHER**, and counting reserve-time fallbacks as failures would be as
wrong as ignoring them.

⇒ **The decode measured here WAS paged attention. 1.307× stands as a paged-KV result.**

⚠ The decider itself nearly lied: the first attempt split the log with `tail -n +$(wc -l ...)`, `wc`
returned leading whitespace, `tail` errored *illegal offset*, the split file was **empty**, and it
printed **four zeros that looked exactly like a measurement**. **Verify the splitter split
something.**

### ⚠ THE CAVEAT, NOT BURIED: the cold-arm check fired WITH `warm=1`

pos1 ran **4.4% slower on prefill and 5.9% slower on decode** than pos4. The prelude killed the
cold-first-arm effect at 8k — **which is where I validated it** — and does not fully kill it at 256k.
**I validated a fix in the regime where it works and shipped it for the regime where it does not:
the control-set failure again, on the fix for the previous control-set failure.**

⇒ EFFECT is therefore biased toward paged, and the honest **warm-vs-warm** numbers are:

```
wall     817.8 vs 849.2  = 1.0384   paged 3.8% slower
decode    20.36 vs 26.62 = 1.3072   paged 30.7% faster   <- the number to quote
```

**Decode survives the correction.** Four independent 256k decode measurements across two sample
sizes: **1.2767 · 1.3692 · 1.4090 · 1.3471.**

### THE BAR — "equal or better", 256k, champion kernel

| metric | verdict |
|---|---|
| **decode** | **BETTER by 31-35%** — resolved, clears drift ~6×, survives the cold-arm correction |
| **prefill** | **INDISTINGUISHABLE** — a 1.7% effect inside a 14.8% band. **Not "worse". Unreadable.** |
| **wall** | tie within noise, by construction (97% prefill, and prefill is the 14.8% metric) |

⇒ **Nothing is measurably worse and decode is decisively better. The bar is MET at 256k.**
Caveats attached rather than hidden: **n=2 per arm** (a drift bound, not a variance estimate), the
**cold-arm signature is still present**, and **prefill is unmeasurable at this sample size**.

### ⚠⚠ AND THE CAVEAT THAT WAS MISSING FROM THIS SECTION UNTIL THE COVERAGE SWEEP: `-np 1`

**Every number above is a SINGLE-SLOT result.** `paged_parity_gate` runs `-np 1`, as do
`arch_serve_gate` and `long_context_gate`. **"Paging is as fast as static" silently carries "with one
slot."**

⚠⚠ **CORRECTED 2026-08-10 07:0x — THE SENTENCE THAT USED TO SIT HERE WAS FALSE WHEN I WROTE IT.**
It read: *"this lane has an **open `-np > 1` defect** — absolute `batch_offsets`/`write_slots`
against local ubatch indices — which no `-np 1` gate can see, and **which fires under ordinary
continuous batching.**"* **That defect was fixed four days earlier and closed by measurement 27
hours earlier.** Dates, because the gap is the whole point:

```
fix    c2f28a79d  "derive paged batch offsets/lens/slots from the UBATCH"   2026-08-06 11:23
close  00cb274    "finding 1c CLOSED BY MEASUREMENT"                        2026-08-09 02:23
close  e404116    "concurrency CLOSED -- four-way warm clean, both models"   2026-08-09 03:19
I WROTE THE CLAIM ABOVE                                                     2026-08-10 05:28
```

Content-level proof it is in the shipped tip, not merely ancestry:
`git show 074672e33:src/llama-graph.cpp | grep -c "DERIVE EVERYTHING FROM THE UBATCH"` → **1**.

⇒ **Mechanism, and it is the INVERSE of the shape `lint_stale_status.sh` catches.** I copied the
defect description out of `FINDINGS-2026-08-06.md` §1's **body** and never re-read the **status
table at the top of the same file**, whose line 11 says `1c … CLOSED BY MEASUREMENT 2026-08-09`.
The lint looks for an open word in a file's HEAD and a resolution word in its TAIL. This was the
resolution in the head, the open claim in a **different file's** body — it scores 0 of 14 and is
structurally right to. Recorded as **shape 5** in that lint's own can't-see list.

⇒ **THE CAVEAT SURVIVES; ONLY ITS REASON CHANGES — AND THE REPLACEMENT IS SHARPER.**
Every number above is still **single-slot**, because `paged_parity_gate` runs `-np 1`. But the
reason is no longer "a live defect fires under ordinary load" (which would make every parity result
describe a configuration nobody runs). The real residual is an **empty composition cell**:

```
concurrency closure ran   -c 8192    block 16   short prompts   -np 2/4   3-4 reps
paged_parity_gate  runs   -c 524288  block 64   fill 400k       -np 1
```

**Neither crosses the other's constant. Nothing has measured multi-slot AT long context.** That is
a named gap, not a known bug, and it is memory-bound: the realistic cell is `-c 262144 -np 2`
(~131k per slot). Boarded below, not queue-jumped.

⇒ **The verdict stands as stated** — the bar was speed at 256k-1M and that is what was measured. But
**the caveat belongs beside the number, not three sections away**, because the number is what gets
quoted. It was missing from this section for two hours after I wrote it down elsewhere.

### ★★★ 2026-08-10 08:30 — THE CELL WAS ENTERED, AND `-np 1` IS NOT A GAP. IT IS A KERNEL CONTRACT.

First clean run of the multi-slot × long-context cell (35B, block 64, `-np 2`, two concurrent
40,016-token prompts, champion on). Static arm **CLEAN**, 625 blocks spanned. **The paged arm
crashed the server:**

```
ggml_metal_op_paged_attn: CHAMP-PAGED REFUSED (n_seq!=1) D=256 bs=64 n_seq=2 n_tokens=2
ggml_abort <- ggml_metal_op_paged_attn <- graph_compute <- llama_decode
```

⚠ **The gate's canned verdict said "1c REPRODUCES on this binary" and that was WRONG.** The evidence
was `<UNPARSEABLE http=000>` — **no HTTP response at all**, i.e. a dead server, not a corrupted
answer. The verdict string was written for the corruption case and fires on any paged-bad /
static-clean split. **A verdict that names a mechanism it did not observe is worse than "FAIL".**

⇒ **THE ABORT IS CORRECT.** `llm_graph_context` **relaxes** the staged-tile bound
(`block_size × head_dim ≤ 8192`) when the champion will serve, so a layer admitted under that
relaxation **cannot legally run on the scalar kernel** — falling back would dispatch out of budget
and return **plausible numbers**, the silent-corruption incident that bound was calibrated against.

⇒ **THE DEFECT IS THE ADMISSION TEST, WHICH WAS NARROWER THAN THE KERNEL'S PRECONDITION:**

```
champ_geometry = champ_on && block_size == 64 && head_dim in {64,96,128,192,256}
                             ^^^^ n_seq appears nowhere
```

**The capability contract said YES to a configuration the kernel refuses.** Fixed in `a4e8aeb08` by
adding `cparams.n_seq_max == 1` — **`n_seq_max`, not the ubatch's live `n_seq`**, because a server
started `-np 2` may co-batch at any moment and admitting on a momentarily-single-sequence ubatch
would re-arm the abort **later in a long run instead of at the first decode**.

★★ **WHAT THIS DOES TO EVERY NUMBER ABOVE — it makes the caveat STRONGER, not weaker:**

> **`-np 1` is not an unmeasured cell. The champion kernel cannot serve multi-slot at all.**
> Before the fix a `-np > 1` server **aborted**; after it, the champion is refused and those layers
> take the static path. **1.9662 at 512k is single-slot BY CONSTRUCTION.**

⚠ **Consistent with the 2026-08-09 concurrency closure, not contradicting it:** that ran **block 16
with the champion OFF** — the *scalar* kernel, which does handle multi-seq and was clean four-way on
two models. **Scalar: multi-slot fine. Champion: single-slot only. Two kernels, two contracts, and
nothing had crossed them until this cell.**

⇒ **The cell therefore SPLITS, and only one half is still open:**
```
champion + multi-slot   IMPOSSIBLE by contract   -- closed, by fix a4e8aeb08
scalar   + multi-slot + LONG CONTEXT   STILL OPEN -- block 16, champ off, -np 2, 40k/seq
```

---

| rung | verdict | evidence |
|---|---|---|
| 256k, champion, **pred_n=14** | prefill UNREADABLE · **decode 1.3692x** · wall UNRESOLVED | 4-arm ABBA. Decode cleared its drift 7.5×, but the window was **0.67 s** — see the inversion below. **Superseded by the verdict above.** |
| 256k / 512k, **bs=64 champ off** | ⛔ **VOID** | the "paged" arm never paged: 3,610 refusals, `64 × 256 > 8192` |
| 8k, 9B, **pred_n=128** | **prefill 1.0075 (clears)** · **decode 0.9017 (clears)** | clean box, cold-arm check +0.1%. The first decode number in this lane with a real window — **and it says paged is 9.8% SLOWER.** |
| 8k, 35B, pred_n=14 | wall 1.0095 · prefill 0.999x · decode 0.878x | decode leg inherits the short-window status; prefill leg survives |
| ~~512k, **n=1 per arm**~~ | ~~decode 1.991 · prefill 1.059 · wall 0.936 — likely an UPPER BOUND (cold static pos1)~~ | **SUPERSEDED by the complete 4-arm ABBA below.** Kept because its prediction was testable and half of it was wrong: the corrected ratio moved **DOWN only 1.2%** (1.991 → 1.966), not the ~3% the 256k precedent implied, **because the cold penalty it was correcting for turned out to be +1.06%.** |
| **512k, 4-arm ABBA, n=2 per arm** | ★★★ **decode 1.9662 · prefill 1.0623 · wall 1.0714 — ALL THREE CLEAR** | **The bar is MET at 512k on every metric.** See the verdict block below. |
| 1M | **not measured — projected below** | Fits in memory (f16 116.1 GiB / q8_0 76.1 GiB of 128). |

### ★ 1M PROJECTION, and it is the first number that makes the case on its own

✅ **RE-DERIVED 2026-08-10 07:55 FROM THE COMPLETE 4-ARM ABBA.** The block below previously used
the **n=1** 512k singles (static tg 8.48, paged tg 16.88 — pos1 and pos2), with a caveat that they
were unbanked and that a cold pos1 would move the ratio down. **Arms 3-4 landed; the inputs are now
n=2 position-balanced means, and the cold penalty they were hedging against measured +1.06%.**
⚠ **This is the number the owner quotes when deciding the 1M spend, so it was re-derived rather than
left with a "superseded" tag** — a stale input inside a live projection is the exact shape named
twice already in this file.

```
                 256k(n=2)  512k(n=2)   decay/doubling    -> 1M projection
static prefill   241.20      124.58        1.936             64.35 tok/s  -> 3.45 h/arm
paged  prefill   237.10      132.345       1.792             73.87 tok/s  -> 3.01 h/arm
static decode     19.76        8.525       2.318            **3.68 tok/s**
paged  decode     26.62       16.762       1.588            **10.55 tok/s**
```

⇒ **A 4-arm ABBA at 1M is ~12.9 h** (prefill-dominated, 800,063-token fill at the 0.763 occupancy
ratio the sweep holds constant), not the 20-24 h in the old note and slightly under the 13.5 h the
n=1 inputs implied.

⇒ **1M decode ratio projects to 2.87×** — down a hair from the 2.9× the singles gave, because the
n=2 paged mean (16.762) is lower than pos2 alone (16.882).

⇒ **And the decode line is the argument:** at 1M, static decode projects to **3.68 tok/s — unusable
for anything interactive — while paged projects to 10.55.** A projected **2.87×**. **That is the
first number in this programme that makes the case for paging on its own, rather than as parity.**

⚠ **PROJECTION, NOT MEASUREMENT, and my last two were BOTH LOW.** I predicted 1.65-1.70 for the 512k
decode ratio and it came in at **1.991**; the foreman's independent ~1.77 was also low. **Every
extrapolation on this curve so far has under-called the widening**, so read these as a floor rather
than a centre — and the 1M rung remains **the owner's spend to authorise**, at 13.5 h.

> ### ⚠⚠ THE DECODE SIGN IS NOT STABLE UNDER SAMPLE SIZE. Everything decode-related is provisional.
>
> `n_predict` is a **ceiling**, not a floor. The needle prompt is answered in **14 tokens** and the
> model emits EOS (`pred_n=14, pred_ms=670, stop_type=eos`, against a requested 512). **Every decode
> number this lane produced before 19:25 was averaged over 0.67 seconds.**
>
> With `ignore_eos` and a real 128-token window, the 9B smoke **inverted**: decode went from ~0.98×
> to **0.9017×**, clearing its own drift. A short window can hold a **stable wrong sign** — the 35B's
> 1.3692× reproduced across two arms to 0.6% and is still not safe.
>
> ⚠ **AND THAT 0.6% IS SPECIFIC TO THE 14-TOKEN RUN — DO NOT GENERALISE IT.** Its paged pair was
> 29.45 / 29.28. The **good-window** 256k run's paged pair is **27.00 / 26.245 = 2.8%**. I quoted
> "paged drift has been 0.6%" for hours from this line, in chat and in expectation-setting, without
> re-deriving it at the rung I was actually comparing against. **A figure carried forward without its
> derivation** — the same class as a rate without its sample size, and it would have made an ordinary
> replicate look like either a triumph or an anomaly on a band four times too narrow.
> (caught by Grok, from the `.res` artifacts)
>
> **Two readings, registered before the re-run:**
> **(a) context-dependent** — block-table indirection is a fixed per-step cost, while static
> attention cost grows with context; at 8k the overhead dominates, at 256k the benefit does. *Both
> numbers true.* **(b)** the 1.37× is an artefact and collapses. **(a) predicts the re-run reproduces
> it.**

> ### ⛔⛔ EVERY 35B PARITY NUMBER FROM 2026-08-09 IS VOID. THE PAGED ARM WAS NOT PAGING.
>
> From the paged arm's own log, all at WARN level:
>
> ```
> DS4P-CHECKOUT           1     pool allocated
> DS4P-SET              110     context attached to the graph
> capability contract  3610     every attention layer REFUSED
>
> "paged layer refused: layer 3: block_size x head_dim exceeds the staged-tile
>  budget (need block_size*head_dim <= 8192)"
> ```
>
> Ornith-35B is `n_embd_head_v = 256`, `n_layer = 40`, and the gate ran `--kv-block-size 64`.
> **64 × 256 = 16,384.** Refused layers 3, 7, 11, …, 39 — every 4th of 40, exactly the full-attention
> set on this hybrid; the other 30 are recurrent. **100% of attention fell back to static.**
>
> ⇒ The 256k tie (1.0003) and the 512k tie (0.9905) were **static vs static-with-an-idle-pool**, which
> explains them perfectly and retroactively: 0.3% paged drift, decode a dead heat, effect always
> inside noise. They were the same code path.
>
> ⚠ The gate's validity check asserted `n_gpu_blocks > 0`. **That proves the pool was BUILT and
> nothing else.** CHECKOUT proves allocation; CONSUME proves consumption — a distinction already
> written down in this project — and the instrument was built on the wrong one.
>
> ⚠ **The 9B numbers are unaffected**: that harness ran `--kv-block-size 16`, and 16 × 256 = 4,096,
> inside the budget by accident of history. **The one parameter never re-derived when the model
> changed is the one that broke**, and because a refused layer falls back to static, the output stayed
> correct the whole time.
>
> **Fixed**: block size is now derived from `n_embd_head_v` by a 10 s geometry probe, and the paged
> arm VOIDs on the engine's own WARN-level no-consumer alarm.
>
> ### ⚠⚠ AND THE FIRST FIX TRADED A SILENT NO-OP FOR A SILENT DOWNGRADE.
>
> The 8192 bound is the **scalar** kernel's. `paged_layer_supported` already relaxes it for the
> champion (`llama-graph.cpp:4548`):
>
> ```cpp
> champ_geometry = champ_on && block_size == 64 &&
>                  (head_dim == 64|96|128|192|256);
> if (!champ_geometry && block_size*head_dim > 8192) reject(...)
> ```
>
> The champion does not stage K/V tiles — flat in nsg — and **contractually requires
> `block_size == 64`**. So at head_dim 256 there are **three** states, not two:
>
> | config | result |
> |---|---|
> | `champ=1, bs=64` | **champion serves it — the configuration paging exists for** |
> | `champ=0, bs=32` | scalar serves it, slowly (wall 1.3035) |
> | `champ=0, bs=64` | **every layer refused, silently static** — 4.5 h of "parity ties" |
>
> My geometry probe clamped 64 → 32 to satisfy a bound that does not apply, **silently selecting the
> slow kernel**. Now champion-aware, and `DS4P_METAL_CHAMP` defaults ON with `champ=` stamped in the
> header.
>
> ⚠⚠ **The code comment predicting the refusal case is dated 2026-08-06:** *"at bs=64/D=256 every
> layer refused and silently took the static path, making a paged run indistinguishable from
> static."* It was found, root-caused and written down three days earlier, **in the very function
> this gate calls**, and the gate walked into it anyway. **The knowledge existed; the harness did not
> carry it.** A finding that lives only in a comment protects the next reader of that function and
> nobody else — which is the argument for encoding it in the instrument, as the geometry probe and
> the no-consumer assertion now do.

**"as correct as": MEASURED and PASSING.** 8/8 at 8k, 5/5 pre-existing grids, and 430 chunks at 225k
with needle PASS — `FINDINGS-paged-cross-request.md`, final section.

> ### ⚠⚠ RETRACTION. The first version of this file said the opposite, in bold, and it was wrong.
>
> It read: *"AS CORRECT AS IS NOT MEASURED AT ANY OF THESE RUNGS, AND IT IS THE BIGGER RISK ...
> FINDINGS-paged-cross-request.md is OPEN and records 2 of 6 runs FAILING at 224,992 tokens."*
>
> **The finding is CLOSED.** Root-caused and fixed in `6391c5e63` (2026-08-08 17:11), which is an
> ancestor of the tree these measurements ran on: oversized final prefill chunk → compute-buffer
> layout → leftover activation floats under `s_copy` → read as a row index into a 1-row state tensor
> → garbage recurrent state from request 2 onward. Induced on demand across five pre-registered
> arms, control included.
>
> The 2-of-6 rate is retracted **inside that same document** — *"an artefact of the harnesses"*, both
> failures coming from harnesses whose warmup made the measured request #2.
>
> ⇒ **I quoted its status header instead of reading it**, and the header had been stale since the fix
> landed. Second time in one hour: `PLAN-paged-arch-support.md` has the same shape, I found it, wrote
> a banner on it, and then repeated the mistake in the next file I opened. **When a document's status
> header and its final section disagree, the final section is the state.**
>
> ⇒ `output_sanity.py` still earns its place — a check that can fail on wrong output is worth having
> whether or not today's instance is closed — but it is **insurance, not a live alarm**, and the
> first version of this file sold it as a live alarm.

---

## ★ TASK: MAKE DEEPSEEK-V4-FLASH WORK (added 2026-08-09 on the owner's order)

Model on disk: `DeepSeek-V4-Flash-0731/UD-Q2_K_XL`, **3 shards, ~96.8 GB**
(5.3 MB + 49.4 GB + 47.4 GB).

### ✅ STEP 0 RESULT (2026-08-10): STATIC WORKS. PAGED ABORTS, ONE LAYER EARLIER THAN PREDICTED.

**STATIC — the owner's goal is MET, with no code written:**

```
loads in 42 s (96.8 GB, 3 shards) -> "model loaded", listening
/v1/chat/completions:
  content   'Paris'
  reasoning '1. The user asks for the capital of France in one word. 2. The capital of
             France is Paris. 3. "Paris" is one word.'
  finish    stop (clean EOS, not a limit)   ·   38 completion tokens
raw /completion: 31.6 tok/s prefill, 22.65 tok/s decode
```

The two upstream Metal fixes in the 42-commit merge were all it needed. **Running it before writing
anything was the right call and it cost ten minutes.**

**PAGED — `--kv-paged` hard-aborts at startup:**

```
E llama_paged_scheduler_init: context does not have a paged KV cache: found a
  non-paged memory type.
server-context.cpp:1575: GGML_ASSERT(paged_sched && "failed to init the paged scheduler") failed
```

⚠⚠ **It never reaches a graph.** The tier analysis below priced the blocker at the *kernel* level —
the missing mask for top-k. **That is still true and it is not the FIRST blocker.**
`llama_kv_cache_dsv4` is a composite memory with four sub-caches and **is not a paged memory type at
all**, so the scheduler refuses before a single layer is built. The markers reflect that:
`capability contract refusals = 0`, `took the STATIC path = 0` — **not because everything paged, but
because nothing was ever asked.**

⇒ **NEW TIER 0, ahead of everything below:** `llama_kv_cache_dsv4` must own a paged pool and
`llama_paged_scheduler_init` must accept it.

**TIER 0 IS FULLY SCOPED, and it is the FOURTH instance of an established pattern — not greenfield.**
`llama_paged_scheduler_init` accepts memory by a chain of `dynamic_cast`, and its own comment on the
most recent addition reads *"Third wrapper, same resolution"*:

```cpp
llama_kv_cache_paged        -> direct
llama_memory_hybrid_iswa    -> get_mem_attn_paged()
llama_memory_hybrid         -> get_mem_attn_paged()
llama_kv_cache_iswa         -> get_mem_attn_paged()     <- "third wrapper, same resolution"
llama_kv_cache_dsv4         -> ** the fourth branch, to be written **
```

Four mechanical pieces, each with a line to copy from:

| # | change | copy from | **insert at** |
|---|---|---|---|
| 1 | `unique_ptr<llama_kv_cache_paged> mem_attn_paged` + `set_attn_paged()` / `get_mem_attn_paged()` on `llama_kv_cache_dsv4` | `llama-kv-cache-iswa.h:99-109` | **decl after `dsv4.h:146`** (`get_lid() const;`), **member after `:169`** (`kv_lid`), `private:` is at `:155` |
| 2 | construct the pool for DSV4 and attach it | `llama-model.cpp:2318-2352` (`pg_head_dim/pg_n_head/pg_block_size` → `new llama_kv_cache_paged(...)` → `init(...)` → `hybrid_iswa->set_attn_paged(paged_attn)`) | **after `llama-model.cpp:2183`**, the `res = new llama_kv_cache_dsv4(...)` construction, closing at `:2197` |
| 3 | `get_attn_paged()` / `set_attn_paged_ctx()` on `llama_kv_cache_dsv4_context` | `llama-kv-cache-iswa.h:155-156`, ctx set at `llama-kv-cache-iswa.cpp:238` | **before the `private:` at `dsv4.h:155`**, beside `get_n_rs_seq()` / `get_rs_idx()` |
| 4 | fourth `dynamic_cast` branch in the scheduler | `llama-paged-scheduler.cpp:50-59` (the `llama_kv_cache_iswa` branch, comment *"Third wrapper, same resolution"*) | **immediately after that branch** |

⚠ **Insertion points located by reading, not by planning — but NOT written.** Writing four pieces of
C++ across four files during a two-hour timed measurement, unable to compile (a `-fsyntax-only` is
still a compile, and the reads-only rule exists because an arm came back 5.2% slow while I built),
is how unverified code accumulates. **The implementation is mechanical once the GPU frees; the
locating was the part worth doing in dead time.**

### ⚠⚠ AND TIER 0 MUST NOT SHIP ALONE EITHER — IT WOULD BUILD THE SILENT-FALLBACK STATE ON PURPOSE

Tier 0 makes the **scheduler** accept DSV4. It does **nothing** about the graph. So on its own it
converts today's **loud abort** into a server that **starts cleanly, allocates a pool, and pages
zero layers** — the exact state that produced 4.5 hours of static-vs-static "parity" on 2026-08-09,
and the exact state LAW 6 exists to catch.

⇒ **Today's failure mode is BETTER than what Tier 0 alone would produce.** An abort is honest. A
green server with an unread pool is not.

⇒ **DSV4 paging is all-or-nothing: Tier 0 + 1 + 2 land together, or nothing lands.** The
`gate_assert_paged_consumed` law is the check that would catch a partial ship, and the acceptance
bar stays what it was: **capability-contract refusals == 0 on every attention layer**, paged ≡ static.

### ⚠ SEPARATE CHEAP DEFECT: a user-passed flag CRASHES the server

`--kv-paged` on any arch whose memory is not paged-capable hits `GGML_ASSERT` and aborts. **A
designed refusal — "this model's memory type does not support paging; remove `--kv-paged`" — is
strictly better than an assert**, and this lane already refuses by design in three other places.
Small, and squarely inside "make the model work in expected ways".

### Step 0 — RUN IT BEFORE WRITING ANY CODE

The two upstream Metal fixes DSV4 needs are **already in the 42-commit merge**:

| commit | what it fixes |
|---|---|
| `e40bf8864` | `threadgroup half4x4[]` is a **COMPILE ERROR** in MSL (matrix types have no zero-arg constructor) — inside `kernel_lightning_indexer`, the DSV4 indexer kernel |
| `a194a75b7` | NORM/RMS_NORM drop partial-simdgroup sums → **wrong mean and variance for the whole row** |

⇒ **So "static DSV4 works" may already be true.** The first action is a serve + one-shot
completion, not a code change. **Writing code before running the binary is how a day gets spent on a
problem that was fixed upstream two days ago.**

### Step 1 — STATIC path, if step 0 fails

Diagnose against the actual error. Known-adjacent risk, rescued from a deleted clone
(`historical/antirez-dsv4-metal/`): **the graph NODE budget.** `GGML_ASSERT(obj_new)` fired during
context reserve at `-ub 1024` on the other DSV4 implementation because its floor only covered ≤512.
Ours is `max(n_tokens*40, 32u*n_tensors())` — **different shape, no floor** — and
`LLM_ARCH_DEEPSEEK4` is in that arch list. **Unmeasured on our code, not excluded.**

### Step 2 — PAGED path, and it splits three ways

`deepseek4.cpp` never calls `build_attn`; it calls **`build_attn_mha` directly, three times**:

| site | shape | pageable? |
|---|---|---|
| `:877` PLAIN | `k = mctx->get_k(ctx0, il)`, implicit causal mask | **YES — this is the shape our funnel already serves** |
| `:786` CSA | `k_all = concat(raw_k, csa_k)`, `kq_mask = concat(raw_mask, **top_k_mask**)` | no — see the blocker |
| `:841` HCA | `k_all = concat(raw_k, hca_k)`, `kq_mask = concat(raw_mask, hca_mask)` | no — same |

⇒ **THE BLOCKER, verified at the signature and not asserted:** `ggml_paged_attn_banded` takes
**no mask tensor**. Visibility is analytic — `causal` + `visibility_window` + `context_lens` — and
`rel_logits` is a distance-indexed bias bounded by `rel_extent`. **A top-k selection is arbitrary
per (q,k) pair; an analytic band cannot express it.** Missing kernel capability, not a wiring gap.

⚠ The earlier framing "dual cache, different pool geometry" was **directionally right and
mechanically wrong**: the two-cache concat is a graph-level `ggml_concat` that paging could feed
from two pools. **The MASK is the homeless piece.**

| tier | scope | cost |
|---|---|---|
| **1** | plain layers only | **small** — the gemma3 pattern. ⚠ **PRECONDITION: read DSV4's actual `swa_type`.** `get_raw()` returns `llama_kv_cache_iswa*`, and today's guard only covers `LLAMA_SWA_TYPE_STANDARD`. **Inferring the type from the cache class is exactly the move that silently disabled gemma4's paging this morning.** |
| **2** | CSA + HCA | **the real work** — optional explicit mask input on `ggml_paged_attn_banded`, the Metal kernel, a CPU reference, `test-paged-vs-cpu` coverage |
| **3** | LID indexer | **skip** — `get_lid()->get_k()` feeds `ggml_mul_mat` + relu + `top_k`. A **scorer**, not attention. |

### ⚠⚠ THE GOAL IS SUPPORT PARITY, NOT THROUGHPUT — AND MY FIRST COST-BENEFIT ANSWERED THE WRONG QUESTION

The owner, verbatim: *"the point is to have the model runnable on the machine in expected states,
and have kv paged supported too as you are doing this work already. **context is not the issue.**
This is just to make the model work in expected ways on Mac."*

The paragraph that used to sit here argued **against** paging on memory math — 96.8 GB of 128 leaves
~31 GB, so the KV pool is small either way. **That reasoning is correct and it answers a question
nobody asked.** It priced paging as a long-context *performance* lever because that is the metric I
had spent the day measuring. **The frame was mine, not the task's** — the same mistake as filing the
GGUF-conversion fix "irrelevant" because one download happened to be prebuilt.

⇒ **Support parity is a COVERAGE claim**, and the acceptance test changes with it: not a wall-clock
ratio, but **`fails the paged capability contract` == 0 on every attention layer**, with paged
output matching static.

### ★★ AND THE LAYER SPLIT SETTLES THE TIER ARGUMENT — MEASURED FROM THE GGUF, NOT ESTIMATED

Read straight out of `deepseek4.attention.compress_ratios` in the metadata shard (5.3 MB, no need to
touch the 96 GB of weights):

| ratio | layers | path | pageable today |
|---|---|---|---|
| 0 | **5** | PLAIN, single cache, implicit causal mask | **yes** |
| 4 | **21** | CSA — concat of two caches + **top-k mask** | no |
| 128 | **20** | HCA — concat of two caches + mask | no |

**Tier 1 covers 5 of 46 layers. Eleven per cent.**

⇒ **That is exactly the state this lane spent 2026-08-09 proving is indistinguishable from static** —
a pool allocated, a handful of layers reading it, the rest silently falling back, and a green that
means nothing. **Shipping Tier 1 alone would be a misleading green, not an increment.**

⇒ **So for the owner's actual goal, Tier 2 is not optional. It is 89% of the model.**

Other facts from the same read, both load-bearing:

- `attention.sliding_window = 128`, and `deepseek4.cpp:67` sets **`swa_type = LLAMA_SWA_TYPE_STANDARD`**.
  ⇒ **The Tier 1 precondition is RESOLVED, by reading the arch rather than inferring from the cache
  class**: STANDARD is exactly the rolling window the analytic band implements, so the
  `visibility_window` path covers those layers.
- `head_count = 64`, `head_count_kv = 1` — MLA-style. GQA ratio 64, integer, so the paged capability
  contract's GQA check passes.
- `indexer.top_k = 512` — the sparse selection that has nowhere to go in the banded kernel.

⚠ Also live upstream and unfixed: **issue #26694, "DeepSeek-V4-Flash degenerates into repetition and
leaks special tokens in long agentic chats (Metal)"** — Mac Studio, `-ngl 99 -fa on`, 262k ctx.
Follow-ups narrow the trigger to **prompt content, not the client**. Budget for it before promising
agentic use.

---

## ★ COVERAGE AUDIT — does each gate's DEFAULT input cross the constant its claim scales with?

Grok's class split: a **COUNTING** defect is one the instrument saw and miscounted (sweep =
enumerate the failure vocabularies); a **COVERAGE** defect is one it never saw (sweep = **name the
structural constant each claim scales with, and check the default input crosses it**). This is the
second sweep, done by reading rather than by a parser — a generic parse across five heterogeneously
named scripts misread four of five on the first attempt, and **a fragile lint is worse than an
honest table.**

| gate | default input | block_size | crosses a block? | crosses `n_swa`? | `-np` |
|---|---|---|---|---|---|
| `arch_serve_gate` | ctx 4096, prompt ~6 + npred **8** | 16 (engine default) | **NO — one block** | **NO** | 1 |
| `long_context_gate` | ctx 32768, **fill 14000** | 32 | yes (~440 blocks) | yes | 1 |
| `multislot_gate` | ctx 8192, short prompts | 16 | short — **unverified** | **NO** | **2** ✓ |
| `warm_multislot_gate` **(defaults)** | ctx 8192, short prompts | 16 | short — **unverified** | **NO** | 2 ✓ |
| `warm_multislot_gate` **(long-context mode, added 2026-08-10)** | `MS_CTX=131072 MS_FILL=32000` | **64** | **YES — 625 blocks, MEASURED not estimated** (`ptok=40006` against a 32000 target: the ~12-tok/line estimate was **25% low**) | **YES** | 2 ✓ |
| `paged_parity_gate` | ctx 262144, **fill 225000**, npred **512** | 64 | yes (~3500 blocks) | yes | 1 |

⇒ **The two long-context gates cross everything. The three short gates cross almost nothing** — and
those three are the ones that carry the **arch matrix** and the **multi-slot** claims.

⚠ **`-np` is the constant only the multislot gates cross.** `arch_serve_gate`, `long_context_gate`
and `paged_parity_gate` are all `-np 1`, so **every parity number and every arch green is
single-slot only.**

⚠⚠ **CORRECTED 2026-08-10 — this row used to justify that with "this lane's open `-np > 1` defect".
It is not open.** Fixed `c2f28a79d` (2026-08-06), closed by measurement `00cb274` + `e404116`
(2026-08-09), 27 hours before I wrote the claim. Full dates and mechanism at the corrected block
above. **The correct statement of the gap is the composition, not a defect:** the closure ran
`-c 8192` / block 16 / short prompts, this gate runs `-np 1` / block 64 / 400k fill, and
**nothing has measured multi-slot at long context.**

⇒ **Not a call to change the defaults.** A 14-token arch gate over 21 architectures is cheap on
purpose, and making it long would trade coverage for a matrix nobody re-runs. **The fix is that each
green names its size** — now printed by `arch_serve_gate`, and recorded here for the rest.

### ⚠ THIS TABLE IS DATED, AND THAT IS THE WHOLE MAINTENANCE PLAN

**Re-read 2026-08-10 08:10 after `warm_multislot_gate` gained `MS_CTX`/`MS_BLK`/`MS_FILL`/`MS_NPRED`/`MS_NGPUB` — the convention below applied to my own change on the same day I wrote it.** Originally read against five gates. A hand table has **no positive control and no reach**: it
goes stale silently the day someone adds a sixth gate or changes a default, and nothing in the tree
will notice — the exact property that makes a stale header dangerous. (raised by Grok)

⇒ **The durable form is a CONVENTION, not a parser**, because the parser I tried misread four of
five and **an instrument that misreads is worse than none — a reader trusts it**:

> **Re-ask the coverage question whenever a gate is added or a default input changes.**
> For the new or changed gate: name the structural constants its claims scale with — `n_swa`,
> `block_size`, `ubatch`, `-np` — and state, in its own output, whether the default input crosses
> each. **Then update the date on this table.**

⚠ This directory holds **four** records of a prose convention failing to bind the next script, which
is why the rule sits **next to the table it governs** rather than in a style note — and why
`arch_serve_gate` prints its own two limits at runtime instead of relying on anyone reading this.

---

## ⚠⚠⚠ WHAT THE ARCH MATRIX'S 21 GREENS ACTUALLY CLAIM (2026-08-10)

`arch_serve_gate.sh` defaults: prompt `"The capital of France is"` (~6 tokens) + `AG_NPRED 8`
= **~14 tokens**, against the engine's default **`block_size 16`**.

⇒ **The entire paged run allocates ONE BLOCK and never crosses a boundary**, and the prompt sits
**inside `n_swa`** on every windowed architecture.

| the green covers | the green does NOT cover |
|---|---|
| the arch serves under `--kv-paged` | **block-table traversal** — one block, no boundary |
| a graph consumed the pool (`DS4P-CONSUME > 0`) | **cross-block indexing, write-slot mapping, reuse, eviction** |
| paged output matches static | **sliding-window correctness** — a wrong or absent `visibility_window` is invisible inside the window |

⇒ **The matrix's real sentence is "21 architectures produce correct output for a single-block prompt
that fits inside any window."** That is a genuine result and a much smaller one than it has been
read as — **including by me, all week.**

⚠ **Not speculative for this fork.** The corruption hunt in `FINDINGS-paged-cross-request.md`
root-caused to **the size of the FINAL PARTIAL PREFILL CHUNK** — a multi-block, boundary-shaped
defect that a 14-token run **cannot express at all**. And the blanket SWA reject shipped on
2026-08-09, which silently disabled gemma4's windowed paging, **would have printed the same green
before and after.**

⇒ Both limits now print themselves at the end of every `arch_serve_gate` run. **The fix is a
sentence, not a threshold**: re-run with `AG_PROMPT`/`AG_NPRED` past a few blocks and past `n_swa`
when those claims are the ones being made.

---

## GPU-BOUND (queues behind the running measurement)

| item | cost | why it needs the GPU |
|---|---|---|
| ★ **DECODE CONTEXT SWEEP — the decisive experiment** | **<1 h total** | The 9B at 8k says paged decode is **0.9017×** (slower); the 35B at 256k says **1.3692×** (faster). **Same kernel, same block size, verified** — `head_dim=256 · CHAMPION · bs=64` on both — so it is not a kernel confound. If the fixed-per-step-indirection-vs-growing-attention-cost story is right there is a **crossover length**, and it is findable cheaply: **one model, ABBA, `ignore_eos`, at 8k · 32k · 64k · 128k.** Prefill at those lengths is minutes. A monotone curve crossing zero supports the story; a flat 0.90 everywhere with a jump at 256k refutes it and points somewhere else. **This answers "why" and gets "is it real" for free at the low end; a second 256k point estimate answers neither.** |
| ✅ ~~clean 256k ABBA~~ | — | **DONE 2026-08-10 03:20.** decode 1.3471 clears · prefill UNREADABLE · wall UNRESOLVED |
| ✅ ~~clean 512k ABBA~~ | — | **DONE 2026-08-10 07:35.** decode **1.9662** · prefill **1.0623** · wall **1.0714** — **all three clear**, cold-arm +1.06%. Bar MET at 512k. |
| `gemma3` paged verification | ~10 min | wired but NOT proven: needs `DS4P-CONSUME > 0` and output matching static |
| `gemma4` SWA-restored check | ~10 min | proves the guard fix re-enabled what it broke; needs CONSUME on a layer where `is_swa(il)` is true |
| ~~paged-corruption rate at 225k~~ | — | **CLOSED**, see the retraction above; no runs needed |
| ★ **MULTI-SLOT AT LONG CONTEXT — the empty cell, named 2026-08-10** | **~1.5 h** | The `-np>1` defect is **fixed and closed** (`c2f28a79d`, `00cb274`, `e404116`) — but the closure ran `-c 8192` / block 16 / short prompts, and every parity number runs `-np 1` / block 64 / 400k fill. **Neither crosses the other's constant, so the joint cell has never been entered.** It is not a suspected defect; it is the configuration a real server runs in, and nothing has measured it. Memory-bound, so the cell must be chosen not assumed: `-c 262144 -np 2` gives **~131k per slot**, both slots crossing block *and* window, at a fill the box holds. The check is the one that already exists — `warm_multislot_gate.sh` semantics (prime → concurrent → third sequential) with `MS_CTX` raised, plus `gate_assert_paged_consumed`. **Cheapest remaining item that retires a caveat riding on every number in this file.** |
| 1M rung | ~20-24 h | owner's authorization |
| NIAH sweeps | long | **marked Pending by the owner**, deliberately deferred until paging matches static |
| inkling paged path | 975B | marker added, never gate-verified |
| CUDA/Metal parity on the box | — | needs the box's GPUs free |

⚠ **The gemma3 and gemma4 checks have a gate-shape requirement, registered before the run:** the
prompt must be **longer than `n_swa`**. Below the window a windowed layer and a full-causal layer see
identical context and produce identical logits, so a short-prompt gate is structurally incapable of
detecting a broken band. It would return a clean green that means nothing.

---

## CPU-ONLY (can proceed now)

| item | state |
|---|---|
| ⛔⛔ **STALE `.res` CAN BE READ AS THIS RUN'S ARM** (found by Grok, 2026-08-09) | If an arm dies, `arm()` returns early, `static.res` is never written, the ABBA rename never happens — and **the previous run's file survives and is loaded as this arm.** Right now `static2.res` on disk is an **8k fixture from my own smoke test**: `n=3665, wall 6.5s, ok=true, needle PASS`. It parses, it passes every check the summariser makes, and it would produce **"paged is 104% SLOWER"** with full confidence. Two one-line fixes: **(a)** assert all four arms share `prompt_n`, else VOID — the field is already recorded and compared against nothing; **(b)** clear `$D/*.res` at gate start. Same class as the stale headers and the stale arch table: **an old file indistinguishable from a new one**, and this instance I created myself by running smoke tests into the measurement directory. |
| ⚠ **the `.res` omits the field a verdict depended on** | The artifact carries wall/pp/tg/n and **no `predicted_n`**, which is why artifact-first verification could not catch the `n_predict` defect — the load-bearing field simply was not there. Add `predicted_n`, `predicted_ms` and the run's own `OUT` path. **An artifact must carry every field its verdict depends on, or it launders assumptions.** |
| ⚠ **`ignore_eos` and `output_sanity.py` COLLIDE — sequence them** | With `ignore_eos: true` the model runs past its answer for the full 512 tokens, and what follows a completed answer is very often repetition. **`output_sanity.py` would grade that DEGENERATE on a perfectly healthy run** — its repetition detector fires at 0.35 and post-answer filler blows through that. Fix them together, not separately: grade only the text **up to the first EOS position**, or send a second short correctness request with `ignore_eos` off. Two fixes that are each right and jointly wrong is how the last three defects in this file were built. |
| wire `output_sanity.py` into `paged_parity_gate.sh` | **next up.** Save the arm's text into its `.res`, grade paged against static in the ABBA summariser. Zero extra GPU cost, turns every future parity run into a corruption sampler. |
| ✅ ~~`ignore_eos`~~ · ~~achieved `pred_n` in the artifact~~ · ~~stale `.res` guard~~ | **all shipped and smoke-verified 2026-08-09** (`3f02818`): `pred_n` 14 → 128 on every arm, the achieved length printed beside the requested one, `decode window actually generated: [...]` in the summary, `*.res` cleared at start **and** a VOID unless all four arms share `prompt_n`. |
| ✅ ~~LAW 6 unproven~~ | **VALIDATED by direct control** (`alarm_control.sh`): arm B proved the marker prints (240 banded, **0 auto**), arm A fired the alarm on 20 refusals. ⚠ Arm B's `auto=0` also **refuted my own published explanation** for the alarm's historic silence — eliminating three alternatives was not evidence for the fourth. That silence is now an **open loose end**, deliberately left unexplained. |
| ⛔ ~~**`ignore_eos` — `PP_NPRED` has never taken effect**~~ (fixed, kept for the record) | **highest-value queued fix.** The request sets `n_predict: 512`, which is a **ceiling, not a floor**, and the model answers the needle in **14 tokens** then emits EOS: `pred_n=14, pred_ms=670, stop_type=eos`. **Every decode number in this lane is sampled over ~0.6 seconds**, which is *shorter* than the 48-token case the change replaced. Fix is one field, `"ignore_eos": true`. Also print the ACHIEVED `predicted_n` on the arm line — the header's `npred=512` records what was *requested*, and **a parameter that silently does not take effect is worse than one that is absent, because it looks like the question was asked.** |
| ⛔ **`"out"` in the `.res` records NOTHING** (found by Grok) | I added it in the same commit that fixed `npred=512` recording a request that never took effect — **and made the identical mistake one line away.** The writer reads `os.environ.get("OUT","")` inside a python heredoc, but the shell never exports `OUT`, so it is always `""`. **A field added specifically to make artifacts self-identifying, which identifies nothing.** Fix: pass it as `argv`, the way every other value in that heredoc is passed. ✅ **DONE — and this row said "Queued, script executing" for hours after it was fixed.** `paged_parity_gate.sh:231-236` carries the fix and its own comment explaining it, and the live run's `.res` files have the field populated. ⚠ **A stale status row inside the document I swept for stale status rows this morning, found by a reader and not by me.** The sweep looked for the `-np>1` vocabulary specifically; **a vocabulary sweep finds the class it was given and reports silence about every other instance** — which reads as coverage. |
| ⚠ **stamp the model's STORAGE LOCATION in the result header** | **queued, script running.** The header records only the GGUF's basename. On 2026-08-09 the model tree under `~/Documents/GitHub/ornith-models` was reorganised **mid-run** (mtime 19:54:40) and the re-run had to be pointed at `/Volumes/KING4TB/...` instead — **a USB volume.** Two result files with identical headers can therefore describe runs whose weights came off different storage. The mmap page-fault path differs, and although the warm-up prelude pulls 21 GB into page cache on a 128 GB box (so all four arms are equal), **"the arms are equal" and "this run is comparable to yesterday's" are different claims.** Record the directory — `/Volumes/...` carries no username, so the scrubber leaves it intact. ⚠ **Downgraded after measuring**: the warm-up arms took **5 s and 6 s** off that USB volume, and a cold 21 GB USB read cannot finish in five seconds — so the pages were already in page cache from the *previous* run reading the *old* path, which is only possible if **both paths are the same physical file**. Inference from timing, not a filesystem fact (the old path is gone, so no inode to compare). The stamp is still worth having: next time this should be readable rather than reconstructed from how fast a warm-up ran. |
| decide `-lv 4` vs `-lv 5` in `paged_parity_gate.sh` | **queued, cannot edit a running script.** The gate runs `-lv 4` and proves consumption via LAW 6 (absence of the engine's WARN alarm). The sibling `arch_serve_gate.sh` runs `-lv 5` **specifically so it can read `DS4P-CONSUME` directly** — its own comment says *"At -lv 4 the marker count read ZERO on an arch that was demonstrably paging."* A direct positive count beats an inferred one; the cost is DEBUG-volume I/O during a timed run, which is itself a confound. Record whichever is chosen **as a choice**, so nobody "fixes" it back. |
| warm-up response is never checked | **known hole.** `warmup()` sends its curl to `/dev/null`, so a 500 or an empty completion still prints "discarded (5s)". Log-line-is-not-work-done, in code I wrote today. ✅ **Edit C in the staged applier closes the half that matters**: it now greps the prelude's own log for the failure vocabulary the measured arms already VOID on, and refuses rather than letting the next arm inherit a cold-arm confound. The response body itself is still unread. |
| ⚠ **CLASS: a fixed `$D` destroys the previous run's logs, and ~25 gates have one** | **Swept 2026-08-10, and deliberately NOT fixed in bulk.** `grep` finds a fixed `${CLAUDE_JOB_DIR:-/tmp}/<name>` in essentially every gate here. Re-running any of them silently overwrites the last run's evidence. **The severity scales with run cost, and only there is the fix clearly worth its risk:** `paged_parity_gate` costs **3.5 h per ABBA** and its logs are the only within-run data this lane has, so it is fixed (edit D). A 90-second gate whose log is a diagnostic you can regenerate by re-running is a different situation, and **24 untested edits across 24 scripts to close a hole that costs 90 seconds is the trade this project's own algorithm says to refuse.** ⇒ **The rule, not the patch: if a gate's run costs more than a few minutes, its `$D` must be per-run.** Recorded here because a class found once and fixed in one place is how the other 24 get missed later — this is the note that says they were seen.
⚠⚠ **AND THE TRIAGE COST ME EVIDENCE WITHIN THE HOUR.** `warm_multislot_gate` was one of the 24 I declined to fix, on the argument that a short run is cheap to repeat. At 08:30 I needed the **champion** run's `paged-r1.log` to check whether the engine's no-consumer alarm had fired — and the **scalar** run had already overwritten it. **The run was repeatable; the specific comparison was not, because by then the binary AND the configuration had both changed.** ⇒ The rule stands, but its threshold was wrong. It is not *"how long does the run take"* — it is **"could this artifact ever be compared against one produced by a different binary or configuration?"** For any gate whose logs feed a cross-run comparison, `$D` must be per-run **regardless of runtime**. |
| the remaining ISWA archs | `gemma2` `gemma3n` `cohere2` `cohere2moe` `phi3` `olmo2` `exaone4` `exaone-moe` `openai-moe` `plamo3` `mellum` `mimo2` `smallthinker` `afmoe` — minus anything CHUNKED or SYMMETRIC, which the band cannot express at all |
| ~~`mimo2` sinks / `dflash` non-causal~~ | ✅ **re-verified in source, genuinely closed.** `mimo2.cpp:187` passes `sinks` into the paged call; `dflash.cpp:467` passes `/*causal=*/false`. Both were checked because `PLAN-paged-arch-support.md`'s top table was stale enough to cause a real bug, and "recorded closed" is not the same as closed. |
| ~~does the new `swa_type` guard break them?~~ | ✅ **no.** Both declare `LLAMA_SWA_TYPE_STANDARD`, which the analytic band implements exactly, so the tightened guard is a no-op for them. Checked rather than assumed — the last guard I added silently disabled gemma4. Window and causal are also orthogonal in the kernel (`lo` from the window, upper bound from `causal`), so `dflash` passing a window **and** `causal=false` is not a contradiction. |
| lint sweep | `lint_paged_consumers` · `lint_scrub_coverage` · `lint_common_laws` — the last reports 6 of 44 gates reach the shared laws |
| **champion-kernel arch coverage** | `arch_serve_gate.sh` passes no `--kv-block-size`, so it runs the engine default of **16** (`common/common.h:565`) on the scalar path. 16 × head_dim ≤ 8192 holds up to head_dim 512, so every arch pages and the matrix greens are valid. **But the CHAMPION is never exercised there** — 21 architectures verified on one kernel and none on the other, while the champion is the kernel the parity numbers come from. Not urgent, and not nothing. |

### Kernel selection, as a table, because getting it wrong is silent

| gate | block_size | champion | how it proves paging |
|---|---|---|---|
| `arch_serve_gate` | 16 (engine default) | off | counts `DS4P-CONSUME` directly — runs `-lv 5` **on purpose** |
| `long_context_gate` | 32 | off | LAW 6 (engine WARN alarm) |
| `multislot_gate` | 16 | off | LAW 6 |
| `warm_multislot_gate` | 16 | off | marker at `MS_LV=5`, else LAW 6 |
| `paged_parity_gate` | **64** | **on** | LAW 6 |

⚠ `paged_parity_gate` is the only one on the champion, and it is the only one whose output is a
**speed** claim. That is deliberate — the bar asks whether paging is as fast as static, and the
answer must come from the kernel paging actually ships with — but it does mean the champion's
correctness rests on far less arch coverage than the scalar path's.

---

## ⚠ THE CAVEAT THAT COST A MEASUREMENT TODAY

**"CPU-only" does not mean "free while a measurement runs."** During arm 1 of the clean 256k run I
executed three `-fsyntax-only` compiles, several python fixture runs and a handful of git operations.
Arm 1 came back at **232.1 tok/s prefill against the earlier cold run's 244.9** — 5.2% slower, on the
run whose entire purpose was to be cleaner than that one. Thermals are an equally plausible cause and
I cannot separate them from here, which is exactly the problem: **the confound is unattributable
after the fact.**

⇒ The rule, and it is stricter than it looks: **while a parity arm is in flight, reads only.** No
compiles, no test fixtures, no fan-out. Doc edits and file reads are fine. Load on *some* arms is
worse than load on all of them, because that is a positional confound and positional confounds are
the one thing ABBA cannot balance.
