# Quantization & Inference Reference

A plain-language glossary and practical playbook, written for the owner.
Every term here appeared in real work on this project. No maths background assumed.

---

## Part 1 — The numbers you keep seeing

### bpw (bits per weight)

A model is a big pile of numbers called **weights**. Originally each weight is stored in
16 bits (bf16). **bpw is how many bits each weight gets after compression.**

| format | bpw | a 27B model becomes |
|---|---|---|
| bf16 (original) | 16.0 | ~51 GB |
| Q8_0 | ~8.5 | ~27 GB |
| Q4_K | ~4.5 | ~15 GB |
| IQ2_KT | 2.145 | ~7 GB |
| IQ1_KT | 1.75 | ~6.6 GB |

Lower bpw = smaller file = fits a smaller GPU = but less faithful to the original.
**bpw is the single most important dial in this whole field.**

### PPL (perplexity)

**A score for how well a model predicts text. Lower is better.**

Intuition: perplexity ≈ "how many words is the model effectively torn between at each
step." PPL 10 means it's about as confused as if choosing among 10 options. PPL 100,000
means it has no idea — it's outputting noise.

It's measured by feeding the model a fixed piece of text (we use `wikitext`) and asking
how surprised it was. Same text + same settings = comparable numbers.

**Critical rule: only compare PPL numbers measured the same way** (same text, same number
of chunks, same model). Comparing PPL across different models or chunk counts is meaningless.
I broke this rule earlier in the project and had to throw out a comparison.

### "PPL ladder"

Just my shorthand for: *take one model, quantize it to many different bpw levels, measure
PPL at each, and look at the curve.* It shows where quality falls off a cliff.

### The specific numbers you asked about

**`98,935 → 95.8` (that's the "1,032×")**
IQ1_KT at 1.75 bpw. Without an imatrix: PPL 98,935 = complete garbage, noise.
With an imatrix: PPL 95.8 = still poor, but a functioning model.
95.8 is 1,032 times smaller than 98,935. **Same file size, same bit-width — the only
difference is calibration.** That is the whole point.

**`65.7 → 14.3`**
IQ3_KT at 3.15 bpw. Without imatrix PPL 65.7 (bad — this looked like a "quality cliff"
in the data). With imatrix PPL 14.3 (fine). **So the "cliff" wasn't real — it was an
artifact of bad calibration.** This is why I stopped trusting no-imatrix measurements.

**`0.149 → 0.25–0.32` (Fable's numbers, different metric)**
This is **acceptance rate** in speculative decoding — a small fast "draft" model guesses
the next tokens and the big model checks them. Acceptance = fraction of guesses accepted.
Higher = faster generation. Their draft model quantized to Q2_K scored 0.149 (bad);
requantized with a proper imatrix it scored 0.25–0.32. **Same story, different metric.**

**`1.6 bpw holds 0.293`**
Their draft model squeezed to 1.6 bpw *still* scored 0.293 acceptance — as good as the
8-bit version. **Extreme compression cost nothing, provided calibration was right.**

**`18.38 vs 16.33`**
My rotation experiment. Rotation gave PPL 18.38; plain imatrix gave 16.33. **Lower is
better, so plain imatrix won.** The "clever" transform lost to good calibration.

---

## Part 2 — Core concepts

### Quantize / dequantize

**Quantize** = compress the weights to fewer bits (bf16 → 4-bit). Done once, offline,
producing the `.gguf` file.

**Dequantize** = expand them back to full numbers *at runtime*, so maths can be done.
Happens constantly during inference, on the fly, in the GPU kernel.

You can think of it like a JPEG: quantize = saving as JPEG, dequantize = decoding it to
pixels when you view it. The loss happened at save time; decoding just unpacks it.

### imatrix (importance matrix)

**A file recording which weights matter most.**

How it's made: run a sample of real text through the *unquantized* model and watch which
weights actually get used strongly. Weights that fire hard on real text are important;
weights that barely activate are not.

Why it matters: when you only have 2 bits per weight, you can't be accurate everywhere.
The imatrix tells the quantizer **"spend your precision here, be sloppy there."**

**This is the single highest-leverage thing in low-bit quantization.** It's why
IQ1_KT went from garbage (98,935) to functional (95.8) at *identical* file size.

### Calibration

The general term for "using sample data to tune the compression." Making an imatrix is
calibration. The quality of your calibration *data* matters:

> GLM-KT's shipped imatrix was 1.29 MB of **English-only** wikitext for a **bilingual
> 744B** model. That is very thin calibration for a giant bilingual model — a likely
> contributor to its poor behaviour.

**Rule of thumb: calibration data should look like what you'll actually use the model for.**

---

## Part 3 — The rotation family

### What "rotate" actually means

Imagine the weights of one layer as a cloud of points in space. **Rotating means viewing
that cloud from a different angle.** The cloud is unchanged; only the coordinates change.

Mathematically: multiply the weight matrix by an *orthogonal* matrix (one that preserves
all lengths and angles). Because it's orthogonal, you can undo it exactly — **no
information is lost.** The model computes the exact same function.

**Why bother?** Because of **outliers.** Weight distributions often have a few enormous
values among many small ones. Quantization hates that: your limited bit budget gets
stretched to cover the huge value, wasting precision on the rest.

Rotation **spreads the outliers out.** After rotating, the values are more evenly sized,
so a fixed number of bits covers them better.

My own measurement of this effect:

| weights | max/rms before | after rotation |
|---|---|---|
| with an outlier channel | 13.42 | **2.55** |

The outlier got flattened. That's rotation earning its keep.

**Crucially: rotation happens BEFORE quantization, on full-precision weights.**
It is preparation, not repair. (You had this backwards — worth locking in.)

### Hadamard

A specific, very cheap rotation matrix made only of `+1` and `-1` entries. It's used
because you can apply it with additions and subtractions instead of full matrix
multiplication — extremely fast. "Hadamard rotation" is the standard workhorse here.

### Incoherence processing

The umbrella term for "rotate the weights so no single coordinate dominates." A weight
matrix is *incoherent* when its information is spread evenly rather than concentrated in
a few outliers. Incoherent matrices quantize much better.

### The named methods

| name | what it does |
|---|---|
| **QuIP#** | Rotation (incoherence) + a smart codebook. Early proof this works at 2-bit. |
| **QTIP** | Rotation + **trellis** coding (see below). The paper KT is based on. |
| **QuaRot** | Rotates the whole network, folding rotations into neighbouring layers so it's free at runtime. Also rotates activations. |
| **SpinQuant** | Like QuaRot, but *learns* the best rotation instead of using a fixed one. |
| **KT (ours)** | ik's trellis quants — IQ1_KT … IQ4_KT. **Trellis without the rotation half.** |

### Trellis coding

Instead of storing each weight independently, a trellis stores a *path* through a
sequence of allowed states, where each step depends on the previous one. It's borrowed
from communications (error-correcting codes). This shares information between neighbouring
weights and packs more meaning into fewer bits than treating each weight separately.

**KT = trellis. QTIP = trellis + rotation. That's the missing half we've been probing.**

### Why my first rotation attempt failed (and what's next)

I rotated the weights but **reused the old imatrix.** The imatrix says "column 7 is
important" — but after rotation, column 7 is a *blend* of all the original columns.
The importance scores no longer point at the right things.

The maths (ordinary linear algebra, not a discovery of mine): after a Hadamard rotation,
the best per-column importance estimate becomes **exactly uniform** — a 780× spread of
importance values collapses to perfectly flat. **Rotation mathematically erases per-column
importance information.**

**The fix, still to be tested: rotate → make a NEW imatrix on the rotated model → then
quantize.** Then importance and weights are in the same coordinate system. This is what
QuaRot/SpinQuant do. This experiment is queued.

---

## Part 4 — Attention, KV cache, and why long context degrades

### Attention scores

At each step the model decides **how much to look at each previous token**. Those weights
are the *attention scores*. High score = "this earlier token is relevant right now."

Every generated token requires computing scores against *all* previous tokens. That's why
long context is expensive, and why errors accumulate.

### KV cache

To avoid recomputing everything for each new token, the model caches two things per past
token: a **Key** (what this token offers) and a **Value** (the content it contributes).
Together: the **KV cache**.

**It grows linearly with context.** At 200K tokens it can dwarf the model itself — which
is why we quantize it too (`-ctk` for keys, `-ctv` for values).

### Keys vs Values — the asymmetry (your fix)

**They are not equally robust.** Measured fidelity vs full precision:

| KV precision | fidelity |
|---|---|
| Q8_0 | ~81.6% |
| Q4_0 | ~8.3% |

Keys tolerate 4-bit reasonably; **values do not.** Keys are used to *compare and rank*
(only the ordering matters much), while values are *summed directly into the output* —
so errors in values land straight in the result.

**⇒ Use `-ctk q4_0 -ctv q8_0`:** aggressive on keys, protective on values. Most of the
memory saving, far less of the damage. You had been running `q4_0` on **both**.

### Why it breaks at depth and not at the start

Each 4-bit rounding error slightly perturbs attention scores. On **one** layer with a
**short** context that's negligible. But:
- errors repeat across every layer (65 layers in this 27B), and
- they compound across hundreds of generated tokens

So output is fine early and degrades as context grows. **Exactly what you observed.**

### ★★★ MEASURED ON THE REAL GLM 5.2 (2026-07-26)

**1. Loop rate is scale-INDEPENDENT.** GLM 5.2 IQ1_KT (~744B MoE, 169 GB) loops **4/8**
under greedy decoding, which is *identical* to the 27B at the same 1.75 bpw. Being 27×
larger bought nothing. Contrast with the coherence floor, which *is* scale-dependent
(4B at 1.75 bpw is noise; 27B and 744B are both fluent). **Two different phenomena —
scale rescues coherence, not loop-resistance.** Do not conflate them.

**2. DRY works, on the real model.** Matched pair, one server load, same temperature,
only the sampler differing:

| arm | loop rate |
|---|---|
| plain sampling @ 0.7 | **3/8** (38%) |
| **DRY @ 0.7** | **1/8** (12%) |

Two-thirds of the looping removed by configuration alone. No requantization, no rebuild,
no VRAM cost. Per-prompt: `doc` (ttr 0.10 → 0.57) and `review` (`period=3 ×5`) both
fixed; every other prompt's diversity improved as well.

**3. One prompt survives everything: long open narrative.** The `story` prompt
("write a very long technical narrative about debugging an outage over three days")
looped on the 27B, on GLM under greedy, *and* under DRY (ttr 0.07 → 0.12 — improved but
still collapsed). Long free-form narrative is the hardest case for a sub-2 bpw model.
If that is your workload, DRY alone is not sufficient.

**Recommended config for a sub-2 bpw model doing long generation:**
```
--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
--temp 0.7  --dry-multiplier 0.8  --dry-base 1.75  --dry-allowed-length 2  -fa on
```

---

### ★★★ MEASURED VERDICT: WEIGHT bit-width causes looping. KV precision does not.

Same 27B model, same 8 loop-prone prompts, greedy (deterministic), **KV pinned at f16 so
weights are the only variable**:

| model | bpw | loop rate |
|---|---|---|
| **Q8_0** | 8.5 | **0/8** (clean everywhere, ttr 0.49–0.81) |
| **IQ1_KT** | 1.75 | **4/8** |

**Zero to fifty percent from bit-width alone.** Combined with the KV result below
(f16 / q8 / q4 all 4/8 — no effect), the conclusion is:

> **Looping is driven by WEIGHT quantization. KV quantization is irrelevant to it.**

### ⚠️ PERPLEXITY DOES NOT CAPTURE LOOPING

The IQ1_KT model scores a respectable **PPL 10.91** — yet it loops on **half** of
long-generation prompts. Q8_0 never loops. **Loop rate is a quality axis orthogonal to
perplexity.**

This explains why quant ladders that look acceptable on PPL disappoint in real agentic
use: PPL measures next-token prediction on *reference text*, but looping is a failure of
*long free generation*, which PPL never exercises.

**⇒ Report loop-rate alongside PPL on any quant card.** It is cheap: 8 exhaustive-
enumeration prompts, greedy, 1500 tokens, repeating-cycle detection.

---

### MEASURED: KV quantization does NOT cause looping

**Controlled experiment** (27B IQ1_KT, 8 loop-prone prompts × 3 KV precisions, **greedy
decoding so every run is deterministic** — one run per cell *is* the answer, no sampling
noise):

| KV precision | loop rate | which prompts looped |
|---|---|---|
| f16/f16 | **4/8** | list, enum, story, review |
| q8_0/q8_0 | **4/8** | list, essay, story, review |
| q4_0/q4_0 | **4/8** | list, essay, story, review |

**Identical rate.** KV precision only shuffles *which* prompts loop (f16 looped on `enum`
but not `essay`; the quantized ones did the reverse) — it does not change *how often*.

**⇒ Practical: do NOT spend VRAM on f16 KV hoping to stop loops. It does not help.
Spend it on context instead.** The cause is attention collapse, which hits full-precision
KV just as hard.

**⇒ RETRACTED (my earlier claims, both wrong):** "you've been quantizing the wrong half of
the KV cache", and the `q8_0` fallback — q8 looped exactly as much as q4. The keys-vs-values
asymmetry is still real in the *literature* for accuracy, but it does **not** govern looping.

**✅ The validated loop fix is the DRY sampler** — 2/2 reproduced loops fixed
(TTR 0.18 → 0.57 and 0.59), no requantization, no rebuild.

**Loop reproducer** (use this to test any future fix): greedy decoding, 1500 tokens, and
prompts that demand long exhaustive enumeration ("list 300 …", "write an exhaustive
5000-word essay …", "enumerate 200 failure modes …"). ~50% of such prompts loop on a
1.75 bpw 27B. Detect with a repeating-tail-cycle check over periods 1–40, **not** a
diversity proxy — and never score an empty generation as a loop.

---

⚠️ **Superseded first attempt (kept as a lesson):** our 27B depth matrix was
**underpowered and its apparent effect was noise** — full results (TTR, higher = better):

| config | 32K | 96K |
|---|---|---|
| f16/f16 | 0.80 | 0.86 |
| q8/q8 | 0.75 | 0.55 |
| **q4/q4** | **0.81** | **0.82** |
| q4/q4 + DRY | 0.76 | 0.85 |

**`q4/q4` beat `q8/q8`, which is physically impossible as a real effect** — so the spread
is sampling noise (one sample per cell, temperature 0.7, 400 tokens). Any story built on
the q8/q8 0.55 outlier is invalid. **A valid design needs ≥5 samples per cell (or temp 0),
longer generations, and a better loop metric than TTR-over-80-words.**

The keys-vs-values asymmetry above is from the *literature*, not from our measurements.
**⚠️ Also measured: `-ctk q4_0 -ctv q8_0` ran at 4.1 t/s vs 57–72 t/s for other configs
and then timed out** — the asymmetric config hits a slow path in this build. Verify
performance before adopting it.

### Attention collapse (the loop mechanism)

From the **LoopGuard** paper (arXiv 2604.10044): loops happen when some attention heads
**lock onto a narrow band of recent tokens.** Once output starts repeating, those repeats
dominate what the heads look at, which produces more repeats — self-reinforcing.

Under RoPE (the position-encoding scheme), periodic repetition makes attention scores
nearly *identical* step to step, creating a stable trap the model can't escape.

Their fix — keep **32 "anchor" tokens** from the start, keep a sparse sample of the middle,
and clean repetitive junk out of the recent window — cut loop rates from **93–100% down
to 1.3–2.7%**.

### Flash Attention (`-fa on`)

A faster, memory-efficient way of computing attention that avoids building the huge score
matrix in memory.

**Hard requirement we hit: a quantized V-cache REQUIRES flash attention.** Without it:

```
quantized V cache was requested, but this requires Flash Attention
```

Note `-fa` takes a value — `-fa on`. Bare `-fa` is a parse error.

---

## Part 5 — Samplers (how the next word gets picked)

The model outputs a probability for **every** word in its vocabulary. Samplers decide how
to choose from that list. They apply **in order**, each narrowing the field.

| sampler | what it does | typical |
|---|---|---|
| **temperature** | Flattens or sharpens the distribution. Low = safe/repetitive, high = creative/chaotic. | 0.6–0.7 |
| **top_k** | Keep only the K most likely words. | 40 |
| **top_p** (nucleus) | Keep the smallest set whose probabilities sum to P. Adapts to confidence. | 0.95 |
| **min_p** | Drop anything less likely than P × (the top choice). Scales with confidence. | 0.05–0.1 |
| **typ_p** (typical) | Prefer words of "typical" information content — avoids both bland and bizarre. | 1.0 (off) |
| **repetition penalty** | Penalize words already used. **See warning below.** | 1.0–1.1 |
| **DRY** | "Don't Repeat Yourself" — detects repeated *sequences* and penalizes continuing them, scaled by match length. | mult 0.8 |
| **XTC** | "Exclude Top Choices" — sometimes drops the most obvious word to break clichés. | 0.1 |

### The repetition-penalty trap (your instinct was right, it is funny)

**Naive repetition penalty can CAUSE loops.** Your sidewalk-pole analogy is genuinely apt.

Mechanism: it penalizes *individual tokens* regardless of context. Common necessary words
("the", "of", a subject's name) get suppressed. That distorts the distribution, pushing
the model off the natural path into a weird region — where it can get stuck in a *different*
repetitive rut. You penalize small repeats and induce large ones.

**DRY is the better tool:** it penalizes repeated *sequences*, not individual words, so
ordinary grammar isn't punished.

Recommended order (from Unsloth's QwQ-32B work):

```
--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
```

---

## Part 6 — Why small models die at low bpw but big ones survive

**The observation:** 4B at 1.75 bpw = noise. 27B at 1.75 bpw = perfectly coherent.
Same bits per weight. Confirmed independently in both our lanes.

**Why:**

1. **Total information.** 4B × 1.75 bpw ≈ 0.9 GB. 27B × 1.75 bpw ≈ 6.6 GB. A language
   model needs some absolute minimum of information to encode grammar, facts and reasoning.
   0.9 GB is below that floor; 6.6 GB is above it.

2. **Redundancy.** Big models store knowledge in a spread-out, duplicated way. Damage a
   few weights and other paths still carry the signal. Small models are densely packed —
   every weight is load-bearing, so damage is unrecoverable.

3. **Error averaging.** More parameters per computation means quantization errors (random,
   independent) partially cancel. Fewer parameters means each error lands undiluted.

**Practical rule: aggressive quantization is a big-model privilege.** For your 1M-context
and giant-model goals this is good news — the models you care about are exactly the ones
that tolerate it. It also means **never validate a low-bit method on a small model** —
it will look broken when it isn't. (This is precisely why I could not verify the IQ1_KT
kernel on a 4B: both the new and reference code paths produced noise, so matching noise
proved nothing. A 27B produces real text, so identical PPL is meaningful evidence.)

---

## Part 7 — Practical config for highly quantized models

```bash
llama-server -m model.gguf \
  -ngl 99 \                         # all layers on GPU
  -fa on \                          # REQUIRED for quantized V-cache
  -ctk q4_0 -ctv q8_0 \             # asymmetric: keys cheap, values protected
  -c 172000 \                       # context; must fit VRAM alongside weights
  -ub 512 -b 2048 \                 # smaller ubatch = smaller compute buffer
  -np 1 \                           # one slot gets the whole context
  --jinja                           # correct chat template / tool calling
```

**Sampling for long generations:**
```
--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
temperature 0.6-0.7, top_k 40, top_p 0.95, min_p 0.05, dry_multiplier 0.8
```

### Gotchas learned the hard way

| symptom | cause / fix |
|---|---|
| `quantized V cache ... requires Flash Attention` | add `-fa on` |
| `-fa` → parse error | it takes a value: `-fa on` |
| OOM at high `-c` despite model fitting | KV cache + compute buffer. Lower `-ub`, or quantize KV |
| Quality collapse at depth | values at q4. Use `-ctv q8_0` |
| Loops appearing only late in generation | error compounding + attention collapse. DRY sampler; protect V-cache |
| Low-bit quantize hard-fails on one layer | MTP/nextn layers get no activations → no imatrix data. `--tensor-type blk.N=q8_0` |
| Quantized model = pure garbage | missing imatrix. Always calibrate at low bpw |
| PPL numbers disagree between runs | different chunk counts. Fix `--chunks` across a comparison |

---

## Part 7b — ★ THE LOOP FIX: attention sinks (TurboQuant / StreamingLLM)

**The looping problem at depth is solved, and the fix is already implemented for llama.cpp.**

### The mechanism, stated simply

Attention uses *softmax*, which forces the attention weights to **add up to 1**. The model
must put that weight *somewhere*, every single step — even when nothing in the context is
especially relevant.

If there is no natural place to dump that leftover weight, it lands on **recent tokens**.
Recent tokens are whatever was just generated. So the model attends to its own last few
words → generates similar words → attends to those → **loop**.

An **attention sink** is a few tokens at the very start, permanently kept, that act as a
drain for this residual weight. With a sink, softmax has somewhere harmless to put the
leftover and never collapses onto the recent window.

**Quantized KV makes this worse**: rounding noise nudges attention scores, tipping the
model into the trap sooner.

### Evidence (two independent sources agree on 32 tokens)

| source | finding |
|---|---|
| **TurboQuant** (empirical) | `--tri-keep-first 0` → *"decode collapses into a token-repetition loop within ~300 tokens."* Sink is **mandatory**. |
| **TurboQuant** (multi-turn) | Without QJL correction, **loops at turn 3–4**; with it, **9+ turns** stable. |
| **LoopGuard** (theory) | Keep **32 anchor tokens** + sparse middle + cleaned recent → loop rate **93–100% → 1.3–2.7%**. |

TurboQuant's note: default `keep_first=4` covers only the chat template; **`32` covers a
full prompt header including injected instructions** — the right choice for agentic use
where a system prompt sits at the front.

### TurboQuant numbers

- **5.2× KV compression** (Qwen3.5-35B: 5,120 MB → 990 MB)
- **PPL 6.829 → 6.850 = +0.3%** — effectively lossless
- Cost: TBQP (with QJL) is slower than TBQ — ~42 t/s vs ~65 t/s on GLM-4.7-Flash

### Flags

| flag | meaning |
|---|---|
| `--triattention FILE` | TriAttention calibration data |
| `--tri-budget N` | per-layer slots retained after eviction |
| `--tri-keep-first N` | **attention sink size — the loop fix.** Default 4, use 32 |

Recommended for 40K+ context:
```
--cache-type-k amx3 --cache-type-v amxv3 \
--triattention calib/model.bin --tri-budget 128 --tri-keep-first 32
```
Plus `DSV4_BATCHED_COMPRESSOR=1` for multi-turn prefills beyond ~100K tokens.

Files touched (relevant to porting into our KT tree): `ggml/src/ggml-cuda.cu`,
`src/llama.cpp` (KV routing + attention-sink logic), `include/llama.h`, `turboquant/`.

---

## Part 8 — Open questions (live work)

1. **Hadamard in KT, done right** — rotate → recalibrate → quantize. Untested; queued.
2. **KV asymmetry × KT quants** — does `-ctk q4 -ctv q8` behave the same when the *weights*
   are 1.75 bpw rather than 4-bit? KT sits below the bit-widths any published KV study covers.
3. **LoopGuard for quantized models** — the paper explicitly does not study quantization.
4. **Ternary (~1.58 bpw)** — the extreme end, where rotation should matter most.

---

*Maintained during the KT-quant work. Numbers here are measured on this hardware unless
attributed to a paper.*

---

## Part 9 — Session-final measured verdicts (2026-07-26)

**The knee, complete ladder (27B, file-level bpw, greedy, f16 KV):**

| rung | file bpw | loop rate |
|---|---|---|
| IQ1_KT | 1.99 | 4/8 |
| ffn_down to IQ2 | 2.18 | 2/8 |
| ffn_down+gate to IQ2 | 2.24 | 4/8 |
| all-FFN to IQ2 | 2.30 | 0/8 (knee) |
| uniform IQ2_KT | 2.48 | 0/8 |

0/8 is first reached at 2.30 file-level and holds above. Sub-knee ordering is
trajectory-chaotic (a strict-subset rung looped less), so per-rung rates below the knee
carry roughly +/- 2-in-8 noise. `ffn_up` is load-bearing: raising only down+gate does not
clear looping.

**The validated production fix (measured on the real GLM 5.2 IQ1_KT):** the story-prompt
holdout that survived standard DRY falls to 0 loops with `dry_allowed_length 1` (one
parameter down from the default 2), and a full 8-prompt regression stays clean. Recommended:

```
--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
--temp 0.7 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 1 -fa on
```

**Mixed KV cache types are a large-context trap.** `-ctk q4_0 -ctv q8_0` measured within
~10% of matched types at 8K context, but at ~100K context the same config collapses
(prefill throughput decayed to the point of a 3600s timeout) while `-ctk q4_0 -ctv q4_0`
completed the identical 98.9K-token prefill at 1428 t/s. Never mix K and V cache types for
long-context serving on this build; match them.

**KV-cache Hadamard (attn-rot) benefit is small on KT weights:** +0.28% perplexity at
q4/q4 KV (9.0329 on vs 9.0579 off). Real but not worth fused-kernel engineering unless the
KV drops below q4.

**Format brittleness is a low-bit-only hazard:** a one-newline prompt-separator change
flipped a sub-2.5 bpw model between generating and immediate EOS at 32K context, but a
Q8_0 model was identical across all separators and depths. Above the knee it vanishes.

---

## Part 6: fused ops, scratch tensors, and the context ceiling

Added 2026-07-29 because these came up while chasing why GLM 5.2 could not go past
64K, and the vocabulary was in the way.

### Operation (op)

One step of maths in the model's compute graph. "Multiply these two matrices",
"add these", "apply softmax". A single token going through GLM 5.2 executes tens of
thousands of ops. ggml (llama.cpp's engine) builds the whole graph first, then runs it.

### Scratch tensor

A **temporary result** that exists only partway through a calculation. Not a weight
you downloaded, not the KV cache you keep, just working space.

Cooking analogy: a weight is an ingredient in the cupboard, the KV cache is the dish
you are building up, and a scratch tensor is the chopping board. You need it while
working, you throw it away after, but **it still takes counter space while it exists**.

llama.cpp reserves one big region for these up front, the **compute buffer**, sized for
the worst moment in the graph. That reservation is real VRAM, unavailable for anything
else, whether or not the peak is ever hit.

### Fused vs unfused

**Unfused** means each op runs separately and writes its result to memory, and the next
op reads it back:

```
step 1: compute scores        -> write a big scratch tensor to VRAM
step 2: read it back, softmax -> write another
step 3: read it back, top-k   -> write another
```

**Fused** means one kernel does all of it in registers and on-chip memory, and only the
final small answer is written:

```
one kernel: scores -> softmax -> top-k, entirely internal -> write only the top-k
```

Same maths, same answer. The difference is that the unfused version has to *materialise*
each intermediate at full size, and the fused version never does. Flash attention is the
famous example: it fuses the attention chain so the full score matrix is never written.

### Why this is your context ceiling

GLM 5.2 uses DSA sparse attention, which has an extra piece called the **lightning
indexer** that picks which keys are worth attending to. Its score tensor is shaped

```
[n_kv, 32 heads, n_ubatch]  in F32
```

`n_kv` grows with context. So this scratch tensor grows with **context x ubatch**, and
2-3 copies are live at once. **Flash attention does not help here**, because the indexer
is a separate op that FA never touches. That is why raising `-ub` blew up your VRAM
while KV itself stayed small: at 64K your KV is only 1.6-3 GB, and the scratch space was
the thing eating the card.

Mainline has a **fused** lightning indexer that never materialises that tensor, and our
build already contains it.

### The switch that may be turning it off

`llama-context.cpp`, `resolve_fused_ops()`. Before using a fused op, llama.cpp checks
that the fused node landed on the same device as the layer it belongs to:

```c
if (device_fused != device_layer) {
    device_mismatch = true;
    break;                                 // stops at the FIRST mismatch
}
...
if (device_mismatch) { enabled = false; }  // disables it for the WHOLE model
```

Two things make this severe:

1. It **breaks on the first mismatch**, so one bad layer ends the search.
2. It then disables the fused op **globally**, for every layer, not just that one.

With `--tensor-split 49,51` the model spans two cards, and there is a boundary where the
scheduler can place the fused node on a different card than the layer. If that happens
once, the fused indexer is off everywhere, every layer falls back to materialising the
big scratch tensor, and the compute buffer balloons.

### REFUTED 2026-07-29, before any silicon was spent on it

I proposed that the two-card split was switching this off. **It is not.** Fable-DSpark
checked real GLM split serve logs on the box before authorising a window:

```
print_info: arch       = glm-dsa
print_info: model type = 744B.A40B
(layer assignment: 218 CUDA0 / 221 CUDA1  -> genuinely split)
resolve_fused_ops: resolving fused Lightning Indexer support:
resolve_fused_ops: Lightning Indexer enabled
```

Enabled, on a real split, and three other logs agree. Zero `is assigned to device`
warnings anywhere on the box.

**Why the mismatch cannot fire.** The scheduler places a node on the device holding its
weights, and `indexer_score` for layer `il` is built from layer `il`'s own tensors, so
`device_fused == dev_layer(il)` by construction. A layer-split never triggers it. The
guard exists for genuine missing-backend-support cases, not for tensor splits.

**So the fused indexer is already ON and the compute-buffer ceiling has another cause.**
The mechanism above is still worth understanding, and the guard's break-on-first-mismatch
plus disable-globally behaviour is still real and still a hazard for other fused ops. But
it is not what is capping this context, and I should have looked for an existing log
before proposing a window. The answer was on disk the whole time.
