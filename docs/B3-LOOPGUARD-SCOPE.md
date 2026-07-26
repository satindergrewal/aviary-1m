# B3: LoopGuard-style detect-and-prune (scope, and an argument for not building it yet)

A decision document for the last open item on the loop program. Written to be rejected
as easily as approved.

---

## What B3 was going to be

A detector inside llama.cpp that watches for the attention-collapse signature behind
repetition loops (heads locking onto the recent window), and intervenes at the mechanism
level: prune or reweight the collapsed attention rather than penalising the output text.
Estimated 3-5 days. Nobody has built it for llama.cpp.

## Why it should be questioned before it is built

The loop program already produced a working fix, and it costs nothing:

```
--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
--temp 0.7 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 1
```

Measured on the production GLM itself: plain sampling 3/8 loops, DRY at
`allowed_length 2` 1/8, DRY at `allowed_length 1` **0/8 with a clean full-prompt
regression**. Zero requantization, zero VRAM, one config line.

So B3 is not competing against a broken baseline. It is competing against a validated
free fix, and it has to earn 3-5 days against that. Right now the honest position is
that **we cannot say what B3 would add, because we have never measured where DRY fails.**

## The gap in our own evidence

Every DRY result we have is from one region of the space:

| dimension | what we tested | what we never tested |
|---|---|---|
| temperature | 0.7 | **temp 0 / greedy**: loops reproduce at temp 0, and agentic tool-calling wants determinism |
| output shape | prose prompts | **structured output**: code, tables, JSON, where legitimate repetition is normal and DRY may damage it |
| depth | ~1.5K on GLM, 32K/128K on the 27B ladder | GLM at its real ~16K ceiling (probe built, window pending) |
| length | 800-1500 tokens | very long single generations, where collapse has more room to develop |

DRY is a *symptom* treatment: it penalises repeated n-grams in the output. B3 targets the
*cause*. That distinction only pays if there is a residual class of loops DRY cannot
reach. Two plausible candidates, both untested:

1. **Greedy loops.** DRY reshapes a distribution; at temp 0 it can only shift the argmax.
   Whether that is enough is unknown.
2. **Loops DRY suppresses at the cost of correctness.** A loop "fixed" by mangling
   legitimate repeated structure is not fixed. Our regression covered 8 prose prompts,
   not code or tables.

## Recommendation: measure the residual first, then decide

Spend roughly half a day, not 3-5 days:

- **R1, greedy arm.** Run the loop reproducer at temp 0 with and without DRY. If DRY
  closes greedy loops too, one of B3's two justifications is gone.
- **R2, structured-output arm.** Add code/table/JSON prompts to the harness and check
  both loop rate *and* output validity under DRY. This tests the damage side, which the
  current prompt set cannot see.
- **R3, long-generation arm.** One long run per config, since collapse needs room.

Decision rule, fixed in advance so the result cannot be rationalised afterwards:

- If R1-R3 leave **no material residual**, **cut B3.** The loop program is done, and its
  deliverable is a config line plus the harness. That is a good outcome, not a failure.
- If they expose a **specific residual class**, build B3 *targeted at that class*, with
  the residual as its acceptance test, which also makes it far smaller than 3-5 days of
  open-ended work.

This is deletion before optimisation. B3 as currently scoped is a solution looking for a
failure we have not demonstrated we still have.

## If B3 is built anyway, here is the shape

So the option is costed rather than hand-waved:

- **Hook:** a per-layer callback after attention weights, cheapest at the graph level
  where imatrix's `cb_eval` already observes tensors.
- **Signal:** entropy or max-weight concentration of attention over the recent window,
  per head, tracked across decode steps. Collapse shows as sustained low entropy pinned
  to a short recent span.
- **Intervention:** the risky part. Options are dropping the collapsed head's
  contribution, temperature-scaling its weights, or forcing a KV-window jump. All change
  model behaviour on non-looping text too, which is exactly what has to be measured.
- **Gate:** loop rate must fall *and* generation success and output validity must not,
  measured with `tools/loop_rate.py` including `--chat` so in-think loops count.
- **Real cost:** the detector is maybe a day. The intervention plus proving it does no
  collateral damage is the other 2-4.

## Bottom line

Recommend **R1-R3 first (about half a day), then decide.** Do not start a 3-5 day build
against a failure mode that a free config change may already have closed.
