# A1: three revisions, ending in a controlled null

> ## ⚠️⚠️ RETRACTION 2026-07-28, same night: the conclusion below is NOT supported
>
> This document concluded "scale buys bit-width tolerance, p = 6.1e-05". **That
> claim is withdrawn.** It rests on comparing two models at 1.75 bpw without ever
> measuring what either model scores at essentially-lossless precision. When that
> control was run, the effect disappeared.
>
> | | Q8_0 floor (lossless) | IQ1_KT (1.75 bpw) | damage attributable to quantization |
> |---|---|---|---|
> | Qwen3-4B | **42%** (10/24) | 100% (21/21) | +58 points |
> | Qwen3.6-27B | **0%** (0/20) | 45% (10/22) | +45 points |
>
> **Interaction test (is the small model damaged more?): difference of differences
> 12.9 points, z = 0.88, p = 0.38. Not significant.** There is no evidence that
> scale changes how much quantization hurts.
>
> What the p = 6.1e-05 actually measured was the *absolute* loop rate at 1.75 bpw.
> That number is real, but the 27B also beats the 4B at **lossless** precision
> (0/20 vs 10/24, p = 0.0009), so the gap is mostly the 27B being a better model,
> not it tolerating low bit-width better.
>
> **Two things that survive and are worth keeping:**
>
> 1. **The 27B's floor is 0%.** It does not loop at all when lossless. So the
>    entire 45% it shows at IQ1_KT *is* quantization damage, cleanly attributed.
>    That is a useful, well-controlled number.
> 2. **The 4B is unusable as a quantization testbed on prompt set v2.** A 42%
>    floor means most of its signal is capability, not bit-width. Any future
>    ladder should use models whose lossless floor is near zero, or match floors
>    across arms.
>
> **The design lesson, now a rule:** run the lossless rung as *cell zero* of any
> bpw ladder, before any comparison. Two numbers mean nothing until you know what
> they are numbers of. This is the second time in one night the same class of
> error survived into a published claim: first "what else differs between the
> arms" (dense vs MoE), then "what does each arm score with no treatment".
>
> **Note on the original A1.** Its conclusion, that scale buys no tolerance, is
> closer to the controlled result than this document's overturn was. It reached
> that for the wrong reason (a dense-vs-MoE confound), but it landed nearer the
> truth. The honest net position after three revisions is **no evidence either
> way, from a design that could finally detect an effect if one existed.**

## (superseded) A1 reopened: scale does buy bit-width tolerance

Run 2026-07-28. the owner challenged the A1 conclusion ("scale bought nothing")
on the grounds that it rested on a single test. He was right, and re-testing
reversed the finding.

## What A1 actually compared

A1 measured Qwen3.6-27B at IQ1_KT (4/8 loops) against GLM 5.2 at IQ1_KT (4/8
loops) and concluded that scale buys no tolerance at a fixed bit-width. Reading
the GGUF metadata shows the two models are not comparable in the way that
conclusion assumed:

| | Qwen3.6-27B | GLM 5.2 |
|---|---|---|
| architecture | `qwen35`, **dense** (no `expert_count` key) | `glm-dsa`, MoE, 256 experts, 8 used |
| FFN width | `feed_forward_length` 17408 | `expert_feed_forward_length` 2048 |
| embedding length | 5120 | 6144 |
| size of each quantized FFN matrix | **89.1 M** | **12.6 M** |

Total parameters rose 27x while the size of the matrix actually being crushed to
1.75 bpw fell 7x. Those two effects push in opposite directions, so a null result
is uninformative: it is consistent with scale helping, scale hurting, or neither.

## The discriminating test

Total-parameter count and matrix size are anti-correlated across the models on
hand, so one experiment separates them.

| model | type | matrix quantized | total params | predicted by scale | predicted by matrix size |
|---|---|---|---|---|---|
| Qwen3-4B | dense | 24.9 M | 4 B | loops **much more** than GLM | loops **no worse** than GLM |

Qwen3-4B has matrices twice the size of GLM's but 1/186th the parameters.

## Result

All at IQ1_KT (1.75 bpw), greedy, prompt set v1, imatrix-quantized, CUDA.

| model | type | matrix | total params | loop rate (depth 0) | loop rate (8K) |
|-------|------|--------|--------------|---------------------|----------------|
| Qwen3-4B  | dense | 24.9 M | 4 B   | **7/7 (100%)** | **7/7 (100%)** |
| Qwen3.6-27B | dense | 89.1 M | 27 B | 4/8 (50%) | 1/6 (17%) |
| GLM 5.2   | MoE   | 12.6 M | 744 B | 4/8 (50%) | not run |

**Scale hypothesis supported. Matrix-size hypothesis refuted.** The 4B carries
matrices twice as large as GLM's and loops twice as often. Matrix size predicted
the reverse; parameter count predicted the observed ordering.

## The corrected reading of A1

GLM 5.2 at 744 B matches a 27 B dense model at the same bit-width *despite* its
quantization units being 7x smaller. That is not "scale bought nothing". Scale
bought exactly enough to offset a 7x smaller quantization unit.

## Confirmed at n = 24

Re-run on prompt set v2, same models, same bit-width, same build.

| comparison | n = 8 | **n = 24** | Fisher exact |
|---|---|---|---|
| **scale**: Qwen3-4B vs Qwen3.6-27B, both 1.75 bpw | 7/7 vs 4/8 | **21/21 (100%) vs 10/22 (45%)** | **p = 6.1e-05** |
| **knee**: 27B IQ1_KT vs IQ2_KT | 4/8 vs 0/8 | 10/22 (45%) vs 2/22 (9%) | p = 0.016 |

Both were p = 0.077 at n = 8. Both are now significant, and the point estimates
barely moved:

| cell | v1 (n=8) | v2 (n=24) |
|------|----------|-----------|
| 4B IQ1_KT   | 7/7 = 100% | 21/21 = 100% |
| 27B IQ1_KT  | 4/8 = 50%  | 10/22 = 45%  |
| 27B IQ2_KT  | 0/8 = 0%   | 2/22 = 9%    |

So v1 was underpowered rather than biased. **Scale buys bit-width tolerance,
p < 0.0001.**

## Second finding: looping and non-termination are separable

The bounded prompts in v2 make non-termination a real measurement, and it does
not follow the loop rate at all:

| model | bpw | loop rate | non-termination (bounded) |
|-------|-----|-----------|---------------------------|
| Qwen3.6-27B IQ1_KT | 1.75  | 45% | 2/6 = 33% |
| Qwen3.6-27B IQ2_KT | 2.145 | 9%  | 2/6 = 33% |

Climbing a bit-width rung cuts looping fivefold (p = 0.016) and moves
non-termination not at all. They are different defects with different remedies.

That is directly relevant to the production GLM: raising bit-width will not fix
the behaviour where the model completes and verifies its task and then runs to
65,535 tokens anyway, and DRY did not fix it either, having been active
throughout that run. A runtime stop-guard is the remaining lever.

## What is still not proven

- **Family is confounded with size.** The models span `qwen3`, `qwen35` and
  `glm-dsa`. Even the 4B to 27B step crosses a model generation. A same-family,
  same-generation ladder would close this; we do not have one on hand.
- The MoE arm of the scale result is still the single GLM datapoint at 4/8.
  Kimi-K3 would be the second, and would test the extrapolation below.

## Consequence for Kimi-K3

Kimi-K3 is 2.77 T parameters, 3.7x GLM 5.2, with expert matrices of
3072 x 3584 = 11.0 M, essentially the same size as GLM's 12.6 M. Under the
supported hypothesis its loop rate at max KT should land **below** GLM's 50%,
not above it. That is the difference between "worth quantizing" and "not worth
the disk", and it is now the leading reason to keep the weights.

## Related correction: the non-termination metric

The same run showed Qwen27B-IQ2_KT at depth 0 scoring 0/8 loops, 8/8 generation
success and 8/8 non-termination. The prompt set demands exhaustive output
("List 300 distinct examples", "an exhaustive 5000-word essay"), so running out
a 1500-token budget on those prompts is correct behaviour, not a defect.

Non-termination is therefore **not diagnostic at shallow depth on prompt set
v1**. It is diagnostic in two places:

- agentic use, where the model has completed and verified its task and keeps
  going anyway (observed: a single response reaching 65,535 tokens), and
- the depth bimodality, where prompts split between emitting EOS at token 1 and
  never emitting one at all.

Prompt set v2 needs items with a natural endpoint so the metric can separate
"correctly kept writing" from "could not stop".
