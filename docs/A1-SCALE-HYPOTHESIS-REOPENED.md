# A1 reopened: scale does buy bit-width tolerance

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

## What is still not proven

- **n = 8 prompts.** Fisher exact two-tailed on 7/7 vs 4/8 gives **p = 0.077**.
  Suggestive, short of conventional significance. It replicates at two depths,
  which helps, but one prompt set of eight is thin.
- **Family is confounded with size.** The three models span `qwen3`, `qwen35`
  and `glm-dsa`. Even the 4B to 27B step crosses a model generation.
- The fix for both is a larger prompt set. At 16 prompts, 14/14 against 8/16
  would give p = 0.006. That is the same change the non-termination metric needs
  (see below), so it is one piece of work.

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
