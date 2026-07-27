# N3: the depth control arm, and what it invalidates

Run 2026-07-28. Qwen3.6-27B IQ1_KT (1.75 bpw), CUDA, greedy, 1500 tokens,
prompt set v1, `llama.cpp-kt/build-cuda`, f16 KV, single GPU.

## Why this run existed

The earlier depth matrix measured the CANDIDATE (IQ2_KT) and reported
"never loops at any depth". No control was run. A loop rate of zero has two
possible causes and that experiment could not tell them apart:

1. the model genuinely stopped looping, or
2. the harness stopped being able to see loops at that depth.

IQ1_KT loops 4/8 at shallow depth, so it is the positive control.

## Result

| depth | loop rate | generation success | non-termination |
|-------|-----------|--------------------|-----------------|
| 0     | **4/8 (50%)** | 8/8 (100%) | 8/8 (100%) |
| 8K    | 1/6 (17%) | 6/8 (75%)  | 5/6 (83%)  |
| 32K   | **0/4 (0%)**  | 4/8 (50%)  | 4/4 (100%) |

Depth 0 reproduces A1's 4/8 exactly, so the harness and the build are sound.

## What it invalidates

**Loop rate falls to zero at depth because the loop-prone prompts stop
generating, not because the looping stops.** Per-prompt, the same prompts carry
the failure and then vanish:

| prompt | depth 0 | depth 8K | depth 32K |
|--------|---------|----------|-----------|
| enum   | LOOP (ttr 0.02) | LOOP (ttr 0.15) | EMPTY (1 token) |
| list   | LOOP (ttr 0.11) | ok (ttr 0.39)   | EMPTY (1 token) |
| review | LOOP (ttr 0.07) | ok, EOS at 548  | ok (ttr 0.43)   |
| story  | LOOP (period 1 x49 "backlogs,") | ok (ttr 0.34) | ok (ttr 0.61) |

So "IQ2_KT never loops at any depth" says nothing about IQ2_KT: the known-bad
IQ1_KT also scores 0/4 at 32K. **Depth loop-rate is uninformative past ~8K for
either quant.** Any depth comparison must report generation success beside it,
which is why that is now a first-class metric rather than a footnote.

## What the new axis shows

Non-termination was invisible to the old harness and it is close to universal
here. Across all 24 generations, **exactly one ended because the model chose to
end it** (`review` at 8K, 548 tokens, `stop=eos`). Every other generation either

- ran the full 1500-token budget without emitting a stop token (17 of 24), or
- emitted EOS at token 1 and produced nothing (6 of 24).

At 1.75 bpw the model is **bimodal**: it says nothing, or it says everything.
The middle, "answer the question and stop", is nearly absent. That is a
different defect from looping and it is the one that makes a quant unusable
agentically, because every turn consumes the whole budget.

Note the direction of the depth effect on stop behaviour. At depth 0 all eight
prompts hit the limit and none emitted EOS. At 32K, four emit EOS at token 1 and
the remaining four never emit one at all. The model does not lose the stop token
gradually; it flips between never using it and using it immediately.

## Reproduce

```bash
# server
cd <BOX>/llama.cpp-kt
CUDA_VISIBLE_DEVICES=1 ./build-cuda/bin/llama-server \
  -m <BOX>/ktdev/Qwen27B-IQ1_KT.gguf -ngl 99 -c 65536 -fa on --port 8091

# harness (note: build/ in that tree is CPU-only, build-cuda/ is the one you want)
<BOX>/ktdev/n3_depth_control.sh
```

Raw rows: `<BOX>/ktdev/n3_results/`.
