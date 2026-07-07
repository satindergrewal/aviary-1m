# Axis 3: Instruction adherence under context saturation (working doc)

Status: research and design phase. Explicitly long work, not a weekend build. Do the industry research before building anything (this doc records it).

## The three axes

1. **Find facts** (our NIAH harness): single-pass retrieval from a haystack. Certified on the fleet.
2. **Combine facts** (NVIDIA RULER): multikey retrieval, variable tracking, aggregation. Still single-pass. In progress on the fleet.
3. **Obey rules over time** (this doc): a system prompt carrying a conditional rule matrix ("do X and Y if A and B hold, otherwise C, never N"), evaluated turn by turn as the context fills with tool outputs and noise. The production failure mode: the model applies the wrong branch, or drops standing constraints, somewhere around turn N, long before retrieval fails.

Observed independently by us and by production users (Discord, July 2026): models get "worse than a toddler" at compound conditional rules over long agentic sessions, while still passing needle tests at the same lengths. Retrieval and adherence are different capabilities with different decay curves.

## Prior art (found, not assumed)

The field is actively working adjacent problems, which validates the axis and narrows our gap:

- [SEQUOR](https://arxiv.org/pdf/2605.06353): automatic benchmark for constraint adherence in long multi-turn conversations. Closest existing work. Finds accuracy declines with conversation length even for a single constraint, and substantially faster when multiple constraints are introduced sequentially.
- [Laban et al. 2025 / "LLMs Lose the Thread"](https://arxiv.org/pdf/2605.12922): LLMs perform 39 percent worse and 112 percent less reliably multi-turn vs equivalent single-turn; adherence to earlier-turn instructions degrades monotonically (o1-preview drops 88 to 71 percent between turns one and three).
- [MMMT-IF](https://arxiv.org/pdf/2409.18216): multimodal multi-turn instruction following.
- ["Models Recall What They Violate"](https://arxiv.org/pdf/2604.28031): models can often retrieve the constraint they are simultaneously violating. Direct evidence that Axis 1 passing does not imply Axis 3 passing.
- [GraphIF](https://arxiv.org/html/2511.10051v2): mitigation via relation-graph prompts.
- MT-Eval, MARS-Bench, LongProc: all report monotonic multi-turn degradation tied to context dilution and constraint-retrieval failure.
- IFEval: verifiable-constraint scoring, but single-turn and short context. Its verifiable-constraint style is worth borrowing for scoring.

## The gap we could actually fill

Existing work is mostly: frontier API models, modest context lengths, simple or sequential constraints. Unoccupied combination:

1. **Conditional rule matrices** (branch logic, not just standing constraints) scored per-turn for correct branch selection
2. **Extreme saturation**: contexts filled to 100K, 500K, 1M with realistic tool-output noise, on the long-context builds we ship
3. **Local open models** (the fleet), where nobody publishes this data
4. **The memory A/B**: same run with and without rule reinjection every N turns, quantifying exactly how much an external memory hack buys. Nobody publishes that delta.

## Satinder's hypothesis (recorded, to be tested)

"This is the industry-wide Memory problem. With a good memory system it would not much matter what model we use; instructions would be followed over long contexts."

Calibration to test rather than assume: external memory (retrieving and reinjecting the relevant rule near the point of action, MemGPT/Letta style; the crude every-N-turns reinjection production users already do) sidesteps context distance and is probably the strongest practical mitigation. But branch misapplication is partly a reasoning failure that persists even with the rule in fresh context; models violate constraints they can recite ("Models Recall What They Violate"). Prediction to verify: memory reinjection recovers a large fraction of adherence but not all of it, and the residual gap is model-capability-dependent. The A/B design above measures exactly this and settles the hypothesis with data.

## Design sketch v0 (to be attacked and revised)

- System prompt: K rules, mixed types: standing prohibitions ("never use exclamation marks"), unconditional obligations ("every reply ends with a status line"), conditional branches ("if the user message contains a color, reply in French; if it contains a number, append a running total; if both, French plus total").
- Driver: scripted N-turn conversation; each turn injects realistic filler (tool outputs, logs, code) to hit a target fill rate; some turns trigger conditions, some do not; trigger schedule is seeded and logged.
- Scoring: per turn, programmatically verifiable (IFEval-style): correct branch taken, standing constraints intact, running state (the total) correct. Output = adherence-vs-turn-depth curve and adherence-vs-context-fill curve.
- Variants: (a) bare, (b) reinjection every N turns, (c) reinjection only of violated rules (oracle memory), across models and context fills.

## Holes to poke (initial, add more)

- Scoring ambiguity: natural language replies can partially comply; verifiable constraints must be crisply checkable or the benchmark measures the judge, not the model.
- Trigger-detection confound: failure to apply a branch could be failure to notice the trigger (Axis 1 style) rather than failure to obey; design must separate the two (e.g., ask the model to also state which triggers it saw).
- Filler realism: random essays may not dilute attention the way genuine tool-call transcripts do; use both and compare.
- Template effects: chat-template handling of long multi-turn histories differs per model family and per server; the llama-server multi-turn path is itself under test (we hit template bugs in Claude Code integration).
- KV cache policy: context-shift/eviction settings can silently delete the system prompt; must pin and document serving config or we measure the server, not the model.
- Contamination with SEQUOR-style publics: fine, we are measuring our models, not claiming a novel task category; cite and differentiate honestly.

## Next actions (when work resumes)

1. Read SEQUOR and Laban et al. properly; steal scoring ideas, note differences.
2. Decide overlap policy: run SEQUOR as-is on the fleet first if runnable locally; build ours only for the gaps (saturation depth, branch matrices, memory A/B).
3. Harness skeleton: driver + verifiable scorer + seeded trigger schedule.
4. Pilot on Gemma4-12B-1M and the qwen36 hybrid at 3 fill levels, then decide sample sizes.
## Field datapoints (production users, recorded as they arrive)

- **CoronelOc (Reaper Labs Discord, July 8, 2026):** reinjects tool-use system instructions every 1500 tokens during tool-call phases on Gemma4; reports zero percent wrong tool-call format after adopting this ("reassert the instructions within the sliding attention window"). Residual failures are semantic (wrong command flags), which matches the prediction that reinjection fixes format adherence but not reasoning correctness. This is the strongest external evidence yet for the memory-reinjection arm of the A/B design; the 1500-token cadence (aligned to Gemma SWA window) is a concrete reinjection policy worth testing as variant (b).
- Same user maintains Gemma4 tool-calling tutorials documenting native tool syntax and template filtering of tool responses ("model blindness" loops): github.com/nnnarvaez/Exploring-Gemma4-and-Gemini. Trigger-detection vs obedience separation in our design should also separate "never saw the tool result" failures (harness bug) from true adherence failures.
