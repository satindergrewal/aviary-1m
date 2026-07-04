# Training plan: a domain-tuned Ornith (draft v1)

Goal, per the project north star: an open-weights coding model with 1M context, usable through Claude Code, at the highest intelligence reachable on owned or briefly-rented hardware. Active training track: SFT on vetted public agent-trace datasets. Speed optimization (speculative drafts) is deferred.

## Two training tracks

### Track 1 (ACTIVE, Satinder's decision): agent-trace SFT

SFT on vetted permissive trace datasets from the shortlist (the Apache-2.0 25k cleaned set first; the MIT 2M set after provenance verification), plus, over time, traces from Satinder's own Claude Code sessions. Teaches tool-call discipline, terse output, Claude-Code-shaped agentic workflows. This is the training work happening now.

### Track 2 (DEFERRED idea, not scheduled): domain corpus midtraining

Continued pretraining (a.k.a. midtraining) on the exact domain the model will serve, followed by a light SFT to restore chat behavior.

Corpus candidates, all license-clean:

| Source | Content | License |
|---|---|---|
| iroh, quinn, libp2p (rust-libp2p), s2n-quic, str0m | full Rust source + docs + tests | MIT/Apache-2.0 |
| tokio, hyper, rustls, ring | the async/TLS substrate the project sits on | MIT/Apache-2.0 |
| IETF RFCs + active drafts (QUIC, TLS, WebTransport, NAT traversal, MASQUE) | spec language the project implements | IETF Trust, freely reproducible |
| Selected papers (transport, DHT, gossip, NAT) | research the project draws on | varies, curate per-paper |
| The Rust Book, Rustonomicon, std docs | idiom grounding | MIT/Apache-2.0 |

Estimated corpus size: 100M to 500M tokens after dedup and cleaning. This is midtraining scale, not pretraining: with MoE efficiency (3B active on the 35B), a single pass is a few hundred GPU-hours worst case and much less on the 9B.

Why high conviction: the failure mode this attacks is hallucinated APIs and shallow RFC knowledge, which is precisely what hurts daily agentic coding on a specialized stack. Nothing in a generic trace set teaches quinn's connection API.

(Deferred by decision on July 5: trace-SFT first. The corpus idea stays documented for a future phase only.)

## Method decisions (current position, revisable)

- **9B first, always.** Every recipe validates on the 9B locally (5090 QLoRA or Runpod full-FT at ~$30-60) before any 35B spend. Same gate philosophy that just saved $187 on DSpark.
- **QLoRA vs full fine-tune:** QLoRA rank 64-128 for iteration; a full-FT run only if QLoRA shows a measurable ceiling. Note: Unsloth does not support qwen35moe yet (open issue), so the 35B path is vanilla PEFT/TRL or axolotl.
- **Sequence length during training: 8-16K**, with document packing. Long-context ability comes from the base model and must be *protected*, not trained: every checkpoint re-runs the NIAH ladder (a fine-tune that scores 90 on HumanEval but drops to 4/10 needles at 524K is a regression, not a win).
- **Anti-forgetting:** mix 10-20 percent general/code replay data into the domain corpus; keep learning rates midtraining-low (1e-5 class for full-FT, higher for LoRA).
- **Base model choice:** start from the official Ornith weights, not the 1M-baked GGUFs (training happens in safetensors land; YaRN metadata is re-baked into the GGUF exports afterward). Train first, abliterate last (decided July 4).

## Evaluation gauntlet (every candidate checkpoint)

1. Tier 1 thermometer: HumanEval + GSM8K-100, A/B vs base.
2. Domain eval (to be built, small but ours): API-recall quiz generated from iroh/quinn/libp2p signatures + RFC detail questions. This is the metric the whole exercise exists to move.
3. NIAH ladder re-cert (32K to 1M) to prove long context survived.
4. The Claude Code session gate: drive a real session via the tested Ollama config on tasks from the actual Rust project; judge tool-call fidelity and terseness by use.
5. Tier 2 SWE-bench-50 A/B for checkpoints that pass 1-4.

## Hardware mapping

| Step | Where | Est. cost |
|---|---|---|
| Corpus building + dedup | Mac, CPU | $0 |
| 9B QLoRA iterations | 5090 | $0, hours each |
| 9B full-FT (if needed) | 2x H100 Runpod | ~$30-60/run |
| 35B QLoRA (winning recipe only) | 1x 96GB Runpod (or the incoming Pro 6000 locally) | ~$25-60/run |
| 35B midtrain full pass (only if LoRA ceilings) | 8x H100 | ~$200-800, decided later |

## Open research questions (next passes)

1. Corpus mix ratios: code vs specs vs papers; dedup strategy (exact + near-dup MinHash).
2. Whether Ornith's own RL-heavy post-training reacts badly to midtraining (catastrophic forgetting risk higher on RL-tuned models; mitigations: lower LR, LoRA-only, replay).
3. Trace-set provenance verification for the MIT 2M set.
4. Domain-eval construction: auto-generate API-recall questions from rustdoc JSON output.
5. Whether the incoming RTX Pro 6000 (96GB) arrives before the 35B step, which would move most Runpod line items to $0.

Status: draft v1, written July 5 while the eval hardware was busy. Revised as each open question closes.
