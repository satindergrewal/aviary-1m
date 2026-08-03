# ds4-gates — standing gate suite for the ds4-ports lane

Every ported feature ships behind a `DS4P_*` env switch (default OFF) and passes its gate
before merge. Gates re-run identically forever — a gate is the artifact that makes a claim
citable. Pattern credit: ds4's speed-bench suite (Entrpi/ds4, gate-name-as-checklist).

## Conventions (P0-4, binding for every script here)

1. **Same-boot ABBA** for perf numbers: A/B/B/A within one server boot where possible; else
   pin clocks and record them in the output artifact.
2. **Exactness counters over log lines**: a gate PASSES only on a counter or byte-compared
   artifact (hashed token streams, /metrics counters proving the path executed). Never on an
   announcement line.
3. **Provenance marks** in every result file: VERIFIED-BY-ME / control-flow-read / arithmetic /
   inherited.
4. **Kill-switch discipline**: each gate runs its feature ON and OFF in the same boot;
   OFF-path must be byte-identical to the pre-feature baseline.
5. Output: `results/<gate>-<yyyymmdd-hhmm>.txt` (this dir, gitignored? NO — results are the
   record, commit them).
6. Box etiquette: exact-PID kills only; never touch dsv4-srv/regen/K3 processes; GPU gates
   only in announced windows (coordinate in #main).

## Gates

| script | item | needs GPU | status |
|---|---|---|---|
| `abort_paths_gate.sh` | P0-3 dead-client abort (STEP 0: measure current tree) | yes (any serve) | WRITTEN, awaiting window |
| `reval_gate.sh` | P0-2 state↔claim binding | CPU ok (small model) | stub |
| `quench_gate.sh` | P0-1 yield-quench 3-regime A/B | yes | stub |
| `hybrid_paged_gate.sh` | 3b inkling/K3 paged serve | yes | stub |
