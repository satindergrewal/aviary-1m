# RULER on llama.cpp: reproducibility bundle

NVIDIA RULER does not ship a llama.cpp backend, so we bridge it. Everything needed to reproduce our numbers:

- RULER commit: `38da79d79519` (github.com/NVIDIA/RULER)
- Data generation: RULER's `scripts/data/prepare.py`, unmodified. Tokenizer `unsloth/gemma-2-9b` (open byte-identical mirror of the gemma tokenizer), `--model_template_type base`, `--random_seed 42`, sample counts stated per published run.
- Scoring: RULER's `scripts/eval/evaluate.py`, unmodified except replacing one dead import (`nemo...read_manifest`, an ASR utility) with a 4-line jsonl reader; diff in `patches.md`.
- The bridge: `ruler_run_clean.py` (this folder, ~40 lines). Reads RULER's validation.jsonl, calls llama-server `/v1/chat/completions` at temperature 0, writes RULER's prediction record shape verbatim. Thinking ON by default (`--no-thinking` for the ablation); per-sample reasoning length recorded in the raw predictions.
- Orchestration: `ruler_clean.sh` (server flags included: `-np 1`, full context, `--jinja`).
- Serving: llama.cpp llama-server (build stated per published run).

Nothing model-facing is ours except the HTTP call. Claude/AI tooling authored the glue; NVIDIA's scripts computed every score.
