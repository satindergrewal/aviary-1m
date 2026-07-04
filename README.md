---
license: mit
base_model: deepreinforce-ai/Ornith-1.0-35B
tags:
- gguf
- long-context
- yarn
- ornith
- llama.cpp
- ollama
---

# Ornith-1.0-35B — 1M Context GGUF (YaRN)

GGUF quantizations of [deepreinforce-ai/Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) with YaRN RoPE scaling **baked into the GGUF metadata** for a **1,048,576-token context window** — 4× the native 262,144. No fine-tuning: weights are bit-identical to the original release; only the rope-scaling metadata differs, so llama.cpp and Ollama apply the extension automatically with no extra flags.

## Validation: 50/50 needles, zero retrieval loss

Multi-needle retrieval test: 10 unique needles per run planted at depths 5%–95% in generated filler prose, retrieved in a single pass at temperature 0. Every needle recovered at every context length, including double and quadruple the native window.

![Needle-in-a-haystack results](niah_heatmap.png)

| Context | Needles | Cold prefill (M3 Max 128GB) |
|---|---|---|
| 32,768 | 10/10 | 38 s |
| 131,072 | 10/10 | 8.3 min |
| 262,144 (native) | 10/10 | 23 min |
| 524,288 (YaRN 2×) | 10/10 | 97 min |
| 1,048,576 (YaRN 4×) | 10/10 | ~6.8 h |

All rungs replicated 10/10 with independent haystacks and needle codes. The **Q4_K_M** file was additionally validated at 524K (10/10) loaded with **no rope flags at all** — confirming the baked metadata alone activates YaRN, which is exactly how Ollama loads it.

![Prefill throughput](prefill_speed.png)

Why 1M context is practical on this model at all: Ornith-1.0-35B is a hybrid — only ~10 of 40 layers are full attention (2 KV heads × 256 head dim), the rest linear attention. KV cache is ~20 KB/token: **~20 GB at 1M tokens**, versus hundreds of GB for a dense-attention 35B. Q8_0 weights + full 1M KV run in ~55–63 GB total.

Retrieval is necessary, not sufficient: NIAH shows the model can address the full window; it does not measure reasoning quality over long documents. Expect some quality tax from YaRN interpolation on tasks harder than retrieval.

## Files

| File | Size | Use |
|---|---|---|
| `ornith-1.0-35b-1M-Q8_0.gguf` | 35 GB | best quality; 64 GB+ unified memory |
| `ornith-1.0-35b-1M-Q4_K_M.gguf` | 20 GB | smaller memory footprint |

## Run with Ollama

```
# Modelfile
FROM ./ornith-1.0-35b-1M-Q4_K_M.gguf
PARAMETER num_ctx 1048576
```

```bash
ollama create ornith-1m -f Modelfile
ollama run ornith-1m
```

Scale `num_ctx` down to what your machine and task need — the KV cache is allocated for the context you request, not the 1M maximum.

## Run with llama.cpp

```bash
llama-server -m ornith-1.0-35b-1M-Q8_0.gguf -c 1048576 -np 1 --jinja
```

Note `-np 1`: parallel slots divide the context; the default would give each request only a quarter of the window.

The same result on the **original** GGUFs, via runtime flags:

```bash
llama-server -m ornith-1.0-35b-Q8_0.gguf -c 1048576 -np 1 --jinja \
  --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 \
  --override-kv qwen35moe.context_length=int:1048576
```

## Reproduce the benchmark

`niah_test.py` (in this repo) runs the multi-needle test against any OpenAI-compatible endpoint:

```bash
python3 niah_test.py --tokens 1048576 --url http://127.0.0.1:8000
```

It disables Ornith's thinking mode for deterministic scoring and varies haystacks by `--seed` (needed for cold timing measurements — llama-server's prompt cache makes identical reruns nearly instant). Raw results: `results.jsonl`, per-batch prefill timings: `prefill_timing.jsonl`.

## Credits & license

All model training credit to [DeepReinforce](https://huggingface.co/deepreinforce-ai) — this repo only changes rope-scaling metadata on their MIT-licensed release, following the YaRN factor-4 configuration recommended for the underlying Qwen3.5 architecture. MIT, same as upstream.
