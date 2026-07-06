<div align="center">

<img src="assets/aviary-logo-transparent.png" alt="aviary-1m" width="520"/>

**A flock of open models extended to a 1,048,576-token context with YaRN, then certified needle by needle: Ornith-1.0 (9B, 35B, 397B), the Gemma 4 uncensored trio, Qwen3.6-35B uncensored, and counting. Plus MTP speculative-decoding grafts and vision hookups. No fine-tuning. Verified everything.**

[![HF: 35B 1M GGUF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ornith--1.0--35B--1M--GGUF-ffd21e?logo=huggingface)](https://huggingface.co/satgeze/Ornith-1.0-35B-1M-GGUF)
[![HF: 9B 1M GGUF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ornith--1.0--9B--1M--GGUF-ffd21e?logo=huggingface)](https://huggingface.co/satgeze/Ornith-1.0-9B-1M-GGUF)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Context: 1M](https://img.shields.io/badge/context-1%2C048%2C576%20tokens-8A2BE2)]()
[![NIAH: 50/50](https://img.shields.io/badge/NIAH-50%2F50%20needles-brightgreen)]()
[![llama.cpp](https://img.shields.io/badge/llama.cpp-compatible-informational)]()
[![Ollama](https://img.shields.io/badge/Ollama-ready-informational)]()

</div>

## What this is

Ornith-1.0 models are Qwen3.5-family hybrids: only ~1 in 4 layers is full attention (the rest linear attention), so the KV cache at 1M tokens is ~20 GB (35B) / ~32 GB (9B) instead of hundreds of GB. That makes million-token context practical on consumer hardware. This repo holds the tooling and evidence; the ready-to-run GGUFs live on Hugging Face with the YaRN rope-scaling metadata baked in, so llama.cpp and Ollama apply the 4× extension automatically.

## Results

10 needles per run at depths 5-95%, single pass, temperature 0: perfect retrieval at every length through 1M, replicated with independent haystacks:

![NIAH heatmap](niah_heatmap.png)

| Context | Needles | Cold prefill (M3 Max 128GB) |
|---|---|---|
| 32,768 | 10/10 | 38 s |
| 131,072 | 10/10 | 8.3 min |
| 262,144 (native) | 10/10 | 23 min |
| 524,288 (YaRN 2×) | 10/10 | 97 min |
| 1,048,576 (YaRN 4×) | 10/10 | ~6.8 h |

![Prefill throughput](prefill_speed.png)

## Contents

| File | Purpose |
|---|---|
| `niah_test.py` | Multi-needle haystack test against any OpenAI-compatible endpoint |
| `bake_yarn.py` | Bake YaRN 1M metadata into any Qwen3.5-family GGUF |
| `make_charts.py` | Render the heatmap + throughput charts from results |
| `ornith9b_quants_pipeline.sh` | Full 9B quant ladder: download → bake → imatrix low-bit quants → upload |
| `results.jsonl`, `prefill_timing.jsonl` | Raw benchmark data |

## Vision

Ornith kept Qwen3.5's multimodal skeleton: attach the bundled mmproj vision tower (in each HF repo, extracted by bartowski) via `llama-server --mmproj` and the models see. Smoke-tested with accurate OCR on the 9B and 35B, including on the 1M-baked GGUFs. llama.cpp only for now; Ollama's qwen35 vision path is pending upstream.

## Model repos

| Model | Architecture | Long-context validation | Vision | Hugging Face |
|---|---|---|---|---|
| **Ornith-1.0-35B** | MoE, 3B active | **50/50 needles through 1M** (needle-perfect) | measured, mmproj included | [satgeze/Ornith-1.0-35B-1M-GGUF](https://huggingface.co/satgeze/Ornith-1.0-35B-1M-GGUF) |
| **Ornith-1.0-9B** | dense | perfect to 524K; 7/10 at 1M (mapped, isolation run in progress) | measured, mmproj included | [satgeze/Ornith-1.0-9B-1M-GGUF](https://huggingface.co/satgeze/Ornith-1.0-9B-1M-GGUF) |
| **Ornith-1.0-397B** | MoE | ladder pending (IQ1_M coherence-verified) | graft available, untested | [satgeze/Ornith-1.0-397B-1M-GGUF](https://huggingface.co/satgeze/Ornith-1.0-397B-1M-GGUF) |

Every repo is mirrored on ModelScope ([35B](https://www.modelscope.ai/models/satgeze/Ornith-1.0-35B-1M-GGUF) | [9B](https://www.modelscope.ai/models/satgeze/Ornith-1.0-9B-1M-GGUF) | [397B](https://www.modelscope.ai/models/satgeze/Ornith-1.0-397B-1M-GGUF)); the complete quant ladders live there permanently. All quants in every repo carry the YaRN 1M metadata baked in. Full verification details live on each model card. Everything together: the [Ornith 1M Context collection](https://huggingface.co/collections/satgeze/ornith-1m-context) on Hugging Face.

## Quick start

```bash
ollama run hf.co/satgeze/Ornith-1.0-35B-1M-GGUF:Q4_K_M
# then: /set parameter num_ctx 1048576   (needs ~20GB free for KV)
```

or llama.cpp, no flags needed beyond context size:

```bash
llama-server -m ornith-1.0-35b-1M-Q8_0.gguf -c 1048576 -np 1 --jinja
```

## Credits

All model training credit to [DeepReinforce](https://huggingface.co/deepreinforce-ai) (Ornith-1.0, MIT). This project only changes rope-scaling metadata and measures the result. MIT, same as upstream.
