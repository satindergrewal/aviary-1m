# PLAN: V4-Vision Graft (Baseten recipe, pilot-first)

**Status 2026-08-02: PARKED on Satinder's word — resume from this doc.**
Owner direction: "we must try this on smaller text-only model first, make and setup our dataset
and tooling right, then see if we can do this for bigger model." GGUF-direct training assessment
explicitly requested by him and folded in (Stage 1 A/B below).

---

## 1. Objective

Give DeepSeek-V4-Flash-0731 (text-only, 284B total / 13B active MoE, 1M ctx) vision by grafting
a pretrained vision encoder onto the **frozen** LLM via a small **trained projector** — the exact
recipe proven by `baseten/GLM-5.2-Vision-NVFP4`. No safetensors-BF16 mega-run unless the
quantized-backward experiment (§5.3) fails.

### 1.1 The Baseten proof (why feasibility is retired)

From the GLM-5.2-Vision-NVFP4 model card:
- **Text backbone:** GLM-5.2 (744B / A40B active, MoE + MLA + DSA), frozen, byte-identical to
  `nvidia/GLM-5.2-NVFP4`.
- **Vision tower:** MoonViT-3d from Kimi-K2.6 (Moonshot) — 27 layers, 1152-dim, frozen,
  byte-identical.
- **Trained component ONLY:** PatchMerger MLP projector, `pre_norm → linear_1 → GELU → linear_2`,
  1152→4608→6144, **49.5M params**.
- **Throughput:** ≤4096 image tokens/image (16,384 MoonViT patches, 2×2 merge). Context 1M.
- Third-party graft (not Z.ai, not Moonshot). Card gives NO training detail (dataset/method
  undocumented) — our recipe must stand on our own eval.

### 1.2 ★ The donor is VIDEO-capable — verified 2026-08-03 (his deduction, confirmed)

Baseten's card presents images only, but the MoonViT-3d tower itself does video. Verified in
Kimi-K2.6's own `kimi_k25_vision_processing.py` (HF repo, custom code): imports
`VideoChunkInput, get_video_meta, navit_resize_video, real_sample_fps_and_max_num_frames,
timestamp_as_str`, frame-samples via `VideoReader`, and configures
`temporal_merge_kernel_size` (temporal patch merging). So K2.6 = image+video IN, and a graft
built on this tower can inherit video understanding **if the projector + training data cover
temporal sequences** — the tower alone doesn't grant it, the recipe must.
**Decision recorded (his direction 2026-08-03): we target the same — video-capable graft.**
Honest caveat: Qwen3-VL also does video, so video availability alone doesn't prove Baseten's
donor choice (their reasons may be temporal-merge design, embedding compatibility, or
familiarity); but it makes MoonViT strictly safer to commit to, and the donor A/B in §8 stays.

## 2. Target architecture (V4 graft)

- **Frozen LLM:** DeepSeek-V4-Flash, our served `UD-Q4_K_XL` GGUF (5 shards,
  `/mnt/nvme0/bigmodels/dsv4-flash/`). Training runs AGAINST THE QUANT WE SERVE — no
  distribution gap between train-time and serve-time base (the external-drafter lesson:
  components trained against a different precision collapse on the served quant).
- **Frozen vision encoder:** MoonViT-3d (default; see §8 open question on Qwen3-VL-ViT as donor).
- **Projector:** ~50M-param PatchMerger-style MLP, **explicitly designed to a shape mtmd can
  already serve** (constraint: zero llama.cpp code changes to run the graft).
- **Serve path:** llama.cpp `llama-server -m V4.gguf --mmproj graft-mmproj.gguf`
  (mmproj = vision encoder + projector in llama.cpp's mmproj layout).

## 3. Assets already in hand

| Asset | Where | Notes |
|---|---|---|
| V4-Flash UD-Q4_K_XL (served quant) | box `/mnt/nvme0/bigmodels/dsv4-flash/` | 5 shards, shard 1 metadata-only |
| transformers `deepseek_v4` (training-grade PyTorch forward) | pip latest (2026-05-02 contrib) | AutoModel registered; mHC + CSA/HCA + hash-layer bootstrap |
| Qwen3-VL-4B captioner (dataset engine) | box `/mnt/8tb/models/Qwen3-VL-4B-Instruct-Q8_0.gguf` + `Qwen3-VL-4B-mmproj-BF16.gguf` | Unsloth stock; serves via `PROJECTOR_TYPE_QWEN3VL` already in our trees |
| His Huihui Qwen3-VL-4B abliterated FP8 | box `/mnt/8tb/models/text_encoders/` | vision tower INTACT (315 tensors); convert later if wanted — our tree's converter lacks qwen3_vl (runtime HAS it); upstream converter pull needed |
| Qwen3-4B pilot base | box `/mnt/8tb/models/text_encoders/Qwen3-4B-UD-Q6_K_XL.gguf` (GGUF) + HF safetensors to fetch for BF16 arm | dense, 8 GB BF16 |
| GGUF parse/dequant code | box: `gguf_ablate.py`, `dsv4_build_minitarget.py` | proven Q8_0/Q4 dequant reads — the engine's front half |
| DeepSpec trainer venv | box `/mnt/data/DeepSpec/.venv` | torch present (DSpark head training runs there) |
| Mac llama-server builds (working) | `~/Documents/GitHub/llama.cpp-{dspark-metal,mackt,yaniss-dsv4-mac}/build/bin/` | ⚠ `~/Documents/GitHub/llama.cpp/build/bin` SEGFAULTS (mixed dylibs 0.15.3/0.17.0) — do not use |
| kv-paged V4 server (live) | box tmux `dsv4-srv`, :8331 | dsv4-kvpaged build, 4×1M, his sessions on it — GPU windows for training must not disturb |

**Storage rule (his correction 2026-08-02):** ALL box file work via `ssh satinder@192.168.0.101`
on `/mnt/8tb|/mnt/nvme0|/mnt/data`. `/Volumes/8tb` on the Mac is only an SMB mount — never use it
for pipeline I/O.

## 4. Stage 0 — dataset tooling (the piece that must be right)

### 4.1 Corpus spec (target 100–200K pairs for the real run; ~20–50K proves pilot convergence)

1. **Public general:** CC3M subset + COCO val2017 (clean licenses) via HF parquet — no scraping.
2. **Domain (our real use case): agentic render/UI verification.** Self-generated, zero copyright
   murk: headless-Chrome captures of three.js demos, WebGL scenes, dashboards, code editors,
   error pages, black-screen/broken renders (NEGATIVE examples — the verifier must say "black
   screen" when it is one), console-log screenshots. Target 5–10K.
3. **Documents/OCR (secondary):** public-doc pages, tables, diagrams — Qwen3-VL's strength,
   useful generality.
4. **★ VIDEO track (added 2026-08-03 per §1.2):** short clips (2–16 s) with action/state-change
   captions. Sources: (a) self-generated screen/WebGL recordings — same zero-copyright-murk
   trick as track 2, and it IS our domain (build-progress recordings, render-failure timelines);
   (b) license-clean public video-caption sets (verify per-set terms before use). Captioner:
   Qwen3-VL-4B handles video natively (its processor has the same resize_video path). Prompt:
   temporal narrative (what happens, in what order, what changes), 150–300 tokens.
   **Cost note: video pairs run ~3–6× image cost** (decode + N frames/sample through the
   captioner) and produce much longer training sequences (frames/merge × per-frame tokens),
   so the track is sized smaller: ~10–20K pairs in the real run, 1–2K for the pilot.

### 4.2 Caption daemon (box-side, resumable)

- Serves Qwen3-VL-4B on the box (CPU or GPU-window; NOT against Satinder's V4 sessions).
  `llama-server -m Qwen3-VL-4B-Instruct-Q8_0.gguf --mmproj Qwen3-VL-4B-mmproj-BF16.gguf -np 4`.
- Fixed structured captioning prompt (describe content, layout, state, errors; 150–250 tokens).
- Output `train_pairs.jsonl`: `{image_relpath, caption, source, w, h}` + `manifest.json`.
- **Error-churn lesson baked in:** resume cursor counts VALID rows only (client patch DeepSpec
  `8bc6843` pattern); transient errors retry; never count errors as progress.
- Throughput estimate: 4B Q8_0 on box CPU ≈ 8–15 t/s → ~10–20 s/caption × 4 workers ≈ 700–1,400/hr;
  on a GPU window ≈ 5–10× that. 50K pairs ≈ 1–3 days background.

## 5. Stage 1 — pilot graft: Qwen3-4B + MoonViT (box, $0)

### 5.1 Why this pilot

- 4B dense BF16 = 8 GB: full-precision forward+backward on one card, zero quant tricks → clean
  reference recipe.
- Same MoonViT donor + same projector class as the V4 graft → tooling carries 1:1.
- Produces the serving artifact too (mmproj for Qwen3-4B) → end-to-end proof in llama.cpp.

### 5.2 Trainer design (new code, DeepSpec venv)

- Parents frozen; only projector params update (AdamW, lr ~1e-3 with warmup/cosine, bf16).
- Loss: next-token CE on caption tokens given image tokens + a fixed instruction wrapper
  (match the wrapper to the chat template the graft will be SERVED with — template honesty).
- Batch: grad-accum to effective ~64–128 samples; activation checkpointing off (4B fits).
- Data: `train_pairs.jsonl` → (image, caption) with train/val split 98/2; val loss + sample
  captions logged every N steps.

### 5.3 ★ The backward-path A/B (his GGUF-direct question, answered with numbers)

Same data, same projector, two engines:
- **Arm A (reference):** frozen Qwen3-4B BF16, exact autograd (transformers qwen3).
- **Arm B (GGUF-direct):** frozen Qwen3-4B with linears swapped for custom autograd layers that
  dequantize GGUF Q4_K/Q6_K blocks on the fly (built on our `gguf_ablate.py`/minitarget dequant
  code). QLoRA mechanics sourced from GGUF, no bnb, no re-quantization.
- **Acceptance (our abliteration criterion):** Arm B within quant floor of Arm A — val-loss delta
  and blind caption-quality comparison. If B ≈ A → **Stage 2 runs ON THE BOX at 155 GB resident,
  $0 RunPod.** If B degrades → Stage 2 fallback §6.2.
- Numerics validation first: dequant-layer forward output vs llama.cpp's dequant on identical
  tensors, tolerance ≤ quant noise. This gate catches grid/unpack bugs before any training.

### 5.4 Exporter + serve proof

- Export MoonViT + trained projector → mmproj GGUF in llama.cpp layout (match the
  `PROJECTOR_TYPE` the projector shape was designed against, §2).
- Serve: `llama-server -m Qwen3-4B.gguf --mmproj pilot-mmproj.gguf` on Mac/box.
- Smoke: caption shot2.png-class inputs; compare against Qwen3-VL's captions of the same images.

### 5.5 Pilot eval (gates before Stage 2)

1. Caption quality: blind pairwise vs Qwen3-VL-4B captions (win/tie/loss, N=100).
2. **Render-verify domain (the metric we actually care about):** 50 held-out render screenshots
   (good renders, black screens, partial renders, console errors) — must classify state correctly
   ≥ 90%.
3. No refusal/loop regressions on text-only prompts (graft must not damage the base).
4. **★ Video probe (added 2026-08-03, §1.2):** 20 held-out short clips (2–16 s, self-generated
   render recordings + public license-clean) — must narrate the action in correct temporal order
   on ≥ 70% (video is the stretch gate: the tower supports it, the projector+data must earn it).

## 6. Stage 2 — V4-Flash graft

### 6.1 Primary path (if 5.3 Arm B passes): box, $0

- Engine: transformers `deepseek_v4` forward + GGUF-Q4-dequant autograd linears (Arm-B engine,
  arch-swapped). V4 stays frozen at UD-Q4_K_XL.
- Memory: 155 GB weights resident + ~20–30 GB activations (small batch, checkpointing) — fits a
  GPU window; NOT alongside his V4 sessions.
- Throughput unknown until measured (dequant-on-the-fly backward through 284B Q4); pilot gives
  the calibration point.

### 6.2 Fallback path (if Arm B fails): single-B200 RunPod, ~$200–900

- Official `deepseek-ai/DeepSeek-V4-Flash` FP8 safetensors + bnb/Unsloth QLoRA-style projector
  training on 1×B200 (180 GB: 4-bit 284B ≈ 145 GB + activations, tight but proven pattern for
  big-MoE QLoRA). 1–3 days at $8–12/hr.
- Only if Arm B fails AND Satinder approves the spend (RunPod rules: gate-on-binary,
  create-to-delete).

### 6.3 Explicitly rejected

- 4×B200/8×H100 BF16 fleet ($500–3,000): the lazy first estimate; superseded by §5.3/§6.1/§6.2.
- bnb NF4 conversion of our GGUF: double-quant (Q4_K→NF4) damage, no upside over §6.1/§6.2.

## 7. Risks / unknowns (honest list)

1. **Dataset size for convergence** — Baseten's is undocumented; LLaVA-style ablations say
   projector-only converges at ~100–500K pairs for strong captioning; pilot will read the curve
   at 20–50K and extrapolate.
2. **mtmd projector-shape constraint** — designing the projector to an existing mtmd type is the
   zero-code path; if quality demands a custom shape, add a small `PROJECTOR_TYPE` to mtmd
   (bounded, in-tree work, fleet rules apply).
3. **Image-token budget** — ≤4096 image tokens/image is the Baseten point; agentic screenshots
   are text-dense; may want higher effective resolution for the domain set (eval will show).
4. **mHC/CSA/HCA backward correctness** in transformers deepseek_v4 under dequant layers — the
   numerics gate (§5.3) is what de-risks this; watch for gradient spikes through the hash-boot
   layers.
5. **Graft-side text regression** — frozen base should be immune; gate 5.5.3 verifies.

## 8. Open questions

1. **Donor: MoonViT vs Qwen3-VL-ViT.** Baseten picked MoonViT (reasons undocumented; likely
   NaViT native-resolution handling, standard 1152-dim output matching mature projector
   ecosystems, Kimi document strength — and, verified 2026-08-03 (§1.2), **native VIDEO via
   temporal-merge**). Qwen3-VL's ViT is mtmd-supported in our tree too, arguably stronger on
   GUI/OCR, and also does video — so donor choice is NOT video-driven; it's
   temporal-merge-design + embedding compatibility + proven recipe. Default MoonViT for the
   pilot; donor A/B is a cheap later branch if caption evals favor it.
2. **Abliterated Qwen3-VL (his Huihui FP8)** as captioner upgrade — needs upstream converter
   class (scratch tree); only if refusals ever appear in captioning (unlikely).
3. **Vision-context budget in serving** — image tokens eat the 1M window; with kv-paged this is
   cheap VRAM-wise, but per-request latency grows; serve-time defaults TBD after pilot.
4. **★ Video token budget (new, §1.2):** a 16 s clip at ~8 fps sampled ≈ 128 frames →
   32 merged frames at temporal_merge 4 ≈ 32 × per-frame patch tokens. At the Baseten point
   (≤4096 tokens/image-frame) that's ~100K+ tokens per clip — full-fidelity video is a
   1M-context citizen, not a 64K one. Pilot clips: cap at 2–4 s / ~8–16 sampled frames until
   the serve-time budget is measured. Phasing decision (recorded): **image-first pilot, video
   track activates in Stage 2 prep** — pilot gates 5.5.1–3 decide the graft; 5.5.4 (video)
   is informative, not blocking.

## 9. Current state / next actions when unparked

- [x] Recipe proven externally (Baseten card analyzed)
- [x] Captioner chosen + downloaded + serve-verified on Mac (Qwen3-VL-4B, accurate GUI captions)
- [x] transformers deepseek_v4 confirmed (stage-2 forward exists)
- [x] Storage rule corrected (box-side only)
- [ ] Dataset builder implemented (box, `/mnt/data/vision-graft/`)
- [ ] Corpus download + domain render generator
- [ ] Video corpus: screen/WebGL recordings + license-clean clips, 1–2K pilot / 10–20K real (§4.1 track 4)
- [ ] Caption daemon run to 20–50K pairs (image) + 1–2K (video)
- [ ] Pilot trainer (§5.2) + numerics gate (§5.3)
- [ ] Arm A vs Arm B (§5.3) → decision
- [ ] Exporter + serve proof (§5.4) + eval gates (§5.5 incl. video probe 5.5.4)
- [ ] Stage 2 go/no-go with Satinder (incl. RunPod fallback decision + video track activation)
